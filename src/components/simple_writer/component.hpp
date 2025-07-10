#include <aligned_mem.hpp>
#include <composite/component.hpp>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <complex>
#include <sys/uio.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <nlohmann/json.hpp>
#include <fstream>
#include <helpers.hpp>


template <typename T>
class simple_file_writer : public composite::component {
    using input_t = aligned::aligned_mem<T>;
    using input_port_t = composite::input_port<std::unique_ptr<input_t>>;
    using output_port_t = composite::output_port<std::unique_ptr<input_t>>;

public:
    explicit simple_file_writer(std::string_view postfix)
        : composite::component("simple_file_writer"), 
          m_postfix(postfix),
          m_output_dir("/data/output"),
          m_stream_id(0),
          m_rotate_interval(1000),
          write_buf(WRITE_BUF_SIZE, 0)
    {
        add_port(&m_in_port);
        add_port(&m_out_port);

    }

    void initialize() override {
        std::cerr << "Initializing and creating directory: " << m_output_dir << "\n";
        std::filesystem::create_directories(m_output_dir);
        open_file();
        writer_thread = std::thread(&simple_file_writer::writer_loop, this);
    }
    
    auto process() -> composite::retval override {
        using enum composite::retval;
    
        auto [data, ts, meta] = m_in_port.get_data();
        if (!data) return NOOP;
        if (meta.has_value()){
            logger()->trace("received metadata:\n{}", meta->to_string());
            m_meta = sigmf::build_sigmf_json(meta.value());
            m_out_port.send_metadata(meta.value());
            logger()->trace("sent metadata:\n{}", meta->to_string());
        }
        
        auto data_clone = std::make_unique<input_t>(*data);

        WriteJob job{ts.seconds, ts.picoseconds, std::move(data)};
    
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            write_queue.push(std::move(job));
        }
        queue_cv.notify_one();
        return NORMAL;
    }

    void writer_loop() {
        composite::timestamp first_ts_this_file{};
        bool first_sample_written = false;
    
        while (true) {
            WriteJob job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this] { return stop_writer || !write_queue.empty(); });
    
                if (stop_writer && write_queue.empty()) break;
    
                job = std::move(write_queue.front());
                write_queue.pop();
            }
    
            // Save the first timestamp after file rotation
            if (!first_sample_written) {
                first_ts_this_file.seconds = job.seconds;
                first_ts_this_file.picoseconds = job.picoseconds;
                first_sample_written = true;
            }
    
            // Only write the data, not the timestamp
            struct iovec iov[1];
            iov[0].iov_base = job.data->data();
            iov[0].iov_len = job.data->size() * sizeof(T);
    
            ssize_t bytes_written = ::writev(m_fd, iov, 1);
            if (bytes_written < 0) perror("writev");
    
            m_written++;
            if (m_written >= m_rotate_interval) {
                // Build and write SigMF metadata
                write_sigmf_meta(first_ts_this_file);
                rotate_file();
                first_sample_written = false;
            }
    
            m_out_port.send_data(std::move(job.data), composite::timestamp{job.seconds, job.picoseconds});
        }
    }
    
    
    ~simple_file_writer() override {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop_writer = true;
        }
        queue_cv.notify_all();
        if (writer_thread.joinable()) writer_thread.join();
    
        if (m_fd >= 0) ::close(m_fd);
    }
    

private:
    struct WriteJob {
        uint32_t seconds;
        uint64_t picoseconds;
        std::unique_ptr<input_t> data;
    };
    void open_file() {
        std::string base_filename = std::format("{}_stream{}_{}", m_postfix, m_stream_id, m_file_index++);
        auto data_filepath = std::filesystem::path(m_output_dir) / (base_filename + ".sigmf-data");
        auto meta_filepath = std::filesystem::path(m_output_dir) / (base_filename + ".sigmf-meta");
    
        // Open the binary data file
        m_fd = ::open(data_filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (m_fd < 0) {
            throw std::runtime_error("Failed to open file: " + data_filepath.string());
        }
        m_written = 0;
    
    
        // Write to JSON file
        std::ofstream meta_out(meta_filepath);
        if (!meta_out.is_open()) {
            throw std::runtime_error("Failed to open meta file: " + meta_filepath.string());
        }
        meta_out << m_meta.dump(4);
    }
    void write_sigmf_meta(const composite::timestamp& ts) {
        std::stringstream datetime_stream;
        uint64_t total_pico = static_cast<uint64_t>(ts.seconds) * 1'000'000'000'000ULL + ts.picoseconds;
    
        // Convert to ISO 8601 string (you can refine this logic)
        std::time_t secs = ts.seconds;
        std::tm tm = *std::gmtime(&secs);
    
        char datetime_buf[64];
        std::strftime(datetime_buf, sizeof(datetime_buf), "%FT%TZ", &tm);
        std::string iso_time(datetime_buf);
    
        nlohmann::json meta = m_meta;  // stored from earlier metadata
        meta["captures"] = {{
            {"core:sample_start", 0},
            {"core:datetime", iso_time}
        }};
    
        std::filesystem::path meta_filepath = std::filesystem::path(m_output_dir) /
            std::format("{}_stream{}_{}.sigmf-meta", m_postfix, m_stream_id, m_file_index - 1);
    
        std::ofstream out(meta_filepath);
        out << meta.dump(2) << std::endl;
    }
    

    void rotate_file() {
        if (m_fd >= 0) {
            ::close(m_fd);
            m_fd = -1;
        }
        open_file();
    }
    static constexpr std::size_t WRITE_BUF_SIZE = 8 * 1024 * 1024;
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    std::string m_output_dir;
    std::string m_postfix;
    std::uint32_t m_stream_id;
    std::uint32_t m_rotate_interval;
    std::uint32_t m_written = 0;
    std::uint32_t m_file_index = 0;
    std::thread writer_thread;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::queue<WriteJob> write_queue;
    bool stop_writer = false;
    int m_fd = -1;
    std::vector<char> write_buf;
    nlohmann::json m_meta;
};
