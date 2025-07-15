#pragma once
#include "readerwriterqueue.h"
#include "interface.hpp"
#include <pcap/pcap.h>
#include <pthread.h>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <utility>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <cstring>

struct PcapPacket {
    std::unique_ptr<uint8_t[]> buffer;
    uint32_t length;
    PcapPacket() : buffer(nullptr), length(0) {}

    PcapPacket(std::unique_ptr<uint8_t[]> src, uint32_t len)
        : buffer(std::move(src)), length(len) {}

    // Disallow copying
    PcapPacket(const PcapPacket&) = delete;
    PcapPacket& operator=(const PcapPacket&) = delete;

    // Allow moving
    PcapPacket(PcapPacket&&) noexcept = default;
    PcapPacket& operator=(PcapPacket&&) noexcept = default;
};

struct PcapFilePaths {
    std::string final_path;
    std::string temp_path;
};


class PcapWriterThread {
public:
    explicit PcapWriterThread(const udp::config& config)
      : 
    //   m_rotate_every(rotate_every_packets),
      m_queue{8192},
      m_counter(0),
      m_stop(false)
    //   m_write_dir(output_dir) 
      {
        m_write_dir = config.write_directory.value_or("/data/output/");
        m_rotate_every = config.packets_per_pcap.value_or(100000);
        open_new_file();
        m_thread = std::jthread(&PcapWriterThread::run, this);
    }

    ~PcapWriterThread() {
        stop();
    }
    
    void enqueue(std::unique_ptr<uint8_t[]> pkt_data, uint32_t pkt_len) {
        m_queue.enqueue(PcapPacket(std::move(pkt_data), pkt_len));
    }
    
    void stop() {
        if (m_stop.exchange(true)) return;
        close_file();
    }

private:
    std::jthread m_thread;
    moodycamel::ReaderWriterQueue<PcapPacket> m_queue;
    std::atomic<bool> m_stop;

    pcap_t* m_pcap = nullptr;
    pcap_dumper_t* m_dumper = nullptr;

    uint64_t m_rotate_every;
    uint64_t m_counter;
    PcapFilePaths m_file_paths;
    std::string m_write_dir; 

    void run() {
        pthread_setname_np(pthread_self(), "pcap_writer");
        PcapPacket pkt;
        struct pcap_pkthdr hdr;
        struct timeval now;
        gettimeofday(&now, nullptr);
        hdr.ts = now;
        while (!m_stop.load()) {
            if (!m_queue.try_dequeue(pkt)) {
                std::this_thread::sleep_for(std::chrono::microseconds(6));
                continue;
            }
    
            // maybe_rotate();
            if (m_counter >= m_rotate_every) {
                close_file();
                open_new_file();
                m_counter = 0;
            }
    
            hdr.caplen = pkt.length;
            hdr.len = pkt.length;
    
            pcap_dump((u_char*)m_dumper, &hdr, pkt.buffer.get());
            ++m_counter;
        }
    }

    void open_new_file() {
        m_file_paths = generate_file_paths();
    
        m_pcap = pcap_open_dead(DLT_EN10MB, 65535);
        if (!m_pcap) throw std::runtime_error("Failed to open pcap dead");
    
        m_dumper = pcap_dump_open(m_pcap, m_file_paths.temp_path.c_str());
        if (!m_dumper) {
            pcap_close(m_pcap);
            throw std::runtime_error("Failed to open pcap file: " + m_file_paths.temp_path);
        }
    }

    void close_file() {
        if (m_dumper) {
            pcap_dump_close(m_dumper);
            m_dumper = nullptr;
        }
        if (m_pcap) {
            pcap_close(m_pcap);
            m_pcap = nullptr;
        }
    
        if (!m_file_paths.temp_path.empty() && !m_file_paths.final_path.empty()) {
            std::rename(m_file_paths.temp_path.c_str(), m_file_paths.final_path.c_str());
        }
    
        m_file_paths = {};
    }

    PcapFilePaths generate_file_paths() {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
        std::string base = ss.str();
    

        return {
            m_write_dir + base + ".pcap",       // final_path
            m_write_dir + "tmp_" + base         // temp_path
        };
    }

};
