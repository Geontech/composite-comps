#pragma once
#include "readerwriterqueue.h"

#include <pcap/pcap.h>
#include <thread>
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

class PcapWriterThread {
public:
    explicit PcapWriterThread(uint64_t rotate_every_packets)
        : m_rotate_every(rotate_every_packets),
        m_queue{8192},
        m_counter(0),
        m_stop(false) {
        open_new_file();
        m_thread = std::thread(&PcapWriterThread::run, this);
    }

    ~PcapWriterThread() {
        stop();
    }
    
    void enqueue(std::unique_ptr<uint8_t[]> pkt_data, uint32_t pkt_len) {
        m_queue.enqueue(PcapPacket(std::move(pkt_data), pkt_len));
    }
    
    

    void stop() {
        if (m_stop.exchange(true)) return;
        // m_cv.notify_one();
        // if (m_thread.joinable()) {
        //     m_thread.join();
        // }

        close_file();
    }

private:
    std::thread m_thread;
    // std::mutex m_mutex;
    // std::condition_variable m_cv;
    // std::queue<PcapPacket> m_queue;
    moodycamel::ReaderWriterQueue<PcapPacket> m_queue;

    
    std::atomic<bool> m_stop;

    pcap_t* m_pcap = nullptr;
    pcap_dumper_t* m_dumper = nullptr;

    uint64_t m_rotate_every;
    uint64_t m_counter;

    void run() {
        pthread_setname_np(pthread_self(), "pcap_writer");
        PcapPacket pkt;
        while (!m_stop.load()) {
            if (!m_queue.try_dequeue(pkt)) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
    
            maybe_rotate();
    
            struct pcap_pkthdr hdr;
            struct timeval now;
            gettimeofday(&now, nullptr);
            hdr.ts = now;
            hdr.caplen = pkt.length;
            hdr.len = pkt.length;
    
            pcap_dump((u_char*)m_dumper, &hdr, pkt.buffer.get());
            ++m_counter;
        }
    }
     
    
    void maybe_rotate() {
        if (m_counter >= m_rotate_every) {
            close_file();
            open_new_file();
            m_counter = 0;
        }
    }

    void open_new_file() {
        m_pcap = pcap_open_dead(DLT_EN10MB, 65535);
        if (!m_pcap) throw std::runtime_error("Failed to open pcap dead");

        std::string filename = generate_filename();
        m_dumper = pcap_dump_open(m_pcap, filename.c_str());
        if (!m_dumper) {
            pcap_close(m_pcap);
            throw std::runtime_error("Failed to open pcap file: " + filename);
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
    }

    std::string generate_filename() {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "/data/output/" << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S") << ".pcap";
        return ss.str();
    }
};
