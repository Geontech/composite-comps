#pragma once
#include <atomic>
#include <chrono>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>

class PcapDirectoryCleanerThread {
public:
    PcapDirectoryCleanerThread(std::string path, uint64_t max_bytes, uint32_t interval_sec)
        : m_path(std::move(path)),
          m_max_bytes(max_bytes),
          m_interval(interval_sec),
          m_stop(false),
          m_thread(&PcapDirectoryCleanerThread::run, this) {}

    ~PcapDirectoryCleanerThread() {
        stop();
    }

    void stop() {
        if (m_stop.exchange(true)) return;

    }

private:
    std::string m_path;
    uint64_t m_max_bytes;
    uint32_t m_interval;
    std::atomic<bool> m_stop;
    std::jthread m_thread;

    void run() {
        pthread_setname_np(pthread_self(), "pcap_cleaner");
        while (!m_stop.load()) {
            try {
                std::this_thread::sleep_for(std::chrono::seconds(m_interval));
                std::vector<std::filesystem::directory_entry> pcap_files;
                uint64_t total_size = 0;

                for (const auto& entry : std::filesystem::directory_iterator(m_path)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".pcap") {
                        pcap_files.push_back(entry);
                        total_size += entry.file_size();
                    }
                }

                if (total_size <= m_max_bytes) {
                    continue;}

                // Sort by last write time (oldest first)
                std::sort(pcap_files.begin(), pcap_files.end(),
                          [](const auto& a, const auto& b) {
                              return std::filesystem::last_write_time(a) < std::filesystem::last_write_time(b);
                          });

                for (const auto& file : pcap_files) {
                    if (total_size <= m_max_bytes) break;

                    uint64_t sz = file.file_size();
                    std::filesystem::remove(file);
                    total_size -= sz;
                }

            } catch (const std::exception& e) {
                continue;
            }
        }
    }
};
