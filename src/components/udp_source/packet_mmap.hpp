#pragma once

#include "aligned_alloc_resource.hpp"

#include <composite/timestamp.hpp>
#include <cstdint>
#include <readerwriterqueue.h>
#include <string_view>
#include <thread>

namespace udp {

class packet_mmap {
public:
    packet_mmap(std::string_view interface, std::string_view addr, uint16_t port);
    ~packet_mmap();

    auto start() -> void;
    auto stop() -> void;
    auto get_data(std::shared_ptr<std::pmr::vector<uint8_t>>&) -> bool;
    
private:
    auto receive(std::stop_token token) -> void;
    void log_stats();
    void get_ethtool_stats(const std::string& iface);

    int m_socket{-1};
    int m_join_socket{-1};
    void* m_ring{nullptr};
    std::jthread m_recv_thread;
    std::jthread m_stat_thread;
    aligned_alloc_resource m_upstream_alloc;
    std::pmr::synchronized_pool_resource m_pool_resource;
    std::unique_ptr<moodycamel::ReaderWriterQueue<std::shared_ptr<std::pmr::vector<uint8_t>>>> m_queue;

    uint16_t m_pkt_count{};

}; // class packet_mmap

} // namespace udp
