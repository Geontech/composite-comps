#pragma once

#include "interface.hpp"
#include "pmr/aligned_alloc_resource.hpp"

#include <atomic>
#include <composite/timestamp.hpp>
#include <cstdint>
#include <readerwriterqueue.h>
#include <string_view>
#include <thread>

namespace udp {

class packet_mmap : public interface {
    using queue_t = moodycamel::ReaderWriterQueue<buffer_ptr_t>;
public:
    packet_mmap(std::string_view interface, std::string_view addr, uint16_t port);
    ~packet_mmap() final;

    auto start() -> void override;
    auto stop() -> void override;
    auto get_data(buffer_ptr_t&) -> bool override;
    auto get_stats() -> statistics override;

private:
    auto receive(std::stop_token token) -> void;

    int m_socket{-1};
    int m_join_socket{-1};
    void* m_ring{nullptr};
    std::jthread m_recv_thread;
    aligned_alloc_resource m_upstream_alloc;
    std::pmr::synchronized_pool_resource m_pool_resource;
    std::unique_ptr<queue_t> m_queue;
    std::atomic<uint32_t> m_pkts_recvd{};

}; // class packet_mmap

} // namespace udp
