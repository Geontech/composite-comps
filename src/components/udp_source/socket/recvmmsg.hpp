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

class recvmmsg : public interface {
    using queue_t = moodycamel::ReaderWriterQueue<buffer_ptr_t>;
public:
    recvmmsg(const config& config);
    ~recvmmsg() final;

    auto start_recv() -> void override;
    auto stop_recv() -> void override;
    auto get_data(buffer_ptr_t&) -> bool override;
    auto get_stats() -> statistics override;

private:
    auto receive(std::stop_token token) -> void;

    int m_socket{-1};
    std::jthread m_recv_thread;
    std::unique_ptr<queue_t> m_queue;
    std::size_t m_batch_size{128};
    std::size_t m_msg_size{};
    std::atomic<uint32_t> m_pkts_recvd{};

}; // class packet_mmap

} // namespace udp
