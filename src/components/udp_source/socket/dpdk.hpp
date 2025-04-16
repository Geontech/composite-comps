#pragma once

#include "interface.hpp"
#include "pmr/ring_resource.hpp"
#include "processing_queue.hpp"

#include <atomic>
#include <composite/timestamp.hpp>
#include <cstdint>
#include <string_view>
#include <thread>
#include <rte_ethdev.h>

namespace udp {

class dpdk_udp : public interface {
    using queue_t = processing_queue<buffer_ptr_t>;
public:
    dpdk_udp(const config& config);
    ~dpdk_udp() final;

    auto start_recv() -> void override;
    auto stop_recv() -> void override;
    auto get_data(std::shared_ptr<buffer_t>&) -> bool override;
    auto get_stats() -> statistics override;
    
private:
    auto receive(std::stop_token token) -> void;

    int m_socket{-1};
    int m_join_socket{-1};
    void* m_ring{nullptr};
    std::jthread m_recv_thread;
    queue_t m_queue;
    uint32_t m_frame_size{};
    uint32_t m_frame_count{};
    ring_resource m_resource;
    std::atomic<uint32_t> m_pkts_recvd{};
    std::atomic<uint64_t> m_cycles{};
    std::atomic<uint32_t> m_iterations{};
    std::atomic<uint32_t> m_pkts_in_burst{};
    std::atomic<uint32_t> m_no_queue{};
    struct rte_eth_stats m_dpdk_stats;
    bool m_eth_dev_configured{false};
    struct rte_mempool* m_mbuf_pool;
    struct rte_eth_conf m_port_conf;
    uint16_t m_selected_port;
    struct udp::config m_config;

}; // class dpdk_udp

} //namespace udp