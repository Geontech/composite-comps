/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * This file is part of composite-comps.
 *
 * composite-comps is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * composite-comps is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include "interface.hpp"
#include "pmr/ring_resource.hpp"
#include "processing_queue.hpp"
#include "pcap_cleaner_thread.hpp"
#include "pcap_writer_thread.hpp"

#include <atomic>
#include <composite/timestamp.hpp>
#include <cstdint>
#include <rte_ethdev.h>
#include <string_view>
#include <thread>

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

    int m_join_socket{-1};
    std::jthread m_recv_thread;
    queue_t m_queue;
    uint32_t m_frame_size{};
    uint32_t m_frame_count{};
    ring_resource m_resource;
    bool m_eth_dev_configured{false};
    struct rte_mempool* m_mbuf_pool = nullptr;
    struct rte_eth_conf m_port_conf;
    uint16_t m_selected_port;
    std::string m_interface;
    uint16_t m_port;
    uint32_t m_ip_addr_le;
    std::string m_ip_addr;
    // std::string m_transport;
    uint16_t m_rx_ring_size{};
    uint32_t m_num_mbufs{};
    uint32_t m_mbuf_cache_size{};
    uint32_t m_burst_size{};
    std::string m_socket_mem{};

    // Stats
    std::atomic<uint32_t> m_pkts_recvd{};
    std::atomic<uint64_t> m_cycles{};
    std::atomic<uint32_t> m_iterations{};
    std::atomic<uint32_t> m_pkts_in_burst{};
    std::atomic<uint32_t> m_no_queue{};
    struct rte_eth_stats m_dpdk_stats;

    // Pcap
    bool m_write_pcap{false};
    std::unique_ptr<PcapWriterThread> m_pcap_writer_thread;
    std::unique_ptr<PcapDirectoryCleanerThread> m_cleaner_thread;

}; // class dpdk_udp

} //namespace udp
