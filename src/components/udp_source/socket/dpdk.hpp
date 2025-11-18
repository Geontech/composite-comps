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
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include "interface.hpp"

#include <cstdint>
#include <string>
#include <thread>
#include <atomic>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <vector>
#include <chrono>
#include <spdlog/spdlog.h>

// Forward declarations to avoid pulling in DPDK headers in header file
struct rte_mempool;
struct rte_mbuf;

// Forward declaration for IGMP manager
namespace udp::igmp {
    class manager;
}

namespace udp {

class dpdk final : public interface {
public:
    struct config {
        std::shared_ptr<spdlog::logger> logger;

        // High-level user configuration (matches other socket types)
        std::string interface;            // Interface name (e.g., "eth0", "mlx5_0")
        std::string ip_addr;              // Expected destination IP (enables filtering + IGMP for multicast)
        uint16_t port = 0;                // Expected UDP port (0 = accept all)

        // DPDK-specific optional overrides
        std::optional<uint16_t> port_id;  // Override auto-detection from interface name
        std::optional<uint16_t> queue_id; // Override auto-allocation
        std::string mempool_name = "mbuf_pool";
        uint16_t burst_size = 32;

        // IGMP configuration (automatically enabled for multicast destinations)
        std::string src_ip;               // Source IP for IGMP (required for multicast)
        bool igmp_respond_to_queries = true;
    };

    explicit dpdk(const config& cfg);
    ~dpdk() override;

    // Disable copy/move
    dpdk(const dpdk&) = delete;
    dpdk& operator=(const dpdk&) = delete;

    auto start_recv(output_port_t* port) -> void override;
    auto stop_recv() -> void override;
    auto get_stats() -> std::map<std::string, std::string> override;

private:
    auto receive(std::stop_token token) -> void;

    // Packet parsing helper
    struct udp_payload {
        uint8_t* data;
        uint16_t length;
        bool valid;
    };
    auto extract_udp_payload(rte_mbuf* mbuf) -> udp_payload;

    config m_config;
    output_port_t* m_out_port{nullptr};

    // Resolved DPDK identifiers (for cleanup)
    uint16_t m_resolved_port_id{0};
    uint16_t m_resolved_queue_id{0};
    bool m_queue_allocated{false};

    // DPDK resources (obtained from framework)
    rte_mempool* m_mempool{nullptr};

    // IGMP manager (optional)
    std::unique_ptr<igmp::manager> m_igmp_mgr;

    // Track multicast MAC addresses added to NIC filter (6 bytes for Ethernet MAC)
    std::array<uint8_t, 6> m_mcast_mac{};
    bool m_mcast_mac_added{false};
    std::optional<uint32_t> m_dst_ip_be;
    std::chrono::steady_clock::time_point m_last_igmp_poll;

    // Threading
    std::jthread m_recv_thread;
    std::atomic<bool> m_running{false};
    std::vector<rte_mbuf*> m_rx_burst;

    // Statistics
    std::atomic<uint64_t> m_pkts_dropped{0};  // Invalid/filtered packets
    std::atomic<uint64_t> m_bursts_empty{0};   // Number of empty rx_burst calls
    std::atomic<uint64_t> m_igmp_queries{0};   // IGMP queries received
};

} // namespace udp
