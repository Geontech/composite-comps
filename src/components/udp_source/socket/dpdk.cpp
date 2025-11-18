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

#include "dpdk.hpp"
#include "igmp.hpp"
#include "net/utils.hpp"

#include <composite/buffers/buffer.hpp>
#include <composite/buffers/external_buffer.hpp>
#include <composite/dpdk/manager.hpp>

#include <spdlog/spdlog.h>

#include <rte_mbuf.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_udp.h>
#include <rte_cycles.h>

#include <chrono>
#include <thread>
#include <cstring>
#include <memory>
#include <utility>
#include <arpa/inet.h>

namespace {
struct mbuf_release {
    rte_mbuf* mbuf{};
    void operator()() {
        if (mbuf) {
            rte_pktmbuf_free(mbuf);
            mbuf = nullptr;
        }
    }
};
} // namespace

namespace udp {

dpdk::dpdk(const config& cfg) :
  interface(cfg.logger),
  m_config(cfg) {
    // Get DPDK manager and acquire resources
    auto& dpdk_mgr = composite::dpdk::manager::instance();

    if (!dpdk_mgr.is_initialized()) {
        auto msg = "DPDK not initialized by framework";
        m_logger->error(msg);
        throw std::runtime_error(msg);
    }

    // Resolve port_id from interface name (if not explicitly provided)
    if (m_config.port_id.has_value()) {
        // User explicitly specified port_id
        m_resolved_port_id = m_config.port_id.value();
        m_logger->debug("using explicitly configured DPDK port_id: {}", m_resolved_port_id);
    } else {
        // Auto-resolve from interface name
        auto port_id_opt = dpdk_mgr.get_port_id_for_interface(m_config.interface);
        if (!port_id_opt.has_value()) {
            auto msg = std::format("interface '{}' not found in DPDK configuration", m_config.interface);
            m_logger->error(msg);
            throw std::runtime_error(msg);
        }
        m_resolved_port_id = port_id_opt.value();
        m_logger->info("auto-resolved interface '{}' to DPDK port_id: {}", m_config.interface, m_resolved_port_id);
    }

    // Resolve queue_id (auto-assign if not specified)
    if (m_config.queue_id.has_value()) {
        // User explicitly specified queue_id
        m_resolved_queue_id = m_config.queue_id.value();

        // Validate and allocate
        if (!dpdk_mgr.is_queue_available(m_config.interface, m_resolved_queue_id)) {
            auto msg = std::format("queue {} on interface '{}' is not available or already allocated",
                                  m_resolved_queue_id, m_config.interface);
            m_logger->error(msg);
            throw std::runtime_error(msg);
        }

        if (!dpdk_mgr.allocate_queue(m_config.interface, m_resolved_queue_id)) {
            auto msg = std::format("failed to allocate queue {} on interface '{}'",
                                  m_resolved_queue_id, m_config.interface);
            m_logger->error(msg);
            throw std::runtime_error(msg);
        }

        m_logger->debug("using explicitly configured DPDK queue_id: {}", m_resolved_queue_id);
    } else {
        // Auto-assign next available queue
        auto queue_id_opt = dpdk_mgr.allocate_next_available_queue(m_config.interface);
        if (!queue_id_opt.has_value()) {
            auto msg = std::format("no available queues on interface '{}'", m_config.interface);
            m_logger->error(msg);
            throw std::runtime_error(msg);
        }
        m_resolved_queue_id = queue_id_opt.value();
        m_logger->info("auto-assigned DPDK queue_id: {} on interface '{}'",
                      m_resolved_queue_id, m_config.interface);
    }
    m_queue_allocated = true;

    // Get mempool from framework
    m_mempool = dpdk_mgr.get_mempool(m_config.mempool_name);
    if (!m_mempool) {
        auto msg = std::format("failed to get mempool '{}'", m_config.mempool_name);
        m_logger->error(msg);
        throw std::runtime_error(msg);
    }

    // Verify port is configured
    if (!dpdk_mgr.is_port_configured(m_resolved_port_id)) {
        auto msg = std::format("DPDK port '{}' is not configured", m_resolved_port_id);
        m_logger->error(msg);
        throw std::runtime_error(msg);
    }

    if (!m_config.ip_addr.empty()) {
        struct in_addr dst{};
        if (inet_pton(AF_INET, m_config.ip_addr.c_str(), &dst) == 1) {
            m_dst_ip_be = dst.s_addr;
        } else {
            m_logger->warn("invalid ip_addr '{}' specified; destination filtering disabled",
                          m_config.ip_addr);
        }
    }

    // Automatically enable IGMP for multicast destinations if src_ip is provided
    auto is_multicast = !m_config.ip_addr.empty() && net::is_ipv4_multicast(m_config.ip_addr);

    if (is_multicast && !m_config.src_ip.empty()) {
        m_logger->debug("multicast destination detected, enabling IGMP");

        // Auto-detect source MAC address - required for IGMP
        rte_ether_addr local_mac;
        if (rte_eth_macaddr_get(m_resolved_port_id, &local_mac) != 0) {
            auto msg = std::format("failed to auto-detect MAC address for port {}", m_resolved_port_id);
            m_logger->error(msg);
            throw std::runtime_error(msg);
        }
        m_logger->info("auto-detected source MAC: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                      local_mac.addr_bytes[0], local_mac.addr_bytes[1],
                      local_mac.addr_bytes[2], local_mac.addr_bytes[3],
                      local_mac.addr_bytes[4], local_mac.addr_bytes[5]);

        // Convert IP string to network byte order uint32_t for IGMP
        struct in_addr addr;
        if (inet_pton(AF_INET, m_config.src_ip.c_str(), &addr) != 1) {
            m_logger->error("invalid src_ip format: {}", m_config.src_ip);
            throw std::runtime_error("IGMP enabled but src_ip is invalid");
        }
        uint32_t local_ipv4_be = addr.s_addr;  // already in network byte order

        // IGMP uses TX queue 0 for control traffic (separate from data)
        igmp::config igmp_cfg{
            .port_id = m_resolved_port_id,
            .tx_queue = 0,  // Queue 0 for control traffic
            .mbuf_pool = m_mempool,
            .local_mac = local_mac,
            .local_ipv4_be = local_ipv4_be,
            .vlan_id = std::nullopt,  // No VLAN for now
            .robustness_variable = 2,
            .unsolicited_report_interval_sec = 10.0,
            .respond_to_queries = m_config.igmp_respond_to_queries
        };

        m_igmp_mgr = std::make_unique<igmp::manager>(igmp_cfg);

        // Send IGMP Join for the multicast group
        m_logger->info("sending IGMP join for multicast group {}", m_config.ip_addr);
        if (!m_igmp_mgr->join_group(m_config.ip_addr)) {
            m_logger->error("failed to join multicast group {} (invalid or not multicast)",
                           m_config.ip_addr);
            throw std::runtime_error("IGMP join failed");
        } else {
            // Configure NIC to accept multicast MAC without clobbering other filters
            m_mcast_mac = net::create_multicast_mac(m_config.ip_addr);

            rte_ether_addr dpdk_mac;
            std::memcpy(dpdk_mac.addr_bytes, m_mcast_mac.data(), m_mcast_mac.size());

            int ret = rte_eth_dev_mac_addr_add(m_resolved_port_id, &dpdk_mac, 0);
            if (ret == 0) {
                m_mcast_mac_added = true;
                m_logger->info("set multicast address filter: {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                                m_mcast_mac[0], m_mcast_mac[1], m_mcast_mac[2],
                                m_mcast_mac[3], m_mcast_mac[4], m_mcast_mac[5]);
            } else {
                m_logger->error("failed to set multicast address filter: {} (ret={})",
                                strerror(-ret), ret);
            }

            // Set packet types for hardware filtering (Ethernet + IPv4 + UDP)
            uint32_t ptypes[] = {RTE_PTYPE_L2_ETHER, RTE_PTYPE_L3_IPV4, RTE_PTYPE_L4_UDP};
            ret = rte_eth_dev_set_ptypes(m_resolved_port_id, RTE_PTYPE_UNKNOWN, ptypes, 3);
            if (ret < 0) {
                m_logger->warn("failed to set packet types for hardware filtering (ret={})", ret);
            } else {
                m_logger->debug("configured hardware packet type filtering (Ethernet+IPv4+UDP)");
            }
        }
    } else if (is_multicast && m_config.src_ip.empty()) {
        m_logger->warn("multicast destination {} detected but src_ip not configured - IGMP disabled. "
                      "Multicast traffic may not be forwarded to this interface.",
                      m_config.ip_addr);
    }

    if (m_config.burst_size == 0) {
        m_logger->warn("burst_size was 0; defaulting to 32 packets");
        m_config.burst_size = 32;
    }
    m_rx_burst.resize(m_config.burst_size);
    m_last_igmp_poll = std::chrono::steady_clock::now();
}

dpdk::~dpdk() {
    stop_recv();

    // Clear multicast address filter if added
    if (m_mcast_mac_added) {
        rte_ether_addr dpdk_mac;
        std::memcpy(dpdk_mac.addr_bytes, m_mcast_mac.data(), m_mcast_mac.size());
        int ret = rte_eth_dev_mac_addr_remove(m_resolved_port_id, &dpdk_mac);
        if (ret == 0) {
            m_logger->debug("cleared multicast address filter");
        } else {
            m_logger->warn("failed to clear multicast address filter: {} (ret={})",
                          strerror(-ret), ret);
        }
    }

    if (m_queue_allocated) {
        auto& dpdk_mgr = composite::dpdk::manager::instance();
        dpdk_mgr.release_queue(m_config.interface, m_resolved_queue_id);
        m_queue_allocated = false;
    }
}

auto dpdk::start_recv(output_port_t* port) -> void {
    if (m_running.exchange(true)) {
        m_logger->warn("dpdk receive already running on port {} queue {}",
                      m_resolved_port_id, m_resolved_queue_id);
        return;
    }
    m_out_port = port;
    m_recv_thread = std::jthread(&dpdk::receive, this);
    if (auto ret = pthread_setname_np(m_recv_thread.native_handle(), "dpdk"); ret != 0) {
        m_logger->warn("failed to set thread name: {}", std::string{strerror(ret)});
    }
    m_logger->debug(
        "dpdk receive started on port {} queue {}",
        m_resolved_port_id, m_resolved_queue_id
    );
}

auto dpdk::stop_recv() -> void {
    if (!m_running.exchange(false)) {
        return;
    }
    m_recv_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
    m_logger->debug("dpdk receive stopped");
}

auto dpdk::receive(std::stop_token token) -> void {
    auto* pkts = m_rx_burst.data();
    while (!token.stop_requested()) [[likely]] {

        // Receive burst of packets from DPDK port
        uint16_t nb_rx = rte_eth_rx_burst(
            m_resolved_port_id,
            m_resolved_queue_id,
            pkts,
            m_config.burst_size
        );

        if (nb_rx == 0) {
            m_bursts_empty.fetch_add(1, std::memory_order_relaxed);

            rte_delay_us_block(1);
            continue;
        }

        // Process each received packet
        for (uint16_t i = 0; i < nb_rx; i++) {
            struct rte_mbuf* mbuf = pkts[i];

            // Check for IGMP query
            if (m_igmp_mgr && m_igmp_mgr->handle_rx(mbuf)) {
                m_igmp_queries.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            // Extract UDP payload from packet
            auto payload = extract_udp_payload(mbuf);

            if (payload.valid) {
                m_pkts_recvd.fetch_add(1, std::memory_order_relaxed);

                auto buffer = std::make_shared<composite::external_buffer<uint8_t, mbuf_release>>(
                    payload.data,
                    payload.length,
                    mbuf_release{mbuf}
                );
                m_out_port->send_data(composite::immutable_buffer<uint8_t>(std::move(buffer)), {});
            } else {
                m_pkts_dropped.fetch_add(1, std::memory_order_relaxed);
                rte_pktmbuf_free(mbuf);
            }
        }

        if (m_igmp_mgr) {
            auto now = std::chrono::steady_clock::now();
            if (now - m_last_igmp_poll > std::chrono::milliseconds(100)) {
                m_igmp_mgr->poll();
                m_last_igmp_poll = now;
            }
        }
    }
}

auto dpdk::extract_udp_payload(rte_mbuf* mbuf) -> udp_payload {
    udp_payload result{nullptr, 0, false};

    // Check for chained (multi-segment) mbufs
    if (mbuf->nb_segs > 1) {
        m_logger->warn(
            "Chained mbuf detected ({} segments), dropping packet. "
            "Consider increasing mempool mbuf size or enabling packet linearization.",
            mbuf->nb_segs
        );
        return result;
    }

    // Get pointer to start of packet data
    uint8_t* pkt_data = rte_pktmbuf_mtod(mbuf, uint8_t*);
    uint32_t pkt_len = rte_pktmbuf_pkt_len(mbuf);

    // Parse Ethernet header
    if (pkt_len < sizeof(rte_ether_hdr)) {
        m_logger->trace("Packet too short for Ethernet header");
        return result;
    }

    auto* eth_hdr = reinterpret_cast<rte_ether_hdr*>(pkt_data);
    uint16_t ether_type = rte_be_to_cpu_16(eth_hdr->ether_type);

    // Skip broadcast packets
    if (rte_is_broadcast_ether_addr(&eth_hdr->dst_addr)) {
        m_logger->trace("Skipping broadcast packet");
        return result;
    }

    // Strip ALL VLAN tags (handles Q-in-Q / 802.1ad double tagging)
    size_t l3_offset = sizeof(rte_ether_hdr);
    while (ether_type == RTE_ETHER_TYPE_VLAN) {
        // Check packet has room for VLAN header
        if (pkt_len < l3_offset + sizeof(rte_vlan_hdr)) {
            m_logger->trace("Packet too short for VLAN header");
            return result;
        }

        auto* vlan_hdr = reinterpret_cast<rte_vlan_hdr*>(pkt_data + l3_offset);
        ether_type = rte_be_to_cpu_16(vlan_hdr->eth_proto);
        l3_offset += sizeof(rte_vlan_hdr);
    }

    // We only handle IPv4 for now
    if (ether_type != RTE_ETHER_TYPE_IPV4) {
        // Log first 32 bytes of packet for diagnostics
        if (m_logger->should_log(spdlog::level::debug)) {
            std::string hex_dump;
            for (uint32_t i = 0; i < std::min(32u, pkt_len); i++) {
                char buf[4];
                snprintf(buf, sizeof(buf), "%02x ", pkt_data[i]);
                hex_dump += buf;
            }
            m_logger->debug("Non-IPv4 packet (ether_type=0x{:04x}): {}", ether_type, hex_dump);
        } else {
            m_logger->trace("Non-IPv4 packet (ether_type=0x{:04x})", ether_type);
        }
        return result;
    }

    // Parse IP header (after Ethernet header and all VLAN tags)
    auto* ip_hdr = reinterpret_cast<rte_ipv4_hdr*>(
        pkt_data + l3_offset
    );

    if (pkt_len < l3_offset + sizeof(rte_ipv4_hdr)) {
        m_logger->trace("Packet too short for IP header");
        return result;
    }

    // Check if it's UDP
    if (ip_hdr->next_proto_id != IPPROTO_UDP) {
        m_logger->trace("Non-UDP packet (proto={})", ip_hdr->next_proto_id);
        return result;
    }

    // Optional: Filter by destination IP
    if (m_dst_ip_be.has_value() && ip_hdr->dst_addr != m_dst_ip_be.value()) {
        if (m_logger->should_log(spdlog::level::trace)) {
            char ip_str[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &ip_hdr->dst_addr, ip_str, sizeof(ip_str));
            m_logger->trace("IP mismatch: {} != {}", ip_str, m_config.ip_addr);
        }
        return result;
    }

    // Parse UDP header
    size_t ip_hdr_len = rte_ipv4_hdr_len(ip_hdr);
    size_t l4_offset = l3_offset + ip_hdr_len;
    auto* udp_hdr = reinterpret_cast<rte_udp_hdr*>(
        pkt_data + l4_offset
    );

    if (pkt_len < l4_offset + sizeof(rte_udp_hdr)) {
        m_logger->trace("Packet too short for UDP header");
        return result;
    }

    // Optional: Filter by destination port
    uint16_t dst_port = rte_be_to_cpu_16(udp_hdr->dst_port);
    if (m_config.port != 0 && dst_port != m_config.port) {
        m_logger->trace("Port mismatch: {} != {}", dst_port, m_config.port);
        return result;
    }

    // Extract UDP payload
    uint16_t udp_len = rte_be_to_cpu_16(udp_hdr->dgram_len);
    uint16_t payload_len = udp_len - sizeof(rte_udp_hdr);

    size_t payload_offset = l4_offset + sizeof(rte_udp_hdr);
    if (pkt_len <= payload_offset) {
        m_logger->trace("No UDP payload");
        return result;
    }

    uint8_t* payload_data = pkt_data + payload_offset;

    // Validate payload length
    if (payload_len == 0 ||
        (pkt_data + pkt_len) < (payload_data + payload_len)) {
        m_logger->trace("Invalid UDP payload length: {}", payload_len);
        return result;
    }

    result.data = payload_data;
    result.length = payload_len;
    result.valid = true;

    return result;
}

auto dpdk::get_stats() -> std::map<std::string, std::string> {
    std::map<std::string, std::string> stats;

    // Cumulative stats
    stats["pkts_recvd"] = std::to_string(m_pkts_recvd.load(std::memory_order_relaxed));
    stats["pkts_dropped"] = std::to_string(m_pkts_dropped.load(std::memory_order_relaxed));
    stats["bursts_empty"] = std::to_string(m_bursts_empty.load(std::memory_order_relaxed));

    // IGMP stats
    if (m_igmp_mgr) {
        stats["igmp_queries"] = std::to_string(
            m_igmp_queries.load(std::memory_order_relaxed)
        );
    }

    // DPDK port-level statistics (hardware counters)
    struct rte_eth_stats eth_stats;
    if (rte_eth_stats_get(m_resolved_port_id, &eth_stats) == 0) {
        stats["hw_pkts_recvd"] = std::to_string(eth_stats.ipackets);
        stats["hw_bytes_recvd"] = std::to_string(eth_stats.ibytes);
        stats["hw_pkts_dropped"] = std::to_string(eth_stats.imissed);
        stats["hw_rx_errors"] = std::to_string(eth_stats.ierrors);
        stats["hw_rx_nombuf"] = std::to_string(eth_stats.rx_nombuf);

        // Per-queue stats (for our queue only)
        if (m_resolved_queue_id < RTE_ETHDEV_QUEUE_STAT_CNTRS) {
            stats["hw_q_pkts"] = std::to_string(eth_stats.q_ipackets[m_resolved_queue_id]);
            stats["hw_q_bytes"] = std::to_string(eth_stats.q_ibytes[m_resolved_queue_id]);
            stats["hw_q_errors"] = std::to_string(eth_stats.q_errors[m_resolved_queue_id]);
        }
    }

    return stats;
}

} // namespace udp
