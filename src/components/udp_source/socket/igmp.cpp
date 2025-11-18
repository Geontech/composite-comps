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

#include "igmp.hpp"
#include "net/utils.hpp"

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <vector>
#include <optional>
#include <limits>
#include <arpa/inet.h>

namespace udp::igmp {

manager::manager(const config& cfg)
    : port_id_(cfg.port_id),
        tx_queue_(cfg.tx_queue),
        mbuf_pool_(cfg.mbuf_pool),
        local_mac_(cfg.local_mac),
        local_ip_be_(cfg.local_ipv4_be),
        vlan_id_(cfg.vlan_id),
        tsc_hz_(rte_get_tsc_hz()),
        robustness_variable_(cfg.robustness_variable),
        unsolicited_report_interval_sec_(cfg.unsolicited_report_interval_sec),
        respond_to_queries_(cfg.respond_to_queries),
        next_due_tsc_(std::numeric_limits<uint64_t>::max())
{}

manager::~manager() {
    // Send IGMP Leave for all joined groups per RFC 2236
    for (uint32_t group_ip : groups_) {
        // Send Leave immediately (robustness_variable times for reliability)
        for (uint8_t i = 0; i < robustness_variable_; ++i) {
            send_igmp_report(group_ip, IGMP_TYPE_V2_LEAVE_GROUP);
        }
    }
}

// Join a multicast group by IP address string.
// Validates and converts to network byte order.
bool manager::join_group(std::string_view group_ip)
{
    // Validate it's a multicast IP and convert to network byte order
    if (!net::is_ipv4_multicast(group_ip)) {
        // Invalid or not multicast
        return false;
    }

    // Convert to network byte order for internal storage
    struct in_addr addr;
    inet_pton(AF_INET, group_ip.data(), &addr);  // Already validated above
    uint32_t group_ip_be = addr.s_addr;

    auto [it, inserted] = groups_.insert(group_ip_be);
    if (!inserted) {
        // Already joined
        return true;
    }

    schedule_unsolicited_reports(group_ip_be, IGMP_TYPE_V2_MEMBERSHIP_REPORT);
    return true;
}

// Leave a multicast group by IP address string.
// Validates and converts to network byte order.
bool manager::leave_group(std::string_view group_ip) {
    // Validate it's a multicast IP and convert to network byte order
    if (!net::is_ipv4_multicast(group_ip)) {
        // Invalid or not multicast
        return false;
    }

    // Convert to network byte order for lookup
    struct in_addr addr;
    inet_pton(AF_INET, group_ip.data(), &addr);  // Already validated above
    uint32_t group_ip_be = addr.s_addr;

    auto it = groups_.find(group_ip_be);
    if (it == groups_.end()) {
        // Not joined
        return false;
    }
    groups_.erase(it);

    for (uint8_t i = 0; i < robustness_variable_; ++i) {
        send_igmp_report(group_ip_be, IGMP_TYPE_V2_LEAVE_GROUP);
    }
    return true;
}

// Call from RX loop. Returns true if packet was IGMP and is consumed (freed).
bool manager::handle_rx(rte_mbuf* m) {
    uint8_t* data = rte_pktmbuf_mtod(m, uint8_t*);
    size_t len = rte_pktmbuf_pkt_len(m); // assume small IGMP is contiguous

    if (len < sizeof(rte_ether_hdr)) {
        return false;
    }

    auto* eth = reinterpret_cast<rte_ether_hdr*>(data);
    uint16_t ether_type = rte_be_to_cpu_16(eth->ether_type);
    uint8_t* ptr = data + sizeof(rte_ether_hdr);

    // Optional VLAN
    if (ether_type == RTE_ETHER_TYPE_VLAN) {
        if (len < sizeof(rte_ether_hdr) + sizeof(rte_vlan_hdr)) {
            return false;
        }
        auto* vlan = reinterpret_cast<rte_vlan_hdr*>(ptr);
        ether_type = rte_be_to_cpu_16(vlan->eth_proto);
        ptr += sizeof(rte_vlan_hdr);
    }

    if (ether_type != RTE_ETHER_TYPE_IPV4) {
        return false;
    }

    if (len < static_cast<size_t>(ptr - data) + sizeof(rte_ipv4_hdr)) {
        return false;
    }

    auto* ip = reinterpret_cast<rte_ipv4_hdr*>(ptr);
    uint8_t ihl = ip->version_ihl & 0x0F;
    if (ihl < 5) {
        return false;
    }

    size_t ip_hdr_len = ihl * 4;
    if (len < static_cast<size_t>(ptr - data) + ip_hdr_len) {
        return false;
    }

    if (ip->next_proto_id != IPPROTO_IGMP) {
        return false;
    }

    uint16_t ip_total_len = rte_be_to_cpu_16(ip->total_length);
    if (ip_total_len < ip_hdr_len + sizeof(igmp_hdr)) {
        return false;
    }

    uint8_t* igmp_ptr = ptr + ip_hdr_len;
    if (len < static_cast<size_t>(igmp_ptr - data) + sizeof(igmp_hdr)) {
        return false;
    }

    auto* igmp = reinterpret_cast<igmp_hdr*>(igmp_ptr);

    if (igmp->type == IGMP_TYPE_MEMBERSHIP_QUERY) {
        if (respond_to_queries_) {
            handle_igmp_query(*igmp);
        }
        rte_pktmbuf_free(m);
        return true;
    }

    if (igmp->type == IGMP_TYPE_V2_MEMBERSHIP_REPORT ||
        igmp->type == IGMP_TYPE_V1_MEMBERSHIP_REPORT) {
        if (groups_.contains(igmp->group_addr)) {
            cancel_pending_reports(igmp->group_addr, IGMP_TYPE_V2_MEMBERSHIP_REPORT);
        }
        rte_pktmbuf_free(m);
        return true;
    }

    // Ignore other IGMP types on RX for now.
    return false;
}

// Call periodically from your main loop to send scheduled reports.
void manager::poll() {
    uint64_t now = rte_rdtsc();

    // Fast path: nothing due yet
    if (now < next_due_tsc_) {
        return;
    }

    uint64_t new_next = std::numeric_limits<uint64_t>::max();

    auto it = pending_reports_.begin();
    while (it != pending_reports_.end()) {
        if (it->due_tsc <= now) {
            // Best effort send; if TX fails, we just drop.
            send_igmp_report(it->group_ip_be, it->igmp_type);
            it = pending_reports_.erase(it);
        } else {
            if (it->due_tsc < new_next) {
                new_next = it->due_tsc;
            }
            ++it;
        }
    }

    next_due_tsc_ = new_next;
}

bool manager::send_igmp_report(uint32_t group_ip_be, uint8_t igmp_type) {
    rte_mbuf* m = rte_pktmbuf_alloc(mbuf_pool_);
    if (!m)
        return false;

    uint8_t* data = rte_pktmbuf_mtod(m, uint8_t*);
    uint8_t* ptr = data;

    auto* eth = reinterpret_cast<rte_ether_hdr*>(ptr);
    ptr += sizeof(rte_ether_hdr);

    // L2
    rte_ether_addr dst_mac = multicast_mac_from_group(group_ip_be);
    rte_ether_addr_copy(&local_mac_, &eth->src_addr);
    rte_ether_addr_copy(&dst_mac, &eth->dst_addr);

    uint16_t ether_type;

    if (vlan_id_) {
        ether_type = RTE_ETHER_TYPE_VLAN;
        eth->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_VLAN);

        auto* vlan = reinterpret_cast<rte_vlan_hdr*>(ptr);
        ptr += sizeof(rte_vlan_hdr);
        vlan->vlan_tci = rte_cpu_to_be_16(*vlan_id_ & 0x0FFF);
        vlan->eth_proto = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);
    } else {
        ether_type = RTE_ETHER_TYPE_IPV4;
        eth->ether_type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);
    }
    (void)ether_type;

    auto* ip = reinterpret_cast<rte_ipv4_hdr*>(ptr);
    ptr += sizeof(rte_ipv4_hdr);

    auto* igmp = reinterpret_cast<igmp_hdr*>(ptr);
    ptr += sizeof(igmp_hdr);

    const uint16_t total_len = sizeof(rte_ipv4_hdr) + sizeof(igmp_hdr);

    // Build IGMP
    igmp->type = igmp_type;
    igmp->max_resp_time = 0; // not used for reports / leaves
    igmp->group_addr = group_ip_be;
    igmp->checksum = 0;
    {
        uint16_t cksum = rte_raw_cksum(igmp, sizeof(igmp_hdr));
        igmp->checksum = ~cksum;
        if (igmp->checksum == 0) {
            igmp->checksum = 0xFFFF;
        }
    }

    // Build IPv4 header
    ip->version_ihl = (4 << 4) | (sizeof(rte_ipv4_hdr) / 4);
    ip->type_of_service = 0;
    ip->total_length = rte_cpu_to_be_16(total_len);
    ip->packet_id = 0;
    ip->fragment_offset = rte_cpu_to_be_16(0);
    ip->time_to_live = 1; // local link
    ip->next_proto_id = IPPROTO_IGMP;
    ip->hdr_checksum = 0;
    ip->src_addr = local_ip_be_;
    ip->dst_addr = group_ip_be; // v2 reports go to group address

    ip->hdr_checksum = rte_ipv4_cksum(ip);

    const uint16_t l2_len = static_cast<uint16_t>(
        vlan_id_ ? sizeof(rte_ether_hdr) + sizeof(rte_vlan_hdr)
                    : sizeof(rte_ether_hdr));

    m->data_len = l2_len + total_len;
    m->pkt_len  = m->data_len;
    m->l2_len   = l2_len;
    m->l3_len   = sizeof(rte_ipv4_hdr);

    uint16_t sent = rte_eth_tx_burst(port_id_, tx_queue_, &m, 1);
    if (sent == 0) {
        rte_pktmbuf_free(m);
        return false;
    }
    return true;
}

void manager::handle_igmp_query(const igmp_hdr& igmp) {
    uint32_t group_be = igmp.group_addr;
    uint8_t max_resp_time = igmp.max_resp_time;

    // General query
    if (group_be == 0) {
        for (uint32_t g : groups_) {
            schedule_query_response(g, IGMP_TYPE_V2_MEMBERSHIP_REPORT, max_resp_time);
        }
    } else {
        // Group-specific
        if (groups_.find(group_be) != groups_.end()) {
            schedule_query_response(group_be, IGMP_TYPE_V2_MEMBERSHIP_REPORT, max_resp_time);
        }
    }
}

// Schedule a randomized response within max_resp_time window (IGMP query).
void manager::schedule_query_response(uint32_t group_ip_be,
                                uint8_t igmp_type,
                                uint8_t max_resp_time_field) {
    double max_delay_sec;
    if (max_resp_time_field == 0) {
        max_delay_sec = 10.0; // default if zero
    } else {
        max_delay_sec = static_cast<double>(max_resp_time_field) / 10.0; // units of 100ms
    }

    uint32_t r = rte_rand();
    double frac = static_cast<double>(r) / static_cast<double>(std::numeric_limits<uint32_t>::max());
    double delay_sec = frac * max_delay_sec;

    uint64_t delay_tsc = static_cast<uint64_t>(delay_sec * static_cast<double>(tsc_hz_));
    uint64_t due = rte_rdtsc() + delay_tsc;

    pending_reports_.push_back(pending_report{group_ip_be, igmp_type, due});
    if (due < next_due_tsc_) {
        next_due_tsc_ = due;
    }
}

// Schedule robustness_variable_ unsolicited reports for joins/leaves.
void manager::schedule_unsolicited_reports(uint32_t group_ip_be, uint8_t igmp_type)
{
    uint64_t now = rte_rdtsc();

    for (uint8_t i = 0; i < robustness_variable_; ++i) {
        double delay_sec = unsolicited_report_interval_sec_ * static_cast<double>(i);
        uint64_t delay_tsc = static_cast<uint64_t>(delay_sec * static_cast<double>(tsc_hz_));
        uint64_t due = now + delay_tsc;

        pending_reports_.push_back(pending_report{group_ip_be, igmp_type, due});
        if (due < next_due_tsc_) {
            next_due_tsc_ = due;
        }
    }
}

void manager::cancel_pending_reports(uint32_t group_ip_be, uint8_t igmp_type) {
    bool removed = false;
    auto it = pending_reports_.begin();
    while (it != pending_reports_.end()) {
        if (it->group_ip_be == group_ip_be && it->igmp_type == igmp_type) {
            it = pending_reports_.erase(it);
            removed = true;
        } else {
            ++it;
        }
    }

    if (!removed) {
        return;
    }

    if (pending_reports_.empty()) {
        next_due_tsc_ = std::numeric_limits<uint64_t>::max();
        return;
    }

    next_due_tsc_ = pending_reports_.front().due_tsc;
    for (const auto& pending : pending_reports_) {
        next_due_tsc_ = std::min(next_due_tsc_, pending.due_tsc);
    }
}

// Convert multicast IP (network byte order) to multicast MAC address.
// RFC 1112: Maps to 01:00:5e:XX:XX:XX where XX:XX:XX = lower 23 bits of IP.
rte_ether_addr manager::multicast_mac_from_group(uint32_t group_ip_be) {
    uint32_t ip_host = ntohl(group_ip_be);

    rte_ether_addr mac;
    mac.addr_bytes[0] = 0x01;
    mac.addr_bytes[1] = 0x00;
    mac.addr_bytes[2] = 0x5e;
    mac.addr_bytes[3] = (ip_host >> 16) & 0x7F;  // Only lower 7 bits of byte 2
    mac.addr_bytes[4] = (ip_host >> 8) & 0xFF;
    mac.addr_bytes[5] = ip_host & 0xFF;

    return mac;
}

} // namespace udp::igmp
