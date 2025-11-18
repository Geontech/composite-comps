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

#include <rte_mbuf.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_ethdev.h>
#include <rte_byteorder.h>
#include <rte_cycles.h>
#include <rte_random.h>

#include <cstdint>
#include <unordered_set>
#include <vector>
#include <optional>
#include <limits>
#include <string_view>

namespace udp::igmp {

struct config {
    uint16_t port_id{};
    uint16_t tx_queue{};
    rte_mempool* mbuf_pool{nullptr};
    rte_ether_addr local_mac{};
    uint32_t local_ipv4_be{};                 // IPv4 in network byte order
    std::optional<uint16_t> vlan_id;        // nullopt = untagged

    // IGMP robustness / timing (IGMPv2-ish behavior)
    uint8_t robustness_variable{2};        // default per RFC
    double unsolicited_report_interval_sec{10.0}; // default per RFC
    bool respond_to_queries{true};
};

class manager {
    static constexpr uint8_t IPPROTO_IGMP = 2;
    // IGMPv2 types
    static constexpr uint8_t IGMP_TYPE_MEMBERSHIP_QUERY      = 0x11;
    static constexpr uint8_t IGMP_TYPE_V1_MEMBERSHIP_REPORT  = 0x12; // not used, here for completeness
    static constexpr uint8_t IGMP_TYPE_V2_MEMBERSHIP_REPORT  = 0x16;
    static constexpr uint8_t IGMP_TYPE_V2_LEAVE_GROUP        = 0x17;
public:
    explicit manager(const config& cfg);
    ~manager();

    // Join a multicast group by IP address string (e.g., "239.1.2.3").
    // Validates that the IP is in the multicast range (224.0.0.0/4).
    // Schedules RobustnessVariable unsolicited membership reports.
    // Returns false if IP is invalid or not multicast.
    bool join_group(std::string_view group_ip);

    // Leave a multicast group by IP address string.
    // Schedules RobustnessVariable unsolicited leave messages.
    // Returns false if IP is invalid or group was not joined.
    bool leave_group(std::string_view group_ip);

    // Call from RX loop. Returns true if packet was IGMP and is consumed (freed).
    bool handle_rx(rte_mbuf* m);

    // Call periodically from your main loop to send scheduled reports.
    void poll();

private:
#pragma pack(push, 1)
    struct igmp_hdr {
        uint8_t   type;
        uint8_t   max_resp_time;  // in 1/10 sec for queries
        rte_be16_t checksum;
        rte_be32_t group_addr;
    };
#pragma pack(pop)

    struct pending_report {
        uint32_t group_ip_be;
        uint8_t  igmp_type;
        uint64_t due_tsc;
    };

    uint16_t port_id_;
    uint16_t tx_queue_;
    rte_mempool* mbuf_pool_;
    rte_ether_addr local_mac_;
    uint32_t local_ip_be_;
    std::optional<uint16_t> vlan_id_;

    uint64_t tsc_hz_;
    uint8_t  robustness_variable_;
    double   unsolicited_report_interval_sec_;
    bool     respond_to_queries_;

    std::unordered_set<uint32_t> groups_; // group IPs in network byte order
    std::vector<pending_report> pending_reports_;
    uint64_t next_due_tsc_;

    // --- Helpers ---

    static rte_ether_addr multicast_mac_from_group(uint32_t group_ip_be);
    bool send_igmp_report(uint32_t group_ip_be, uint8_t igmp_type);
    void handle_igmp_query(const igmp_hdr& igmp);
    void schedule_query_response(uint32_t group_ip_be,
                                 uint8_t igmp_type,
                                 uint8_t max_resp_time_field);
    void schedule_unsolicited_reports(uint32_t group_ip_be, uint8_t igmp_type);
    void cancel_pending_reports(uint32_t group_ip_be, uint8_t igmp_type);
};

} // namespace udp::igmp
