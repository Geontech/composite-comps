/*
 * Copyright (C) 2024 Geon Technologies, LLC
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

#include "socket/interface.hpp"

#include <composite/composite.hpp>
#include <composite/metrics/metrics.hpp>

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace struct_props {

struct overrides {
    std::optional<uint32_t> msg_size;
}; // struct overrides

struct dpdk_config {
    std::optional<uint16_t> port_id;        // Optional: auto-resolved from interface
    std::optional<uint16_t> queue_id;       // Optional: auto-assigned
    std::string mempool_name{"mbuf_pool"};
    uint16_t burst_size{32};

    // IGMP configuration (automatically enabled for multicast if src_ip provided)
    bool igmp_respond_to_queries{true};
    std::string src_ip;                     // Source IP for IGMP (enables IGMP if dst is multicast)
}; // struct dpdk_config

} // namespace struct_props

class udp_source : public composite::component {
    static constexpr std::string_view RECVMMSG = "recvmmsg";
    static constexpr std::string_view PACKET_MMAP = "packet_mmap";
    static constexpr std::string_view DPDK = "dpdk";

    using output_port_t = composite::output_port<composite::immutable_buffer<uint8_t>>;
public:
    explicit udp_source(std::string_view id);
    ~udp_source() override = default;
    auto property_change_handler() -> void override;
    auto start() -> void override;
    auto stop() -> void override;
    auto process() -> composite::retval override;

private:
    // Ports
    output_port_t m_out_port{"data_out"};

    // Properties
    bool m_active{false};
    std::string m_socket_type{RECVMMSG};
    std::string m_interface;
    std::string m_ip_addr;
    uint16_t m_port{};
    uint32_t m_num_msgs{};
    uint32_t m_frame_count{8192};
    uint32_t m_recv_buf_size{};
    uint32_t m_autodiscovery_timeout{10};
    struct_props::overrides m_overrides;
    struct_props::dpdk_config m_dpdk;

    // Members
    std::unique_ptr<udp::interface> m_receiver;
    std::jthread m_stat_thread;
    std::mutex m_receiver_mtx;
    bool m_component_running{false};
    bool m_receiver_running{false};

    // Metrics
    composite::metrics::counter<uint64_t>* m_packets_received{nullptr};
    composite::metrics::counter<uint64_t>* m_bytes_received{nullptr};
    composite::metrics::counter<uint64_t>* m_packets_dropped{nullptr};
    composite::metrics::histogram* m_batch_sizes{nullptr};

    auto start_receiver_locked() -> void;
    auto stop_receiver_locked() -> void;
    auto create_metrics() -> udp::metrics;

}; // class udp_source

// Struct property trait specializations
template<>
struct composite::properties::property_traits<struct_props::overrides> {
    static void register_fields(composite::properties::property_set& ps, struct_props::overrides& prop) {
        using enum composite::properties::config_type;
        ps.add("msg_size", prop.msg_size, RUNTIME);
    }
};

template<>
struct composite::properties::property_traits<struct_props::dpdk_config> {
    static void register_fields(composite::properties::property_set& ps, struct_props::dpdk_config& prop) {
        using enum composite::properties::config_type;
        ps.add("port_id", prop.port_id);
        ps.add("queue_id", prop.queue_id);
        ps.add("mempool_name", prop.mempool_name);
        ps.add("burst_size", prop.burst_size);
        ps.add("igmp_respond_to_queries", prop.igmp_respond_to_queries);
        ps.add("src_ip", prop.src_ip);
    }
};