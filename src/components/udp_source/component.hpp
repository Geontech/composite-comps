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
    // PACKET_MMAP only: Unix-socket path to receive a pre-opened AF_PACKET fd over
    // SCM_RIGHTS from a privileged helper, so THIS process needs no CAP_NET_RAW (the
    // socket() call is the backend's only privileged operation). Empty = open the socket
    // directly (requires CAP_NET_RAW here).
    std::string packet_fd_path;
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

struct recvmmsg_config {
    // Zero selects num_msgs. Partial batches are retained across receive calls and
    // published when either this size or max_batch_delay_us is reached.
    uint32_t receive_batch_wait_us{100};
    uint32_t output_batch_size{};
    uint32_t max_batch_delay_us{1000};
}; // struct recvmmsg_config

} // namespace struct_props

COMPOSITE_STRUCT(struct_props::overrides, msg_size, packet_fd_path);
COMPOSITE_STRUCT(struct_props::dpdk_config,
    port_id, queue_id, mempool_name, burst_size, igmp_respond_to_queries, src_ip);
COMPOSITE_STRUCT(struct_props::recvmmsg_config,
    receive_batch_wait_us, output_batch_size, max_batch_delay_us);

class udp_source : public composite::component {
    static constexpr std::string_view RECVMMSG = "recvmmsg";
    static constexpr std::string_view PACKET_MMAP = "packet_mmap";
    static constexpr std::string_view DPDK = "dpdk";

    using output_port_t = composite::output_port<composite::immutable_buffer<uint8_t>>;
public:
    explicit udp_source(std::string_view id);
    ~udp_source() override = default;
    auto property_change_handler(const composite::properties::json& diff) -> void override;
    auto process() -> composite::retval override;

protected:
    // The receiver and the stats thread are worker-scoped resources, so they hang off the
    // framework's worker lifecycle hooks (start()/stop() are final). The hooks run on EVERY
    // start/stop path — including the application/REST enabled-reconcile, which the old
    // start()/stop() overrides were bypassed by (a REST disable left the receiver running).
    auto on_worker_start() -> void override;
    auto on_worker_stop() -> void override;
    auto on_park_requested() -> void override;

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
    struct_props::recvmmsg_config m_recvmmsg;
    struct_props::dpdk_config m_dpdk;

    // Members
    // Component-owned abort eventfd, handed to each receiver as config.abort_fd so a long wait
    // inside start_recv() (packet-size autodiscovery) can be interrupted from OUTSIDE
    // m_receiver_mtx and without touching m_receiver — a plain fd write races nothing. Signalled
    // from on_park_requested() (a stop or property writer needing the worker to yield while a
    // worker-thread reactivation is mid-discovery — the case the park cannot interrupt itself)
    // and from on_worker_stop() (belt-and-braces; see there). Not reachable in time when the
    // discovery runs inside on_worker_start(), because stop is serialized behind the lifecycle
    // lock — that path's bound is the discovery deadline itself.
    // Declared BEFORE m_auto_stop: members destruct in reverse order, so the fd outlives the
    // auto_stop-triggered stop() that may still write it.
    class abort_event {
    public:
        abort_event();
        ~abort_event();
        abort_event(const abort_event&) = delete;
        abort_event& operator=(const abort_event&) = delete;
        [[nodiscard]] auto fd() const noexcept -> int { return m_fd; }
        auto signal() const noexcept -> void;
        auto drain() const noexcept -> void;
    private:
        int m_fd{-1};
    };
    abort_event m_abort;
    std::unique_ptr<udp::interface> m_receiver;
    std::jthread m_stat_thread;
    std::mutex m_receiver_mtx;
    bool m_component_running{false};
    bool m_receiver_running{false};

    // Metrics
    composite::metrics::counter<uint64_t>* m_packets_received{nullptr};
    composite::metrics::counter<uint64_t>* m_bytes_received{nullptr};
    composite::metrics::counter<uint64_t>* m_packets_dropped{nullptr};
    composite::metrics::counter<uint64_t>* m_kernel_drops{nullptr};
    composite::metrics::histogram* m_batch_sizes{nullptr};

    auto start_receiver_locked() -> void;
    auto stop_receiver_locked() -> void;
    auto create_metrics() -> udp::metrics;

    // MUST be last: stops the worker + (via stop()) the receiver and the stats jthread before any
    // member above destructs (the base ~component stops too late). See fft for rationale.
    composite::component::auto_stop m_auto_stop{*this};

}; // class udp_source
