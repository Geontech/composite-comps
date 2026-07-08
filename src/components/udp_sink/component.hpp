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

#include "socket/interface.hpp"

#include <composite/composite.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

/**
 * @brief UDP sink component for multi-destination packet egress
 *
 * Sends packets to multiple UDP destinations based on metadata.
 * Supports multiple socket backends for different performance profiles:
 *   - "send": Basic sendto() per packet
 *   - "sendmmsg": Batched sending with sendmmsg() syscall
 *
 * Uses a connected socket pool with idle-timeout cleanup.
 *
 * Destination routing comes from metadata annotations (keys configurable):
 *   - dest_ip: destination IP address (e.g., "239.1.1.1")
 *   - dest_port: destination port (e.g., 5000)
 *   - stream_id: selects the per-stream latched destination
 * The last seen destination is latched per stream, and the shared metadata
 * instance is used for change detection: packets carrying the same instance
 * as the previous one (the steady state) skip the annotation lookups entirely.
 */
class udp_sink : public composite::component {
    friend struct UdpSinkTestFixture;
    using input_port_t = composite::input_port<composite::immutable_buffer<uint8_t>>;

    static constexpr std::string_view SEND = "send";
    static constexpr std::string_view SENDMMSG = "sendmmsg";

public:
    explicit udp_sink(std::string_view id);
    ~udp_sink() override = default;
    auto property_change_handler(const composite::properties::json& diff) -> void override;
    auto process() -> composite::retval override;
    // Batched sends buffer packets; flush them when the stream ends so the tail
    // is not stranded in the batch.
    auto on_end_of_stream() -> void override;

protected:
    // Worker lifecycle hooks (run on EVERY start/stop path, including the app's
    // enabled-reconcile — overriding start()/stop() would be bypassed there).
    auto on_worker_start() -> void override;
    auto on_worker_stop() -> void override;

private:
    // Per-stream state for destination latching
    struct stream_state {
        std::string last_dest_ip;
        uint16_t last_dest_port{0};
    };

    // Ports
    input_port_t m_in_port{"data_in"};

    // Properties
    std::string m_socket_type{SENDMMSG};      // Socket backend type
    std::string m_dest_ip_key{"dest_ip"};     // Metadata key for destination IP
    std::string m_dest_port_key{"dest_port"}; // Metadata key for destination port
    std::string m_stream_id_key{"stream_id"}; // Metadata key for stream ID
    uint32_t m_socket_timeout_s{30};          // Close idle sockets after N seconds
    uint32_t m_send_buf_size{0};              // Socket send buffer size (0 = system default)
    uint32_t m_batch_size{64};                // Max packets per sendmmsg() call
    uint32_t m_batch_timeout_us{1000};        // Flush batch after N microseconds
    uint32_t m_max_packet_size{4096};         // Max expected packet size (for buffer pre-allocation)
    std::string m_bind_interface;             // Optional: bind to specific interface
    std::string m_default_dest_ip;            // Default destination IP if not in metadata
    uint16_t m_default_dest_port{0};          // Default destination port if not in metadata

    // Sender backend. The mutex serializes the worker's send path against the idle-cleanup
    // thread and against sender rebuilds from property writes (the cleanup thread is NOT
    // parked by the property machinery, so the lock is required, not belt-and-braces).
    std::unique_ptr<udp_tx::interface> m_sender;
    std::mutex m_sender_mtx;

    // Per-stream destination state
    std::unordered_map<uint32_t, stream_state> m_stream_states;

    // Steady-state fast path: destination resolved for this shared metadata instance.
    // Upstream latches its metadata_ptr, so pointer equality means "same annotations" —
    // no map lookups, no string parsing per packet.
    composite::metadata_ptr m_last_meta;
    std::string m_last_ip;
    uint16_t m_last_port{0};

    // Cleanup thread for idle sockets (runs only while the worker runs)
    std::jthread m_cleanup_thread;

    // Helper methods
    auto get_stream_id(const composite::metadata& metadata) -> uint32_t;
    auto get_destination(uint32_t stream_id, const composite::metadata& metadata)
        -> std::pair<std::string, uint16_t>;
    auto create_sender() -> std::unique_ptr<udp_tx::interface>;

    // MUST be last: stops the framework worker before any member above destructs.
    composite::component::auto_stop m_auto_stop{*this};

}; // class udp_sink
