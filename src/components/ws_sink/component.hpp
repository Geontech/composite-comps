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

#include "websocket_server.hpp"

#include <composite/composite.hpp>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// Per-client connection state: FPS throttling, backpressure, and metrics counters.
// Touched from three sides — the component worker (stream sends), the websocket
// threads (connect/message callbacks), and the metrics thread — so the mutable FPS
// state lives under state_mutex and the counters are relaxed atomics.
struct client_state {
    // Connection info
    std::string client_id;

    // FPS management (stream data only) — guarded by state_mutex. default_fps is the
    // configured starting rate and the ceiling backpressure recovery climbs back to.
    float default_fps{12.0f};
    float stream_fps{12.0f};
    std::chrono::microseconds stream_frame_interval{83333}; // 1/12 second
    std::chrono::steady_clock::time_point last_stream_frame_sent;

    // Backpressure state
    std::atomic<size_t> send_queue_size{0};
    size_t send_queue_hwm{5}; // written once at connect, before the client is published

    enum class flow_state { healthy, backpressure, recovering };
    flow_state state{flow_state::healthy}; // guarded by state_mutex

    // Metrics counters (worker writes, metrics thread reads)
    std::atomic<uint64_t> histogram_frames_sent{0};
    std::atomic<uint64_t> stream_frames_sent{0};
    std::atomic<uint64_t> stream_frames_skipped{0};

    // Rolling FPS calculation — guarded by state_mutex
    std::deque<std::chrono::steady_clock::time_point> recent_stream_frame_times;
    static constexpr size_t fps_window_size = 30;

    mutable std::mutex state_mutex;

    // Methods
    auto should_send_stream_frame() const -> bool;
    auto set_stream_fps(float new_fps) -> void;
    auto adjust_stream_fps(flow_state new_state) -> void;
    auto current_flow_state() const -> flow_state;
    auto calculate_stream_fps() const -> float;
    auto update_stream_fps_metrics() -> void;
};

// WebSocket sink: streams spectrum data (binary, FPS-throttled per client) and
// histogram data (JSON) to any number of websocket clients, with per-client
// adaptive backpressure. When no clients are connected the input port depths are
// dropped to 0 so upstream packets are discarded at the ring, not processed.
template <typename T>
class ws_sink : public composite::component {
    using stream_port_t = composite::input_port<composite::immutable_buffer<T>>;
    using histogram_port_t = composite::input_port<composite::immutable_buffer<uint64_t>>;

    static constexpr size_t HISTOGRAM_PORT_DEPTH = 5;

public:
    explicit ws_sink(std::string_view id);
    ~ws_sink() override = default;

    auto initialize() -> void override;
    auto process() -> composite::retval override;

protected:
    // Worker lifecycle hooks (run on EVERY start/stop path, including the app's
    // enabled-reconcile — overriding start()/stop() would be bypassed there).
    auto on_worker_start() -> void override;
    auto on_worker_stop() -> void override;

private:
    // Helper methods
    auto update_port_depths() -> void;
    auto update_cached_metadata(const composite::metadata_ptr& meta) -> void;
    auto make_stream_header(composite::timestamp ts) -> nlohmann::json;
    auto generate_metrics_message(const client_state& client) -> nlohmann::json;
    auto check_client_backpressure(client_state& client) -> void;

    // Connection callbacks (websocket threads)
    auto on_client_connect(std::shared_ptr<websocket_session> session) -> void;
    auto on_client_disconnect(const std::string& client_id) -> void;
    auto on_client_message(const std::string& client_id, const std::string& message) -> void;

    // Ports
    stream_port_t m_data_in_port{"data_in"};
    histogram_port_t m_hist_in_port{"histogram_in"};

    // Properties
    uint16_t m_ws_port{8080};
    std::string m_bind_address{"0.0.0.0"};
    float m_default_stream_fps{12.0f};
    float m_max_stream_fps{60.0f};
    size_t m_send_queue_hwm{5};
    size_t m_max_clients{100};
    size_t m_stream_port_depth{1};
    bool m_compression{false};

    // Members
    std::unique_ptr<websocket_server> m_server;
    std::atomic<std::shared_ptr<const std::vector<std::shared_ptr<client_state>>>> m_clients;
    std::mutex m_clients_mutex; // Protects RCU updates to m_clients

    // 1 Hz per-client metrics, off the data path (the worker parks on the doorbell
    // when idle, so it cannot pace wall-clock sends itself).
    std::jthread m_metrics_thread;

    // Cached stream header fields (worker thread only). m_last_meta enables the
    // steady-state fast path: same shared metadata instance => nothing to re-parse.
    composite::metadata_ptr m_last_meta;
    double m_sample_rate{0.0};
    double m_bandwidth{0.0};
    double m_center_frequency{0.0};
    size_t m_fft_size{0};
    size_t m_frame_size{65536};
    bool m_recheck_frame_size{false}; // set when frame size or fft_size changed
    static constexpr size_t m_data_size = sizeof(T);

    // MUST be last: stops the framework worker before any member above destructs.
    composite::component::auto_stop m_auto_stop{*this};

}; // class ws_sink
