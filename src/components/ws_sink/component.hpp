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
#include <aligned_mem.hpp>

#include <atomic>
#include <chrono>
#include <complex>
#include <composite/component.hpp>
#include <deque>
#include <memory>
#include <nlohmann/json.hpp>
#include <shared_mutex>
#include <string>
#include <vector>

// Client state for per-client FPS management
struct client_state {
    // Connection info
    std::string client_id;
    std::shared_ptr<websocket_session> session;

    // FPS management (stream data only)
    float stream_fps{12.0f};
    std::chrono::microseconds stream_frame_interval{83333}; // 1/12 second
    std::chrono::steady_clock::time_point last_stream_frame_sent;

    // Fixed-rate timers (1 Hz)
    std::chrono::steady_clock::time_point last_metrics_sent;
    static constexpr auto metrics_interval = std::chrono::seconds(1);

    // Backpressure state
    std::atomic<size_t> send_queue_size{0};
    size_t send_queue_hwm{5};

    enum class flow_state { healthy, backpressure, recovering };
    flow_state state{flow_state::healthy};

    // Metrics (non-atomic, don't need to be exact)
    uint64_t histogram_frames_sent{0};
    uint64_t stream_frames_sent{0};
    uint64_t stream_frames_skipped{0};

    // Rolling FPS calculation
    std::deque<std::chrono::steady_clock::time_point> recent_stream_frame_times;
    static constexpr size_t fps_window_size = 30;

    mutable std::mutex state_mutex;

    // Methods
    auto should_send_stream_frame() const -> bool;
    auto should_send_metrics() const -> bool;
    auto set_stream_fps(float new_fps) -> void;
    auto adjust_stream_fps(flow_state new_state) -> void;
    auto calculate_stream_fps() const -> float;
    auto update_stream_fps_metrics() -> void;
};

template <typename T>
class ws_sink : public composite::component {
    using stream_data_t = aligned::aligned_mem<T>;
    using stream_port_t = composite::input_port<std::unique_ptr<stream_data_t>>;
    using histogram_data_t = std::vector<uint64_t>;
    using histogram_port_t = composite::input_port<std::shared_ptr<histogram_data_t>>;
    using enum composite::properties::config_type;

  public:
    ws_sink();
    ~ws_sink() override = default;

    auto initialize() -> void override;
    auto start() -> void override;
    auto stop() -> void override;
    auto process() -> composite::retval override;

  private:
    // Helper methods
    auto update_port_depths() -> void;
    auto make_stream_header(composite::timestamp ts) -> nlohmann::json;
    auto generate_metrics_message(const client_state& client) -> nlohmann::json;
    auto check_client_backpressure(client_state& client) -> void;

    // Connection callbacks
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

    // Cached metadata
    composite::metadata m_stream_metadata;
    double m_sample_rate{0.0};
    double m_bandwidth{0.0};
    double m_center_frequency{0.0};
    size_t m_fft_size{0};
    size_t m_frame_size{65536};
    size_t m_data_size{sizeof(T)};

}; // class ws_sink
