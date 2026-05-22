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

#include "component.hpp"

#include <algorithm>
#include <format>
#include <string_view>

// ========== client_state implementation ==========

auto client_state::should_send_stream_frame() const -> bool {
    auto now = std::chrono::steady_clock::now();
    return (now - last_stream_frame_sent) >= stream_frame_interval;
}

auto client_state::should_send_metrics() const -> bool {
    auto now = std::chrono::steady_clock::now();
    return (now - last_metrics_sent) >= metrics_interval;
}

auto client_state::set_stream_fps(float new_fps) -> void {
    std::lock_guard lock(state_mutex);

    stream_fps = new_fps;
    stream_frame_interval =
        std::chrono::microseconds(static_cast<int64_t>(1'000'000.0f / stream_fps));
}

auto client_state::adjust_stream_fps(flow_state new_state) -> void {
    std::lock_guard lock(state_mutex);

    constexpr float MIN_FPS = 1.0f;
    state = new_state;

    switch (state) {
    case flow_state::backpressure:
        // Reduce by 50%, floor at 1 FPS
        stream_fps = std::max(MIN_FPS, stream_fps * 0.5f);
        stream_frame_interval =
            std::chrono::microseconds(static_cast<int64_t>(1'000'000.0f / stream_fps));
        break;

    case flow_state::recovering:
        // Slowly increase by 25%
        if (stream_fps < 12.0f) { // Don't go above default unless client requests
            stream_fps = std::min(12.0f, stream_fps * 1.25f);
            stream_frame_interval =
                std::chrono::microseconds(static_cast<int64_t>(1'000'000.0f / stream_fps));
        }
        break;

    case flow_state::healthy:
        // Already at stable FPS
        break;
    }
}

auto client_state::calculate_stream_fps() const -> float {
    std::lock_guard lock(state_mutex);

    if (recent_stream_frame_times.size() < 2) {
        return 0.0f;
    }

    auto duration = recent_stream_frame_times.back() - recent_stream_frame_times.front();
    auto duration_sec = std::chrono::duration<float>(duration).count();

    if (duration_sec <= 0.0f)
        return 0.0f;

    return (recent_stream_frame_times.size() - 1) / duration_sec;
}

auto client_state::update_stream_fps_metrics() -> void {
    std::lock_guard lock(state_mutex);

    auto now = std::chrono::steady_clock::now();
    recent_stream_frame_times.push_back(now);

    if (recent_stream_frame_times.size() > fps_window_size) {
        recent_stream_frame_times.pop_front();
    }

    last_stream_frame_sent = now;
}

// ========== ws_sink implementation ==========

template <typename T>
ws_sink<T>::ws_sink() : composite::component("ws_sink") {
    add_port(&m_hist_in_port);
    add_port(&m_data_in_port);

    add_property("port", &m_ws_port);
    add_property("bind_address", &m_bind_address);
    add_property("default_stream_fps", &m_default_stream_fps).units("Hz");
    add_property("max_stream_fps", &m_max_stream_fps).units("Hz");
    add_property("send_queue_hwm", &m_send_queue_hwm);
    add_property("max_clients", &m_max_clients);
    add_property("stream_port_depth", &m_stream_port_depth);
    add_property("compression", &m_compression);
}

template <typename T>
auto ws_sink<T>::initialize() -> void {
    logger()->info("Initializing WebSocket sink");

    m_server = std::make_unique<websocket_server>();
    m_clients.store(std::make_shared<std::vector<std::shared_ptr<client_state>>>());

    // Setup callbacks
    m_server->set_connect_handler([this](auto session) { on_client_connect(session); });
    m_server->set_disconnect_handler([this](auto id) { on_client_disconnect(id); });
    m_server->set_message_handler([this](auto id, auto msg) { on_client_message(id, msg); });
    m_server->set_queue_size_callback([this](auto id, auto size) {
        // Update client queue size for backpressure detection
        auto clients = m_clients.load(std::memory_order_acquire);
        for (auto& client : *clients) {
            if (client->client_id == id) {
                client->send_queue_size.store(size, std::memory_order_relaxed);
                break;
            }
        }
    });
}

template <typename T>
auto ws_sink<T>::start() -> void {
    logger()->info("Starting WebSocket server on {}:{} (compression: {})", m_bind_address,
                   m_ws_port, m_compression ? "enabled" : "disabled");

    m_server->start(m_ws_port, m_bind_address, m_compression);
    update_port_depths();

    composite::component::start();
}

template <typename T>
auto ws_sink<T>::stop() -> void {
    logger()->info("Stopping WebSocket server");

    m_server->stop();
    composite::component::stop();
}

template <typename T>
auto ws_sink<T>::process() -> composite::retval {
    using enum composite::retval;

    // === STREAM DATA ===
    auto [stream_data, stream_ts, stream_meta] = m_data_in_port.get_data();
    if (stream_data != nullptr) {
        m_frame_size = stream_data->size();

        // Handle metadata changes
        if (stream_meta.has_value()) {
            logger()->trace("Received stream metadata:\n{}", stream_meta->to_string());

            // Extract fft_size from annotations
            size_t meta_fft_size = m_fft_size;
            if (stream_meta->annotations.contains("fft_size")) {
                try {
                    meta_fft_size = std::stoul(stream_meta->annotations.at("fft_size"));
                } catch (...) {
                }
            }

            // Check if metadata changed
            if (m_fft_size != meta_fft_size || m_bandwidth != stream_meta->bandwidth ||
                m_center_frequency != stream_meta->center_frequency ||
                m_sample_rate != stream_meta->sample_rate) {

                logger()->info("Metadata changed: cf={:.0f}->{:.0f}, bw={:.0f}->{:.0f}, "
                               "sr={:.0f}->{:.0f}, fft_size={}->{}",
                               m_center_frequency, stream_meta->center_frequency, m_bandwidth,
                               stream_meta->bandwidth, m_sample_rate, stream_meta->sample_rate,
                               m_fft_size, meta_fft_size);

                m_center_frequency = stream_meta->center_frequency;
                m_sample_rate = stream_meta->sample_rate;
                m_bandwidth = stream_meta->bandwidth;
                m_fft_size = meta_fft_size;
            }

            m_stream_metadata = stream_meta.value();
        }

        if (m_frame_size != m_fft_size) {
            logger()->warn("Frame size {} does not match FFT size {}", m_frame_size, m_fft_size);
        }

        // Create header JSON (once per frame)
        nlohmann::json header_json{};
        header_json["event"] = "header";
        header_json["payload"] = make_stream_header(stream_ts);
        auto header_msg = header_json.dump();

        // Binary data (zero-copy via shared_ptr)
        auto binary_data = std::make_shared<const std::vector<uint8_t>>(
            reinterpret_cast<const uint8_t*>(stream_data->data()),
            reinterpret_cast<const uint8_t*>(stream_data->data()) +
                stream_data->size() * sizeof(T));

        // Send to clients
        auto clients = m_clients.load(std::memory_order_acquire);
        for (auto& client : *clients) {
            if (!client->should_send_stream_frame()) {
                continue;
            }

            // Check backpressure state regardless of whether we send
            check_client_backpressure(*client);

            if (client->send_queue_size.load(std::memory_order_relaxed) >= client->send_queue_hwm) {
                client->stream_frames_skipped++;
                continue;
            }

            // Send header (text), then binary data
            m_server->send_to_client(client->client_id, ws_message_data_t{header_msg}, false);
            m_server->send_to_client(client->client_id, ws_message_data_t{binary_data}, true);

            client->stream_frames_sent++;
            client->update_stream_fps_metrics();
        }
    }

    // === HISTOGRAM DATA ===
    // Only read histogram if data is available to avoid blocking
    if (m_hist_in_port.size() > 0) {
        auto [hist_data, hist_ts, hist_meta] = m_hist_in_port.get_data();
        if (hist_data != nullptr) {
            nlohmann::json histogram_json{};
            histogram_json["event"] = "histogram";
            histogram_json["payload"] = *hist_data;

            auto message = histogram_json.dump();

            // Send to all connected clients immediately
            auto clients = m_clients.load(std::memory_order_acquire);
            for (auto& client : *clients) {
                m_server->send_to_client(client->client_id, ws_message_data_t{message}, false);
                client->histogram_frames_sent++;
            }
        }
    }

    // === METRICS ===
    auto now = std::chrono::steady_clock::now();
    auto clients = m_clients.load(std::memory_order_acquire);
    for (auto& client : *clients) {
        if (client->should_send_metrics()) {
            auto metrics_json = generate_metrics_message(*client);
            m_server->send_to_client(client->client_id, ws_message_data_t{metrics_json.dump()},
                                     false);
            client->last_metrics_sent = now;
        }
    }

    return NORMAL;
}

template <typename T>
auto ws_sink<T>::update_port_depths() -> void {
    const size_t client_count = m_clients.load()->size();

    if (client_count == 0) {
        m_data_in_port.depth(0);
        m_hist_in_port.depth(0);
        logger()->info("No clients connected, disabled upstream processing");
    } else {
        m_data_in_port.depth(m_stream_port_depth);
        m_hist_in_port.depth(5);
        logger()->info("{} clients connected, stream_depth={}, histogram_depth=5", client_count,
                       m_stream_port_depth);
    }
}

template <typename T>
auto ws_sink<T>::make_stream_header(composite::timestamp ts) -> nlohmann::json {
    nlohmann::json header{};

    // Format string based on type
    if constexpr (std::is_same_v<T, float>) {
        header["format"] = "SF";
    } else if constexpr (std::is_same_v<T, double>) {
        header["format"] = "SD";
    } else if constexpr (std::is_same_v<std::complex<float>, std::complex<T>>) {
        header["format"] = "CF";
    } else if constexpr (std::is_same_v<std::complex<double>, std::complex<T>>) {
        header["format"] = "CD";
    }

    header["sampleRate"] = m_sample_rate;
    header["bandwidth"] = m_bandwidth;
    header["frequency"] = m_center_frequency;
    header["fftsize"] = m_fft_size;
    header["ape"] = m_frame_size;               // atoms per element
    header["bpa"] = m_data_size;                // bytes per atom
    header["bpe"] = m_frame_size * m_data_size; // bytes per element
    header["timecode_int"] = ts.seconds;
    header["timecode_frac"] = ts.picoseconds;

    return header;
}

template <typename T>
auto ws_sink<T>::generate_metrics_message(const client_state& client) -> nlohmann::json {
    nlohmann::json metrics{};

    metrics["event"] = "metrics";
    metrics["payload"] = {};

    // Connection status
    metrics["payload"]["activeConnections"] = m_clients.load()->size();
    metrics["payload"]["droppingFrames"] = (client.state == client_state::flow_state::backpressure);

    // Frame rate
    auto fps = client.calculate_stream_fps();
    metrics["payload"]["framesPerSecond"] = std::format("{:.1f}", fps);

    // Frame counts
    auto frames_sent = client.stream_frames_sent;
    auto frames_dropped = client.stream_frames_skipped;
    metrics["payload"]["framesSent"] = frames_sent;
    metrics["payload"]["framesDropped"] = frames_dropped;

    // Drop ratio as percentage
    auto total_attempts = frames_sent + frames_dropped;
    auto drop_ratio =
        (total_attempts > 0) ? static_cast<double>(frames_dropped) / total_attempts : 0.0;
    metrics["payload"]["dropRatio"] = std::format("{:.1f}", drop_ratio * 100.0);

    return metrics;
}

template <typename T>
auto ws_sink<T>::check_client_backpressure(client_state& client) -> void {
    const size_t queue_size = client.send_queue_size.load(std::memory_order_relaxed);
    const size_t hwm = client.send_queue_hwm;

    if (queue_size >= hwm && client.state != client_state::flow_state::backpressure) {
        logger()->warn("Client {}: Backpressure detected (queue size: {})", client.client_id,
                       queue_size);
        client.adjust_stream_fps(client_state::flow_state::backpressure);
    } else if (queue_size < hwm / 2 && client.state == client_state::flow_state::backpressure) {
        logger()->info("Client {}: Recovering from backpressure", client.client_id);
        client.adjust_stream_fps(client_state::flow_state::recovering);
    } else if (queue_size < hwm / 4 && client.state == client_state::flow_state::recovering) {
        logger()->info("Client {}: Back to healthy state", client.client_id);
        client.adjust_stream_fps(client_state::flow_state::healthy);
    }
}

template <typename T>
auto ws_sink<T>::on_client_connect(std::shared_ptr<websocket_session> session) -> void {
    auto client = std::make_shared<client_state>();
    client->client_id = session->get_id();
    client->session = session;
    client->stream_fps = m_default_stream_fps;
    client->stream_frame_interval =
        std::chrono::microseconds(static_cast<int64_t>(1'000'000.0f / client->stream_fps));
    client->send_queue_hwm = m_send_queue_hwm;

    auto now = std::chrono::steady_clock::now();
    client->last_stream_frame_sent = now;
    client->last_metrics_sent = now;

    // RCU-style update: copy-on-write (mutex protects concurrent updates)
    std::shared_ptr<std::vector<std::shared_ptr<client_state>>> new_clients;
    {
        std::lock_guard lock(m_clients_mutex);
        auto old_clients = m_clients.load(std::memory_order_acquire);
        new_clients = std::make_shared<std::vector<std::shared_ptr<client_state>>>(*old_clients);
        new_clients->push_back(client);
        m_clients.store(new_clients, std::memory_order_release);
    }

    update_port_depths();

    logger()->info("Client {} connected (total: {}, default FPS: {})", client->client_id,
                   new_clients->size(), client->stream_fps);

    // Send welcome message
    nlohmann::json welcome{};
    welcome["type"] = "welcome";
    welcome["client_id"] = client->client_id;
    welcome["stream_fps"] = client->stream_fps;
    welcome["max_stream_fps"] = m_max_stream_fps;
    welcome["stream_encoding"] = "binary";
    welcome["metrics_rate_hz"] = 1.0;

    m_server->send_to_client(client->client_id, ws_message_data_t{welcome.dump()}, false);
}

template <typename T>
auto ws_sink<T>::on_client_disconnect(const std::string& client_id) -> void {
    // RCU-style update: copy-on-write (mutex protects concurrent updates)
    std::shared_ptr<std::vector<std::shared_ptr<client_state>>> new_clients;
    {
        std::lock_guard lock(m_clients_mutex);
        auto old_clients = m_clients.load(std::memory_order_acquire);
        new_clients = std::make_shared<std::vector<std::shared_ptr<client_state>>>();

        for (auto& client : *old_clients) {
            if (client->client_id != client_id) {
                new_clients->push_back(client);
            }
        }

        m_clients.store(new_clients, std::memory_order_release);
    }

    update_port_depths();

    logger()->info("Client {} disconnected (remaining: {})", client_id, new_clients->size());
}

template <typename T>
auto ws_sink<T>::on_client_message(const std::string& client_id, const std::string& message)
    -> void {
    try {
        auto msg = nlohmann::json::parse(message);

        if (msg["type"] == "set_stream_fps") {
            auto clients = m_clients.load(std::memory_order_acquire);
            for (auto& client : *clients) {
                if (client->client_id == client_id) {
                    if (msg.contains("fps")) {
                        float requested_fps = msg["fps"];

                        // Clamp to allowed range
                        constexpr float MIN_FPS = 1.0f;
                        float new_fps = std::clamp(requested_fps, MIN_FPS, m_max_stream_fps);

                        client->set_stream_fps(new_fps);
                        logger()->info("Client {}: Stream FPS set to {:.1f} (requested: {:.1f})",
                                       client_id, new_fps, requested_fps);

                        // Send acknowledgment
                        nlohmann::json ack{};
                        ack["type"] = "stream_fps_updated";
                        ack["fps"] = new_fps;
                        ack["requested_fps"] = requested_fps;
                        m_server->send_to_client(client_id, ws_message_data_t{ack.dump()}, false);
                    }
                    break;
                }
            }
        }
    } catch (const std::exception& e) {
        logger()->error("Error parsing client message from {}: {}", client_id, e.what());
    }
}

extern "C" {
auto create(std::string_view type) -> std::shared_ptr<composite::component> {
    if (type == "f32") {
        return std::make_shared<ws_sink<float>>();
    } else if (type == "cf32") {
        return std::make_shared<ws_sink<std::complex<float>>>();
    } else if (type == "f64") {
        return std::make_shared<ws_sink<double>>();
    } else if (type == "cf64") {
        return std::make_shared<ws_sink<std::complex<double>>>();
    }
    throw std::invalid_argument("Invalid type. Supported: f32, cf32, f64, cf64");
}
}
