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

#include <composite/core/register.hpp>

#include <algorithm>
#include <charconv>
#include <complex>
#include <format>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace {

// Read an annotation as a size: typed integers directly, strings parsed (the fft
// component stamps "fft_size" as a string annotation).
auto annotation_as_size(const composite::annotation_value& v) -> std::optional<size_t> {
    if (v.holds<std::int64_t>()) {
        const auto i = v.get<std::int64_t>();
        return i >= 0 ? std::optional<size_t>{static_cast<size_t>(i)} : std::nullopt;
    }
    const auto s = v.to_string();
    size_t out{};
    const auto* end = s.data() + s.size();
    if (auto [p, ec] = std::from_chars(s.data(), end, out); ec == std::errc{} && p == end) {
        return out;
    }
    return std::nullopt;
}

auto interval_for(float fps) -> std::chrono::microseconds {
    return std::chrono::microseconds(static_cast<int64_t>(1'000'000.0f / fps));
}

} // namespace

// ========== client_state implementation ==========

auto client_state::should_send_stream_frame() const -> bool {
    std::lock_guard lock(state_mutex);
    auto now = std::chrono::steady_clock::now();
    return (now - last_stream_frame_sent) >= stream_frame_interval;
}

auto client_state::set_stream_fps(float new_fps) -> void {
    std::lock_guard lock(state_mutex);

    stream_fps = new_fps;
    default_fps = new_fps; // an explicit request also becomes the recovery ceiling
    stream_frame_interval = interval_for(stream_fps);
}

auto client_state::adjust_stream_fps(flow_state new_state) -> void {
    std::lock_guard lock(state_mutex);

    constexpr float MIN_FPS = 1.0f;
    state = new_state;

    switch (state) {
    case flow_state::backpressure:
        // Reduce by 50%, floor at 1 FPS
        stream_fps = std::max(MIN_FPS, stream_fps * 0.5f);
        stream_frame_interval = interval_for(stream_fps);
        break;

    case flow_state::recovering:
        // Slowly increase by 25%, back up to the configured/requested rate
        if (stream_fps < default_fps) {
            stream_fps = std::min(default_fps, stream_fps * 1.25f);
            stream_frame_interval = interval_for(stream_fps);
        }
        break;

    case flow_state::healthy:
        // Queue fully drained: restore the configured/requested rate (without this,
        // one transient backpressure episode degrades the client permanently)
        stream_fps = default_fps;
        stream_frame_interval = interval_for(stream_fps);
        break;
    }
}

auto client_state::current_flow_state() const -> flow_state {
    std::lock_guard lock(state_mutex);
    return state;
}

auto client_state::calculate_stream_fps() const -> float {
    std::lock_guard lock(state_mutex);

    if (recent_stream_frame_times.size() < 2) {
        return 0.0f;
    }

    auto duration = recent_stream_frame_times.back() - recent_stream_frame_times.front();
    auto duration_sec = std::chrono::duration<float>(duration).count();

    if (duration_sec <= 0.0f) {
        return 0.0f;
    }

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
ws_sink<T>::ws_sink(std::string_view id) : composite::component(id) {
    add_port(&m_hist_in_port);
    add_port(&m_data_in_port);

    add_property("port", m_ws_port);
    add_property("bind_address", m_bind_address);
    // FPS bounds >= 1: client requests are clamped to [1, max_stream_fps], and
    // std::clamp requires lo <= hi.
    add_property("default_stream_fps", m_default_stream_fps)
        .validate([](const float& v) { return v >= 1.0f; })
        .units("Hz");
    add_property("max_stream_fps", m_max_stream_fps)
        .validate([](const float& v) { return v >= 1.0f; })
        .units("Hz");
    // hwm >= 2: the backpressure ladder recovers at queue < hwm/2 and hwm/4 —
    // integer-zero thresholds (hwm 0 or 1) could never be satisfied, wedging a
    // client at the FPS floor (or skipping every frame) forever.
    add_property("send_queue_hwm", m_send_queue_hwm)
        .validate([](const size_t& v) { return v >= 2; });
    add_property("max_clients", m_max_clients)
        .validate([](const size_t& v) { return v >= 1; });
    add_property("stream_port_depth", m_stream_port_depth)
        .validate([](const size_t& v) { return v >= 1; });
    add_property("compression", m_compression);

    // A UI-facing terminal sink: upstream end-of-stream must NOT finish the worker
    // (the completion tail would stop the websocket server, dropping every client
    // and closing the listen socket for good). The stream ending is a state the UI
    // observes, not the end of this component's life. Deployments can override via
    // the standard finish_at_end property.
    set_properties(composite::properties::json{{"finish_at_end", false}});
}

template <typename T>
auto ws_sink<T>::initialize() -> void {
    logger()->info("Initializing WebSocket sink");

    m_server = std::make_unique<websocket_server>();
    m_server->set_logger(logger());
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

    // Size the rings NOW, while both sides are idle (properties are applied before
    // initialize, and nothing is started yet). A live ring must never be reallocated
    // from the websocket threads, so update_port_depths() at runtime only ever moves
    // the soft limit between 0 and these values — a plain atomic store.
    m_data_in_port.depth(m_stream_port_depth);
    m_hist_in_port.depth(HISTOGRAM_PORT_DEPTH);
    {
        std::lock_guard lock(m_clients_mutex);
        update_port_depths(); // no clients yet -> both paused (depth 0)
    }
}

template <typename T>
auto ws_sink<T>::on_worker_start() -> void {
    logger()->info("Starting WebSocket server on {}:{} (compression: {})", m_bind_address,
                   m_ws_port, m_compression ? "enabled" : "disabled");

    m_server->start(m_ws_port, m_bind_address, m_compression, m_max_clients);
    {
        std::lock_guard lock(m_clients_mutex);
        update_port_depths();
    }

    // 1 Hz per-client metrics. The worker parks on the doorbell when no data is
    // flowing, so wall-clock-paced sends have to come from their own thread.
    m_metrics_thread = std::jthread([this](std::stop_token token) {
        std::mutex mtx;
        std::condition_variable_any cv;
        std::unique_lock lock(mtx);
        while (!token.stop_requested()) {
            // Stop-token-aware wait: a plain sleep would stall every stop path
            // (shutdown, disable, restart) for up to the full second.
            cv.wait_for(lock, token, std::chrono::seconds(1), [] { return false; });
            if (token.stop_requested()) {
                break;
            }
            auto clients = m_clients.load(std::memory_order_acquire);
            if (!clients) {
                continue;
            }
            for (const auto& client : *clients) {
                // Don't grow a stalled client's queue: metrics are droppable.
                if (client->send_queue_size.load(std::memory_order_relaxed) >=
                    client->send_queue_hwm) {
                    continue;
                }
                auto metrics_json = generate_metrics_message(*client);
                m_server->send_to_client(client->client_id, ws_message_data_t{metrics_json.dump()},
                                         false);
            }
        }
    });
}

template <typename T>
auto ws_sink<T>::on_worker_stop() -> void {
    m_metrics_thread.request_stop();
    if (m_metrics_thread.joinable()) {
        m_metrics_thread.join();
    }

    logger()->info("Stopping WebSocket server");
    m_server->stop();

    // The server closed every session; drop the roster so a restart begins with a
    // clean list (stale ids would keep the ports enabled and silently eat sends).
    {
        std::lock_guard lock(m_clients_mutex);
        m_clients.store(std::make_shared<std::vector<std::shared_ptr<client_state>>>(),
                        std::memory_order_release);
        update_port_depths();
    }
}

template <typename T>
auto ws_sink<T>::process() -> composite::retval {
    using enum composite::retval;

    auto did_work = false;

    // === STREAM DATA ===
    if (auto pkt = m_data_in_port.try_get()) {
        did_work = true;
        auto& [data, ts, meta] = *pkt;

        // Steady-state fast path: the same shared metadata instance as the previous
        // packet means nothing to re-parse.
        if (meta && meta != m_last_meta) {
            update_cached_metadata(meta);
        }

        if (data.size() != m_frame_size) {
            m_frame_size = data.size();
            m_recheck_frame_size = true;
        }
        if (m_recheck_frame_size) {
            m_recheck_frame_size = false;
            if (m_fft_size != 0 && m_frame_size != m_fft_size) {
                logger()->warn("Frame size {} does not match FFT size {}", m_frame_size,
                               m_fft_size);
            }
        }

        auto clients = m_clients.load(std::memory_order_acquire);
        if (clients && !clients->empty()) {
            // Built lazily on the FIRST client that will actually take this frame:
            // input arrives at the pipeline rate but clients are FPS-throttled, so
            // most frames must not pay the JSON dump + full payload copy.
            std::string header_msg;
            std::shared_ptr<const std::vector<uint8_t>> binary_data;

            for (auto& client : *clients) {
                if (!client->should_send_stream_frame()) {
                    continue;
                }

                // Check backpressure state regardless of whether we send
                check_client_backpressure(*client);

                if (client->send_queue_size.load(std::memory_order_relaxed) >=
                    client->send_queue_hwm) {
                    client->stream_frames_skipped.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                if (!binary_data) {
                    // Header JSON + one payload copy per frame, shared across clients
                    nlohmann::json header_json{};
                    header_json["event"] = "header";
                    header_json["payload"] = make_stream_header(ts);
                    header_msg = header_json.dump();
                    binary_data = std::make_shared<const std::vector<uint8_t>>(
                        reinterpret_cast<const uint8_t*>(data.data()),
                        reinterpret_cast<const uint8_t*>(data.data()) + data.size() * sizeof(T));
                }

                // Send header (text), then binary data
                m_server->send_to_client(client->client_id, ws_message_data_t{header_msg}, false);
                m_server->send_to_client(client->client_id, ws_message_data_t{binary_data}, true);

                client->stream_frames_sent.fetch_add(1, std::memory_order_relaxed);
                client->update_stream_fps_metrics();
            }
        }
    }

    // === HISTOGRAM DATA ===
    if (auto pkt = m_hist_in_port.try_get()) {
        did_work = true;
        auto& [hist, ts, meta] = *pkt;
        (void)ts;
        (void)meta;

        auto clients = m_clients.load(std::memory_order_acquire);
        if (clients && !clients->empty()) {
            nlohmann::json histogram_json{};
            histogram_json["event"] = "histogram";
            histogram_json["payload"] = std::vector<uint64_t>(hist.begin(), hist.end());

            auto message = histogram_json.dump();

            for (auto& client : *clients) {
                // Same queue bound as the stream path: histograms and metrics are
                // droppable, and the session queue is unbounded — without this, one
                // stalled client grows its queue (and process memory) without limit.
                if (client->send_queue_size.load(std::memory_order_relaxed) >=
                    client->send_queue_hwm) {
                    continue;
                }
                m_server->send_to_client(client->client_id, ws_message_data_t{message}, false);
                client->histogram_frames_sent.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    return did_work ? NORMAL : NOOP;
}

// Call with m_clients_mutex held: the roster load and the depth writes must be
// atomic against a racing connect/disconnect, or a stale snapshot could pause the
// ports while a client is connected (or leave them open with none).
template <typename T>
auto ws_sink<T>::update_port_depths() -> void {
    const auto clients = m_clients.load(std::memory_order_acquire);
    const size_t client_count = clients ? clients->size() : 0;

    if (client_count == 0) {
        m_data_in_port.depth(0);
        m_hist_in_port.depth(0);
        logger()->info("No clients connected, disabled upstream processing");
    } else {
        m_data_in_port.depth(m_stream_port_depth);
        m_hist_in_port.depth(HISTOGRAM_PORT_DEPTH);
        logger()->info("{} clients connected, stream_depth={}, histogram_depth={}", client_count,
                       m_stream_port_depth, HISTOGRAM_PORT_DEPTH);
    }
}

template <typename T>
auto ws_sink<T>::update_cached_metadata(const composite::metadata_ptr& meta) -> void {
    const auto& md = *meta;
    logger()->trace("Received stream metadata:\n{}", md.to_string());

    // Extract fft_size from annotations
    auto meta_fft_size = m_fft_size;
    if (auto it = md.annotations.find("fft_size"); it != md.annotations.end()) {
        if (auto v = annotation_as_size(it->second)) {
            meta_fft_size = *v;
        }
    }

    // Check if metadata changed
    if (m_fft_size != meta_fft_size || m_bandwidth != md.bandwidth ||
        m_center_frequency != md.center_frequency || m_sample_rate != md.sample_rate) {

        logger()->info("Metadata changed: cf={:.0f}->{:.0f}, bw={:.0f}->{:.0f}, "
                       "sr={:.0f}->{:.0f}, fft_size={}->{}",
                       m_center_frequency, md.center_frequency, m_bandwidth, md.bandwidth,
                       m_sample_rate, md.sample_rate, m_fft_size, meta_fft_size);

        m_center_frequency = md.center_frequency;
        m_sample_rate = md.sample_rate;
        m_bandwidth = md.bandwidth;
        if (m_fft_size != meta_fft_size) {
            m_fft_size = meta_fft_size;
            m_recheck_frame_size = true; // re-verify the frame/FFT size match
        }
    }

    m_last_meta = meta;
}

template <typename T>
auto ws_sink<T>::make_stream_header(composite::timestamp ts) -> nlohmann::json {
    nlohmann::json header{};

    // Format string based on type
    if constexpr (std::is_same_v<T, float>) {
        header["format"] = "SF";
    } else if constexpr (std::is_same_v<T, double>) {
        header["format"] = "SD";
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        header["format"] = "CF";
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
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
    const auto clients = m_clients.load(std::memory_order_acquire);
    metrics["payload"]["activeConnections"] = clients ? clients->size() : 0;
    metrics["payload"]["droppingFrames"] =
        (client.current_flow_state() == client_state::flow_state::backpressure);

    // Frame rate
    auto fps = client.calculate_stream_fps();
    metrics["payload"]["framesPerSecond"] = std::format("{:.1f}", fps);

    // Frame counts
    auto frames_sent = client.stream_frames_sent.load(std::memory_order_relaxed);
    auto frames_dropped = client.stream_frames_skipped.load(std::memory_order_relaxed);
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
    const auto state = client.current_flow_state();

    if (queue_size >= hwm && state != client_state::flow_state::backpressure) {
        logger()->warn("Client {}: Backpressure detected (queue size: {})", client.client_id,
                       queue_size);
        client.adjust_stream_fps(client_state::flow_state::backpressure);
    } else if (queue_size < hwm / 2 && state == client_state::flow_state::backpressure) {
        logger()->info("Client {}: Recovering from backpressure", client.client_id);
        client.adjust_stream_fps(client_state::flow_state::recovering);
    } else if (queue_size < hwm / 4 && state == client_state::flow_state::recovering) {
        logger()->info("Client {}: Back to healthy state", client.client_id);
        client.adjust_stream_fps(client_state::flow_state::healthy);
    }
}

template <typename T>
auto ws_sink<T>::on_client_connect(std::shared_ptr<websocket_session> session) -> void {
    auto client = std::make_shared<client_state>();
    client->client_id = session->get_id();
    client->default_fps = m_default_stream_fps;
    client->stream_fps = m_default_stream_fps;
    client->stream_frame_interval = interval_for(client->stream_fps);
    client->send_queue_hwm = m_send_queue_hwm;

    auto now = std::chrono::steady_clock::now();
    client->last_stream_frame_sent = now;

    // Feed this session's queue size straight into the client's atomic — replaces
    // the roster-scan fallback wired at initialize (O(clients) per message). Safe
    // to rebind here: nothing can send to the session until it is published below.
    session->set_queue_size_callback([client](const std::string&, size_t size) {
        client->send_queue_size.store(size, std::memory_order_relaxed);
    });

    // RCU-style update: copy-on-write (mutex protects concurrent updates). The port
    // depths are updated under the SAME lock so they always match the roster a
    // racing connect/disconnect published last.
    std::shared_ptr<std::vector<std::shared_ptr<client_state>>> new_clients;
    {
        std::lock_guard lock(m_clients_mutex);
        auto old_clients = m_clients.load(std::memory_order_acquire);
        if (old_clients && old_clients->size() >= m_max_clients) {
            logger()->warn("Client {} rejected: max_clients ({}) reached", client->client_id,
                           m_max_clients);
            session->close();
            return;
        }
        new_clients = std::make_shared<std::vector<std::shared_ptr<client_state>>>(*old_clients);
        new_clients->push_back(client);
        m_clients.store(new_clients, std::memory_order_release);
        update_port_depths();
    }

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

        if (old_clients) {
            for (auto& client : *old_clients) {
                if (client->client_id != client_id) {
                    new_clients->push_back(client);
                }
            }
        }

        m_clients.store(new_clients, std::memory_order_release);
        update_port_depths(); // under the lock: depths always match the roster
    }

    logger()->info("Client {} disconnected (remaining: {})", client_id, new_clients->size());
}

template <typename T>
auto ws_sink<T>::on_client_message(const std::string& client_id, const std::string& message)
    -> void {
    try {
        auto msg = nlohmann::json::parse(message);

        if (msg.value("type", std::string{}) == "set_stream_fps") {
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

COMPOSITE_REGISTER_COMPONENT([](std::string_view id, const composite::create_args& args)
                                 -> std::shared_ptr<composite::component> {
    const auto type = args.type();
    if (type == "f32") {
        return std::make_shared<ws_sink<float>>(id);
    } else if (type == "cf32") {
        return std::make_shared<ws_sink<std::complex<float>>>(id);
    } else if (type == "f64") {
        return std::make_shared<ws_sink<double>>(id);
    } else if (type == "cf64") {
        return std::make_shared<ws_sink<std::complex<double>>>(id);
    }
    throw std::runtime_error(
        std::format("unknown type '{}' for ws_sink component (supported: f32, cf32, f64, cf64)", type));
})
