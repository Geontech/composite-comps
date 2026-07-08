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

// End-to-end loopback tests: run the real component with a live websocket server on
// 127.0.0.1 and a real IXWebSocket client — verify the welcome handshake, the
// header+binary stream protocol (with metadata-derived header fields), histogram
// events, the 1 Hz metrics broadcast, and the no-clients port-depth gating.

#include "component.hpp"

#include <catch2/catch_test_macros.hpp>

#include <composite/composite.hpp>
#include <composite/ports/output_port.hpp>

#include <ixwebsocket/IXWebSocket.h>
#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace std::chrono_literals;
using json = nlohmann::json;

namespace {

// Grab a currently-free TCP port from the kernel (bind to 0, read it back, close).
auto free_tcp_port() -> uint16_t {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    REQUIRE(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
    socklen_t len = sizeof(addr);
    REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
    auto port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

// A test websocket client that logs everything it receives.
struct ws_client {
    struct received {
        std::string data;
        bool binary{false};
    };

    ix::WebSocket socket;
    std::mutex mtx;
    std::condition_variable cv;
    std::vector<received> messages;
    bool open{false};

    explicit ws_client(uint16_t port) {
        socket.setUrl("ws://127.0.0.1:" + std::to_string(port));
        socket.disableAutomaticReconnection();
        socket.setOnMessageCallback([this](const ix::WebSocketMessagePtr& msg) {
            std::lock_guard lock(mtx);
            if (msg->type == ix::WebSocketMessageType::Open) {
                open = true;
            } else if (msg->type == ix::WebSocketMessageType::Message) {
                messages.push_back({msg->str, msg->binary});
            }
            cv.notify_all();
        });
        socket.start();
    }

    ~ws_client() { socket.stop(); }

    auto wait_for_open(std::chrono::milliseconds timeout) -> bool {
        std::unique_lock lock(mtx);
        return cv.wait_for(lock, timeout, [this] { return open; });
    }

    // Wait until a message satisfying @p pred arrives; returns it, or nullopt on timeout.
    template <typename Pred>
    auto wait_for_message(Pred pred, std::chrono::milliseconds timeout) -> std::optional<received> {
        std::unique_lock lock(mtx);
        size_t scanned = 0;
        std::optional<received> found;
        cv.wait_for(lock, timeout, [&] {
            for (; scanned < messages.size(); ++scanned) {
                if (pred(messages[scanned])) {
                    found = messages[scanned];
                    return true;
                }
            }
            return false;
        });
        return found;
    }
};

// Matches a text message whose JSON has the given value at the given key.
auto text_event = [](std::string_view key, std::string_view value) {
    return [key = std::string(key), value = std::string(value)](const ws_client::received& m) {
        if (m.binary) {
            return false;
        }
        auto j = json::parse(m.data, nullptr, false);
        return !j.is_discarded() && j.value(key, std::string{}) == value;
    };
};

// The component under test with connected feeder ports, initialized and started.
struct sink_fixture {
    uint16_t port{free_tcp_port()};
    std::shared_ptr<ws_sink<float>> comp{std::make_shared<ws_sink<float>>("ws_sink_test")};
    composite::output_port<composite::immutable_buffer<float>> data_src{"data_src"};
    composite::output_port<composite::immutable_buffer<uint64_t>> hist_src{"hist_src"};

    sink_fixture() {
        using composite::properties::config_type;
        comp->set_properties(composite::properties::json{{"port", port},
                                                         {"bind_address", "127.0.0.1"},
                                                         // high FPS so back-to-back test frames
                                                         // are not throttled away
                                                         {"default_stream_fps", 500.0},
                                                         {"max_stream_fps", 1000.0},
                                                         {"stream_port_depth", 4}},
                             config_type::INITIALIZE);
        comp->initialize();
        auto* data_in =
            comp->get_port<composite::input_port<composite::immutable_buffer<float>>>("data_in");
        auto* hist_in = comp->get_port<composite::input_port<composite::immutable_buffer<uint64_t>>>(
            "histogram_in");
        REQUIRE(data_in != nullptr);
        REQUIRE(hist_in != nullptr);
        data_src.connect(data_in);
        hist_src.connect(hist_in);
        comp->start();
    }

    ~sink_fixture() { comp->stop(); }

    auto data_in_depth() -> size_t {
        return comp->get_port<composite::input_port<composite::immutable_buffer<float>>>("data_in")
            ->depth();
    }
};

auto make_float_buffer(size_t n) -> composite::immutable_buffer<float> {
    auto v = std::make_shared<std::vector<float>>(n);
    for (size_t i = 0; i < n; ++i) {
        (*v)[i] = static_cast<float>(i);
    }
    return composite::immutable_buffer<float>(v);
}

// Poll until @p cond holds (or timeout); the connect/disconnect callbacks run on the
// websocket threads, so the effects are observed asynchronously.
template <typename Cond>
auto eventually(Cond cond, std::chrono::milliseconds timeout = 2000ms) -> bool {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (cond()) {
            return true;
        }
        std::this_thread::sleep_for(10ms);
    }
    return cond();
}

} // namespace

TEST_CASE("ws_sink gates input depth on client presence") {
    sink_fixture fx;

    // No clients: both inputs paused (depth 0 -> upstream packets dropped at the ring).
    REQUIRE(fx.data_in_depth() == 0);

    {
        ws_client client(fx.port);
        REQUIRE(client.wait_for_open(3000ms));
        REQUIRE(client.wait_for_message(text_event("type", "welcome"), 3000ms).has_value());
        REQUIRE(eventually([&] { return fx.data_in_depth() == 4; }));
    }

    // Client gone: back to paused.
    REQUIRE(eventually([&] { return fx.data_in_depth() == 0; }));
}

TEST_CASE("ws_sink streams welcome, header+binary, histogram, and metrics to a client") {
    constexpr size_t FRAME = 16;
    sink_fixture fx;

    ws_client client(fx.port);
    REQUIRE(client.wait_for_open(3000ms));

    // Welcome handshake carries the negotiated FPS bounds.
    auto welcome = client.wait_for_message(text_event("type", "welcome"), 3000ms);
    REQUIRE(welcome.has_value());
    auto welcome_json = json::parse(welcome->data);
    CHECK(welcome_json["stream_fps"].get<float>() == 500.0f);
    CHECK(welcome_json["max_stream_fps"].get<float>() == 1000.0f);

    // Wait for the depth to open before feeding, or the ring drops the packets.
    REQUIRE(eventually([&] { return fx.data_in_depth() == 4; }));

    // Stream frames with metadata; the fft component stamps fft_size as a string
    // annotation, so exercise that exact form.
    composite::metadata md;
    md.sample_rate = 1.0e6;
    md.bandwidth = 0.8e6;
    md.center_frequency = 2.4e9;
    md.annotations["fft_size"] = std::to_string(FRAME);
    auto md_ptr = composite::make_metadata(std::move(md));

    // Feed until the client sees a header (the first frame after connect can be
    // FPS-throttled or race the depth toggle).
    std::optional<ws_client::received> header;
    for (int i = 0; i < 100 && !header; ++i) {
        fx.data_src.send_data(make_float_buffer(FRAME), composite::timestamp{123, 456}, md_ptr);
        header = client.wait_for_message(text_event("event", "header"), 100ms);
    }
    REQUIRE(header.has_value());

    auto header_json = json::parse(header->data)["payload"];
    CHECK(header_json["format"] == "SF");
    CHECK(header_json["sampleRate"].get<double>() == 1.0e6);
    CHECK(header_json["bandwidth"].get<double>() == 0.8e6);
    CHECK(header_json["frequency"].get<double>() == 2.4e9);
    CHECK(header_json["fftsize"].get<size_t>() == FRAME);
    CHECK(header_json["ape"].get<size_t>() == FRAME);
    CHECK(header_json["bpa"].get<size_t>() == sizeof(float));
    CHECK(header_json["bpe"].get<size_t>() == FRAME * sizeof(float));
    CHECK(header_json["timecode_int"].get<uint32_t>() == 123);
    CHECK(header_json["timecode_frac"].get<uint64_t>() == 456);

    // The binary frame follows the header and carries the raw samples.
    auto binary = client.wait_for_message([](const ws_client::received& m) { return m.binary; },
                                          3000ms);
    REQUIRE(binary.has_value());
    REQUIRE(binary->data.size() == FRAME * sizeof(float));
    std::vector<float> samples(FRAME);
    std::memcpy(samples.data(), binary->data.data(), binary->data.size());
    for (size_t i = 0; i < FRAME; ++i) {
        CHECK(samples[i] == static_cast<float>(i));
    }

    // Histogram packets go out as JSON events to every client.
    auto bins = std::make_shared<std::vector<uint64_t>>(std::vector<uint64_t>{1, 2, 3, 42});
    fx.hist_src.send_data(composite::immutable_buffer<uint64_t>(bins), composite::timestamp{}, nullptr);
    auto hist = client.wait_for_message(text_event("event", "histogram"), 3000ms);
    REQUIRE(hist.has_value());
    CHECK(json::parse(hist->data)["payload"] == json({1, 2, 3, 42}));

    // Metrics broadcast at 1 Hz from the metrics thread.
    auto metrics = client.wait_for_message(text_event("event", "metrics"), 3000ms);
    REQUIRE(metrics.has_value());
    auto metrics_json = json::parse(metrics->data)["payload"];
    CHECK(metrics_json.contains("activeConnections"));
    CHECK(metrics_json.contains("framesSent"));
    CHECK(metrics_json["activeConnections"].get<size_t>() == 1);
}

TEST_CASE("ws_sink survives upstream end-of-stream") {
    constexpr size_t FRAME = 8;
    sink_fixture fx;

    ws_client client(fx.port);
    REQUIRE(client.wait_for_open(3000ms));
    REQUIRE(client.wait_for_message(text_event("type", "welcome"), 3000ms).has_value());
    REQUIRE(eventually([&] { return fx.data_in_depth() == 4; }));

    // Stream a frame, then end both input streams. A UI-facing sink must NOT
    // self-finish (which would stop the websocket server and drop every client).
    std::optional<ws_client::received> header;
    for (int i = 0; i < 100 && !header; ++i) {
        fx.data_src.send_data(make_float_buffer(FRAME), composite::timestamp{1, 0}, nullptr);
        header = client.wait_for_message(text_event("event", "header"), 100ms);
    }
    REQUIRE(header.has_value());

    fx.data_src.send_eos();
    fx.hist_src.send_eos();

    // The worker idles at end-of-stream but the server stays up: the client stays
    // connected and the control channel still answers.
    std::this_thread::sleep_for(500ms);
    client.socket.sendText(json{{"type", "set_stream_fps"}, {"fps", 30.0}}.dump());
    auto ack = client.wait_for_message(text_event("type", "stream_fps_updated"), 3000ms);
    REQUIRE(ack.has_value());

    // And a NEW client can still connect after EOS.
    ws_client late(fx.port);
    REQUIRE(late.wait_for_open(3000ms));
    REQUIRE(late.wait_for_message(text_event("type", "welcome"), 3000ms).has_value());
}

TEST_CASE("ws_sink applies a client FPS change request") {
    sink_fixture fx;

    ws_client client(fx.port);
    REQUIRE(client.wait_for_open(3000ms));
    REQUIRE(client.wait_for_message(text_event("type", "welcome"), 3000ms).has_value());

    // Request above the max: the ack reports the clamped value.
    client.socket.sendText(json{{"type", "set_stream_fps"}, {"fps", 5000.0}}.dump());
    auto ack = client.wait_for_message(text_event("type", "stream_fps_updated"), 3000ms);
    REQUIRE(ack.has_value());
    auto ack_json = json::parse(ack->data);
    CHECK(ack_json["fps"].get<float>() == 1000.0f);
    CHECK(ack_json["requested_fps"].get<float>() == 5000.0f);
}
