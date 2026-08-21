/*
 * Copyright (C) 2026 Geon Technologies, LLC
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
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */

// Runtime reactivation with autodiscovery must end with a RECEIVING source.
//
// udp_source::on_park_requested() signals the abort eventfd so a stop can cut packet-size
// autodiscovery short — but the framework pokes that hook on EVERY park, property writes
// included, and a reactivation (the `active=true` write) runs inside the very park whose poke
// signalled the fd. Without a drain immediately before start_recv(), the write's own signal
// aborted the discovery it was starting: any runtime write to a receiver-rebuilding property
// silently left udp_source with no running receiver, behind one warning. This test drives the
// exact documented reactivation path over UDP loopback and asserts data flows afterwards.

#include "../component.hpp"

#include <composite/composite.hpp>
#include <composite/ports/input_port.hpp>

#include <catch2/catch_test_macros.hpp>

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <netinet/in.h>
#include <random>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

using namespace std::chrono_literals;

namespace {

// Minimal consumer: counts packets delivered by udp_source's output port.
class counting_sink : public composite::component {
public:
    explicit counting_sink(std::string_view id) : composite::component(id) { add_port(&m_in); }
    auto process() -> composite::retval override {
        while (m_in.try_get()) {
            m_packets.fetch_add(1, std::memory_order_acq_rel);
        }
        return composite::retval::NOOP;
    }
    composite::input_port<composite::immutable_buffer<uint8_t>> m_in{"in", 1024};
    std::atomic<int> m_packets{0};
    composite::component::auto_stop m_auto_stop{*this};
};

// Blast sizing-candidate datagrams (>512 bytes, first byte not a V49 context type) at the
// loopback port until told to stop, so autodiscovery always has traffic to size from.
class loopback_sender {
public:
    explicit loopback_sender(uint16_t port) : m_port(port) {
        m_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        REQUIRE(m_fd >= 0);
        m_thread = std::jthread([this](std::stop_token token) {
            sockaddr_in dst{};
            dst.sin_family = AF_INET;
            dst.sin_port = htons(m_port);
            dst.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            std::array<uint8_t, 600> payload{}; // first byte 0x00: passes the sizing filter
            while (!token.stop_requested()) {
                (void)::sendto(m_fd, payload.data(), payload.size(), 0,
                               reinterpret_cast<const sockaddr*>(&dst), sizeof(dst));
                std::this_thread::sleep_for(5ms);
            }
        });
    }
    ~loopback_sender() {
        m_thread.request_stop();
        if (m_thread.joinable()) {
            m_thread.join();
        }
        ::close(m_fd);
    }

private:
    uint16_t m_port;
    int m_fd{-1};
    std::jthread m_thread;
};

auto wait_for_packets(counting_sink& sink, int n, std::chrono::milliseconds timeout) -> bool {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (sink.m_packets.load(std::memory_order_acquire) < n && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(5ms);
    }
    return sink.m_packets.load(std::memory_order_acquire) >= n;
}

} // namespace

TEST_CASE("runtime reactivation with autodiscovery still receives", "[udp_source]") {
    using composite::properties::config_type;
    using json = composite::properties::json;

    // Ephemeral-ish port with retries: the receiver binds it, and a busy port must not fail the
    // test spuriously. Construction happens inside the active=true write below, so probe the
    // port availability here first with a throwaway bind.
    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> dist(20000, 59999);
    uint16_t port = 0;
    for (int attempt = 0; attempt < 16 && port == 0; ++attempt) {
        const auto candidate = static_cast<uint16_t>(dist(rng));
        const int probe = ::socket(AF_INET, SOCK_DGRAM, 0);
        REQUIRE(probe >= 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(candidate);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(probe, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0) {
            port = candidate;
        }
        ::close(probe); // SO_REUSEADDR in the receiver makes the immediate rebind safe
    }
    REQUIRE(port != 0);

    auto src = std::make_shared<udp_source>("udp_reactivation_uut");
    auto sink = std::make_shared<counting_sink>("udp_reactivation_sink");
    REQUIRE(src->connect("data_out", sink, "in"));

    // INITIALIZE with the source INACTIVE and autodiscovery in play (no msg_size override) —
    // the default configuration shape, and the shape the regression needs.
    src->set_properties(json{{"ip_addr", "127.0.0.1"},
                             {"port", port},
                             {"autodiscovery_timeout", 3},
                             {"active", false}},
                        config_type::INITIALIZE);
    sink->start();
    src->start();

    loopback_sender sender{port};

    // THE regression path: a RUNTIME write parks the worker (poking on_park_requested, which
    // signals the abort fd) and then rebuilds + starts the receiver inside that same park.
    // Pre-fix, the write's own poke aborted autodiscovery instantly and the source came back
    // with no receiver; post-fix, discovery sizes from the loopback traffic and data flows.
    src->set_properties(json{{"active", true}}, config_type::RUNTIME);

    CHECK(wait_for_packets(*sink, 10, 5000ms));

    // And the reactivation is repeatable: deactivate (drops the receiver), reactivate again.
    src->set_properties(json{{"active", false}}, config_type::RUNTIME);
    sink->m_packets.store(0);
    src->set_properties(json{{"active", true}}, config_type::RUNTIME);
    CHECK(wait_for_packets(*sink, 10, 5000ms));

    src->stop();
    sink->stop();
}
