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
        while (auto pkt = m_in.try_get()) {
            {
                std::scoped_lock lk{m_md_mtx};
                m_last_md = std::get<2>(*pkt);
            }
            m_packets.fetch_add(1, std::memory_order_acq_rel);
        }
        return composite::retval::NOOP;
    }
    auto last_session() -> std::string {
        std::scoped_lock lk{m_md_mtx};
        if (m_last_md == nullptr) { return {}; }
        const auto it = m_last_md->annotations.find("stream_session");
        return it == m_last_md->annotations.end() ? std::string{} : it->second.to_string();
    }
    composite::input_port<composite::immutable_buffer<uint8_t>> m_in{"in", 1024};
    std::atomic<int> m_packets{0};
    std::mutex m_md_mtx;
    composite::metadata_ptr m_last_md;
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
    // Every packet carries the IN-BAND stream-session annotation...
    const auto first_session = sink->last_session();
    CHECK(!first_session.empty());

    // And the reactivation is repeatable: deactivate (drops the receiver), reactivate again.
    src->set_properties(json{{"active", false}}, config_type::RUNTIME);
    sink->m_packets.store(0);
    src->set_properties(json{{"active", true}}, config_type::RUNTIME);
    CHECK(wait_for_packets(*sink, 10, 5000ms));
    // ...and the rebuilt receiver is a NEW session (downstream resets its stream state).
    CHECK(sink->last_session() != first_session);
    CHECK(!sink->last_session().empty());

    src->stop();
    sink->stop();
}

TEST_CASE("frame pool grows after a protocol flip to larger datagrams", "[udp_source]") {
    using composite::properties::config_type;
    using json = composite::properties::json;

    // Port probing as above.
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
        ::close(probe);
    }
    REQUIRE(port != 0);

    auto src = std::make_shared<udp_source>("udp_growth_uut");
    auto sink = std::make_shared<counting_sink>("udp_growth_sink");
    REQUIRE(src->connect("data_out", sink, "in"));
    src->set_properties(json{{"ip_addr", "127.0.0.1"},
                             {"port", port},
                             {"autodiscovery_timeout", 3},
                             {"active", true}},
                        config_type::INITIALIZE);
    sink->start();

    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    REQUIRE(fd >= 0);
    sockaddr_in dst{};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(port);
    dst.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    auto blast = [&](std::size_t bytes, int count) {
        std::vector<uint8_t> payload(bytes, 0);  // first byte 0x00: passes the sizing filter
        for (int i = 0; i < count; ++i) {
            (void)::sendto(fd, payload.data(), payload.size(), 0,
                           reinterpret_cast<const sockaddr*>(&dst), sizeof(dst));
            std::this_thread::sleep_for(2ms);
        }
    };

    // Phase 1: SDDS-sized traffic, flowing BEFORE the source starts so autodiscovery (which
    // runs during activation) has packets to size from: 1080-byte datagrams -> 2048-byte
    // frames after bit_ceil.
    std::jthread phase1([&](std::stop_token token) {
        while (!token.stop_requested()) { blast(1080, 1); }
    });
    src->start();
    REQUIRE(wait_for_packets(*sink, 10, 5000ms));
    phase1.request_stop();
    phase1 = {};

    // Phase 2: the pipeline is re-steered to a protocol with LARGER datagrams (a multi-KB
    // V49 stream). Pre-fix, every one of these arrived MSG_TRUNC and was dropped forever —
    // the pipeline died at the socket. Post-fix, truncation triggers pool growth (doubling,
    // 2048 -> 4096 -> 8192) and reception resumes.
    sink->m_packets.store(0);
    std::jthread phase2([&](std::stop_token token) {
        while (!token.stop_requested()) { blast(6000, 1); }
    });
    CHECK(wait_for_packets(*sink, 10, 5000ms));
    phase2.request_stop();
    phase2 = {};

    ::close(fd);
    src->stop();
    sink->stop();
}

TEST_CASE("a recreated source under the same id starts a new stream session", "[udp_source]") {
    using composite::properties::config_type;
    using json = composite::properties::json;

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
        ::close(probe);
    }
    REQUIRE(port != 0);

    // Component ids are unique among LIVE components only: a destroyed-and-recreated source
    // with the SAME id must still mint a fresh session (process-wide counter), or a parser
    // that outlived the swap would never reset.
    auto run_once = [&](std::string* session_out) {
        auto src = std::make_shared<udp_source>("udp_recreate_uut");
        auto sink = std::make_shared<counting_sink>("udp_recreate_sink_" + std::to_string(rand()));
        REQUIRE(src->connect("data_out", sink, "in"));
        src->set_properties(json{{"ip_addr", "127.0.0.1"},
                                 {"port", port},
                                 {"autodiscovery_timeout", 3},
                                 {"active", true}},
                            config_type::INITIALIZE);
        sink->start();
        loopback_sender sender{port};
        src->start();
        REQUIRE(wait_for_packets(*sink, 5, 5000ms));
        *session_out = sink->last_session();
        src->stop();
        sink->stop();
    };

    std::string first;
    std::string second;
    run_once(&first);
    run_once(&second);   // same component id, new lifetime
    CHECK(!first.empty());
    CHECK(!second.empty());
    CHECK(first != second);
}
