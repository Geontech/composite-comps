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

// End-to-end loopback tests: drive the real component's process() through a connected
// port and verify the datagrams actually arrive on bound 127.0.0.1 sockets — the
// default-destination path, metadata-driven routing with per-stream latching, and the
// sendmmsg batch flush on idle.

#include "component.hpp"

#include <catch2/catch_test_macros.hpp>

#include <composite/composite.hpp>
#include <composite/ports/output_port.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

// A bound localhost UDP receiver with a receive timeout.
struct udp_receiver {
    int fd{-1};
    uint16_t port{0};

    udp_receiver() {
        fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        REQUIRE(fd >= 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0; // ephemeral
        REQUIRE(::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
        socklen_t len = sizeof(addr);
        REQUIRE(::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len) == 0);
        port = ntohs(addr.sin_port);
        timeval tv{.tv_sec = 0, .tv_usec = 500'000};
        ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    }
    ~udp_receiver() {
        if (fd >= 0) {
            ::close(fd);
        }
    }
    udp_receiver(const udp_receiver&) = delete;
    auto operator=(const udp_receiver&) -> udp_receiver& = delete;

    // Receive one datagram; empty vector on timeout.
    auto recv_one() -> std::vector<uint8_t> {
        std::vector<uint8_t> buf(65536);
        auto n = ::recv(fd, buf.data(), buf.size(), 0);
        if (n <= 0) {
            return {};
        }
        buf.resize(static_cast<std::size_t>(n));
        return buf;
    }
};

auto make_packet(std::size_t size, uint8_t seed) -> composite::immutable_buffer<uint8_t> {
    auto v = std::make_shared<std::vector<uint8_t>>(size);
    std::iota(v->begin(), v->end(), seed);
    return composite::immutable_buffer<uint8_t>(v);
}

} // namespace

// Global scope so the `friend struct UdpSinkTestFixture;` in udp_sink grants access.
struct UdpSinkTestFixture {
    static auto next_id() -> std::string {
        static std::atomic<int> n{0};
        return "udp_sink_test_" + std::to_string(n.fetch_add(1));
    }

    std::shared_ptr<udp_sink> uut{std::make_shared<udp_sink>(next_id())};
    composite::output_port<composite::immutable_buffer<uint8_t>> src{"src"};

    UdpSinkTestFixture() { src.connect(&uut->m_in_port); }

    void configure(const composite::properties::json& props) {
        uut->set_properties(props, composite::properties::config_type::INITIALIZE);
    }

    // Send one packet through the port and run one process() cycle.
    void feed(composite::immutable_buffer<uint8_t> pkt, composite::metadata_ptr md = nullptr) {
        src.send_data(std::move(pkt), composite::timestamp{0, 0}, std::move(md));
        REQUIRE(uut->process() == composite::retval::NORMAL);
    }

    // Run process() on an empty ring: returns NOOP and flushes any batched packets.
    void drain() { REQUIRE(uut->process() == composite::retval::NOOP); }

    // Friendship does not extend to the classes TEST_CASE_METHOD derives from this fixture,
    // so private state is surfaced through fixture members.
    auto stream_states_size() const -> std::size_t { return uut->m_stream_states.size(); }
};

TEST_CASE_METHOD(UdpSinkTestFixture, "udp_sink sends to the configured default destination",
                 "[udp_sink][integration]") {
    udp_receiver rx;
    configure({{"socket_type", "send"},
               {"default_dest_ip", "127.0.0.1"},
               {"default_dest_port", rx.port}});

    feed(make_packet(64, 1));
    feed(make_packet(128, 2));

    auto p1 = rx.recv_one();
    auto p2 = rx.recv_one();
    REQUIRE(p1.size() == 64);
    REQUIRE(p2.size() == 128);
    REQUIRE(p1[0] == 1);
    REQUIRE(p2[0] == 2);
}

TEST_CASE_METHOD(UdpSinkTestFixture, "udp_sink routes by metadata annotations and latches per stream",
                 "[udp_sink][integration]") {
    udp_receiver rx_default;
    udp_receiver rx_routed;
    configure({{"socket_type", "send"},
               {"default_dest_ip", "127.0.0.1"},
               {"default_dest_port", rx_default.port}});

    // Metadata carrying a destination overrides the default...
    composite::metadata md;
    md.annotations["dest_ip"] = "127.0.0.1";
    md.annotations["dest_port"] = static_cast<std::int64_t>(rx_routed.port);
    auto routed = composite::make_metadata(std::move(md));
    feed(make_packet(32, 3), routed);
    auto p = rx_routed.recv_one();
    REQUIRE(p.size() == 32);
    REQUIRE(p[0] == 3);

    // ...the same shared instance keeps hitting the fast path...
    feed(make_packet(32, 4), routed);
    REQUIRE(rx_routed.recv_one().size() == 32);

    // ...and a packet WITHOUT metadata keeps the latched destination (not the default).
    feed(make_packet(32, 5));
    REQUIRE(rx_routed.recv_one().size() == 32);
    REQUIRE(rx_default.recv_one().empty()); // nothing ever went to the default
}

TEST_CASE_METHOD(UdpSinkTestFixture, "an invalid dest_ip annotation is ignored, not latched",
                 "[udp_sink][integration]") {
    // An unparsable IP used to be latched verbatim and then failed EVERY send for the
    // stream with a per-packet error log; now it is rejected at the latch and the
    // configured default carries the traffic.
    udp_receiver rx_default;
    configure({{"socket_type", "send"},
               {"default_dest_ip", "127.0.0.1"},
               {"default_dest_port", rx_default.port}});

    composite::metadata md;
    md.annotations["dest_ip"] = "not-an-ip";
    feed(make_packet(48, 7), composite::make_metadata(std::move(md)));

    auto p = rx_default.recv_one();
    REQUIRE(p.size() == 48);
    REQUIRE(p[0] == 7);
}

TEST_CASE_METHOD(UdpSinkTestFixture, "a double-typed dest_port annotation routes correctly",
                 "[udp_sink][integration]") {
    // JSON-sourced metadata often types numbers as double; an exact-integral double must
    // parse as a port rather than fail the '5000.000000' string round-trip.
    udp_receiver rx;
    configure({{"socket_type", "send"}, {"default_dest_ip", "127.0.0.1"}, {"default_dest_port", 1}});

    composite::metadata md;
    md.annotations["dest_ip"] = "127.0.0.1";
    md.annotations["dest_port"] = static_cast<double>(rx.port);
    feed(make_packet(24, 9), composite::make_metadata(std::move(md)));

    auto p = rx.recv_one();
    REQUIRE(p.size() == 24);
    REQUIRE(p[0] == 9);
}

TEST_CASE_METHOD(UdpSinkTestFixture, "per-stream destination state is bounded",
                 "[udp_sink][integration]") {
    // stream_id comes from untrusted metadata: cycling it must not grow the latch map
    // without bound. Above the cap the least-recently-resolved stream is evicted.
    udp_receiver rx;
    configure({{"socket_type", "send"},
               {"default_dest_ip", "127.0.0.1"},
               {"default_dest_port", rx.port}});

    for (int i = 0; i < 300; ++i) {
        composite::metadata md;
        md.annotations["stream_id"] = static_cast<std::int64_t>(i);
        feed(make_packet(8, 1), composite::make_metadata(std::move(md)));
    }
    CHECK(stream_states_size() <= 256);
    // Traffic itself is unaffected: every packet still went out.
    std::size_t received = 0;
    while (!rx.recv_one().empty()) {
        ++received;
    }
    CHECK(received == 300);
}

TEST_CASE_METHOD(UdpSinkTestFixture, "udp_sink flushes the sendmmsg batch when the input goes idle",
                 "[udp_sink][integration]") {
    udp_receiver rx;
    configure({{"socket_type", "sendmmsg"},
               {"batch_size", 64}, // far larger than the 3 packets below: only a flush delivers them
               {"default_dest_ip", "127.0.0.1"},
               {"default_dest_port", rx.port}});

    feed(make_packet(16, 1));
    feed(make_packet(16, 2));
    feed(make_packet(16, 3));
    drain(); // empty ring -> NOOP path flushes the batch

    for (int i = 1; i <= 3; ++i) {
        auto p = rx.recv_one();
        REQUIRE(p.size() == 16);
        REQUIRE(p[0] == static_cast<uint8_t>(i));
    }
}
