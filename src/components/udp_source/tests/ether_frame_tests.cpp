/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

// Hermetic tests for the Ethernet/VLAN walk the DPDK receiver uses (ether_frame.hpp), and for
// its composition with the shared IP/UDP classifier (classify_frame) — together these are the
// complete untrusted-bytes parse of the DPDK data path, exercised here without DPDK: the
// backend itself cannot be compiled (COMPS_HAS_DPDK off) or run in CI, so the pure helpers
// carry all of its parser coverage.

#include "socket/ether_frame.hpp"
#include "socket/packet_mmap_frame.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <vector>

namespace {

// -- Ethernet/VLAN/IPv4/UDP frame builder (all lengths caller-controlled: tests forge
//    malformed values on purpose) --

auto push_be16(std::vector<std::uint8_t>& v, std::uint16_t x) -> void {
    v.push_back(static_cast<std::uint8_t>(x >> 8));
    v.push_back(static_cast<std::uint8_t>(x & 0xFF));
}

struct udp_frame_opts {
    std::vector<std::uint16_t> vlan_tags{};
    std::uint32_t dst_ip_be{0x0100A8C0};  // 192.168.0.1 in network order (LE host bytes)
    std::uint16_t dst_port{5000};
    std::uint16_t frag_off_field{0};      // host-order value of the IPv4 frag_off field
    std::uint8_t ihl{5};
    std::uint16_t udp_len_override{0};    // 0 = correct (8 + payload)
};

// Build [ethernet][vlan tags: outer 802.1ad, innermost 802.1Q][ipv4][udp][payload], with
// every forgeable field caller-controlled.
auto make_udp_frame(std::size_t payload_len, const udp_frame_opts& o = {})
    -> std::vector<std::uint8_t> {
    std::vector<std::uint8_t> f;
    f.insert(f.end(), 6, 0x02);
    f.insert(f.end(), 6, 0x04);
    for (std::size_t i = 0; i < o.vlan_tags.size(); ++i) {
        push_be16(f, i + 1 < o.vlan_tags.size() ? udp::k_ethertype_qinq : udp::k_ethertype_vlan);
        push_be16(f, o.vlan_tags[i]);
    }
    push_be16(f, udp::k_ethertype_ipv4);

    const std::size_t ihl_bytes = static_cast<std::size_t>(o.ihl) * 4;
    const std::uint16_t udp_len =
        o.udp_len_override != 0 ? o.udp_len_override
                                : static_cast<std::uint16_t>(8 + payload_len);
    // IPv4 header
    f.push_back(static_cast<std::uint8_t>(0x40 | o.ihl));  // version 4, IHL
    f.push_back(0);                                         // TOS
    push_be16(f, static_cast<std::uint16_t>(ihl_bytes + 8 + payload_len));  // total length
    push_be16(f, 0x1234);                                   // ID
    push_be16(f, o.frag_off_field);                         // flags/fragment offset
    f.push_back(64);                                        // TTL
    f.push_back(17);                                        // protocol = UDP
    push_be16(f, 0);                                        // checksum (unchecked)
    push_be16(f, 0x0A00); push_be16(f, 0x0001);             // src 10.0.0.1
    for (int i = 0; i < 4; ++i) {                           // dst ip (network byte order)
        f.push_back(reinterpret_cast<const std::uint8_t*>(&o.dst_ip_be)[i]);
    }
    for (std::size_t i = 5 * 4; i < ihl_bytes; ++i) {       // IP options padding
        f.push_back(0x00);
    }
    // UDP header
    push_be16(f, 4000);        // src port
    push_be16(f, o.dst_port);  // dst port
    push_be16(f, udp_len);     // length
    push_be16(f, 0);           // checksum
    for (std::size_t i = 0; i < payload_len; ++i) {
        f.push_back(static_cast<std::uint8_t>(i));
    }
    return f;
}

} // namespace

TEST_CASE("locate_ipv4: bare IPv4 frame", "[ether_frame]") {
    const auto f = make_udp_frame(32);
    const auto r = udp::locate_ipv4(f.data(), f.size());
    REQUIRE(r.verdict == udp::l3_locate::kind::ipv4);
    CHECK(r.offset == 14);
    CHECK(r.ether_type == udp::k_ethertype_ipv4);
}

TEST_CASE("locate_ipv4: single 802.1Q tag", "[ether_frame]") {
    const auto f = make_udp_frame(32, {.vlan_tags = {100}});
    const auto r = udp::locate_ipv4(f.data(), f.size());
    REQUIRE(r.verdict == udp::l3_locate::kind::ipv4);
    CHECK(r.offset == 18);
}

TEST_CASE("locate_ipv4: Q-in-Q (802.1ad outer + 802.1Q inner)", "[ether_frame]") {
    // The pre-fix walk only matched TPID 0x8100, so an 802.1ad outer tag (0x88A8) made the
    // whole frame read as non-IPv4 and dropped — despite the comment claiming Q-in-Q support.
    const auto f = make_udp_frame(32, {.vlan_tags = {200, 100}});
    const auto r = udp::locate_ipv4(f.data(), f.size());
    REQUIRE(r.verdict == udp::l3_locate::kind::ipv4);
    CHECK(r.offset == 22);
}

TEST_CASE("locate_ipv4: non-IPv4 ethertype reports the resolved type", "[ether_frame]") {
    std::vector<std::uint8_t> f;
    f.insert(f.end(), 6, 0x02);
    f.insert(f.end(), 6, 0x04);
    push_be16(f, 0x0806);  // ARP
    f.insert(f.end(), 28, 0x00);
    const auto r = udp::locate_ipv4(f.data(), f.size());
    REQUIRE(r.verdict == udp::l3_locate::kind::non_ipv4);
    CHECK(r.ether_type == 0x0806);
}

TEST_CASE("locate_ipv4: IPv6 behind a VLAN tag is non_ipv4", "[ether_frame]") {
    std::vector<std::uint8_t> f;
    f.insert(f.end(), 6, 0x02);
    f.insert(f.end(), 6, 0x04);
    push_be16(f, udp::k_ethertype_vlan);
    push_be16(f, 100);      // TCI
    push_be16(f, 0x86DD);   // IPv6
    f.insert(f.end(), 40, 0x00);
    const auto r = udp::locate_ipv4(f.data(), f.size());
    REQUIRE(r.verdict == udp::l3_locate::kind::non_ipv4);
    CHECK(r.ether_type == 0x86DD);
}

TEST_CASE("locate_ipv4: truncation", "[ether_frame]") {
    SECTION("shorter than an Ethernet header") {
        const std::vector<std::uint8_t> f(13, 0x00);
        CHECK(udp::locate_ipv4(f.data(), f.size()).verdict == udp::l3_locate::kind::truncated);
    }
    SECTION("VLAN TPID present but tag body cut off") {
        std::vector<std::uint8_t> f;
        f.insert(f.end(), 6, 0x02);
        f.insert(f.end(), 6, 0x04);
        push_be16(f, udp::k_ethertype_vlan);
        f.push_back(0x00);  // 1 of the 4 tag bytes
        CHECK(udp::locate_ipv4(f.data(), f.size()).verdict == udp::l3_locate::kind::truncated);
    }
    SECTION("a runaway stack of tags never reads past the frame") {
        // 64 nested TPIDs then nothing: each iteration is bounds-checked.
        std::vector<std::uint8_t> f;
        f.insert(f.end(), 6, 0x02);
        f.insert(f.end(), 6, 0x04);
        for (int i = 0; i < 64; ++i) {
            push_be16(f, udp::k_ethertype_vlan);
            push_be16(f, 1);
        }
        // ends mid-stack: last encapsulated type says VLAN again but no bytes follow
        CHECK(udp::locate_ipv4(f.data(), f.size()).verdict == udp::l3_locate::kind::truncated);
    }
}

// -- Composition: exactly the DPDK receive path (locate_ipv4 then classify_frame) --

namespace {
auto classify_ether(const std::vector<std::uint8_t>& f, std::optional<std::uint32_t> dst_ip_be,
                    std::uint16_t dst_port) -> udp::frame_verdict {
    const auto l3 = udp::locate_ipv4(f.data(), f.size());
    REQUIRE(l3.verdict == udp::l3_locate::kind::ipv4);
    return udp::classify_frame(f.data() + l3.offset, f.size() - l3.offset, dst_ip_be, dst_port);
}
} // namespace

TEST_CASE("dpdk parse composition: matching datagram forwards the exact payload",
          "[ether_frame][classify]") {
    const auto f = make_udp_frame(100, {.vlan_tags = {42}});
    const auto v = classify_ether(f, 0x0100A8C0, 5000);
    REQUIRE(v.verdict == udp::frame_verdict::kind::forward);
    const auto l3 = udp::locate_ipv4(f.data(), f.size());
    CHECK(l3.offset + v.payload_off == f.size() - 100);
    CHECK(v.payload_len == 100);
    CHECK(f[l3.offset + v.payload_off] == 0);
    CHECK(f[l3.offset + v.payload_off + 99] == 99);
}

TEST_CASE("dpdk parse composition: IP options (IHL > 5) shift the payload correctly",
          "[ether_frame][classify]") {
    const auto f = make_udp_frame(64, {.ihl = 8});  // 12 bytes of options
    const auto v = classify_ether(f, 0x0100A8C0, 5000);
    REQUIRE(v.verdict == udp::frame_verdict::kind::forward);
    CHECK(v.payload_off == 8 * 4 + 8);
    CHECK(v.payload_len == 64);
}

TEST_CASE("dpdk parse composition: fragments are rejected, not parsed as UDP",
          "[ether_frame][classify]") {
    // Pre-fix, the DPDK path had NO fragment check: a non-first fragment's payload bytes were
    // parsed as a UDP header, so attacker-controlled payload could match the port filter and
    // forward garbage into the stream.
    SECTION("non-first fragment (nonzero offset)") {
        const auto f = make_udp_frame(64, {.frag_off_field = 0x00B9});  // offset 185*8
        CHECK(classify_ether(f, 0x0100A8C0, 5000).verdict == udp::frame_verdict::kind::malformed);
    }
    SECTION("first fragment (MF set)") {
        const auto f = make_udp_frame(64, {.frag_off_field = 0x2000});  // MF
        CHECK(classify_ether(f, 0x0100A8C0, 5000).verdict == udp::frame_verdict::kind::malformed);
    }
    SECTION("DF alone is not a fragment") {
        const auto f = make_udp_frame(64, {.frag_off_field = 0x4000});  // DF
        CHECK(classify_ether(f, 0x0100A8C0, 5000).verdict == udp::frame_verdict::kind::forward);
    }
}

TEST_CASE("dpdk parse composition: UDP length forgeries", "[ether_frame][classify]") {
    SECTION("dgram_len < 8 cannot underflow into a huge payload") {
        // Pre-fix: payload_len = uint16(udp_len - 8) wrapped to ~65530 for udp_len < 8.
        for (const std::uint16_t bad : {std::uint16_t{1}, std::uint16_t{7}}) {
            const auto f = make_udp_frame(64, {.udp_len_override = bad});
            CHECK(classify_ether(f, 0x0100A8C0, 5000).verdict
                  == udp::frame_verdict::kind::malformed);
        }
    }
    SECTION("dgram_len claiming more than the captured bytes is rejected") {
        const auto f = make_udp_frame(64, {.udp_len_override = 8 + 65});
        CHECK(classify_ether(f, 0x0100A8C0, 5000).verdict == udp::frame_verdict::kind::malformed);
    }
    SECTION("dgram_len shorter than the capture forwards only the claimed bytes") {
        const auto f = make_udp_frame(64, {.udp_len_override = 8 + 32});
        const auto v = classify_ether(f, 0x0100A8C0, 5000);
        REQUIRE(v.verdict == udp::frame_verdict::kind::forward);
        CHECK(v.payload_len == 32);
    }
}

TEST_CASE("dpdk parse composition: destination filters", "[ether_frame][classify]") {
    const auto f = make_udp_frame(16);
    SECTION("ip mismatch is filtered") {
        CHECK(classify_ether(f, 0x0200A8C0, 5000).verdict == udp::frame_verdict::kind::filtered);
    }
    SECTION("port mismatch is filtered") {
        CHECK(classify_ether(f, 0x0100A8C0, 5001).verdict == udp::frame_verdict::kind::filtered);
    }
    SECTION("port 0 accepts any port, nullopt ip accepts any ip") {
        CHECK(classify_ether(f, std::nullopt, 0).verdict == udp::frame_verdict::kind::forward);
    }
}
