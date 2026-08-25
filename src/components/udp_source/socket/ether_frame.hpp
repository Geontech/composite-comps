/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace udp {

/**
 * Pure Ethernet-header walk for receivers that capture at L2 (the DPDK backend; AF_PACKET
 * SOCK_DGRAM starts at L3 and never needs this). No DPDK types and no state, so it is
 * unit-testable in a build with DPDK disabled — the same reasoning as packet_mmap_frame.hpp.
 *
 * Loads are byte-wise, so the helper is alignment- and aliasing-clean on an arbitrary
 * capture pointer.
 */

inline constexpr std::uint16_t k_ethertype_ipv4 = 0x0800;
inline constexpr std::uint16_t k_ethertype_vlan = 0x8100;  ///< 802.1Q
inline constexpr std::uint16_t k_ethertype_qinq = 0x88A8;  ///< 802.1ad outer (Q-in-Q)
inline constexpr std::size_t k_ether_hdr_len = 14;
inline constexpr std::size_t k_vlan_tag_len = 4;

struct l3_locate {
    enum class kind : std::uint8_t {
        ipv4,       ///< IPv4 header begins at `offset`
        non_ipv4,   ///< well-formed Ethernet, but the resolved ethertype is not IPv4
        truncated,  ///< frame too short for the Ethernet header or a claimed VLAN tag
    };
    kind verdict{kind::truncated};
    std::size_t offset{};        ///< L3 byte offset from frame start (valid for ipv4)
    std::uint16_t ether_type{};  ///< resolved post-VLAN ethertype (valid unless truncated)
};

/// Locate the IPv4 header behind an Ethernet header and any stack of VLAN tags. Both tag
/// ethertypes are stripped: 802.1Q (0x8100) and the 802.1ad outer tag (0x88A8), so Q-in-Q
/// double tagging resolves to the inner protocol. Every read is bounds-checked against the
/// captured length first — the bytes are untrusted.
inline auto locate_ipv4(const std::uint8_t* base, std::size_t len) -> l3_locate {
    using kind = l3_locate::kind;
    const auto load_be16 = [base](std::size_t off) {
        return static_cast<std::uint16_t>((static_cast<std::uint16_t>(base[off]) << 8) | base[off + 1]);
    };
    if (len < k_ether_hdr_len) {
        return {.verdict = kind::truncated};
    }
    auto ether_type = load_be16(k_ether_hdr_len - 2);
    std::size_t l3_offset = k_ether_hdr_len;
    while (ether_type == k_ethertype_vlan || ether_type == k_ethertype_qinq) {
        if (len < l3_offset + k_vlan_tag_len) {
            return {.verdict = kind::truncated};
        }
        ether_type = load_be16(l3_offset + 2);  // TCI (2 bytes), then the encapsulated ethertype
        l3_offset += k_vlan_tag_len;
    }
    if (ether_type != k_ethertype_ipv4) {
        return {.verdict = kind::non_ipv4, .ether_type = ether_type};
    }
    return {.verdict = kind::ipv4, .offset = l3_offset, .ether_type = ether_type};
}

} // namespace udp
