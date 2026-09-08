/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#pragma once

#include <bit>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <linux/filter.h>
#include <linux/if_packet.h>
// Only <netinet/in.h> (IPPROTO_UDP, ntohl): classify_frame reads header fields with
// byte-wise loads rather than casting struct iphdr/udphdr over the bytes, so no ip/udp
// header is needed — which also keeps this header includable next to DPDK's rte_ip.h
// (the kernel-UAPI <linux/ip.h> twin redefines struct iphdr against it).
#include <netinet/in.h>
#include <optional>
#include <vector>

namespace udp {

/**
 * Pure helpers for the PACKET_MMAP receiver: ring geometry, frame classification, and the
 * kernel filter program. No I/O and no state, so each is unit-testable without CAP_NET_RAW —
 * the reason the receive path itself has no test coverage (an AF_PACKET socket cannot be
 * opened in an unprivileged CI job).
 */

/// Ring frame size for a maximum UDP payload. A ring frame holds MORE than the payload:
/// the TPACKET v2 header + sockaddr_ll (TPACKET2_HDRLEN), an alignment gap so the packet
/// data lands 16-byte aligned, then — at tp_net, for SOCK_DGRAM — the IPv4 header (up to 60
/// bytes with options), the UDP header, and only then payload. Sizing the frame as
/// bit_ceil(payload) (as this receiver used to) meant every power-of-two payload maximum
/// (2048, 8192, ...) was ALWAYS truncated even when the operator supplied the exact
/// documented msg_size — and using raw TPACKET2_HDRLEN without the alignment terms still
/// truncated boundary payloads (e.g. 1928 bytes). The kernel's own SOCK_DGRAM layout is
/// netoff = TPACKET_ALIGN(tp_hdrlen) + 16 (+ tp_reserve, which this receiver does not use);
/// see af_packet.c tpacket_rcv and Documentation/networking/packet_mmap.
inline constexpr std::size_t k_frame_overhead =
    TPACKET_ALIGN(TPACKET2_HDRLEN) + 16 /* kernel SOCK_DGRAM gap */
    + 60 /* max IPv4 header */ + 8 /* UDP header */;

inline constexpr auto ring_frame_size(std::size_t max_udp_payload) -> std::uint32_t {
    return static_cast<std::uint32_t>(std::bit_ceil(max_udp_payload + k_frame_overhead));
}

/// PACKET_RX_RING geometry. The kernel REQUIRES tp_frame_nr == frames_per_block * tp_block_nr
/// exactly; deriving block_nr by flooring frame_count*frame_size/block_size (as this receiver
/// used to) made any frame_count that doesn't tile whole blocks fail construction with a
/// cryptic EINVAL. Normalize instead: round the frame count UP to fill whole blocks and use
/// the normalized count everywhere (ring addressing, slot bookkeeping).
struct ring_geometry {
    std::uint32_t block_nr{};
    std::uint32_t frame_nr{};  ///< normalized: frames_per_block * block_nr

    static constexpr auto compute(std::uint32_t block_size, std::uint32_t frame_size,
                                  std::uint32_t requested_frames) -> ring_geometry {
        const auto frames_per_block = block_size / frame_size;  // frame_size is a pow2 <= block_size
        const auto blocks = (requested_frames + frames_per_block - 1) / frames_per_block;
        const auto block_nr = blocks == 0 ? 1U : blocks;
        return {.block_nr = block_nr, .frame_nr = frames_per_block * block_nr};
    }
};

/// Verdict for one captured frame (bytes start at the IPv4 header: AF_PACKET SOCK_DGRAM).
struct frame_verdict {
    enum class kind : std::uint8_t {
        forward,    ///< UDP datagram matching the filters: payload at [payload_off, +payload_len)
        filtered,   ///< valid but not ours (protocol / dst ip / dst port mismatch)
        malformed,  ///< bounds or length-arithmetic failure (or a fragment; see below)
    };
    kind verdict{kind::malformed};
    std::size_t payload_off{};
    std::size_t payload_len{};
};

/// Classify a captured frame. Every header field is bounds-checked against the captured
/// length BEFORE it is read (the bytes are untrusted), and IP FRAGMENTS are never parsed
/// as UDP: a non-first fragment carries payload bytes where a UDP header would be read —
/// its "dst port" is arbitrary payload, so it could match the filter and forward garbage.
/// First fragments do carry a real UDP header, but their udp->len covers the whole
/// unreassembled datagram and can never fit inside the fragment's IP total length, so the
/// length check drops them; rejecting on the fragment bits directly makes that explicit.
///
/// All loads are BYTE-WISE (or memcpy), never a struct cast: the AF_PACKET ring aligns the
/// network header 16-byte, but the DPDK path hands classify_frame the IP header at Ethernet
/// offset 14 (+4 per VLAN tag) — 2-byte aligned, where dereferencing an iphdr* is undefined
/// behavior (UBSan traps it; strict-alignment machines fault).
///
/// The IP TOTAL LENGTH is validated and is the authoritative payload bound, not the capture
/// length: sub-60-byte Ethernet frames arrive PADDED and the capture includes the padding,
/// so bounding the UDP length by `avail` alone would let a forged udp->len claim padding
/// bytes as stream payload.
inline auto classify_frame(const std::uint8_t* base, std::size_t avail,
                           std::optional<std::uint32_t> dst_ip_be, std::uint16_t dst_port)
    -> frame_verdict {
    using kind = frame_verdict::kind;
    constexpr std::size_t k_min_ip_hdr = 20;
    constexpr std::size_t k_udp_hdr = 8;
    const auto load_be16 = [base](std::size_t off) {
        return static_cast<std::uint16_t>((static_cast<std::uint16_t>(base[off]) << 8) | base[off + 1]);
    };
    if (avail < k_min_ip_hdr) {
        return {.verdict = kind::malformed};
    }
    const std::uint8_t version_ihl = base[0];
    const std::size_t ihl_bytes = static_cast<std::size_t>(version_ihl & 0x0F) * 4U;
    if ((version_ihl >> 4) != 4 || ihl_bytes < k_min_ip_hdr) {
        return {.verdict = kind::malformed};
    }
    // tot_len must hold at least the claimed IP header + a UDP header, and must fit the
    // capture; together these bound every read below (ihl_bytes + 8 <= tot_len <= avail).
    const std::uint16_t tot_len = load_be16(2);
    if (tot_len < ihl_bytes + k_udp_hdr || tot_len > avail) {
        return {.verdict = kind::malformed};
    }
    // Fragment bits: any nonzero fragment offset, or More-Fragments set, means the UDP
    // payload is split across IP packets this receiver does not reassemble.
    if ((load_be16(6) & 0x3FFF) != 0) {  // offset (13 bits) | MF
        return {.verdict = kind::malformed};
    }
    std::uint32_t daddr{};  // network byte order, matching dst_ip_be
    std::memcpy(&daddr, base + 16, sizeof(daddr));
    if (base[9] != IPPROTO_UDP || (dst_ip_be && daddr != *dst_ip_be)) {
        return {.verdict = kind::filtered};
    }
    if (dst_port != 0 && load_be16(ihl_bytes + 2) != dst_port) {
        return {.verdict = kind::filtered};
    }
    const std::uint16_t udp_total = load_be16(ihl_bytes + 4);
    if (udp_total < k_udp_hdr || ihl_bytes + udp_total > tot_len) {
        return {.verdict = kind::malformed};
    }
    return {.verdict = kind::forward,
            .payload_off = ihl_bytes + k_udp_hdr,
            .payload_len = udp_total - k_udp_hdr};
}

/// Classic BPF program mirroring classify_frame's accept condition, for SO_ATTACH_FILTER:
/// the kernel then drops non-matching traffic BEFORE it consumes a ring slot, which matters
/// whenever the interface carries anything besides this stream (the userspace classifier
/// stays as the authoritative backstop). Offsets are relative to the IPv4 header — for
/// SOCK_DGRAM packet sockets the filter runs with the data pointer at the network header.
/// Attaching a filter to an already-open socket requires NO privileges.
inline auto build_kernel_filter(std::optional<std::uint32_t> dst_ip_be, std::uint16_t dst_port)
    -> std::vector<struct sock_filter> {
    std::vector<struct sock_filter> prog;
    const auto drop_at = [&prog](std::vector<std::size_t>& fixups) {
        // placeholder jf targets are fixed up once the DROP index is known
        fixups.push_back(prog.size() - 1);
    };
    std::vector<std::size_t> to_drop;

    // ldb [9]; jne IPPROTO_UDP -> drop
    prog.push_back({BPF_LD | BPF_B | BPF_ABS, 0, 0, 9});
    prog.push_back({BPF_JMP | BPF_JEQ | BPF_K, 0, 0, IPPROTO_UDP});  // jt fallthrough, jf fixup
    drop_at(to_drop);
    // ldh [6]; jset (offset|MF) -> drop
    prog.push_back({BPF_LD | BPF_H | BPF_ABS, 0, 0, 6});
    prog.push_back({BPF_JMP | BPF_JSET | BPF_K, 0, 0, 0x3FFF});  // jt = drop, jf = fallthrough
    // for JSET the DROP branch is jt: record with a marker (fix up jt instead of jf)
    const auto jset_idx = prog.size() - 1;
    if (dst_ip_be) {
        // ld [16]; jne dst -> drop (BPF loads big-endian bytes as a host-order value)
        prog.push_back({BPF_LD | BPF_W | BPF_ABS, 0, 0, 16});
        prog.push_back({BPF_JMP | BPF_JEQ | BPF_K, 0, 0, ntohl(*dst_ip_be)});
        drop_at(to_drop);
    }
    if (dst_port != 0) {
        // ldx 4*([0]&0xf); ldh [x+2]; jne port -> drop
        prog.push_back({BPF_LDX | BPF_B | BPF_MSH, 0, 0, 0});
        prog.push_back({BPF_LD | BPF_H | BPF_IND, 0, 0, 2});
        prog.push_back({BPF_JMP | BPF_JEQ | BPF_K, 0, 0, dst_port});
        drop_at(to_drop);
    }
    // accept: return the full packet
    const auto accept_idx = static_cast<std::uint8_t>(prog.size());
    prog.push_back({BPF_RET | BPF_K, 0, 0, 0xFFFFFFFF});
    // drop: return 0
    const auto drop_idx = static_cast<std::uint8_t>(prog.size());
    prog.push_back({BPF_RET | BPF_K, 0, 0, 0});

    for (const auto idx : to_drop) {
        prog[idx].jf = static_cast<std::uint8_t>(drop_idx - idx - 1);
    }
    prog[jset_idx].jt = static_cast<std::uint8_t>(drop_idx - jset_idx - 1);
    (void)accept_idx;
    return prog;
}

} // namespace udp
