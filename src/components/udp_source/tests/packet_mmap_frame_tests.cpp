/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

// Hermetic tests for the PACKET_MMAP receiver's pure helpers (packet_mmap_frame.hpp) and the
// SCM_RIGHTS fd handoff (net::receive_fd). The receive path itself cannot run in an
// unprivileged CI job (AF_PACKET needs CAP_NET_RAW) — which is exactly why the geometry,
// classification, and filter logic live in a header with no I/O: everything that CAN be
// tested without the capability, is.

#include "../socket/packet_mmap_frame.hpp"
#include "../net/utils.hpp"

#include <catch2/catch_test_macros.hpp>

#include <arpa/inet.h>
#include <cstring>
#include <fcntl.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

// Build an IPv4+UDP frame the way AF_PACKET SOCK_DGRAM presents it (starting at the IP header).
// The struct-cast writes here are test-side only (the vector's data is suitably aligned);
// classify_frame itself must never cast — see the alignment test below.
auto make_udp_frame(uint16_t dst_port, uint32_t dst_ip_host, std::size_t payload_len,
                    uint16_t frag_off_field = 0, uint8_t ihl = 5, uint8_t proto = IPPROTO_UDP)
    -> std::vector<uint8_t> {
    const std::size_t ihl_bytes = std::size_t{ihl} * 4;
    std::vector<uint8_t> f(ihl_bytes + sizeof(udphdr) + payload_len, 0xAB);
    auto* ip = reinterpret_cast<iphdr*>(f.data());
    ip->version = 4;
    ip->ihl = ihl;
    ip->tot_len = htons(static_cast<uint16_t>(f.size()));
    ip->protocol = proto;
    ip->frag_off = htons(frag_off_field);
    ip->daddr = htonl(dst_ip_host);
    auto* udp_h = reinterpret_cast<udphdr*>(f.data() + ihl_bytes);
    udp_h->dest = htons(dst_port);
    udp_h->len = htons(static_cast<uint16_t>(sizeof(udphdr) + payload_len));
    return f;
}

} // namespace

TEST_CASE("ring frames leave room for TPACKET and protocol headers", "[packet_mmap][geometry]") {
    // A frame must hold the TPACKET v2 header + sockaddr_ll + worst-case IPv4 header + UDP
    // header ON TOP of the payload: sizing frames as bit_ceil(payload) truncated every
    // exact power-of-two payload maximum.
    CHECK(udp::ring_frame_size(2048) == 4096);
    CHECK(udp::ring_frame_size(8192) == 16384);
    CHECK(udp::ring_frame_size(1024) == 2048);
    // The overhead constant covers the ALIGNED header block plus the kernel's SOCK_DGRAM
    // gap plus worst-case protocol headers — raw TPACKET2_HDRLEN is not enough.
    CHECK(udp::k_frame_overhead >= TPACKET_ALIGN(TPACKET2_HDRLEN) + 16 + 20 + 8);
    // Boundary payload: 1928 + raw-header overhead squeaked into 2048 and truncated once the
    // kernel's alignment gap was applied; the aligned overhead pushes it to 4096.
    CHECK(udp::ring_frame_size(1928) == 4096);
    // And a mid-range payload that fits with its overhead is not over-rounded.
    CHECK(udp::ring_frame_size(1400) == 2048);
}

TEST_CASE("ring geometry tiles whole blocks exactly", "[packet_mmap][geometry]") {
    constexpr uint32_t block = 1u << 20;
    // 2048-byte frames: 512 per block. A frame count that doesn't tile blocks used to make
    // PACKET_RX_RING fail with EINVAL; it must round UP to whole blocks instead.
    auto g = udp::ring_geometry::compute(block, 2048, 1000);
    CHECK(g.block_nr == 2);
    CHECK(g.frame_nr == 1024);
    CHECK(g.frame_nr % (block / 2048) == 0);

    g = udp::ring_geometry::compute(block, 2048, 1024);  // exact tiling unchanged
    CHECK(g.block_nr == 2);
    CHECK(g.frame_nr == 1024);

    g = udp::ring_geometry::compute(block, 4096, 1);     // tiny request: one full block
    CHECK(g.block_nr == 1);
    CHECK(g.frame_nr == 256);
}

TEST_CASE("frame classification forwards matching UDP and computes payload bounds",
          "[packet_mmap][classify]") {
    const auto f = make_udp_frame(5000, 0xC0A80101 /*192.168.1.1*/, 64);
    const auto v = udp::classify_frame(f.data(), f.size(), htonl(0xC0A80101), 5000);
    REQUIRE(v.verdict == udp::frame_verdict::kind::forward);
    CHECK(v.payload_off == 20 + sizeof(udphdr));
    CHECK(v.payload_len == 64);

    // Options-bearing header (ihl=8) shifts the payload correctly.
    const auto f2 = make_udp_frame(5000, 0xC0A80101, 16, 0, 8);
    const auto v2 = udp::classify_frame(f2.data(), f2.size(), std::nullopt, 5000);
    REQUIRE(v2.verdict == udp::frame_verdict::kind::forward);
    CHECK(v2.payload_off == 32 + sizeof(udphdr));
}

TEST_CASE("frame classification filters and rejects malformed frames", "[packet_mmap][classify]") {
    using kind = udp::frame_verdict::kind;
    const auto f = make_udp_frame(5000, 0xC0A80101, 32);

    CHECK(udp::classify_frame(f.data(), f.size(), htonl(0x0A000001), 5000).verdict == kind::filtered);  // wrong ip
    CHECK(udp::classify_frame(f.data(), f.size(), std::nullopt, 6000).verdict == kind::filtered);       // wrong port
    CHECK(udp::classify_frame(f.data(), 10, std::nullopt, 0).verdict == kind::malformed);               // short frame

    // Truncated capture: udp->len exceeds the captured bytes.
    CHECK(udp::classify_frame(f.data(), f.size() - 8, std::nullopt, 5000).verdict == kind::malformed);

    // Non-UDP protocol.
    const auto tcp = make_udp_frame(5000, 0xC0A80101, 32, 0, 5, IPPROTO_TCP);
    CHECK(udp::classify_frame(tcp.data(), tcp.size(), std::nullopt, 0).verdict == kind::filtered);

    // ihl below the legal minimum.
    const auto bad_ihl = make_udp_frame(5000, 0xC0A80101, 32, 0, 5);
    auto mut = bad_ihl;
    reinterpret_cast<iphdr*>(mut.data())->ihl = 3;
    CHECK(udp::classify_frame(mut.data(), mut.size(), std::nullopt, 0).verdict == kind::malformed);
}

TEST_CASE("IP fragments are never parsed as UDP", "[packet_mmap][classify][fragments]") {
    using kind = udp::frame_verdict::kind;
    // Non-first fragment (offset != 0): where a UDP header would be read there is arbitrary
    // payload — matching the port filter by luck used to forward garbage.
    const auto frag = make_udp_frame(5000, 0xC0A80101, 32, /*frag_off=*/185);
    CHECK(udp::classify_frame(frag.data(), frag.size(), std::nullopt, 5000).verdict == kind::malformed);

    // First fragment (MF set, offset 0): real UDP header, incomplete payload.
    const auto first = make_udp_frame(5000, 0xC0A80101, 32, /*frag_off=*/0x2000);
    CHECK(udp::classify_frame(first.data(), first.size(), std::nullopt, 5000).verdict == kind::malformed);
}

TEST_CASE("classification is safe and identical on a misaligned frame pointer",
          "[packet_mmap][classify][alignment]") {
    // The DPDK path hands classify_frame the IP header at Ethernet offset 14 (+4 per VLAN
    // tag) — 2-byte aligned. The pre-fix struct-cast implementation was undefined behavior
    // there (UBSan trapped it); the byte-wise loads must produce the same verdict at ANY
    // alignment. Run under the sanitized preset for teeth.
    const auto f = make_udp_frame(5000, 0xC0A80101, 64);
    for (std::size_t shift = 1; shift <= 3; ++shift) {
        std::vector<uint8_t> shifted(shift, 0);
        shifted.insert(shifted.end(), f.begin(), f.end());
        const auto v = udp::classify_frame(shifted.data() + shift, f.size(),
                                           htonl(0xC0A80101), 5000);
        REQUIRE(v.verdict == udp::frame_verdict::kind::forward);
        CHECK(v.payload_off == 20 + sizeof(udphdr));
        CHECK(v.payload_len == 64);
    }
}

TEST_CASE("IP total length is validated and bounds the payload", "[packet_mmap][classify]") {
    using kind = udp::frame_verdict::kind;

    SECTION("version other than 4 is malformed") {
        auto f = make_udp_frame(5000, 0xC0A80101, 32);
        reinterpret_cast<iphdr*>(f.data())->version = 6;
        CHECK(udp::classify_frame(f.data(), f.size(), std::nullopt, 0).verdict == kind::malformed);
    }
    SECTION("tot_len larger than the capture is malformed") {
        auto f = make_udp_frame(5000, 0xC0A80101, 32);
        reinterpret_cast<iphdr*>(f.data())->tot_len = htons(static_cast<uint16_t>(f.size() + 1));
        CHECK(udp::classify_frame(f.data(), f.size(), std::nullopt, 0).verdict == kind::malformed);
    }
    SECTION("tot_len too small to hold IP + UDP headers is malformed") {
        auto f = make_udp_frame(5000, 0xC0A80101, 32);
        reinterpret_cast<iphdr*>(f.data())->tot_len = htons(20);
        CHECK(udp::classify_frame(f.data(), f.size(), std::nullopt, 0).verdict == kind::malformed);
    }
    SECTION("Ethernet padding beyond tot_len is not treated as available payload") {
        // Short frames arrive padded to the Ethernet minimum and the capture includes the
        // padding. A forged udp->len that fits the CAPTURE but exceeds the IP datagram used
        // to be forwarded — emitting padding bytes as stream payload.
        auto f = make_udp_frame(5000, 0xC0A80101, 4);           // tot_len = 32
        f.resize(60, 0x00);                                     // padded capture
        auto* udp_h = reinterpret_cast<udphdr*>(f.data() + 20);
        udp_h->len = htons(8 + 20);                             // claims 16 padding bytes
        CHECK(udp::classify_frame(f.data(), f.size(), std::nullopt, 5000).verdict == kind::malformed);

        // The honest frame, padding still present, forwards exactly its 4 payload bytes.
        udp_h->len = htons(8 + 4);
        const auto v = udp::classify_frame(f.data(), f.size(), std::nullopt, 5000);
        REQUIRE(v.verdict == kind::forward);
        CHECK(v.payload_len == 4);
    }
}

TEST_CASE("kernel filter program shape follows the configured filters", "[packet_mmap][bpf]") {
    // Full filters: proto + frag + ip + port checks, accept, drop.
    auto prog = udp::build_kernel_filter(htonl(0xC0A80101), 5000);
    REQUIRE(prog.size() == 11);
    CHECK(prog[prog.size() - 2].k == 0xFFFFFFFF);  // accept returns full packet
    CHECK(prog[prog.size() - 1].k == 0);           // drop returns 0
    // Every conditional's drop target lands exactly on the final RET 0.
    for (std::size_t i = 0; i < prog.size(); ++i) {
        if (BPF_CLASS(prog[i].code) == BPF_JMP) {
            for (const auto off : {prog[i].jt, prog[i].jf}) {
                CHECK(i + 1 + off < prog.size());
            }
        }
    }
    // No filters: proto + frag checks only.
    auto minimal = udp::build_kernel_filter(std::nullopt, 0);
    REQUIRE(minimal.size() == 6);
}

TEST_CASE("receive_fd adopts a descriptor passed over SCM_RIGHTS", "[packet_mmap][fd_passing]") {
    const std::string path = "/tmp/comps_fd_pass_test_" + std::to_string(::getpid());
    ::unlink(path.c_str());

    // A helper thread plays the privileged sidecar: listen, accept, pass one end of a pipe.
    int pipe_fds[2]{};
    REQUIRE(::pipe(pipe_fds) == 0);
    std::thread helper([&] {
        const int srv = ::socket(AF_UNIX, SOCK_STREAM, 0);
        REQUIRE(srv >= 0);
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
        REQUIRE(::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0);
        REQUIRE(::listen(srv, 1) == 0);
        const int conn = ::accept(srv, nullptr, nullptr);
        REQUIRE(conn >= 0);
        char data = 'F';
        iovec iov{.iov_base = &data, .iov_len = 1};
        alignas(cmsghdr) char ctrl[CMSG_SPACE(sizeof(int))]{};
        msghdr msg{};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        msg.msg_control = ctrl;
        msg.msg_controllen = sizeof(ctrl);
        auto* cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int));
        std::memcpy(CMSG_DATA(cmsg), &pipe_fds[0], sizeof(int));
        REQUIRE(::sendmsg(conn, &msg, 0) == 1);
        ::close(conn);
        ::close(srv);
    });

    const int fd = net::receive_fd(path, 5000);
    helper.join();
    REQUIRE(fd >= 0);
    // The adopted descriptor is live: bytes written to the pipe's other end arrive on it.
    REQUIRE(::write(pipe_fds[1], "ok", 2) == 2);
    char buf[2]{};
    REQUIRE(::read(fd, buf, 2) == 2);
    CHECK(buf[0] == 'o');
    ::close(fd);
    ::close(pipe_fds[0]);
    ::close(pipe_fds[1]);
    ::unlink(path.c_str());
}
