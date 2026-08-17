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

#include <composite/buffers/external_buffer.hpp>

#include "net/utils.hpp"
#include "packet_mmap.hpp"

#include <algorithm>
#include <atomic>
#include <arpa/inet.h>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <format>
#include <immintrin.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <poll.h>
#include <span>
#include <stdexcept>
#include <thread>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace {
constexpr std::size_t DEFAULT_FRAME_SIZE_BYTES = 2048;
constexpr std::size_t FORWARD_BATCH_SIZE = 128;
constexpr std::size_t MAX_IDLE_SPINS = 1024;
constexpr auto IDLE_BACKOFF = std::chrono::microseconds(50);

struct frame_release {
    tpacket2_hdr* hdr{nullptr};
    udp::packet_mmap::ring_buffer_ptr ring{nullptr};
    uint32_t slot{0};

    void operator()(uint8_t* /*payload*/) const {
        if (!hdr) { return; }
        // Return the slot to the kernel FIRST, then clear the held flag. Order matters: if we
        // cleared held first, a recv loop wrapping onto this slot could observe (TP_STATUS_USER,
        // !held) — the slot still carries the OLD bytes (not yet recycled) and would be re-read as
        // a fresh packet (a duplicate). With KERNEL stored first, any interleaving the recv loop
        // sees is safe: (KERNEL, *) short-circuits to not-ready, and (USER, held) backs off. The
        // only cost is that if the kernel refills between these two stores, the loop backs off for
        // a few iterations until held clears — a bounded delay, never corruption.
        std::atomic_ref<uint32_t>(hdr->tp_status).store(TP_STATUS_KERNEL, std::memory_order_release);
        if (ring && ring->slot_held) {
            ring->slot_held[slot].store(false, std::memory_order_release);
        }
    }
};
}

namespace udp {

packet_mmap::ring_buffer::~ring_buffer() {
    if (ring != nullptr) {
        ::munmap(ring, packet_mmap::block_size * block_nr);
    }
}

packet_mmap::packet_mmap(const config& config) :
  interface(config.logger, config.metrics),
  m_frame_count(config.frame_count) {
    auto requested_size = config.msg_size;
    if (requested_size == 0) {
        requested_size = DEFAULT_FRAME_SIZE_BYTES;
        m_logger->debug("msg_size not provided; defaulting to {} bytes for PACKET_MMAP", requested_size);
    }
    m_frame_size = std::bit_ceil(requested_size);
    // Create socket
    m_logger->trace("opening af_packet udp socket");
    m_socket = ::socket(AF_PACKET, SOCK_DGRAM, htons(ETH_P_IP));
    if (m_socket < 0) {
        throw std::runtime_error(std::format("failed to create socket: {}", std::string{strerror(errno)}));
    }

    // Set packet version to v2
    int version = TPACKET_V2;
    m_logger->trace("setting tpacket version: {}", version);
    if (::setsockopt(m_socket, SOL_PACKET, PACKET_VERSION, &version, sizeof(version)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to set packet version: {}", std::string{strerror(errno)}));
    }

    // Set receive buffer size
    if (config.recv_buf_size > 0) {
        m_logger->trace("setting socket receive buffer size to {}", config.recv_buf_size);
        net::set_socket_recv_buffer(m_socket, config.recv_buf_size);
    }

    // Bind the socket
    auto sll = sockaddr_ll{};
    sll.sll_family = AF_PACKET;
    sll.sll_ifindex = net::get_interface_index(config.interface);
    sll.sll_protocol = htons(ETH_P_IP);
    if (sll.sll_ifindex == 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format(
            "failed to get ifindex for interface {}: {}",
            config.interface, std::string{strerror(errno)}
        ));
    }
    if (::bind(m_socket, reinterpret_cast<struct sockaddr*>(&sll), sizeof(sll)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to bind socket: {}", std::string{strerror(errno)}));
    }

    // Set socket ring properties
    auto blocks = (m_frame_count * m_frame_size) / block_size;
    auto block_nr = std::max<uint32_t>(1, blocks);
    struct tpacket_req req{
        .tp_block_size = block_size,
        .tp_block_nr   = block_nr,
        .tp_frame_size = m_frame_size,
        .tp_frame_nr   = m_frame_count
    };
    if (::setsockopt(m_socket, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to set packet_rx_ring: {}", std::string{strerror(errno)}));
    }

    // Memory-map the ring buffer
    auto ring_size = req.tp_block_size * req.tp_block_nr;
    void* ring_ptr = ::mmap(0, ring_size,  PROT_READ | PROT_WRITE, MAP_SHARED, m_socket, 0);
    if (ring_ptr == MAP_FAILED) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to create mmap buffer: {}", std::string{strerror(errno)}));
    }

    // Wrap in shared_ptr for automatic lifetime management
    m_ring_buffer = std::make_shared<ring_buffer>();
    m_ring_buffer->ring = ring_ptr;
    m_ring_buffer->block_nr = block_nr;
    // One held-flag per ring slot, all initially free. Sized to m_frame_count (the same slot count
    // the recv loop wraps on).
    m_ring_buffer->slot_held = std::make_unique<std::atomic<bool>[]>(m_frame_count);
    for (uint32_t i = 0; i < m_frame_count; ++i) {
        m_ring_buffer->slot_held[i].store(false, std::memory_order_relaxed);
    }

    // Request Transparent Huge Pages
    ::madvise(m_ring_buffer->ring, ring_size, MADV_HUGEPAGE);

    if (net::is_ipv4_multicast(config.ip_addr)) {
        // Enable multicast mode on the interface
        auto pkt_mreq = net::create_packet_mreq(config.interface, config.ip_addr);
        if (::setsockopt(m_socket, SOL_PACKET, PACKET_ADD_MEMBERSHIP, &pkt_mreq, sizeof(pkt_mreq)) < 0) {
            ::close(m_socket);
            throw std::runtime_error(std::format("failed to add multicast membership: {}", std::string{strerror(errno)}));
        }

        // Create join socket
        m_join_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (m_join_socket < 0) {
            throw std::runtime_error(std::format("failed to create join socket: {}", std::string{strerror(errno)}));
        }
        // Allow rapid rebind during reconfiguration
        int reuse = 1;
        if (::setsockopt(m_join_socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
            m_logger->warn("failed to set SO_REUSEADDR on join socket: {}", std::string{strerror(errno)});
        }
        // Join multicast group
        auto ip_mreq = net::create_ip_mreq(m_join_socket, config.interface, config.ip_addr);
        if (::setsockopt(m_join_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &ip_mreq, sizeof(ip_mreq)) < 0) {
            ::close(m_join_socket);
            ::close(m_socket);
            throw std::runtime_error(std::format("failed to join multicast group: {}", std::string{strerror(errno)}));
        }
    }

    // Destination filter for receive(): only forward UDP datagrams addressed to the configured
    // IP/port (mirrors the DPDK path). config.ip_addr is the multicast group or a specific unicast
    // IP to accept; an empty / 0.0.0.0 address accepts any dst IP. port 0 accepts any port.
    m_dst_port = config.port;
    if (struct in_addr dst{}; !config.ip_addr.empty()
        && ::inet_pton(AF_INET, config.ip_addr.c_str(), &dst) == 1
        && dst.s_addr != INADDR_ANY) {
        m_dst_ip_be = dst.s_addr;
    }
}

packet_mmap::~packet_mmap() {
    m_recv_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
    if (m_join_socket != -1) {
        ::close(m_join_socket);
    }
    if (m_socket != -1) {
        ::close(m_socket);
    }
    // Ring buffer automatically unmaps when last shared_ptr is destroyed
}

auto packet_mmap::start_recv(output_port_t* port) -> void {
    m_out_port = port;
    m_recv_thread = std::jthread(&packet_mmap::receive, this);
    if (auto ret = pthread_setname_np(m_recv_thread.native_handle(), "packet_mmap"); ret != 0) {
        m_logger->warn("failed to set thread name: {}", std::string{strerror(ret)});
    }
}

auto packet_mmap::stop_recv() -> void {
    m_recv_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
}

auto packet_mmap::get_stats() -> std::map<std::string, std::string> {
    auto stats = std::map<std::string, std::string>{};
    auto tp_stats = tpacket_stats{};
    socklen_t len = sizeof(tp_stats);
    if (getsockopt(m_socket, SOL_PACKET, PACKET_STATISTICS, &tp_stats, &len) == 0) {
        stats["pkts_recvd_kernel"] = std::to_string(tp_stats.tp_packets);
        stats["pkts_dropped_kernel"] = std::to_string(tp_stats.tp_drops);
    }
    stats["pkts_recvd"] = std::to_string(m_pkts_recvd.load());
    return stats;
}

auto packet_mmap::receive(std::stop_token token) -> void {
    auto frame_idx = std::size_t{};
    std::size_t idle_spins = 0;
    std::size_t stalled_slot = SIZE_MAX;  // slot we last warned about being held, or none

    // Forward at most a quarter of the ring at once. Each queued external buffer pins its
    // PACKET_MMAP slot until the downstream consumer releases it, so a fixed batch of 128 would
    // create avoidable self-backpressure for small configured rings.
    const auto forward_batch_size = std::max<std::size_t>(
        1, std::min<std::size_t>(FORWARD_BATCH_SIZE, m_frame_count / 4));

    // Metric accumulators. This is a single-frame-at-a-time poll loop with no syscall batch, so
    // instead sum counts locally while draining a run of ready frames and record them with one
    // atomic add each when the drain ends (a stall, catching up to the kernel, or exit) — rather
    // than a locked RMW per packet. flush() is a no-op once drained, so the idle spin only pays
    // the atomics on its first iteration after a burst.
    uint64_t acc_pkts = 0, acc_bytes = 0, acc_dropped = 0;
    std::vector<composite::immutable_buffer<uint8_t>> output_buffers;
    output_buffers.reserve(forward_batch_size);
    auto flush = [&] {
        if (!output_buffers.empty()) {
            m_metrics.batch_sizes.record(static_cast<double>(output_buffers.size()));
            m_out_port->send_batch(std::span{output_buffers}, {});
            output_buffers.clear();
        }
        if (acc_pkts != 0) {
            m_pkts_recvd.fetch_add(acc_pkts, std::memory_order_relaxed);
            m_metrics.packets_received.add(acc_pkts);
        }
        if (acc_bytes != 0) { m_metrics.bytes_received.add(acc_bytes); }
        if (acc_dropped != 0) { m_metrics.packets_dropped.add(acc_dropped); }
        acc_pkts = acc_bytes = acc_dropped = 0;
    };

    while (!token.stop_requested()) {
        // Get pointer to current frame
        auto* hdr = (struct tpacket2_hdr*)((uint8_t*)m_ring_buffer->ring + (frame_idx * m_frame_size));

        // Check "held" BEFORE tp_status (both acquire) — the order is load-bearing for
        // correctness, not just an early-out. TP_STATUS_USER alone can't tell a fresh kernel packet
        // from a slot we already forwarded that a slow consumer still holds: its frame_release
        // hasn't run, so the ORIGINAL fill's USER status lingers. Re-reading such a slot would
        // deliver a duplicate and create a SECOND frame_release for it, double-returning it to the
        // kernel -> UAF. Observing held==false here synchronizes-with frame_release's release-store
        // of held, which is sequenced AFTER its TP_STATUS_KERNEL store, so the tp_status load below
        // can no longer observe the stale original USER (it sees KERNEL, or the kernel's later
        // refill). Loading tp_status first would leave a window where recv sees (USER-original,
        // held-just-cleared) and re-reads stale bytes.
        if (m_ring_buffer->slot_held[frame_idx].load(std::memory_order_acquire)) {
            // Flush BEFORE waiting: the current accumulator may itself own this slot after a ring
            // rotation. Publishing those buffers lets downstream release it; moving this flush
            // below the wait would turn self-backpressure into a permanent stall.
            flush();
            if (frame_idx != stalled_slot) {
                stalled_slot = frame_idx;
                m_logger->warn("packet_mmap: ring slot {} still held downstream a full rotation "
                               "later; stalling (kernel is dropping). Increase frame_count if this "
                               "persists.", frame_idx);
            }
            // Head-of-line by nature (the kernel can't refill a held slot either): back off and do
            // NOT advance — the in-order ring needs THIS slot recycled before the next packet.
            if (idle_spins < MAX_IDLE_SPINS) {
                ++idle_spins;
                std::this_thread::yield();
            } else {
                idle_spins = 0;
                std::this_thread::sleep_for(IDLE_BACKOFF);
            }
            continue;  // re-check the SAME slot
        }

        auto status = std::atomic_ref<uint32_t>(hdr->tp_status).load(std::memory_order_acquire);

        // Check for ready
        if (status & TP_STATUS_USER) [[likely]] {
            stalled_slot = SIZE_MAX;  // slot is fresh — clear any stall latch
            idle_spins = 0;
            ++acc_pkts;

            // The captured frame is UNTRUSTED (raw network). Every header field is bounds-checked
            // against the captured length BEFORE it is dereferenced, and the datagram is filtered
            // to the configured destination — otherwise a short/malformed packet causes an OOB
            // read, and ntohs(udp->len) - sizeof(udphdr) underflows to a huge length. With
            // SOCK_DGRAM tp_net points at the IP header and tp_snaplen is the captured byte count.
            const std::size_t avail = hdr->tp_snaplen;
            auto* base = reinterpret_cast<uint8_t*>(hdr) + hdr->tp_net;
            bool forwarded = false;

            if (avail >= sizeof(struct iphdr)) {
                auto* ip_hdr = reinterpret_cast<struct iphdr*>(base);
                const std::size_t ihl_bytes = static_cast<std::size_t>(ip_hdr->ihl) * 4U;
                if (ip_hdr->protocol == IPPROTO_UDP
                    && ip_hdr->ihl >= 5                                   // min IPv4 header
                    && avail >= ihl_bytes + sizeof(struct udphdr)         // UDP header fits
                    && (!m_dst_ip_be || ip_hdr->daddr == *m_dst_ip_be)) { // dst-IP filter
                    auto* udp_hdr = reinterpret_cast<struct udphdr*>(base + ihl_bytes);
                    const std::uint16_t udp_total = ntohs(udp_hdr->len);
                    const std::uint16_t dst_port = ntohs(udp_hdr->dest);
                    if ((m_dst_port == 0 || dst_port == m_dst_port)       // dst-port filter
                        && udp_total >= sizeof(struct udphdr)             // no length underflow
                        && static_cast<std::size_t>(ihl_bytes) + udp_total <= avail) { // payload captured
                        const std::size_t payload_len = udp_total - sizeof(struct udphdr);
                        auto* payload = base + ihl_bytes + sizeof(struct udphdr);
                        acc_bytes += payload_len;
                        // Mark the slot held BEFORE publishing the buffer: frame_release (which
                        // clears it) can only run once a consumer has received and dropped this
                        // buffer, which is strictly after send_data below — so held is always set
                        // before it can be cleared.
                        m_ring_buffer->slot_held[frame_idx].store(true, std::memory_order_release);
                        // Zero-copy: wrap the payload; the frame is released (and held cleared) when
                        // the buffer dies.
                        auto buffer = composite::external_buffer<uint8_t>(
                            payload, payload_len,
                            frame_release{hdr, m_ring_buffer, static_cast<uint32_t>(frame_idx)});
                        output_buffers.emplace_back(std::move(buffer));
                        forwarded = true;
                        if (output_buffers.size() == forward_batch_size) {
                            flush();
                        }
                    }
                }
            }
            if (!forwarded) {
                // Non-UDP, malformed, or filtered-out: count + release the frame immediately.
                ++acc_dropped;
                std::atomic_ref<uint32_t>(hdr->tp_status).store(TP_STATUS_KERNEL, std::memory_order_release);
            }
            frame_idx = (frame_idx + 1) % m_frame_count;
        } else {
            flush();  // caught up to the kernel — publish accumulated counts, then idle
            if (idle_spins < MAX_IDLE_SPINS) {
                ++idle_spins;
                std::this_thread::yield();
            } else {
                idle_spins = 0;
                std::this_thread::sleep_for(IDLE_BACKOFF);
            }
        }
    }
    flush();  // stop requested mid-drain: publish any counts accumulated since the last flush
}

} // namespace udp
