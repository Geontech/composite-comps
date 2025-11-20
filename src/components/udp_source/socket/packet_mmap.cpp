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
#include <thread>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {
constexpr std::size_t DEFAULT_FRAME_SIZE_BYTES = 2048;
constexpr std::size_t MAX_IDLE_SPINS = 1024;
constexpr auto IDLE_BACKOFF = std::chrono::microseconds(50);

struct frame_release {
    tpacket2_hdr* hdr{nullptr};
    udp::packet_mmap::ring_buffer_ptr ring{nullptr};

    void operator()() const {
        if (hdr) {
            std::atomic_ref<uint32_t>(hdr->tp_status).store(TP_STATUS_KERNEL, std::memory_order_release);
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
  interface(config.logger),
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
        // Join multicast group
        auto ip_mreq = net::create_ip_mreq(m_join_socket, config.interface, config.ip_addr);
        if (::setsockopt(m_join_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &ip_mreq, sizeof(ip_mreq)) < 0) {
            ::close(m_join_socket);
            ::close(m_socket);
            throw std::runtime_error(std::format("failed to join multicast group: {}", std::string{strerror(errno)}));
        }
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

    while (!token.stop_requested()) {
        // Get pointer to current frame
        auto* hdr = (struct tpacket2_hdr*)((uint8_t*)m_ring_buffer->ring + (frame_idx * m_frame_size));

        auto status = std::atomic_ref<uint32_t>(hdr->tp_status).load(std::memory_order_acquire);

        // Check for ready
        if (status & TP_STATUS_USER) [[likely]] {
            idle_spins = 0;
            m_pkts_recvd.fetch_add(1, std::memory_order_relaxed);

            // Validate protocol
            // Note: With SOCK_DGRAM, tp_net points to IP header (no Ethernet header)
            auto ip_hdr = (struct iphdr*)((uint8_t*)hdr + hdr->tp_net);
            if (ip_hdr->protocol == IPPROTO_UDP) [[likely]] {
                // Extract UDP payload
                auto* udp_hdr = (struct udphdr*)((uint8_t*)(ip_hdr) + ip_hdr->ihl * 4);
                auto* payload = (uint8_t*)(udp_hdr) + sizeof(struct udphdr);
                size_t payload_len = ntohs(udp_hdr->len) - sizeof(struct udphdr);

                auto buffer = std::make_shared<composite::external_buffer<uint8_t>>(
                    payload,
                    payload_len,
                    frame_release{hdr, m_ring_buffer}
                );
                m_out_port->send_data(composite::immutable_buffer<uint8_t>(std::move(buffer)), {});
            } else {
                // Non-UDP packet, release frame immediately
                std::atomic_ref<uint32_t>(hdr->tp_status).store(TP_STATUS_KERNEL, std::memory_order_release);
            }
            frame_idx = (frame_idx + 1) % m_frame_count;
        } else {
            if (idle_spins < MAX_IDLE_SPINS) {
                ++idle_spins;
                std::this_thread::yield();
            } else {
                idle_spins = 0;
                std::this_thread::sleep_for(IDLE_BACKOFF);
            }
        }
    }
}

} // namespace udp
