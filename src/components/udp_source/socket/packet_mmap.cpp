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

#include "helpers.hpp"
#include "packet_mmap.hpp"
#include "pmr/ring_resource.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <format>
#include <immintrin.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <poll.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#include <iostream>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <sys/ioctl.h>

namespace udp {

packet_mmap::packet_mmap(const config& config) :
  interface(config.logger),
  m_frame_size(std::bit_ceil(config.msg_size)),
  m_frame_count(config.frame_count),
  m_resource({.frame_size=m_frame_size, .frame_count=m_frame_count, .alignment=64}) {
    // Create socket
    m_logger->trace("opening af_packet udp socket");
    m_socket = ::socket(AF_PACKET, SOCK_DGRAM, htons(ETH_P_IP));
    if (m_socket < 0) {
        throw std::runtime_error(std::format("failed to create socket: {}", std::string{strerror(errno)}));
    }

    // Set packet version to v2
    // int version = TPACKET_V3;
    int version = TPACKET_V2;
    m_logger->trace("setting tpacket version: {}", version);
    if (::setsockopt(m_socket, SOL_PACKET, PACKET_VERSION, &version, sizeof(version)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to set packet version to v3: {}", std::string{strerror(errno)}));
    }

    // Set receive buffer size
    if (config.recv_buf_size > 0) {
        m_logger->trace("setting socket receive buffer size to {}", config.recv_buf_size);
        if (::setsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, (char*)&config.recv_buf_size, sizeof(config.recv_buf_size)) < 0) {
            ::close(m_socket);
            throw std::runtime_error(std::format("failed to set receive buffer size: {}", std::string{strerror(errno)}));
        }
    }

    // Bind the socket
    auto sll = sockaddr_ll{};
    sll.sll_family = AF_PACKET;
    sll.sll_ifindex = if_nametoindex(config.interface.c_str());
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
    m_block_nr = m_frame_count * m_frame_size / block_size;
    struct tpacket_req req{
        .tp_block_size = block_size,
        .tp_block_nr   = m_block_nr,
        .tp_frame_size = m_frame_size,
        .tp_frame_nr   = m_frame_count
    };
    if (::setsockopt(m_socket, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to set packet_rx_ring: {}", std::string{strerror(errno)}));
    }

    // Memory-map the ring buffer
    auto ring_size = req.tp_block_size * req.tp_block_nr;
    m_ring = ::mmap(0, ring_size,  PROT_READ | PROT_WRITE, MAP_SHARED, m_socket, 0);
    if (m_ring == MAP_FAILED) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to create mmap buffer: {}", std::string{strerror(errno)}));
    }

    // Request Transparent Huge Pages
    ::madvise(m_ring, ring_size, MADV_HUGEPAGE);

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
    ::munmap(m_ring, block_size * m_block_nr);
}

auto packet_mmap::start_recv(output_port_t* port) -> void {
    m_out_port = port;
    m_recv_thread = std::jthread(&packet_mmap::receive, this);
    pthread_setname_np(m_recv_thread.native_handle(), "packet_mmap");
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
    stats["pkts_recvd"] = std::to_string(m_pkts_recvd.exchange(0));
    return stats;
}

auto packet_mmap::receive(std::stop_token token) -> void {
    auto frame_idx = std::size_t{};
    auto allocator = std::pmr::polymorphic_allocator<std::uint8_t>(&m_resource);

    while (!token.stop_requested()) {
        // Get pointer to current frame
        auto* hdr = (struct tpacket2_hdr*)((uint8_t*)m_ring + (frame_idx * m_frame_size));

        // Check for ready
        if (hdr->tp_status & TP_STATUS_USER) [[likely]] {
            m_pkts_recvd.fetch_add(1, std::memory_order_relaxed);

            // Validate protocol
            auto ip_hdr = (struct iphdr*)((uint8_t*)hdr + hdr->tp_mac);
            if (ip_hdr->protocol == IPPROTO_UDP) [[likely]] {
                // Extract UDP payload
                auto* udp_hdr = (struct udphdr*)((uint8_t*)(ip_hdr) + ip_hdr->ihl * 4);
                auto* payload = (uint8_t*)(udp_hdr) + sizeof(struct udphdr);
                size_t payload_len = ntohs(udp_hdr->len) - sizeof(struct udphdr);

                // Create a pmr vector and copy udp payload into it
                auto vec = std::make_shared<buffer_t>(allocator);
                vec->resize(payload_len);
                std::memcpy(vec->data(), payload, payload_len);

                // Send data vector
                m_out_port->send_data(std::move(vec), {});
            }

            // Release the frame
            hdr->tp_status = TP_STATUS_KERNEL;
            frame_idx = (frame_idx + 1) % m_frame_count;
            hdr = (struct tpacket2_hdr*)((uint8_t*)m_ring + (frame_idx * m_frame_size));
        } else {
            struct timespec ts{.tv_sec=0, .tv_nsec=1};
            nanosleep(&ts, nullptr);
            // std::this_thread::yield();
        }
    }
}

} // namespace udp
