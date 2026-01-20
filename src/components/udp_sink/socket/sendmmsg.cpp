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

#include "sendmmsg.hpp"

#include <arpa/inet.h>
#include <cstring>
#include <format>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>

namespace udp_tx {

sendmmsg_tx::sendmmsg_tx(const config& cfg) : m_config(cfg) {
    m_socket_fd = create_socket();
    if (m_socket_fd < 0) {
        throw std::runtime_error("Failed to create UDP socket");
    }

    // Pre-allocate batch structures
    m_batch_queue.reserve(m_config.batch_size);
    m_iovecs.resize(m_config.batch_size);
    m_msgs.resize(m_config.batch_size);

    m_config.logger->info("sendmmsg_tx initialized: batch_size={}, batch_timeout_us={}",
                          m_config.batch_size, m_config.batch_timeout_us);
}

sendmmsg_tx::~sendmmsg_tx() {
    // Flush any remaining packets
    flush();

    if (m_socket_fd >= 0) {
        ::close(m_socket_fd);
        m_socket_fd = -1;
    }
}

auto sendmmsg_tx::send(const std::string& ip, uint16_t port, std::span<const uint8_t> data) -> ssize_t {
    std::scoped_lock lock(m_mutex);

    // Start batch timer on first packet
    if (m_batch_queue.empty()) {
        m_batch_start_time = std::chrono::steady_clock::now();
    }

    // Create destination address
    struct sockaddr_in dest_addr{};
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip.c_str(), &dest_addr.sin_addr) != 1) {
        m_config.logger->error("Invalid IP address: {}", ip);
        m_total_errors++;
        return -1;
    }

    // Queue the packet
    m_batch_queue.emplace_back(queued_packet{
        .data = std::vector<uint8_t>(data.begin(), data.end()),
        .dest_addr = dest_addr
    });

    auto bytes_queued = static_cast<ssize_t>(data.size());

    // Update destination stats
    auto dest_key = make_dest_key(ip, port);
    auto& stats = m_dest_stats[dest_key];
    stats.last_used = std::chrono::steady_clock::now();

    // Check if we should flush
    bool should_flush = false;

    // Flush if batch is full
    if (m_batch_queue.size() >= m_config.batch_size) {
        should_flush = true;
    }

    // Flush if batch timeout exceeded
    if (!should_flush && m_config.batch_timeout_us > 0) {
        auto elapsed = std::chrono::steady_clock::now() - m_batch_start_time;
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        if (elapsed_us >= m_config.batch_timeout_us) {
            should_flush = true;
        }
    }

    if (should_flush) {
        flush_locked();
    }

    return bytes_queued;
}

auto sendmmsg_tx::flush() -> void {
    std::scoped_lock lock(m_mutex);
    flush_locked();
}

auto sendmmsg_tx::flush_locked() -> void {
    if (m_batch_queue.empty()) {
        return;
    }

    auto batch_size = m_batch_queue.size();

    // Set up iovec and mmsghdr for each packet
    for (size_t i = 0; i < batch_size; i++) {
        auto& pkt = m_batch_queue[i];

        m_iovecs[i].iov_base = pkt.data.data();
        m_iovecs[i].iov_len = pkt.data.size();

        std::memset(&m_msgs[i], 0, sizeof(struct mmsghdr));
        m_msgs[i].msg_hdr.msg_name = &pkt.dest_addr;
        m_msgs[i].msg_hdr.msg_namelen = sizeof(pkt.dest_addr);
        m_msgs[i].msg_hdr.msg_iov = &m_iovecs[i];
        m_msgs[i].msg_hdr.msg_iovlen = 1;
    }

    // Send all packets in one syscall
    int sent = ::sendmmsg(m_socket_fd, m_msgs.data(), static_cast<unsigned int>(batch_size), 0);

    if (sent < 0) {
        m_config.logger->error("sendmmsg failed: {} (batch_size={})", strerror(errno), batch_size);
        m_total_errors += batch_size;
    } else {
        // Update stats for successfully sent packets
        for (int i = 0; i < sent; i++) {
            auto& pkt = m_batch_queue[i];
            auto bytes = m_msgs[i].msg_len;

            m_total_packets++;
            m_total_bytes += bytes;

            // Update per-destination stats
            auto dest_key = make_dest_key(
                inet_ntoa(pkt.dest_addr.sin_addr),
                ntohs(pkt.dest_addr.sin_port)
            );
            if (auto it = m_dest_stats.find(dest_key); it != m_dest_stats.end()) {
                it->second.packets_sent++;
                it->second.bytes_sent += bytes;
            }
        }

        if (sent < static_cast<int>(batch_size)) {
            m_config.logger->warn("sendmmsg partial send: {}/{} packets", sent, batch_size);
            m_total_errors += (batch_size - sent);
        }
    }

    m_total_flushes++;
    m_batch_queue.clear();
}

auto sendmmsg_tx::cleanup_idle_sockets() -> void {
    // With single socket design, we just clean up stale destination stats
    std::scoped_lock lock(m_mutex);

    auto now = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(m_config.socket_timeout_s);

    std::vector<std::string> to_remove;
    for (const auto& [key, stats] : m_dest_stats) {
        if (now - stats.last_used > timeout) {
            to_remove.push_back(key);
        }
    }

    for (const auto& key : to_remove) {
        m_config.logger->debug("Removing idle destination stats: {}", key);
        m_dest_stats.erase(key);
    }
}

auto sendmmsg_tx::get_stats() const -> std::map<std::string, std::string> {
    std::map<std::string, std::string> stats;
    stats["total_packets"] = std::to_string(m_total_packets.load());
    stats["total_bytes"] = std::to_string(m_total_bytes.load());
    stats["total_errors"] = std::to_string(m_total_errors.load());
    stats["total_flushes"] = std::to_string(m_total_flushes.load());
    stats["active_destinations"] = std::to_string(m_dest_stats.size());
    stats["pending_batch"] = std::to_string(m_batch_queue.size());
    return stats;
}

auto sendmmsg_tx::make_dest_key(const std::string& ip, uint16_t port) -> std::string {
    return std::format("{}:{}", ip, port);
}

auto sendmmsg_tx::create_socket() -> int {
    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        m_config.logger->error("Failed to create socket: {}", strerror(errno));
        return -1;
    }

    // Set send buffer size if specified
    if (m_config.send_buf_size > 0) {
        int buf_size = static_cast<int>(m_config.send_buf_size);
        if (setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf_size, sizeof(buf_size)) < 0) {
            m_config.logger->warn("Failed to set SO_SNDBUF to {}: {}", buf_size, strerror(errno));
        }
    }

    // Bind to specific interface if specified
    if (!m_config.bind_interface.empty()) {
        struct ifreq ifr{};
        std::strncpy(ifr.ifr_name, m_config.bind_interface.c_str(), IFNAMSIZ - 1);
        if (setsockopt(fd, SOL_SOCKET, SO_BINDTODEVICE, &ifr, sizeof(ifr)) < 0) {
            m_config.logger->warn("Failed to bind to interface {}: {}", m_config.bind_interface, strerror(errno));
        }
    }

    return fd;
}

} // namespace udp_tx
