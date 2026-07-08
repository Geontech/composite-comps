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

#include "send.hpp"

#include <arpa/inet.h>
#include <cstring>
#include <format>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>

namespace udp_tx {

send_tx::send_tx(const config& cfg) : m_config(cfg) {
    m_socket_fd = create_socket();
    if (m_socket_fd < 0) {
        throw std::runtime_error("Failed to create UDP socket");
    }
    m_config.logger->info("send_tx initialized (simple sendto mode)");
}

send_tx::~send_tx() {
    if (m_socket_fd >= 0) {
        ::close(m_socket_fd);
        m_socket_fd = -1;
    }
}

auto send_tx::send(const std::string& ip, uint16_t port, std::span<const uint8_t> data) -> ssize_t {
    // Create destination address
    struct sockaddr_in dest_addr{};
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip.c_str(), &dest_addr.sin_addr) != 1) {
        m_config.logger->error("Invalid IP address: {}", ip);
        m_total_errors++;
        return -1;
    }

    // Send the packet
    ssize_t sent = ::sendto(m_socket_fd, data.data(), data.size(), 0,
                            reinterpret_cast<struct sockaddr*>(&dest_addr),
                            sizeof(dest_addr));

    if (sent < 0) {
        m_config.logger->error("sendto failed: {}", strerror(errno));
        m_total_errors++;
        return -1;
    }

    // Update stats
    m_total_packets++;
    m_total_bytes += sent;

    auto dest_key = make_dest_key(ip, port);
    std::scoped_lock lock(m_mutex);
    auto& stats = m_dest_stats[dest_key];
    stats.last_used = std::chrono::steady_clock::now();
    stats.packets_sent++;
    stats.bytes_sent += sent;

    return sent;
}

auto send_tx::flush() -> void {
    // No-op for simple send - packets are sent immediately
}

auto send_tx::cleanup_idle_sockets() -> void {
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

auto send_tx::get_stats() const -> std::map<std::string, std::string> {
    std::map<std::string, std::string> stats;
    stats["total_packets"] = std::to_string(m_total_packets.load());
    stats["total_bytes"] = std::to_string(m_total_bytes.load());
    stats["total_errors"] = std::to_string(m_total_errors.load());
    stats["active_destinations"] = std::to_string(m_dest_stats.size());
    return stats;
}

auto send_tx::make_dest_key(const std::string& ip, uint16_t port) -> std::string {
    return std::format("{}:{}", ip, port);
}

auto send_tx::create_socket() -> int {
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
