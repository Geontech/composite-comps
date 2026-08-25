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
    std::scoped_lock lock(m_mutex);

    // Resolve the destination, memoized: steady state is one endpoint, so a string compare
    // replaces inet_pton per packet.
    if (ip != m_memo_ip || port != m_memo_port) {
        struct sockaddr_in parsed{};
        parsed.sin_family = AF_INET;
        parsed.sin_port = htons(port);
        if (inet_pton(AF_INET, ip.c_str(), &parsed.sin_addr) != 1) {
            // One-shot: a poisoned destination repeats per packet; the counter carries the rate.
            if (!m_invalid_ip_warned) {
                m_invalid_ip_warned = true;
                m_config.logger->error("Invalid IP address: {} (further occurrences counted, not logged)", ip);
            }
            m_total_errors++;
            if (m_config.send_errors != nullptr) { m_config.send_errors->inc(); }
            return -1;
        }
        m_memo_ip = ip;
        m_memo_port = port;
        m_memo_addr = parsed;
    }
    struct sockaddr_in dest_addr = m_memo_addr;

    // Send the packet
    ssize_t sent = ::sendto(m_socket_fd, data.data(), data.size(), 0,
                            reinterpret_cast<struct sockaddr*>(&dest_addr),
                            sizeof(dest_addr));

    if (sent < 0) {
        if (!m_send_error_warned) {
            m_send_error_warned = true;
            m_config.logger->error("sendto failed: {} (further send errors counted, not logged)",
                                   strerror(errno));
        }
        m_total_errors++;
        if (m_config.send_errors != nullptr) { m_config.send_errors->inc(); }
        return -1;
    }

    // Update stats
    m_total_packets++;
    m_total_bytes += sent;
    if (m_config.packets_sent != nullptr) { m_config.packets_sent->inc(); }
    if (m_config.bytes_sent != nullptr) { m_config.bytes_sent->add(static_cast<uint64_t>(sent)); }

    const auto dest_key = pack_dest_key(dest_addr);
    if (auto it = m_dest_stats.find(dest_key); it != m_dest_stats.end()) {
        it->second.last_used = std::chrono::steady_clock::now();
        it->second.packets_sent++;
        it->second.bytes_sent += static_cast<uint64_t>(sent);
    } else if (m_dest_stats.size() < MAX_TRACKED_DESTS) {
        auto& stats = m_dest_stats[dest_key];
        stats.last_used = std::chrono::steady_clock::now();
        stats.packets_sent = 1;
        stats.bytes_sent = static_cast<uint64_t>(sent);
    }

    return sent;
}

auto send_tx::flush() -> void {
    // No-op for simple send - packets are sent immediately
}

auto send_tx::cleanup_idle_sockets() -> void {
    std::scoped_lock lock(m_mutex);

    auto now = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(m_config.socket_timeout_s);

    std::vector<uint64_t> to_remove;
    for (const auto& [key, stats] : m_dest_stats) {
        if (now - stats.last_used > timeout) {
            to_remove.push_back(key);
        }
    }

    for (const auto key : to_remove) {
        m_config.logger->debug("Removing idle destination stats: {}", format_dest_key(key));
        m_dest_stats.erase(key);
    }
}

auto send_tx::get_stats() const -> std::map<std::string, std::string> {
    std::scoped_lock lock(m_mutex);  // m_dest_stats is not atomic
    std::map<std::string, std::string> stats;
    stats["total_packets"] = std::to_string(m_total_packets.load());
    stats["total_bytes"] = std::to_string(m_total_bytes.load());
    stats["total_errors"] = std::to_string(m_total_errors.load());
    stats["active_destinations"] = std::to_string(m_dest_stats.size());
    return stats;
}

auto send_tx::pack_dest_key(const struct sockaddr_in& addr) -> uint64_t {
    return (static_cast<uint64_t>(addr.sin_addr.s_addr) << 16) | addr.sin_port;
}

auto send_tx::format_dest_key(uint64_t key) -> std::string {
    struct in_addr ip{.s_addr = static_cast<uint32_t>(key >> 16)};
    char buf[INET_ADDRSTRLEN]{};
    inet_ntop(AF_INET, &ip, buf, sizeof(buf));
    return std::format("{}:{}", buf, ntohs(static_cast<uint16_t>(key & 0xFFFF)));
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

    // Bind to specific interface if specified. Fail CLOSED (see sendmmsg.cpp): silently
    // egressing via the default route after an EPERM defeats the operator's steering.
    if (!m_config.bind_interface.empty()) {
        struct ifreq ifr{};
        std::strncpy(ifr.ifr_name, m_config.bind_interface.c_str(), IFNAMSIZ - 1);
        if (setsockopt(fd, SOL_SOCKET, SO_BINDTODEVICE, &ifr, sizeof(ifr)) < 0) {
            const auto msg = std::format("failed to bind to interface {}: {}",
                                         m_config.bind_interface, strerror(errno));
            m_config.logger->error(msg);
            ::close(fd);
            throw std::runtime_error(msg);
        }
    }

    return fd;
}

} // namespace udp_tx
