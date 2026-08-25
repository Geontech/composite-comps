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

#pragma once

#include "interface.hpp"

#include <netinet/in.h>

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

namespace udp_tx {

/**
 * @brief Simple UDP TX using sendto() per packet
 *
 * Uses a single unconnected socket with sendto() for each packet.
 * Simpler than sendmmsg but less efficient for high throughput.
 */
class send_tx : public interface {
public:
    explicit send_tx(const config& cfg);
    ~send_tx() override;

    auto send(const std::string& ip, uint16_t port, std::span<const uint8_t> data) -> ssize_t override;
    auto flush() -> void override;
    auto cleanup_idle_sockets() -> void override;
    auto get_stats() const -> std::map<std::string, std::string> override;

private:
    // Per-destination stats, keyed numerically (ip_be, port) — no per-packet string
    // formatting; the printable form is built only for logging/reporting. Capped: new
    // destinations beyond the cap are delivered but not tracked.
    static constexpr std::size_t MAX_TRACKED_DESTS = 4096;
    struct dest_stats {
        std::chrono::steady_clock::time_point last_used{};
        uint64_t packets_sent{0};
        uint64_t bytes_sent{0};
    };

    config m_config;
    int m_socket_fd{-1};
    mutable std::mutex m_mutex;

    std::unordered_map<uint64_t, dest_stats> m_dest_stats;

    // Memoized destination parse (steady state = one endpoint: string compare, not inet_pton).
    std::string m_memo_ip;
    uint16_t m_memo_port{0};
    struct sockaddr_in m_memo_addr{};

    // One-shot log latches; the counters carry the ongoing rate.
    bool m_invalid_ip_warned{false};
    bool m_send_error_warned{false};

    std::atomic<uint64_t> m_total_packets{0};
    std::atomic<uint64_t> m_total_bytes{0};
    std::atomic<uint64_t> m_total_errors{0};

    static auto pack_dest_key(const struct sockaddr_in& addr) -> uint64_t;
    static auto format_dest_key(uint64_t key) -> std::string;
    auto create_socket() -> int;

}; // class send_tx

} // namespace udp_tx
