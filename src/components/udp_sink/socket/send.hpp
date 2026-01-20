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
    // Per-destination stats
    struct dest_stats {
        std::chrono::steady_clock::time_point last_used{};
        uint64_t packets_sent{0};
        uint64_t bytes_sent{0};
    };

    config m_config;
    int m_socket_fd{-1};
    std::mutex m_mutex;

    std::unordered_map<std::string, dest_stats> m_dest_stats;

    std::atomic<uint64_t> m_total_packets{0};
    std::atomic<uint64_t> m_total_bytes{0};
    std::atomic<uint64_t> m_total_errors{0};

    auto make_dest_key(const std::string& ip, uint16_t port) -> std::string;
    auto create_socket() -> int;

}; // class send_tx

} // namespace udp_tx
