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

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>

#include <spdlog/logger.h>

namespace udp_tx {

/**
 * @brief Configuration for UDP TX backends
 */
struct config {
    std::shared_ptr<spdlog::logger> logger;
    uint32_t send_buf_size{0};          // Socket send buffer size (0 = system default)
    uint32_t socket_timeout_s{30};      // Close idle sockets after N seconds
    uint32_t batch_size{64};            // Max packets per sendmmsg() call
    uint32_t batch_timeout_us{1000};    // Flush batch after N microseconds
    std::string bind_interface;         // Optional: bind to specific interface
};

/**
 * @brief Abstract interface for UDP TX backends
 *
 * Implementations handle socket management and sending packets
 * to multiple destinations efficiently.
 */
class interface {
public:
    virtual ~interface() = default;

    /**
     * @brief Send a packet to the specified destination
     * @param ip Destination IP address
     * @param port Destination port
     * @param data Packet data to send
     * @return Number of bytes sent, or -1 on error
     */
    virtual auto send(const std::string& ip, uint16_t port, std::span<const uint8_t> data) -> ssize_t = 0;

    /**
     * @brief Flush any buffered packets (for batching implementations)
     */
    virtual auto flush() -> void = 0;

    /**
     * @brief Clean up idle sockets that haven't been used recently
     */
    virtual auto cleanup_idle_sockets() -> void = 0;

    /**
     * @brief Get statistics for logging/monitoring
     * @return Map of stat name to value string
     */
    virtual auto get_stats() const -> std::map<std::string, std::string> = 0;

}; // class interface

} // namespace udp_tx
