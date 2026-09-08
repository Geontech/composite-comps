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
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace udp_tx {

/**
 * @brief High-performance UDP TX using sendmmsg() with batching
 *
 * Uses a single unconnected socket with sendmmsg() to send batches
 * of packets to multiple destinations in a single syscall.
 * Each message in the batch specifies its own destination address.
 */
class sendmmsg_tx : public interface {
public:
    explicit sendmmsg_tx(const config& cfg);
    ~sendmmsg_tx() override;

    auto send(const std::string& ip, uint16_t port, std::span<const uint8_t> data) -> ssize_t override;
    auto flush() -> void override;
    auto cleanup_idle_sockets() -> void override;
    auto get_stats() const -> std::map<std::string, std::string> override;

private:
    // Queued packet metadata (data stored in pre-allocated buffer)
    struct queued_packet {
        size_t data_offset;  // Offset into m_data_buffer
        size_t data_size;    // Size of this packet's data
        struct sockaddr_in dest_addr;
    };

    // Maximum expected packet size (configurable via config)
    size_t m_max_packet_size;

    // Per-destination stats and last-used tracking. Keyed NUMERICALLY (ip_be, port packed
    // into a uint64) so neither the send path nor the flush loop ever formats a string or
    // calls inet_ntoa (non-reentrant) per packet; the printable form is built only when a
    // key is logged or reported.
    struct dest_stats {
        std::chrono::steady_clock::time_point last_used{};
        uint64_t packets_sent{0};
        uint64_t bytes_sent{0};
    };

    config m_config;
    int m_socket_fd{-1};
    mutable std::mutex m_mutex;

    // Batch queue (packets waiting to be sent)
    std::vector<queued_packet> m_batch_queue;
    std::chrono::steady_clock::time_point m_batch_start_time{};

    // Pre-allocated data buffer (avoids per-packet malloc)
    std::vector<uint8_t> m_data_buffer;
    size_t m_data_buffer_pos{0};

    // Pre-allocated sendmmsg structures (reused across flushes)
    std::vector<struct iovec> m_iovecs;
    std::vector<struct mmsghdr> m_msgs;

    // Per-destination stats for monitoring (see dest_stats for the numeric-key rationale).
    // Insertion is capped: beyond MAX_TRACKED_DESTS new destinations are simply not tracked
    // (delivery is unaffected) — an upstream cycling destination annotations must not grow
    // this map without bound, and evicting per insert would itself be a hot-path scan.
    static constexpr std::size_t MAX_TRACKED_DESTS = 4096;
    std::unordered_map<uint64_t, dest_stats> m_dest_stats;

    // Memoized destination parse: steady state sends to one destination, so a string
    // compare replaces inet_pton per packet.
    std::string m_memo_ip;
    uint16_t m_memo_port{0};
    struct sockaddr_in m_memo_addr{};

    // One-shot log latches: the first occurrence logs, the counters carry the rate. A
    // malformed-metadata or unreachable-destination flood must not become a log flood.
    bool m_invalid_ip_warned{false};
    bool m_oversize_warned{false};
    bool m_send_error_warned{false};

    // Global stats
    std::atomic<uint64_t> m_total_packets{0};
    std::atomic<uint64_t> m_total_bytes{0};
    std::atomic<uint64_t> m_total_errors{0};
    std::atomic<uint64_t> m_total_flushes{0};

    // Helper methods
    static auto pack_dest_key(const struct sockaddr_in& addr) -> uint64_t;
    static auto format_dest_key(uint64_t key) -> std::string;
    auto create_socket() -> int;
    auto flush_locked() -> void;

}; // class sendmmsg_tx

} // namespace udp_tx
