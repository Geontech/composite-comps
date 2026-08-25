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
#include <stdexcept>
#include <cstring>
#include <format>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>

namespace udp_tx {

namespace {
// Staging-buffer ceilings. A UDP payload cannot exceed 65507 bytes, and the kernel caps a single
// sendmmsg() vector at UIO_MAXIOV (1024), so neither dimension has a legitimate reason to be
// larger. Bounding both is what makes batch_size * max_packet_size safe to compute.
constexpr uint32_t MAX_BATCH_SIZE = 1024;
constexpr uint32_t MAX_PACKET_SIZE = 65507;
} // namespace

sendmmsg_tx::sendmmsg_tx(const config& cfg) : m_config(cfg) {
    // Validate the staging-buffer dimensions before anything else. The component validates these
    // properties too, but this class must not trust a config it did not build: every bound below
    // is load-bearing for the memcpy in send().
    //
    // batch_size == 0 would resize the data buffer to nothing (making every send an overflow) and
    // make `m_batch_queue.size() >= batch_size` true immediately, flushing on every packet.
    // max_packet_size == 0 would do the same to the per-datagram budget.
    if (m_config.batch_size == 0 || m_config.max_packet_size == 0) {
        throw std::invalid_argument(std::format(
            "sendmmsg_tx: batch_size and max_packet_size must be non-zero (got {} and {})",
            m_config.batch_size, m_config.max_packet_size));
    }
    if (m_config.batch_size > MAX_BATCH_SIZE || m_config.max_packet_size > MAX_PACKET_SIZE) {
        throw std::invalid_argument(std::format(
            "sendmmsg_tx: batch_size {} exceeds {} or max_packet_size {} exceeds {}",
            m_config.batch_size, MAX_BATCH_SIZE, m_config.max_packet_size, MAX_PACKET_SIZE));
    }
    // Both operands are bounded above, so the product cannot overflow size_t on any supported
    // platform; compute it in size_t regardless rather than in the uint32_t the config uses.
    const auto buffer_bytes =
        static_cast<std::size_t>(m_config.batch_size) * static_cast<std::size_t>(m_config.max_packet_size);

    // Allocate BEFORE creating the socket: the staging buffer can be up to ~64 MiB, and a
    // bad_alloc after the socket existed would leak the fd (a ctor throw skips the dtor).
    m_batch_queue.reserve(m_config.batch_size);
    m_iovecs.resize(m_config.batch_size);
    m_msgs.resize(m_config.batch_size);
    m_max_packet_size = m_config.max_packet_size;
    m_data_buffer.resize(buffer_bytes);
    m_data_buffer_pos = 0;

    m_socket_fd = create_socket();
    if (m_socket_fd < 0) {
        throw std::runtime_error("Failed to create UDP socket");
    }

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

    // Resolve the destination address. Steady state sends every packet to the same
    // endpoint, so memoize the parse: one string compare instead of inet_pton per packet.
    if (ip != m_memo_ip || port != m_memo_port) {
        struct sockaddr_in parsed{};
        parsed.sin_family = AF_INET;
        parsed.sin_port = htons(port);
        if (inet_pton(AF_INET, ip.c_str(), &parsed.sin_addr) != 1) {
            // One-shot: an invalid destination repeats per packet until upstream fixes it,
            // and the error counter carries the ongoing rate.
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
    const struct sockaddr_in dest_addr = m_memo_addr;

    // Copy packet data into the pre-allocated staging buffer.
    auto data_size = data.size();

    // A datagram larger than the configured maximum can never be staged: the flush below frees
    // at most the whole buffer, so retrying would not help. Reject it here, before any buffer
    // arithmetic. Previously this was unchecked and the memcpy ran off the end of the
    // allocation with an upstream-controlled length.
    if (data_size > m_max_packet_size) {
        if (!m_oversize_warned) {
            m_oversize_warned = true;
            m_config.logger->error(
                "dropping {}-byte datagram: exceeds max_packet_size={} (further occurrences "
                "counted, not logged)", data_size, m_max_packet_size);
        }
        m_total_errors++;
        if (m_config.send_errors != nullptr) { m_config.send_errors->inc(); }
        return -1;
    }

    if (m_data_buffer_pos + data_size > m_data_buffer.size()) {
        // Not enough room left in this batch: flush what is queued, which resets the write
        // position to the start of the buffer.
        flush_locked();
        m_data_buffer_pos = 0;
    }

    // Re-check after the flush. flush_locked() is not guaranteed to have freed space -- it can
    // fail, and a caller-supplied buffer smaller than one datagram would leave nothing to free.
    // The datagram is already known to fit within max_packet_size, so this can only trip on a
    // pathological configuration; failing the send beats corrupting the heap.
    if (m_data_buffer_pos + data_size > m_data_buffer.size()) {
        m_config.logger->error(
            "dropping {}-byte datagram: staging buffer has {} of {} bytes free after flush",
            data_size, m_data_buffer.size() - m_data_buffer_pos, m_data_buffer.size());
        m_total_errors++;
        if (m_config.send_errors != nullptr) { m_config.send_errors->inc(); }
        return -1;
    }

    std::memcpy(m_data_buffer.data() + m_data_buffer_pos, data.data(), data_size);

    // Queue packet metadata (no allocation - just offset/size)
    m_batch_queue.emplace_back(queued_packet{
        .data_offset = m_data_buffer_pos,
        .data_size = data_size,
        .dest_addr = dest_addr
    });

    m_data_buffer_pos += data_size;
    auto bytes_queued = static_cast<ssize_t>(data_size);

    // Touch destination stats (numeric key: no string formatting on the send path). New
    // destinations beyond the cap are delivered but not tracked.
    const auto dest_key = pack_dest_key(dest_addr);
    if (auto it = m_dest_stats.find(dest_key); it != m_dest_stats.end()) {
        it->second.last_used = std::chrono::steady_clock::now();
    } else if (m_dest_stats.size() < MAX_TRACKED_DESTS) {
        m_dest_stats[dest_key].last_used = std::chrono::steady_clock::now();
    }

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

        m_iovecs[i].iov_base = m_data_buffer.data() + pkt.data_offset;
        m_iovecs[i].iov_len = pkt.data_size;

        std::memset(&m_msgs[i], 0, sizeof(struct mmsghdr));
        m_msgs[i].msg_hdr.msg_name = &pkt.dest_addr;
        m_msgs[i].msg_hdr.msg_namelen = sizeof(pkt.dest_addr);
        m_msgs[i].msg_hdr.msg_iov = &m_iovecs[i];
        m_msgs[i].msg_hdr.msg_iovlen = 1;
    }

    // Drain the whole batch. sendmmsg() stops at the first message that fails, returning
    // how many it sent — the remainder was never ATTEMPTED, so treating a short return as
    // "the rest errored" (as this used to) silently dropped valid queued datagrams on any
    // transient mid-batch failure. Resume after each short return; when the head message
    // itself fails (-1), skip that one message and keep going, so one bad destination
    // cannot take the rest of the batch down with it. Each iteration advances by at least
    // one message, so the loop is bounded by the batch size.
    std::size_t offset = 0;
    std::size_t failed = 0;
    int last_errno = 0;
    uint64_t bytes_sent_total = 0;
    while (offset < batch_size) {
        const int sent = ::sendmmsg(m_socket_fd, m_msgs.data() + offset,
                                    static_cast<unsigned int>(batch_size - offset), 0);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            last_errno = errno;
            ++failed;
            ++offset;  // skip the failing head message, keep the rest of the batch
            continue;
        }
        for (int i = 0; i < sent; i++) {
            const auto idx = offset + static_cast<std::size_t>(i);
            const auto bytes = m_msgs[idx].msg_len;
            m_total_packets++;
            m_total_bytes += bytes;
            bytes_sent_total += bytes;
            if (auto it = m_dest_stats.find(pack_dest_key(m_batch_queue[idx].dest_addr));
                it != m_dest_stats.end()) {
                it->second.packets_sent++;
                it->second.bytes_sent += bytes;
            }
        }
        offset += static_cast<std::size_t>(sent);
    }

    if (failed != 0) {
        m_total_errors += failed;
        if (m_config.send_errors != nullptr) { m_config.send_errors->add(failed); }
        if (!m_send_error_warned) {
            m_send_error_warned = true;
            m_config.logger->error(
                "sendmmsg: {}/{} packets failed, last error: {} (further send errors "
                "counted, not logged)", failed, batch_size, strerror(last_errno));
        }
    }
    const auto delivered = batch_size - failed;
    if (delivered != 0 && m_config.packets_sent != nullptr) {
        m_config.packets_sent->add(delivered);
    }
    if (bytes_sent_total != 0 && m_config.bytes_sent != nullptr) {
        m_config.bytes_sent->add(bytes_sent_total);
    }

    m_total_flushes++;
    m_batch_queue.clear();
    m_data_buffer_pos = 0;  // Reset buffer position for next batch
}

auto sendmmsg_tx::cleanup_idle_sockets() -> void {
    // With single socket design, we just clean up stale destination stats
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

auto sendmmsg_tx::get_stats() const -> std::map<std::string, std::string> {
    std::scoped_lock lock(m_mutex);  // m_dest_stats/m_batch_queue are not atomics
    std::map<std::string, std::string> stats;
    stats["total_packets"] = std::to_string(m_total_packets.load());
    stats["total_bytes"] = std::to_string(m_total_bytes.load());
    stats["total_errors"] = std::to_string(m_total_errors.load());
    stats["total_flushes"] = std::to_string(m_total_flushes.load());
    stats["active_destinations"] = std::to_string(m_dest_stats.size());
    stats["pending_batch"] = std::to_string(m_batch_queue.size());
    return stats;
}

auto sendmmsg_tx::pack_dest_key(const struct sockaddr_in& addr) -> uint64_t {
    // Both fields kept in network byte order; the key is opaque until formatted.
    return (static_cast<uint64_t>(addr.sin_addr.s_addr) << 16) | addr.sin_port;
}

auto sendmmsg_tx::format_dest_key(uint64_t key) -> std::string {
    struct in_addr ip{.s_addr = static_cast<uint32_t>(key >> 16)};
    char buf[INET_ADDRSTRLEN]{};
    inet_ntop(AF_INET, &ip, buf, sizeof(buf));
    return std::format("{}:{}", buf, ntohs(static_cast<uint16_t>(key & 0xFFFF)));
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

    // Bind to specific interface if specified. Fail CLOSED: SO_BINDTODEVICE needs
    // CAP_NET_RAW, and continuing after an EPERM (the usual failure in an unprivileged
    // container) would silently egress this stream via the default route — onto a network
    // the operator explicitly steered it away from.
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
