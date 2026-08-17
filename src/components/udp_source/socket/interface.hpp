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

#include <atomic>
#include <bit>
#include <composite/buffers/buffer.hpp>
#include <composite/metrics/metrics.hpp>
#include <composite/ports/output_port.hpp>
#include <map>
#include <memory>
#include <memory_resource>
#include <composite/core/logger.hpp>

namespace udp {

/**
 * @brief Metrics for UDP receiver implementations.
 *
 * These are created by the component and passed to the receiver via config.
 * All metrics are required - receivers always record to these.
 */
struct metrics {
    composite::metrics::counter<uint64_t>& packets_received;
    composite::metrics::counter<uint64_t>& bytes_received;
    composite::metrics::counter<uint64_t>& packets_dropped;
    composite::metrics::counter<uint64_t>& kernel_drops;
    composite::metrics::histogram& batch_sizes;
}; // struct metrics

/**
 * @brief Configuration parameters for a UDP receiver interface.
 *
 * This struct contains all user-defined parameters necessary to initialize
 * and operate a UDP receive interface, including buffer sizes, IP settings,
 * and batching preferences.
 */
struct config {

    /**
     * @brief Logger instance
     */
    std::shared_ptr<composite::logger> logger;

    /**
     * @brief Network interface name (e.g., "eth0").
     */
    std::string interface;

    /**
     * @brief IP address to bind to (e.g., "239.0.0.1" for multicast).
     */
    std::string ip_addr;

    /**
     * @brief UDP port to listen on.
     */
    uint16_t port{};

    /**
     * @brief Size of the socket receive buffer (in bytes).
     */
    std::size_t recv_buf_size{};

    /**
     * @brief Number of packets to receive per system call (batch size).
     */
    std::size_t batch_size{};

    /**
     * @brief Size of each UDP message payload (in bytes).
     */
    std::size_t msg_size{};

    /**
     * @brief Number of frame buffers to preallocate.
     */
    std::size_t frame_count{};

    /**
     * @brief Timeout in seconds for packet size auto-discovery.
     * Only applies to recvmmsg socket type when msg_size is not set.
     */
    std::size_t autodiscovery_timeout{10};

    /**
     * Adaptive recvmmsg coalescing controls. A target of zero selects 75% of batch_size.
     * The receive thread estimates the stream packet rate and waits long enough after an
     * empty-to-readable transition to build approximately this many queued datagrams.
     */
    std::size_t coalesce_target_batch{};
    std::size_t min_coalesce_us{};
    std::size_t max_coalesce_us{};
    std::size_t adaptation_interval_ms{250};

    /**
     * @brief Optional metrics for the receiver.
     * If provided, the receiver will record metrics to these counters.
     */
    udp::metrics metrics;

}; // struct config

/**
 * @brief Abstract base class representing a generic UDP receive interface.
 *
 * Derived classes are expected to implement the actual receive logic using
 * different transport backends (e.g., raw sockets, DPDK, AF_PACKET).
 * The interface handles lifecycle control (start/stop), buffer delivery,
 * and runtime statistics reporting.
 */
class interface {
public:
    using buffer_t = std::pmr::vector<uint8_t>;
    using buffer_ptr_t = std::shared_ptr<buffer_t>;
    using output_port_t = composite::output_port<composite::immutable_buffer<uint8_t>>;

    // Non-copyable and non-movable
    interface(const interface&) = delete;
    interface(interface&&) = delete;
    interface& operator=(const interface&) = delete;
    interface& operator=(interface&&) = delete;

    /**
     * @brief Virtual destructor.
     */
    virtual ~interface() = default;

    /**
     * @brief Start the receive loop.
     *
     * This method must be implemented by derived classes to begin
     * packet capture, buffer population, and data output.
     */
    virtual auto start_recv(output_port_t*) -> void = 0;

    /**
     * @brief Stop the receive loop.
     *
     * This method is expected to terminate any active receive operations.
     */
    virtual auto stop_recv() -> void = 0;

    /**
     * @brief Retrieve runtime statistics for the receiver.
     *
     * @return A snapshot of internal statistics such as packets received,
     *         dropped, errors, etc.
     */
    virtual auto get_stats() -> std::map<std::string, std::string> = 0;

protected:
    /**
     * @brief Constructor.
     *
     * @param logger spdlog logger instance
     * @param metrics metrics for recording
     */
    interface(std::shared_ptr<composite::logger> logger, udp::metrics metrics)
        : m_logger(logger), m_metrics(metrics) {}

    /**
     * @brief Logger instance
     */
    std::shared_ptr<composite::logger> m_logger;

    /**
     * @brief Metrics for recording
     */
    udp::metrics m_metrics;

    std::atomic<uint64_t> m_pkts_recvd{0};      ///< Total packets received

}; // class interface

} // namespace udp
