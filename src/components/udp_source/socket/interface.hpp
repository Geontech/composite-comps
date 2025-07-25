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
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */

 #pragma once

#include "statistics.hpp"

#include <bit>
#include <composite/output_port.hpp>
#include <memory>
#include <memory_resource>
#include <spdlog/spdlog.h>

namespace udp {

enum class transport : uint8_t {
    unknown,
    sdds,
    vita49
}; // enum class transport

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
    std::shared_ptr<spdlog::logger> logger;

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
     * @brief The transport encapsulation of the UDP data (e.g., "sdds" or "vita49")
     */
    std::string transport;

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
     * @brief Size of the RX ring (receive queue) for a given port.
     */
    uint16_t rx_ring_size{};

    /**
     * @brief Total number of memory buffers (mbufs) to allocate in the mempool.
     */
    uint32_t num_mbufs{};

    /**
     * @brief Size of the per-core object cache in the mempool.
     */
    uint16_t mbuf_cache_size{};

    /**
     * @brief Number of packets to retrieve or send in a single burst.
     */
    uint32_t burst_size{};

    /**
     * @brief How much memory dpdk will allocate on the socket. --socket-mem=
     */
    std::string socket_mem{};

    /**
     * @brief Record PCAP
     */
    bool write_pcap{};

    /**
     * @brief Max directory size for pcap recording in integer gigabytes
     */
    std::optional<uint64_t> dir_size_GB;
    /**
     * @brief Directory to write pcaps to
     */
    std::optional<std::string> write_directory;
    /**
     * @brief Number of packets per pcap
     * Lower values may slow performance due to file rotation
     */
    std::optional<uint64_t> packets_per_pcap; 

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
    /**
     * @brief Type alias for a polymorphic vector of bytes.
     */
    using buffer_t = std::pmr::vector<uint8_t>;

    /**
     * @brief Type alias for a unique pointer to a receive buffer.
     */
    using buffer_ptr_t = std::shared_ptr<buffer_t>;
    using output_port_t = composite::output_port<buffer_ptr_t>;

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

    // /**
    //  * @brief Retrieve the next available received buffer.
    //  *
    //  * @param out Reference to a shared pointer that will be populated
    //  *        with the next available data buffer, if any.
    //  * @return True if a buffer was successfully retrieved, false otherwise.
    //  */
    // virtual auto get_data(std::shared_ptr<buffer_t>&) -> bool = 0;

    /**
     * @brief Retrieve runtime statistics for the receiver.
     *
     * @return A snapshot of internal statistics such as packets received,
     *         dropped, errors, etc.
     */
    virtual auto get_stats() -> statistics = 0;

    /**
     * @brief Get the discovered transport protocol being used
     * 
     * @return The transport protocol being received
     */
    auto get_transport() const -> transport {
        return m_transport;
    }

protected:
    /**
     * @brief Default constructor.
     *
     * @param logger spdlog logger instance
     */
    explicit interface(std::shared_ptr<spdlog::logger> logger) : m_logger(logger) {}

    /**
     * @brief Logger instance
     */
    std::shared_ptr<spdlog::logger> m_logger;

    /**
     * @brief Trasnport protocol being received
     */
    transport m_transport;

}; // class interface

} // namespace udp
