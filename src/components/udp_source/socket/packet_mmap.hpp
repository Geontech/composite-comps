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

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <thread>

namespace udp {

class packet_mmap final : public interface {
    static constexpr auto block_size = uint32_t{1 << 20};
public:
    struct ring_buffer {
        void* ring{nullptr};
        uint32_t block_nr{0};
        // Per-slot "held downstream" flags (sized to frame_count). Set true when a slot's payload
        // is forwarded (a live external_buffer references it) and cleared by frame_release when that
        // buffer dies. Lets the recv loop distinguish a fresh kernel packet (TP_STATUS_USER, !held)
        // from an already-forwarded slot still held by a slow consumer on a ring wrap
        // (TP_STATUS_USER, held): re-reading the latter would deliver a duplicate and create a
        // second frame_release for the same slot, double-returning it to the kernel -> UAF.
        std::unique_ptr<std::atomic<bool>[]> slot_held;

        ~ring_buffer();
    };
    using ring_buffer_ptr = std::shared_ptr<ring_buffer>;

    packet_mmap(const config& config);
    ~packet_mmap() final;

    auto start_recv(output_port_t*) -> void override;
    auto stop_recv() -> void override;
    auto get_stats() -> std::map<std::string, std::string> override;

private:

    auto receive(std::stop_token token) -> void;

    output_port_t* m_out_port{nullptr};
    int m_socket{-1};
    int m_join_socket{-1};
    ring_buffer_ptr m_ring_buffer;
    std::jthread m_recv_thread;
    uint32_t m_frame_size{};
    uint32_t m_frame_count{};

    // Destination filter (mirrors the DPDK path): only forward UDP datagrams to the configured
    // dst IP/port. m_dst_ip_be is nullopt (accept any dst IP) unless a specific IPv4 was
    // configured (the multicast group, or a unicast host IP); m_dst_port == 0 accepts any port.
    std::optional<uint32_t> m_dst_ip_be{};  // network byte order
    uint16_t m_dst_port{};                  // host byte order

}; // class packet_mmap

} // namespace udp
