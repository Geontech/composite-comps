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

#include <cstdint>
#include <memory>
#include <string_view>
#include <thread>

namespace udp {

class packet_mmap final : public interface {
    static constexpr auto block_size = uint32_t{1 << 20};
public:
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
    void* m_ring{nullptr};
    std::jthread m_recv_thread;
    uint32_t m_frame_size{};
    uint32_t m_frame_count{};
    uint32_t m_block_nr{};

}; // class packet_mmap

} // namespace udp
