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

#include <composite/buffers/slab_pool.hpp>

#include <cstdint>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace udp {

class recvmmsg final : public interface {
public:
    explicit recvmmsg(const config& config);
    ~recvmmsg() final;
    recvmmsg(const recvmmsg&) = delete;
    recvmmsg(recvmmsg&&) = delete;
    recvmmsg& operator=(const recvmmsg&) = delete;
    recvmmsg& operator=(recvmmsg&&) = delete;

    auto start_recv(output_port_t*) -> void override;
    auto stop_recv() -> void override;
    auto get_stats() -> std::map<std::string, std::string> override;

private:
    auto receive(std::stop_token token) -> void;

    output_port_t* m_out_port{nullptr};
    int m_socket{-1};
    std::jthread m_recv_thread;
    std::size_t m_frame_size{};
    std::size_t m_frame_count{};
    std::size_t m_autodiscovery_timeout{};
    std::shared_ptr<composite::slab_pool<uint8_t>> m_pool;
    std::size_t m_batch_size{128};

}; // class recvmmsg

} // namespace udp
