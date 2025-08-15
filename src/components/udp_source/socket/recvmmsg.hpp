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
#include "pmr/ring_resource.hpp"

#include <atomic>
#include <composite/timestamp.hpp>
#include <cstdint>
#include <string_view>
#include <thread>

namespace udp {

class recvmmsg : public interface {
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
    std::unique_ptr<ring_resource> m_resource;
    bool m_log_frame_warn{true};
    std::size_t m_batch_size{128};
    std::atomic<uint32_t> m_pkts_recvd{};

}; // class recvmmsg

} // namespace udp
