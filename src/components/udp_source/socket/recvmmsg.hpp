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
#include <chrono>
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
    auto signal_stop() noexcept -> void;
    auto clear_stop_signal() noexcept -> void;
    auto record_kernel_drop_snapshot(uint32_t drops) noexcept -> void;
    auto note_rmem_observation(uint64_t bytes) noexcept -> void;

    output_port_t* m_out_port{nullptr};
    int m_socket{-1};
    int m_stop_fd{-1};
    std::jthread m_recv_thread;
    std::size_t m_frame_size{};
    std::size_t m_frame_count{};
    std::size_t m_autodiscovery_timeout{};
    std::shared_ptr<composite::slab_pool<uint8_t>> m_pool;
    std::size_t m_batch_size{128};
    std::chrono::microseconds m_receive_batch_wait{100};
    std::size_t m_output_batch_size{};
    std::chrono::microseconds m_max_batch_delay{1000};
    std::size_t m_effective_recv_buf{};
    std::size_t m_conservative_packet_charge{};
    bool m_recv_buf_explicit{false};

    std::atomic<uint64_t> m_recv_syscalls{0};
    std::atomic<uint64_t> m_wait_syscalls{0};
    std::atomic<uint64_t> m_full_receive_vectors{0};
    std::atomic<uint64_t> m_output_batches{0};
    std::atomic<uint64_t> m_partial_batch_flushes{0};
    std::atomic<uint64_t> m_pending_output_packets{0};
    std::atomic<uint64_t> m_socket_rmem_bytes{0};
    std::atomic<uint64_t> m_socket_rmem_peak_bytes{0}; // max since last stats report
    std::atomic<uint64_t> m_kernel_drops{0};
    std::atomic<uint32_t> m_accounted_kernel_drops{0};
    std::atomic<uint64_t> m_pkts_truncated{0}; ///< datagrams larger than the frame, dropped (MSG_TRUNC)
    int m_abort_fd{-1}; ///< owner's abort eventfd (config.abort_fd); polled during discovery, never closed/drained here

    std::atomic<uint64_t> m_congest_pool_stall{0};
    std::atomic<uint64_t> m_pool_stall_backoff_us{0};

}; // class recvmmsg

} // namespace udp
