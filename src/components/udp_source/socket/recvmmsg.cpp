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

#include <composite/buffers/external_buffer.hpp>

#include "net/utils.hpp"
#include "recvmmsg.hpp"

#include <arpa/inet.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <format>
#include <limits>
#include <linux/sock_diag.h>
#include <netinet/in.h>
#include <net/if.h>
#include <poll.h>
#include <stdexcept>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace udp {

recvmmsg::recvmmsg(const config& config) :
  interface(config.logger, config.metrics),
  m_frame_count(config.frame_count),
  m_autodiscovery_timeout(config.autodiscovery_timeout),
  m_coalesce_target_batch(config.coalesce_target_batch),
  m_min_coalesce(std::chrono::microseconds(config.min_coalesce_us)),
  m_max_coalesce(std::chrono::microseconds(config.max_coalesce_us)),
  m_adaptation_interval(std::chrono::milliseconds(config.adaptation_interval_ms)),
  m_recv_buf_explicit(config.recv_buf_size > 0) {
    // Create socket
    m_logger->trace("opening udp socket");
    m_socket = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_socket < 0) {
        throw std::runtime_error(std::format("failed to create socket: {}", std::string{strerror(errno)}));
    }

    // Allow rapid rebind (avoids "address already in use" during reconfiguration)
    int reuse = 1;
    if (::setsockopt(m_socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        m_logger->warn("failed to set SO_REUSEADDR: {}", std::string{strerror(errno)});
    }

    // Set receive buffer size
    if (config.recv_buf_size > 0) {
        m_logger->trace("setting socket receive buffer size to {}", config.recv_buf_size);
        net::set_socket_recv_buffer(m_socket, config.recv_buf_size);
    }
    m_effective_recv_buf = net::get_socket_recv_buffer(m_socket);
    if (m_recv_buf_explicit) {
        const auto expected = static_cast<uint64_t>(config.recv_buf_size) * 2;
        if (m_effective_recv_buf < expected) {
            m_logger->warn("SO_RCVBUF request of {} bytes was clamped; effective kernel accounting "
                           "limit is {} bytes", config.recv_buf_size, m_effective_recv_buf);
        } else {
            m_logger->debug("SO_RCVBUF requested={} bytes, effective={} bytes",
                            config.recv_buf_size, m_effective_recv_buf);
        }
    } else {
        m_logger->debug("using system-default SO_RCVBUF (effective={} bytes); adaptive recvmmsg "
                        "coalescing is disabled unless recv_buf_size is explicitly configured",
                        m_effective_recv_buf);
    }

    // Set batch size
    if (config.batch_size > 0) {
        m_batch_size = config.batch_size;
    }
    m_logger->trace("using recvmmsg batch size of {}", m_batch_size);

    // Bind the socket
    auto bind_addr = inet_addr(config.ip_addr.data());
    auto is_multicast = net::is_ipv4_multicast(config.ip_addr);
    if (is_multicast) {
        bind_addr = INADDR_ANY;
    }
    if (bind_addr == INADDR_ANY) {
        m_logger->trace("binding socket to 0.0.0.0:{}", config.port);
    } else {
        m_logger->trace("binding socket to {}:{}", config.ip_addr, config.port);
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = bind_addr;
    addr.sin_port = htons(config.port);
    if (::bind(m_socket, (const struct sockaddr*)&addr, sizeof(addr)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to bind socket: {}", std::string{strerror(errno)}));
    }

    // Join multicast group
    if (is_multicast) {
        m_logger->trace("subscribing to multicast group {} on interface {}", config.ip_addr, config.interface);
        auto ip_mreq = net::create_ip_mreq(m_socket, config.interface, config.ip_addr);
        if (::setsockopt(m_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &ip_mreq, sizeof(ip_mreq)) < 0) {
            ::close(m_socket);
            throw std::runtime_error(std::format("failed to join multicast group: {}", std::string{strerror(errno)}));
        }
    }

    // User has overridden properties - allocate frame pool
    if (config.msg_size > 0) {
        m_frame_size = std::bit_ceil(config.msg_size);
        m_pool = composite::slab_pool<uint8_t>::create(m_frame_size, m_frame_count);
    }

    // eventfd makes both the indefinite WAIT and the adaptive COALESCE delay immediately
    // interruptible. It is process-local and requires no privileges.
    m_stop_fd = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    if (m_stop_fd < 0) {
        auto err = std::string{strerror(errno)};
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to create recvmmsg stop event: {}", err));
    }
}

recvmmsg::~recvmmsg() {
    m_recv_thread.request_stop();
    signal_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
    if (m_stop_fd != -1) {
        ::close(m_stop_fd);
    }
    if (m_socket != -1) {
        ::close(m_socket);
    }
}

auto recvmmsg::start_recv(output_port_t* port) -> void {
    // User has not overriden properties (or requesting auto-discovery)
    if (m_frame_size == 0) {
        // Minimum data packet size in an attempt to exclude context packets. Deliberately signed
        // to match recvfrom's ssize_t return: as std::size_t it would make the comparison below
        // unsigned, promoting a -1 error return to SIZE_MAX and passing the size check.
        static constexpr ssize_t MIN_DATA_PACKET_SIZE = 512;

        // VITA 49 packet type identifiers
        static constexpr uint8_t V49_CONTEXT_PACKET = 0x40;
        static constexpr uint8_t V49_EXT_CONTEXT_PACKET = 0x50;

        // Discover the size of the incoming packets from the wire
        std::array<uint8_t, 9000> buffer{}; // Use a jumbo frame buffer
        auto attempts = 0;
        const auto max_attempts = static_cast<int>(m_autodiscovery_timeout);

        while (attempts < max_attempts) {
            struct pollfd pfd{
                .fd = m_socket,
                .events = POLLIN,
                .revents = 0
            };
            if (auto poll_res = ::poll(&pfd, 1, 1000/*ms*/); poll_res <= 0) {
                ++attempts;
                m_logger->debug(
                    "waiting for data to know how to size internal buffers... (attempt {}/{})",
                    attempts, max_attempts
                );
                continue;
            }
            // MSG_DONTWAIT: poll() provides the wait; a poll-ready datagram discarded at receive
            // time (deferred UDP checksum validation) must EAGAIN back to poll, not block stop()
            // for the whole discovery window.
            if (auto recvd = ::recvfrom(m_socket, buffer.data(), buffer.size(), MSG_DONTWAIT, nullptr, nullptr); recvd > MIN_DATA_PACKET_SIZE) {
                // Skip V49 context packets, wait for data packet
                auto pkt_type = buffer[0] & 0xF0;
                if (pkt_type == V49_CONTEXT_PACKET || pkt_type == V49_EXT_CONTEXT_PACKET) {
                    continue;
                }
                m_logger->trace("using discovered msg_size of: {} bytes", recvd);
                m_frame_size = std::bit_ceil(static_cast<std::size_t>(recvd));
                m_pool = composite::slab_pool<uint8_t>::create(m_frame_size, m_frame_count);
                break;
            }
        }

        if (m_frame_size == 0) {
            throw std::runtime_error(
                std::format(
                    "failed to discover packet size after {} seconds - no valid data packets received. "
                    "Consider setting 'overrides.msg_size' explicitly in the configuration.",
                    m_autodiscovery_timeout
                )
            );
        }
    }
    if (!m_pool) {
        m_pool = composite::slab_pool<uint8_t>::create(m_frame_size, m_frame_count);
    }
    if (m_frame_count < m_batch_size) {
        m_logger->warn("frame_count ({}) smaller than batch_size ({}); clamping batch_size to frame count",
                       m_frame_count, m_batch_size);
        m_batch_size = m_frame_count;
    }
    if (m_batch_size == 0) {
        throw std::runtime_error("recvmmsg frame_count and batch_size must be greater than zero");
    }
    if (m_max_coalesce < m_min_coalesce) {
        m_logger->warn("recvmmsg min_coalesce_us exceeds max_coalesce_us; clamping the minimum");
        m_min_coalesce = m_max_coalesce;
    }
    if (m_adaptation_interval <= std::chrono::milliseconds::zero()) {
        m_adaptation_interval = std::chrono::milliseconds(250);
    }

    // Linux charges receive-buffer space by skb allocation, not UDP payload bytes. Use a
    // deliberately conservative power-of-two charge estimate and reserve 75% of SO_RCVBUF for
    // scheduler jitter, drain work, and bursts outside the intentional coalescing window.
    m_conservative_packet_charge = std::max<std::size_t>(
        4096, std::bit_ceil(m_frame_size + std::size_t{1024}));
    const auto safe_packet_budget = m_effective_recv_buf / 4 / m_conservative_packet_charge;
    auto requested_target = m_coalesce_target_batch == 0
        ? (m_batch_size * 3) / 4 : m_coalesce_target_batch;
    requested_target = std::clamp(requested_target, std::size_t{1}, m_batch_size);

    if (m_max_coalesce > std::chrono::microseconds::zero() && !m_recv_buf_explicit) {
        m_logger->warn("adaptive recvmmsg coalescing requested without an explicit recv_buf_size; "
                       "disabling coalescing to protect the system-default socket buffer");
        m_max_coalesce = std::chrono::microseconds::zero();
        m_min_coalesce = std::chrono::microseconds::zero();
    }
    if (m_max_coalesce > std::chrono::microseconds::zero() && safe_packet_budget < 2) {
        m_logger->warn("effective SO_RCVBUF ({}) is too small for safe adaptive coalescing; disabling it",
                       m_effective_recv_buf);
        m_max_coalesce = std::chrono::microseconds::zero();
        m_min_coalesce = std::chrono::microseconds::zero();
    }
    m_coalesce_target_batch = m_max_coalesce > std::chrono::microseconds::zero()
        ? std::min(requested_target, safe_packet_budget)
        : std::size_t{1};
    if (m_max_coalesce > std::chrono::microseconds::zero()
        && m_coalesce_target_batch < requested_target) {
        m_logger->warn("recvmmsg target_batch {} exceeds the 25% SO_RCVBUF safety budget; "
                       "clamping target to {}", requested_target, m_coalesce_target_batch);
    }
    m_out_port = port;
    clear_stop_signal();
    m_recv_thread = std::jthread(&recvmmsg::receive, this);
    if (auto ret = pthread_setname_np(m_recv_thread.native_handle(), "recvmmsg"); ret != 0) {
        m_logger->warn("failed to set thread name: {}", std::string{strerror(ret)});
    }
}

auto recvmmsg::stop_recv() -> void {
    m_recv_thread.request_stop();
    signal_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
}

auto recvmmsg::signal_stop() noexcept -> void {
    if (m_stop_fd == -1) { return; }
    eventfd_t value = 1;
    if (::eventfd_write(m_stop_fd, value) < 0 && errno != EAGAIN) {
        // There is no useful recovery action here. The stop token still terminates a receiver
        // that is actively draining; only a thread asleep in poll/ppoll depends on this signal.
    }
}

auto recvmmsg::clear_stop_signal() noexcept -> void {
    if (m_stop_fd == -1) { return; }
    eventfd_t value{};
    while (::eventfd_read(m_stop_fd, &value) == 0) {}
}

auto recvmmsg::get_stats() -> std::map<std::string, std::string> {
    // Pull socket-memory diagnostics on the low-rate stats path as well as the adaptive receive
    // path. This keeps the default (coalescing-disabled) hot path free of extra getsockopt calls.
    std::array<uint32_t, SK_MEMINFO_VARS> memory{};
    socklen_t memory_len = sizeof(memory);
    if (::getsockopt(m_socket, SOL_SOCKET, SO_MEMINFO, memory.data(), &memory_len) == 0) {
        const auto count = static_cast<std::size_t>(memory_len) / sizeof(memory[0]);
        if (count > SK_MEMINFO_RMEM_ALLOC) {
            m_socket_rmem_bytes.store(memory[SK_MEMINFO_RMEM_ALLOC], std::memory_order_relaxed);
        }
        if (count > SK_MEMINFO_DROPS) {
            m_kernel_drops.store(memory[SK_MEMINFO_DROPS], std::memory_order_relaxed);
            record_kernel_drop_snapshot(memory[SK_MEMINFO_DROPS]);
        }
    }
    auto stats = std::map<std::string, std::string>{};
    stats["pkts_recvd"] = std::to_string(m_pkts_recvd.load());
    stats["recv_syscalls"] = std::to_string(m_recv_syscalls.load());
    stats["estimated_pps"] = std::to_string(m_estimated_pps.load());
    stats["coalesce_us"] = std::to_string(m_coalesce_us.load());
    stats["coalesce_target_batch"] = std::to_string(m_coalesce_target_batch);
    stats["effective_recv_buf"] = std::to_string(m_effective_recv_buf);
    stats["estimated_packet_charge"] = std::to_string(m_conservative_packet_charge);
    stats["socket_rmem_bytes"] = std::to_string(m_socket_rmem_bytes.load());
    stats["kernel_drops"] = std::to_string(m_kernel_drops.load());
    return stats;
}

auto recvmmsg::record_kernel_drop_snapshot(uint32_t drops) noexcept -> void {
    auto previous = m_accounted_kernel_drops.load(std::memory_order_relaxed);
    while (previous != drops) {
        const auto delta = static_cast<uint32_t>(drops - previous);
        // A small modular distance handles one u32 wrap. A distance in the upper half of the
        // range means a concurrent observer already published a newer snapshot, so ignore this
        // older sample. This assumes fewer than 2^31 drops occur between observations.
        if (delta > std::numeric_limits<uint32_t>::max() / 2) { return; }
        if (m_accounted_kernel_drops.compare_exchange_weak(
                previous, drops, std::memory_order_relaxed, std::memory_order_relaxed)) {
            m_metrics.kernel_drops.add(delta);
            return;
        }
    }
}

auto recvmmsg::receive(std::stop_token token) -> void {
    if (!m_pool) {
        m_logger->error("recvmmsg pool not initialized");
        return;
    }

    // Pre-allocate structures for batch operations
    auto iovecs = std::vector<struct iovec>(m_batch_size);
    auto msgs = std::vector<struct mmsghdr>(m_batch_size);
    using buffer_opt_t = std::optional<composite::external_buffer<uint8_t>>;
    auto buffers = std::vector<buffer_opt_t>(m_batch_size);
    auto output_buffers = std::vector<composite::immutable_buffer<uint8_t>>{};
    output_buffers.reserve(m_batch_size);
    bool pool_stalled = false;

    // Acquire buffers from pool and set up iovecs
    auto acquire_buffer = [&](std::size_t idx) -> bool {
        constexpr auto ACQUIRE_BACKOFF = std::chrono::microseconds(50);
        bool first_attempt = true;
        while (!token.stop_requested()) {
            if (auto buf = m_pool->acquire()) {
                buffers[idx] = std::move(buf);
                iovecs[idx].iov_base = buffers[idx]->data();
                iovecs[idx].iov_len = buffers[idx]->size();
                msgs[idx].msg_hdr.msg_iov = &iovecs[idx];
                msgs[idx].msg_hdr.msg_iovlen = 1;
                msgs[idx].msg_hdr.msg_control = nullptr;
                msgs[idx].msg_hdr.msg_controllen = 0;
                msgs[idx].msg_hdr.msg_name = nullptr;
                msgs[idx].msg_hdr.msg_namelen = 0;
                msgs[idx].msg_len = 0;
                return true;
            }
            if (first_attempt) {
                pool_stalled = true;
                first_attempt = false;
            }
            std::this_thread::sleep_for(ACQUIRE_BACKOFF);
        }
        return false;
    };

    // Initial batch acquisition
    for (std::size_t i = 0; i < m_batch_size; ++i) {
        if (!acquire_buffer(i)) { return; }
    }
    pool_stalled = false; // initial pool population is not downstream congestion

    using clock = std::chrono::steady_clock;
    constexpr double RATE_EWMA_ALPHA = 0.2;
    constexpr auto IDLE_RATE_RESET = std::chrono::seconds(1);
    constexpr std::size_t CONGESTION_CLEAN_CYCLES = 8;
    constexpr uint64_t COALESCE_RMEM_DIVISOR = 4; // intentional delay may use at most 25%

    auto rate_window_start = clock::now();
    uint64_t rate_window_packets = 0;
    uint64_t recv_syscalls = 0;
    double estimated_pps = 0.0;
    double nominal_coalesce_us = 0.0;
    std::size_t congestion_clean_cycles_remaining = 0;
    const bool adaptive_enabled = m_max_coalesce > std::chrono::microseconds::zero();
    std::optional<clock::time_point> last_receive_error_log;
    uint64_t suppressed_receive_errors = 0;
    m_coalesce_us.store(0, std::memory_order_relaxed);

    struct socket_memory_snapshot {
        uint64_t allocated{};
        uint64_t limit{};
        uint64_t drops{};
    };
    bool meminfo_supported = true;
    bool have_drop_baseline = false;
    uint64_t last_kernel_drops = 0;

    auto read_socket_memory = [&]() -> std::optional<socket_memory_snapshot> {
        if (!adaptive_enabled || !meminfo_supported) { return std::nullopt; }
        std::array<uint32_t, SK_MEMINFO_VARS> values{};
        socklen_t len = sizeof(values);
        if (::getsockopt(m_socket, SOL_SOCKET, SO_MEMINFO, values.data(), &len) < 0) {
            meminfo_supported = false;
            m_logger->warn("SO_MEMINFO unavailable for recvmmsg safety feedback: {}",
                           std::string{strerror(errno)});
            return std::nullopt;
        }
        const auto count = static_cast<std::size_t>(len) / sizeof(values[0]);
        if (count <= SK_MEMINFO_RMEM_ALLOC) {
            meminfo_supported = false;
            m_logger->warn("SO_MEMINFO returned too few fields for recvmmsg safety feedback");
            return std::nullopt;
        }
        return socket_memory_snapshot{
            .allocated = values[SK_MEMINFO_RMEM_ALLOC],
            .limit = count > SK_MEMINFO_RCVBUF
                ? values[SK_MEMINFO_RCVBUF] : m_effective_recv_buf,
            .drops = count > SK_MEMINFO_DROPS ? values[SK_MEMINFO_DROPS] : last_kernel_drops
        };
    };

    auto enter_congestion = [&] {
        congestion_clean_cycles_remaining = CONGESTION_CLEAN_CYCLES;
        nominal_coalesce_us = 0.0;
        m_coalesce_us.store(0, std::memory_order_relaxed);
    };

    auto observe_socket_memory = [&](bool& cycle_congested)
        -> std::optional<socket_memory_snapshot> {
        auto snapshot = read_socket_memory();
        if (!snapshot) { return std::nullopt; }
        m_socket_rmem_bytes.store(snapshot->allocated, std::memory_order_relaxed);
        m_kernel_drops.store(snapshot->drops, std::memory_order_relaxed);
        record_kernel_drop_snapshot(static_cast<uint32_t>(snapshot->drops));

        if (have_drop_baseline && snapshot->drops != last_kernel_drops) {
            // SK_MEMINFO_DROPS is u32. Unsigned subtraction preserves the delta across its wrap
            // (assuming fewer than 2^32 drops between observations, which is unavoidable here).
            const auto delta = static_cast<uint32_t>(snapshot->drops)
                - static_cast<uint32_t>(last_kernel_drops);
            if (congestion_clean_cycles_remaining == 0) {
                m_logger->warn("UDP socket dropped {} packet(s); disabling coalescing until {} clean drains",
                               delta, CONGESTION_CLEAN_CYCLES);
            }
            cycle_congested = true;
            enter_congestion();
        }
        last_kernel_drops = snapshot->drops;
        have_drop_baseline = true;

        if (snapshot->limit > 0
            && snapshot->allocated >= snapshot->limit / COALESCE_RMEM_DIVISOR) {
            cycle_congested = true;
            enter_congestion();
        }
        return snapshot;
    };

    auto update_rate_and_interval = [&](std::size_t packets) {
        rate_window_packets += packets;
        const auto now = clock::now();
        const auto elapsed = now - rate_window_start;
        if (elapsed < m_adaptation_interval) { return; }

        const auto elapsed_seconds = std::chrono::duration<double>(elapsed).count();
        // Backpressure stretches wall-clock receive intervals and would make this estimate falsely
        // low. Do not teach the feed-forward controller while a safety latch is active.
        if (congestion_clean_cycles_remaining == 0) {
            const auto measured_pps = static_cast<double>(rate_window_packets) / elapsed_seconds;
            estimated_pps = estimated_pps == 0.0
                ? measured_pps
                : RATE_EWMA_ALPHA * measured_pps + (1.0 - RATE_EWMA_ALPHA) * estimated_pps;

            double desired_us = static_cast<double>(m_min_coalesce.count());
            if (estimated_pps > 0.0 && m_coalesce_target_batch > 1) {
                desired_us = (static_cast<double>(m_coalesce_target_batch - 1) * 1'000'000.0)
                    / estimated_pps;
            }
            desired_us = std::clamp(desired_us,
                static_cast<double>(m_min_coalesce.count()),
                static_cast<double>(m_max_coalesce.count()));
            // Keep a floating-point accumulator so sub-microsecond convergence is not lost to
            // integral duration truncation.
            nominal_coalesce_us += RATE_EWMA_ALPHA * (desired_us - nominal_coalesce_us);
        }
        m_estimated_pps.store(static_cast<uint64_t>(estimated_pps), std::memory_order_relaxed);
        m_recv_syscalls.store(recv_syscalls, std::memory_order_relaxed);
        rate_window_packets = 0;
        rate_window_start = now;
    };

    while (!token.stop_requested()) {
        // WAIT: sleep without periodic wakeups until either data or an explicit stop arrives.
        struct pollfd wait_fds[2]{
            {.fd = m_socket, .events = POLLIN, .revents = 0},
            {.fd = m_stop_fd, .events = POLLIN, .revents = 0}
        };
        const auto wait_started = clock::now();
        int poll_result{};
        do {
            poll_result = ::poll(wait_fds, 2, -1);
        } while (poll_result < 0 && errno == EINTR && !token.stop_requested());

        if (token.stop_requested() || (wait_fds[1].revents & POLLIN)) { return; }
        if (poll_result < 0) {
            m_logger->error("recvmmsg wait failed: {} (errno={})", std::string{strerror(errno)}, errno);
            return;
        }
        if (wait_fds[0].revents & POLLNVAL) {
            m_logger->error("recvmmsg socket became invalid while waiting");
            return;
        }
        if (!(wait_fds[0].revents & (POLLIN | POLLERR))) { continue; }

        if (clock::now() - wait_started >= IDLE_RATE_RESET) {
            // A stopped/restarted or intermittent source should not inherit a stale rate from the
            // previous run. Start conservatively and relearn from the new stream.
            estimated_pps = 0.0;
            nominal_coalesce_us = 0.0;
            rate_window_packets = 0;
            rate_window_start = clock::now();
            m_estimated_pps.store(0, std::memory_order_relaxed);
            m_coalesce_us.store(0, std::memory_order_relaxed);
        }

        bool cycle_congested = false;
        pool_stalled = false;
        auto memory_before = observe_socket_memory(cycle_congested);

        // COALESCE: the EWMA supplies a nominal delay, but socket capacity and the congestion latch
        // have absolute priority. With no SO_MEMINFO support, retain the same conservative static
        // budget using the verified SO_RCVBUF value and one packet already queued.
        uint64_t allocated = memory_before
            ? memory_before->allocated : m_conservative_packet_charge;
        uint64_t limit = memory_before && memory_before->limit > 0
            ? memory_before->limit : m_effective_recv_buf;
        double actual_coalesce_us = 0.0;
        if (congestion_clean_cycles_remaining == 0
            && estimated_pps > 0.0
            && nominal_coalesce_us > 0.0
            && limit / COALESCE_RMEM_DIVISOR > allocated) {
            const auto safe_bytes = limit / COALESCE_RMEM_DIVISOR - allocated;
            const auto capacity_us = static_cast<double>(safe_bytes) * 1'000'000.0
                / (estimated_pps * static_cast<double>(m_conservative_packet_charge));
            actual_coalesce_us = std::min(nominal_coalesce_us, capacity_us);
            if (actual_coalesce_us < static_cast<double>(m_min_coalesce.count())) {
                actual_coalesce_us = 0.0; // never violate the capacity ceiling to satisfy a minimum
            }
            actual_coalesce_us = std::min(
                actual_coalesce_us, static_cast<double>(m_max_coalesce.count()));
        }
        const auto current_coalesce = std::chrono::microseconds(
            static_cast<int64_t>(std::max(0.0, actual_coalesce_us)));
        m_coalesce_us.store(current_coalesce.count(), std::memory_order_relaxed);

        if (current_coalesce > std::chrono::microseconds::zero()) {
            struct pollfd stop_pfd{.fd = m_stop_fd, .events = POLLIN, .revents = 0};
            auto timeout = std::chrono::duration_cast<std::chrono::nanoseconds>(current_coalesce);
            struct timespec ts{
                .tv_sec = static_cast<time_t>(timeout.count() / 1'000'000'000),
                .tv_nsec = static_cast<long>(timeout.count() % 1'000'000'000)
            };
            int coalesce_result{};
            do {
                coalesce_result = ::ppoll(&stop_pfd, 1, &ts, nullptr);
            } while (coalesce_result < 0 && errno == EINTR && !token.stop_requested());
            if (token.stop_requested() || (stop_pfd.revents & POLLIN)) { return; }
            if (coalesce_result < 0) {
                m_logger->error("recvmmsg coalesce wait failed: {} (errno={})",
                                std::string{strerror(errno)}, errno);
                return;
            }
        }

        // DRAIN: recvmmsg is always nonblocking. This makes a stale poll readiness indication
        // harmless and keeps draining full-rate streams without returning through WAIT/COALESCE.
        while (!token.stop_requested()) {
            ++recv_syscalls;
            auto recvd = ::recvmmsg(m_socket, msgs.data(), m_batch_size, MSG_DONTWAIT, nullptr);
            if (recvd < 0) {
                if (errno == EINTR) { continue; }
                if (errno == EAGAIN || errno == EWOULDBLOCK) { break; }
                const auto recv_errno = errno;
                if (recv_errno == EBADF || recv_errno == EINVAL) {
                    m_logger->error("recvmmsg failed fatally: {} (errno={})",
                                    std::string{strerror(recv_errno)}, recv_errno);
                    return;
                }
                const auto now = clock::now();
                if (!last_receive_error_log
                    || now - *last_receive_error_log >= std::chrono::seconds(1)) {
                    if (suppressed_receive_errors == 0) {
                        m_logger->error("recvmmsg failed: {} (errno={})",
                                        std::string{strerror(recv_errno)}, recv_errno);
                    } else {
                        m_logger->error("recvmmsg failed: {} (errno={}); suppressed {} repeated error(s)",
                                        std::string{strerror(recv_errno)}, recv_errno,
                                        suppressed_receive_errors);
                    }
                    last_receive_error_log = now;
                    suppressed_receive_errors = 0;
                } else {
                    ++suppressed_receive_errors;
                }
                break;
            }
            if (recvd == 0) { break; }

            const auto msgs_recvd = static_cast<std::size_t>(recvd);
            // Backlog signals are evaluated before rate sampling so stalls/full vectors cannot
            // train the EWMA toward a longer delay.
            if (msgs_recvd == m_batch_size) {
                cycle_congested = true;
                enter_congestion();
            }
            update_rate_and_interval(msgs_recvd);

            m_pkts_recvd.fetch_add(msgs_recvd, std::memory_order_relaxed);
            m_metrics.packets_received.add(msgs_recvd);
            m_metrics.batch_sizes.record(static_cast<double>(msgs_recvd));

            uint64_t batch_bytes = 0;
            output_buffers.clear();
            for (std::size_t i = 0; i < msgs_recvd; ++i) {
                if (!buffers[i].has_value()) {
                    m_metrics.packets_dropped.inc();
                    m_logger->critical("recvmmsg buffer invariant violated for received slot {}", i);
                    return;
                }
                const auto len = msgs[i].msg_len;
                batch_bytes += len;
                output_buffers.emplace_back(
                    composite::immutable_buffer<uint8_t>(std::move(buffers[i].value())).slice(0, len));
            }

            // One immutable consumer gets one downstream ring publication for the entire receive
            // batch. Fan-out/mutable connections retain output_port's safe per-buffer fallback.
            m_out_port->send_batch(output_buffers, {});
            output_buffers.clear(); // release our shares before waiting for pool replacements
            m_metrics.bytes_received.add(batch_bytes);

            for (std::size_t i = 0; i < msgs_recvd; ++i) {
                buffers[i].reset();
                if (!acquire_buffer(i)) { return; }
            }
            if (pool_stalled) {
                cycle_congested = true;
                enter_congestion();
            }
        }

        // The next cycle's pre-coalesce observation catches drops that occurred during this drain.
        // Keeping that sample (rather than reusing a stale post-drain value across WAIT) preserves
        // the occupancy guard while avoiding a second SO_MEMINFO syscall per cycle.
        if (!cycle_congested && congestion_clean_cycles_remaining > 0) {
            --congestion_clean_cycles_remaining;
        }
        // This is a single-writer statistic. Publish once per complete drain rather than paying
        // for an atomic RMW on every syscall in the hot path.
        m_recv_syscalls.store(recv_syscalls, std::memory_order_relaxed);
    }
}

} // namespace udp
