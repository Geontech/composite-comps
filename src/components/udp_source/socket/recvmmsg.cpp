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
#include "receive_batch_policy.hpp"
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
  m_receive_batch_wait(std::chrono::microseconds(config.receive_batch_wait_us)),
  m_output_batch_size(config.output_batch_size),
  m_max_batch_delay(std::chrono::microseconds(config.max_batch_delay_us)),
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
        m_logger->debug("using system-default SO_RCVBUF (effective={} bytes)",
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
    // Linux charges receive-buffer space by skb allocation, not UDP payload bytes. Keep the
    // conservative estimate as useful operational telemetry even though this implementation no
    // longer deliberately accumulates packets in the kernel socket queue.
    m_conservative_packet_charge = std::max<std::size_t>(
        4096, std::bit_ceil(m_frame_size + std::size_t{1024}));
    if (m_output_batch_size == 0) { m_output_batch_size = m_batch_size; }
    m_output_batch_size = std::clamp(
        m_output_batch_size, std::size_t{1}, std::max<std::size_t>(1, m_frame_count));
    constexpr auto MAX_RECEIVE_BATCH_WAIT = std::chrono::microseconds(5000);
    if (m_receive_batch_wait > MAX_RECEIVE_BATCH_WAIT) {
        m_logger->warn("recvmmsg receive_batch_wait_us {} exceeds the 5000 us safety limit; "
                       "clamping it", m_receive_batch_wait.count());
        m_receive_batch_wait = MAX_RECEIVE_BATCH_WAIT;
    }
    m_logger->debug("recvmmsg fixed batching: receive_wait_us={}, output_batch_size={}, "
                    "max_delay_us={}", m_receive_batch_wait.count(), m_output_batch_size,
                    m_max_batch_delay.count());
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
    // Pull socket-memory diagnostics on the low-rate stats path as well as the periodic receive
    // path, so reports retain visibility even while the source is idle.
    std::array<uint32_t, SK_MEMINFO_VARS> memory{};
    socklen_t memory_len = sizeof(memory);
    if (::getsockopt(m_socket, SOL_SOCKET, SO_MEMINFO, memory.data(), &memory_len) == 0) {
        const auto count = static_cast<std::size_t>(memory_len) / sizeof(memory[0]);
        if (count > SK_MEMINFO_RMEM_ALLOC) {
            m_socket_rmem_bytes.store(memory[SK_MEMINFO_RMEM_ALLOC], std::memory_order_relaxed);
            note_rmem_observation(memory[SK_MEMINFO_RMEM_ALLOC]);
        }
        if (count > SK_MEMINFO_DROPS) {
            m_kernel_drops.store(memory[SK_MEMINFO_DROPS], std::memory_order_relaxed);
            record_kernel_drop_snapshot(memory[SK_MEMINFO_DROPS]);
        }
    }
    auto stats = std::map<std::string, std::string>{};
    stats["pkts_recvd"] = std::to_string(m_pkts_recvd.load());
    stats["recv_syscalls"] = std::to_string(m_recv_syscalls.load());
    stats["wait_syscalls"] = std::to_string(m_wait_syscalls.load());
    stats["total_receive_syscalls"] = std::to_string(
        m_recv_syscalls.load() + m_wait_syscalls.load());
    stats["full_receive_vectors"] = std::to_string(m_full_receive_vectors.load());
    stats["receive_batch_wait_us"] = std::to_string(m_receive_batch_wait.count());
    stats["output_batch_size"] = std::to_string(m_output_batch_size);
    stats["max_batch_delay_us"] = std::to_string(m_max_batch_delay.count());
    stats["output_batches"] = std::to_string(m_output_batches.load());
    stats["partial_batch_flushes"] = std::to_string(m_partial_batch_flushes.load());
    stats["pending_output_packets"] = std::to_string(m_pending_output_packets.load());
    stats["effective_recv_buf"] = std::to_string(m_effective_recv_buf);
    stats["estimated_packet_charge"] = std::to_string(m_conservative_packet_charge);
    stats["socket_rmem_bytes"] = std::to_string(m_socket_rmem_bytes.load());
    // Interval maximum, reset on read: the instantaneous sample above can miss pressure that
    // occurred between the five-second reports.
    stats["socket_rmem_peak_bytes"] = std::to_string(
        m_socket_rmem_peak_bytes.exchange(0, std::memory_order_relaxed));
    stats["kernel_drops"] = std::to_string(m_kernel_drops.load());
    stats["congest_pool_stall"] = std::to_string(m_congest_pool_stall.load());
    stats["pool_stall_backoff_us"] = std::to_string(m_pool_stall_backoff_us.load());
    if (m_pool) {
        stats["pool_capacity"] = std::to_string(m_pool->capacity());
        stats["pool_outstanding"] = std::to_string(m_pool->outstanding());
        stats["pool_available"] = std::to_string(m_pool->available());
    }
    return stats;
}

auto recvmmsg::note_rmem_observation(uint64_t bytes) noexcept -> void {
    auto prev = m_socket_rmem_peak_bytes.load(std::memory_order_relaxed);
    while (prev < bytes && !m_socket_rmem_peak_bytes.compare_exchange_weak(
               prev, bytes, std::memory_order_relaxed, std::memory_order_relaxed)) {}
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

    using clock = receive_batch_policy::clock;
    auto policy = receive_batch_policy(receive_batch_policy::settings{
        .output_batch_size = m_output_batch_size,
        .receive_vector_size = m_batch_size,
        .receive_batch_wait = m_receive_batch_wait,
        .max_batch_delay = m_max_batch_delay,
    });

    // The receive vector and the output accumulator are deliberately independent:
    // num_msgs controls one recvmmsg call, while output_batch_size controls one ring publish.
    auto iovecs = std::vector<struct iovec>(m_batch_size);
    auto msgs = std::vector<struct mmsghdr>(m_batch_size);
    using buffer_opt_t = std::optional<composite::external_buffer<uint8_t>>;
    auto buffers = std::vector<buffer_opt_t>(m_batch_size);
    auto acquired_buffers = std::vector<composite::external_buffer<uint8_t>>{};
    acquired_buffers.reserve(m_batch_size);
    auto output_buffers = std::vector<composite::immutable_buffer<uint8_t>>{};
    output_buffers.reserve(m_output_batch_size);

    auto flush_output = [&](bool partial) {
        if (output_buffers.empty()) { return; }
        if (partial && m_max_batch_delay > std::chrono::microseconds::zero()
            && output_buffers.size() < m_output_batch_size) {
            m_partial_batch_flushes.fetch_add(1, std::memory_order_relaxed);
        }
        m_out_port->send_batch(output_buffers, {});
        output_buffers.clear();
        policy.note_batch_flushed();
        m_pending_output_packets.store(0, std::memory_order_relaxed);
        m_output_batches.fetch_add(1, std::memory_order_relaxed);
    };

    auto prepare_slot = [&](std::size_t idx) {
        msgs[idx] = {};
        iovecs[idx].iov_base = buffers[idx]->data();
        iovecs[idx].iov_len = buffers[idx]->size();
        msgs[idx].msg_hdr.msg_iov = &iovecs[idx];
        msgs[idx].msg_hdr.msg_iovlen = 1;
    };

    auto acquire_buffers = [&](std::size_t first, std::size_t count) -> bool {
        constexpr auto ACQUIRE_BACKOFF = std::chrono::microseconds(50);
        std::optional<clock::time_point> stall_start;
        std::size_t filled = 0;
        while (filled < count && !token.stop_requested()) {
            acquired_buffers.clear();
            const auto acquired = m_pool->acquire_batch(count - filled, acquired_buffers);
            for (std::size_t i = 0; i < acquired; ++i) {
                const auto idx = first + filled + i;
                buffers[idx].emplace(std::move(acquired_buffers[i]));
                prepare_slot(idx);
            }
            filled += acquired;
            if (filled == count) { break; }
            if (!stall_start) {
                m_congest_pool_stall.fetch_add(1, std::memory_order_relaxed);
                stall_start = clock::now();
                // A partial userspace batch may itself hold the only buffers that can refill
                // the receive vector (notably when frame_count == num_msgs). Publish it before
                // waiting so output aggregation can never deadlock its own pool refill.
                if (policy.flush_for(output_buffers.size(), clock::now(),
                                     receive_batch_policy::event::pool_stall)
                    == receive_batch_policy::flush_reason::pool_stall) {
                    flush_output(true);
                }
            }
            std::this_thread::sleep_for(ACQUIRE_BACKOFF);
        }
        if (stall_start) {
            m_pool_stall_backoff_us.fetch_add(static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    clock::now() - *stall_start).count()),
                std::memory_order_relaxed);
        }
        return filled == count;
    };

    if (!acquire_buffers(0, m_batch_size)) { return; }
    uint64_t recv_syscalls = 0;
    uint64_t wait_syscalls = 0;
    std::optional<clock::time_point> last_receive_error_log;
    uint64_t suppressed_receive_errors = 0;

    enum class delay_result { expired, stopped, failed };
    auto interruptible_delay = [&](std::chrono::microseconds delay) -> delay_result {
        const auto deadline = clock::now() + delay;
        while (!token.stop_requested()) {
            const auto now = clock::now();
            if (now >= deadline) { return delay_result::expired; }
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now);
            struct timespec timeout{
                .tv_sec = static_cast<time_t>(ns.count() / 1'000'000'000),
                .tv_nsec = static_cast<long>(ns.count() % 1'000'000'000)
            };
            struct pollfd stop_fd{.fd = m_stop_fd, .events = POLLIN, .revents = 0};
            ++wait_syscalls;
            m_wait_syscalls.store(wait_syscalls, std::memory_order_relaxed);
            const auto result = ::ppoll(&stop_fd, 1, &timeout, nullptr);
            if (result == 0) { return delay_result::expired; }
            if (result > 0 && (stop_fd.revents & POLLIN)) { return delay_result::stopped; }
            if (result < 0 && errno == EINTR) { continue; }
            if (result < 0) {
                m_logger->error("recvmmsg batch wait failed: {} (errno={})",
                                std::string{strerror(errno)}, errno);
                return delay_result::failed;
            }
        }
        return delay_result::stopped;
    };

    struct socket_memory_snapshot {
        uint64_t allocated{};
        uint64_t drops{};
    };
    bool meminfo_supported = true;
    bool have_drop_baseline = false;
    uint64_t last_kernel_drops = 0;
    auto last_safety_observe = clock::now();

    auto read_socket_memory = [&]() -> std::optional<socket_memory_snapshot> {
        if (!meminfo_supported) { return std::nullopt; }
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
            .drops = count > SK_MEMINFO_DROPS ? values[SK_MEMINFO_DROPS] : last_kernel_drops
        };
    };

    auto observe_socket_memory = [&]() -> std::optional<socket_memory_snapshot> {
        last_safety_observe = clock::now();
        auto snapshot = read_socket_memory();
        if (!snapshot) { return std::nullopt; }
        m_socket_rmem_bytes.store(snapshot->allocated, std::memory_order_relaxed);
        note_rmem_observation(snapshot->allocated);
        m_kernel_drops.store(snapshot->drops, std::memory_order_relaxed);
        record_kernel_drop_snapshot(static_cast<uint32_t>(snapshot->drops));

        if (have_drop_baseline && snapshot->drops != last_kernel_drops) {
            // SK_MEMINFO_DROPS is u32. Unsigned subtraction preserves the delta across its wrap
            // (assuming fewer than 2^32 drops between observations, which is unavoidable here).
            const auto delta = static_cast<uint32_t>(snapshot->drops)
                - static_cast<uint32_t>(last_kernel_drops);
            m_logger->warn("UDP socket dropped {} packet(s)", delta);
        }
        last_kernel_drops = snapshot->drops;
        have_drop_baseline = true;
        return snapshot;
    };

    while (!token.stop_requested()) {
        const auto now = clock::now();
        if (policy.flush_for(output_buffers.size(), now)
            == receive_batch_policy::flush_reason::deadline) {
            flush_output(true);
            continue;
        }

        auto socket_readable_at = now;
        if (!policy.backlog_likely()) {
            // Idle/normal path: wait for data, the oldest output packet's deadline, or stop.
            struct pollfd wait_fds[2]{
                {.fd = m_socket, .events = POLLIN, .revents = 0},
                {.fd = m_stop_fd, .events = POLLIN, .revents = 0}
            };
            int poll_result{};
            do {
                struct timespec timeout{};
                struct timespec* timeout_ptr = nullptr;
                if (const auto remaining = policy.deadline_remaining(clock::now())) {
                    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(*remaining);
                    timeout.tv_sec = static_cast<time_t>(ns.count() / 1'000'000'000);
                    timeout.tv_nsec = static_cast<long>(ns.count() % 1'000'000'000);
                    timeout_ptr = &timeout;
                }
                ++wait_syscalls;
                m_wait_syscalls.store(wait_syscalls, std::memory_order_relaxed);
                poll_result = ::ppoll(wait_fds, 2, timeout_ptr, nullptr);
            } while (poll_result < 0 && errno == EINTR && !token.stop_requested());

            if (token.stop_requested() || (wait_fds[1].revents & POLLIN)) {
                flush_output(true);
                return;
            }
            if (poll_result < 0) {
                m_logger->error("recvmmsg wait failed: {} (errno={})",
                                std::string{strerror(errno)}, errno);
                flush_output(true);
                return;
            }
            if (poll_result == 0) {
                flush_output(true);
                continue;
            }
            if (wait_fds[0].revents & POLLNVAL) {
                m_logger->error("recvmmsg socket became invalid while waiting");
                flush_output(true);
                return;
            }
            if (!(wait_fds[0].revents & (POLLIN | POLLERR))) { continue; }

            socket_readable_at = clock::now();
            const auto batch_wait = policy.receive_wait(socket_readable_at);
            if (batch_wait > std::chrono::microseconds::zero()) {
                const auto delay = interruptible_delay(batch_wait);
                if (delay != delay_result::expired) {
                    flush_output(true);
                    return;
                }
            }
            // The fixed receive window consumes the same total latency budget as userspace
            // accumulation. Flush an older partial output batch before receiving more.
            if (policy.flush_for(output_buffers.size(), clock::now())
                == receive_batch_policy::flush_reason::deadline) {
                flush_output(true);
            }
        }

        ++recv_syscalls;
        m_recv_syscalls.store(recv_syscalls, std::memory_order_relaxed);
        auto recvd = ::recvmmsg(m_socket, msgs.data(), m_batch_size, MSG_DONTWAIT, nullptr);
        if (recvd < 0) {
            if (errno == EINTR) { continue; }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                policy.note_receive_result(0);
                continue;
            }
            policy.note_receive_result(0);
            const auto recv_errno = errno;
            if (recv_errno == EBADF || recv_errno == EINVAL) {
                m_logger->error("recvmmsg failed fatally: {} (errno={})",
                                std::string{strerror(recv_errno)}, recv_errno);
                flush_output(true);
                return;
            }
            const auto error_now = clock::now();
            if (!last_receive_error_log
                || error_now - *last_receive_error_log >= std::chrono::seconds(1)) {
                if (suppressed_receive_errors == 0) {
                    m_logger->error("recvmmsg failed: {} (errno={})",
                                    std::string{strerror(recv_errno)}, recv_errno);
                } else {
                    m_logger->error("recvmmsg failed: {} (errno={}); suppressed {} repeated error(s)",
                                    std::string{strerror(recv_errno)}, recv_errno,
                                    suppressed_receive_errors);
                }
                last_receive_error_log = error_now;
                suppressed_receive_errors = 0;
            } else {
                ++suppressed_receive_errors;
            }
            continue;
        }
        if (recvd == 0) {
            policy.note_receive_result(0);
            continue;
        }

        const auto msgs_recvd = static_cast<std::size_t>(recvd);
        policy.note_receive_result(msgs_recvd);
        if (msgs_recvd == m_batch_size) {
            m_full_receive_vectors.fetch_add(1, std::memory_order_relaxed);
        }
        const auto receive_now = clock::now();
        m_pkts_recvd.fetch_add(msgs_recvd, std::memory_order_relaxed);
        m_metrics.packets_received.add(msgs_recvd);
        m_metrics.batch_sizes.record(static_cast<double>(msgs_recvd));

        uint64_t batch_bytes = 0;
        for (std::size_t i = 0; i < msgs_recvd; ++i) {
            if (!buffers[i].has_value()) {
                m_metrics.packets_dropped.inc();
                m_logger->critical("recvmmsg buffer invariant violated for received slot {}", i);
                flush_output(true);
                return;
            }
            const auto len = msgs[i].msg_len;
            batch_bytes += len;
            // After a size flush, a remainder from this same receive inherits the call's original
            // readability time. That may flush it slightly early, but can never violate the
            // configured latency budget by stamping it later than it was actually available.
            if (output_buffers.empty()) { policy.note_batch_started(socket_readable_at); }
            output_buffers.emplace_back(composite::immutable_buffer<uint8_t>(
                std::move(buffers[i].value()).take(len)));
            if (policy.flush_for(output_buffers.size(), clock::now())
                == receive_batch_policy::flush_reason::size) {
                flush_output(false);
            }
        }
        m_pending_output_packets.store(output_buffers.size(), std::memory_order_relaxed);
        m_metrics.bytes_received.add(batch_bytes);

        if (policy.flush_for(output_buffers.size(), clock::now())
            == receive_batch_policy::flush_reason::deadline) {
            flush_output(true);
        }

        for (std::size_t i = 0; i < msgs_recvd; ++i) {
            buffers[i].reset();
        }
        if (!acquire_buffers(0, msgs_recvd)) {
            flush_output(true);
            return;
        }
        // Socket diagnostics are observational only. They do not alter batching policy.
        if (receive_now - last_safety_observe >= std::chrono::milliseconds(250)) {
            observe_socket_memory();
        }
    }

    flush_output(true);
}

} // namespace udp
