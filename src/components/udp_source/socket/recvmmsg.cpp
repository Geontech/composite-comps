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
#include <array>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <format>
#include <limits>
#include <netinet/in.h>
#include <net/if.h>
#include <poll.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace udp {

recvmmsg::recvmmsg(const config& config) :
  interface(config.logger),
  m_frame_count(config.frame_count),
  m_autodiscovery_timeout(config.autodiscovery_timeout) {
    // Create socket
    m_logger->trace("opening udp socket");
    m_socket = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_socket < 0) {
        throw std::runtime_error(std::format("failed to create socket: {}", std::string{strerror(errno)}));
    }

    // Set receive buffer size
    if (config.recv_buf_size > 0) {
        m_logger->trace("setting socket receive buffer size to {}", config.recv_buf_size);
        net::set_socket_recv_buffer(m_socket, config.recv_buf_size);
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
}

recvmmsg::~recvmmsg() {
    m_recv_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
    if (m_socket != -1) {
        ::close(m_socket);
    }
}

auto recvmmsg::start_recv(output_port_t* port) -> void {
    // User has not overriden properties (or requesting auto-discovery)
    if (m_frame_size == 0) {
        // Minimum data packet size in an attempt to exclude context packets
        static constexpr std::size_t MIN_DATA_PACKET_SIZE = 512;

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
            if (auto recvd = ::recvfrom(m_socket, buffer.data(), buffer.size(), 0, nullptr, nullptr); recvd > MIN_DATA_PACKET_SIZE) {
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
    m_out_port = port;
    m_recv_thread = std::jthread(&recvmmsg::receive, this);
    if (auto ret = pthread_setname_np(m_recv_thread.native_handle(), "recvmmsg"); ret != 0) {
        m_logger->warn("failed to set thread name: {}", std::string{strerror(ret)});
    }
}

auto recvmmsg::stop_recv() -> void {
    m_recv_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
}

auto recvmmsg::get_stats() -> std::map<std::string, std::string> {
    auto stats = std::map<std::string, std::string>{};
    stats["pkts_recvd"] = std::to_string(m_pkts_recvd.load());
    return stats;
}

auto recvmmsg::receive(std::stop_token token) -> void {
    if (!m_pool) {
        m_logger->error("recvmmsg pool not initialized");
        return;
    }

    struct pollfd pfd{
        .fd = m_socket,
        .events = POLLIN,
        .revents = 0
    };

    // Pre-allocate structures for batch operations
    auto iovecs = std::vector<struct iovec>(m_batch_size);
    auto msgs = std::vector<struct mmsghdr>(m_batch_size);
    using buffer_opt_t = std::optional<composite::external_buffer<uint8_t>>;
    auto buffers = std::vector<buffer_opt_t>(m_batch_size);

    // Acquire buffers from pool and set up iovecs
    auto acquire_buffer = [&](std::size_t idx) -> bool {
        constexpr auto ACQUIRE_BACKOFF = std::chrono::microseconds(50);
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
            std::this_thread::sleep_for(ACQUIRE_BACKOFF);
        }
        return false;
    };

    // Initial batch acquisition
    for (std::size_t i = 0; i < m_batch_size; ++i) {
        if (!acquire_buffer(i)) { return; }
    }

    while (!token.stop_requested()) {
        if (auto poll_res = ::poll(&pfd, 1, 1); poll_res <= 0) {
            continue;
        }

        if (pfd.revents & POLLIN) [[likely]] {
            struct timespec ts{.tv_sec=0, .tv_nsec=100'000};
            auto recvd = ::recvmmsg(m_socket, msgs.data(), m_batch_size, 0, &ts);
            if (recvd < 0) {
                if (errno == EINTR) { continue; }
                m_logger->error("recvmmsg failed: {} (errno={})", std::string{strerror(errno)}, errno);
                if (errno == EBADF || errno == EINVAL) { return; }
                continue;
            }
            if (recvd == 0) {
                continue;
            }

            auto msgs_recvd = static_cast<std::size_t>(recvd);
            m_pkts_recvd.fetch_add(msgs_recvd, std::memory_order_relaxed);

            // Process received messages
            for (std::size_t i = 0; i < msgs_recvd; ++i) {
                if (!buffers[i].has_value()) {
                    continue;
                }

                // Create length-adjusted view using take()
                auto len = msgs[i].msg_len;
                auto sized_buffer = composite::immutable_buffer<uint8_t>(
                    std::make_shared<composite::external_buffer<uint8_t>>(
                        std::move(buffers[i].value())
                    )
                ).slice(0, len);

                // Send data downstream
                m_out_port->send_data(sized_buffer, {});

                // Acquire replacement buffer for next batch
                buffers[i].reset();
                if (!acquire_buffer(i)) { return; }
            }
        }
    }
}

} // namespace udp
