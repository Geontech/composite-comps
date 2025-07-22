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

#include "helpers.hpp"
#include "recvmmsg.hpp"
#include "pmr/ring_resource.hpp"

#include <arpa/inet.h>
#include <array>
#include <cerrno>
#include <cstring>
#include <format>
#include <netinet/in.h>
#include <net/if.h>
#include <poll.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

namespace udp {

recvmmsg::recvmmsg(const config& config) :
  interface(config.logger),
  m_frame_count(config.frame_count) {
    // Create socket
    m_logger->trace("opening udp socket");
    m_socket = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_socket < 0) {
        throw std::runtime_error(std::format("failed to create socket: {}", std::string{strerror(errno)}));
    }

    // Set receive buffer size
    if (config.recv_buf_size > 0) {
        m_logger->trace("setting socket receive buffer size to {}", config.recv_buf_size);
        if (::setsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, (char*)&config.recv_buf_size, sizeof(config.recv_buf_size)) < 0) {
            ::close(m_socket);
            throw std::runtime_error(std::format("failed to set receive buffer size: {}", std::string{strerror(errno)}));
        }
    }

    // Set batch size
    if (config.batch_size > 0) {
        m_batch_size = config.batch_size;
    }
    m_logger->trace("using recvmmsg batch size of {}", m_batch_size);

    // Bind the socket
    auto bind_addr = htonl(inet_addr(config.ip_addr.data()));
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
    addr.sin_addr.s_addr = ntohl(bind_addr);
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

    // User has overriden properties
    if (config.msg_size > 0) {
        m_frame_size = std::bit_ceil(config.msg_size);
        m_resource = std::make_unique<ring_resource>(ring_resource::ring_config{.frame_size=m_frame_size, .frame_count=m_frame_count, .alignment=64});
        m_logger->trace("using user-provided msg_size of: {} bytes", config.msg_size);
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
    // User has overriden properties
    if (m_frame_size == 0) {
        // Discover the size of the incoming packets from the wire
        std::array<uint8_t, 9000> buffer{}; // Use a jumbo frame buffer
        while (true) {
            if (auto recvd = ::recvfrom(m_socket, buffer.data(), buffer.size(), 0, nullptr, nullptr); recvd > 0) {
                // Assume anything over 1000 bytes is a valid data packet (either SDDS or V49)
                if (recvd < 512) {
                    continue;
                }
                if (buffer[0] & 0x40 || buffer[0] & 0x50) { // ignore v49 context signatures
                    continue;
                }
                m_logger->trace("using discovered msg_size of: {} bytes", recvd);
                m_frame_size = std::bit_ceil(static_cast<std::size_t>(recvd));
                m_resource = std::make_unique<ring_resource>(ring_resource::ring_config{.frame_size=m_frame_size, .frame_count=m_frame_count, .alignment=64});
                break;
            }
        }
    }
    m_out_port = port;
    m_recv_thread = std::jthread(&recvmmsg::receive, this);
    pthread_setname_np(m_recv_thread.native_handle(), "recvmmsg");
}

auto recvmmsg::stop_recv() -> void {
    m_recv_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
}

auto recvmmsg::get_stats() -> statistics {
    auto stats = statistics{};
    stats.pkts_recvd = m_pkts_recvd.exchange(0);
    return stats;
}

auto recvmmsg::receive(std::stop_token token) -> void {
    struct pollfd pfd{
        .fd = m_socket,
        .events = POLLIN,
        .revents = 0
    };

    // Allocator for pmr vectors
    auto allocator = std::pmr::polymorphic_allocator<std::uint8_t>(m_resource.get());

    // Buffers
    buffer_ptr_t buffers[m_batch_size];
    struct mmsghdr msgs[m_batch_size]{};
    struct iovec iovecs[m_batch_size]{};

    // Initialize buffers and iovecs
    for (std::size_t i = 0; i < m_batch_size; ++i) {
        buffers[i] = std::make_shared<buffer_t>(allocator);
        buffers[i]->resize(m_frame_size);
        iovecs[i].iov_base = buffers[i]->data();
        iovecs[i].iov_len = buffers[i]->size();
        msgs[i].msg_hdr.msg_iov = &iovecs[i];
        msgs[i].msg_hdr.msg_iovlen = 1;
    }

    auto msgs_recvd = std::size_t{};
    while (!token.stop_requested()) {
        if (auto poll_res = ::poll(&pfd, 1, 1/*ms*/); poll_res <= 0) {
            continue;
        }

        if (pfd.revents & POLLIN) [[likely]] {
            // Call recvmmsg
            struct timespec ts{.tv_sec=0, .tv_nsec=100'000}; // 100 us
            if (auto recvd = ::recvmmsg(m_socket, &msgs[msgs_recvd], m_batch_size - msgs_recvd, 0, &ts); recvd > 0) {
                msgs_recvd += recvd;
                m_pkts_recvd.fetch_add(recvd, std::memory_order_relaxed);
            }
            if (msgs_recvd < m_batch_size) {
                continue;
            }

            // Place onto queue
            for (auto i=0u; i<msgs_recvd; ++i) {
                buffers[i]->resize(msgs[i].msg_len); // trim to actual recvd size
                m_out_port->send_data(std::move(buffers[i]), {});

                // Reallocate for next loop
                buffers[i] = std::make_shared<buffer_t>(allocator);
                buffers[i]->resize(m_frame_size);
                iovecs[i].iov_base = buffers[i]->data();
            }
            msgs_recvd = 0;
        }
    }

}

} // namespace udp
