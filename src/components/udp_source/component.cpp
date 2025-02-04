/*
 * Copyright (C) 2024 Geon Technologies, LLC
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

#include "component.hpp"

#include <arpa/inet.h>
#include <array>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <netinet/in.h>
#include <net/if.h>
#include <source_location>
#include <spdlog/spdlog.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

namespace udpsrc::net {

auto get_interface_ip(int fd, std::string_view interface) -> std::string {
    struct ifreq ifr{};
    ifr.ifr_addr.sa_family = AF_INET;
    strncpy(ifr.ifr_name, interface.data(), interface.size());
    if (auto res = ioctl(fd, SIOCGIFADDR, &ifr); res != -1) {
        return std::string{inet_ntoa(((struct sockaddr_in*)&ifr.ifr_addr)->sin_addr)};
    }
    return {};
}

} // namespace udpsrc::net


udp_source::udp_source() : composite::component("udp_source") {
    add_port(m_out_port.get());
    using enum composite::properties::config_type;
    add_property("interface", &m_interface).configurability(RUNTIME).change_listener([this]() {
        m_new_socket_required = true;
        return true;
    });
    add_property("ip_addr", &m_ip_addr).configurability(RUNTIME).change_listener([this]() {
        m_new_socket_required = true;
        return true;
    });
    add_property("port", &m_port).configurability(RUNTIME).change_listener([this]() {
        m_new_socket_required = true;
        return true;
    });
    add_property("recv_buf_size", &m_recv_buf_size).units("bytes");
    add_property("msg_size", &m_msg_size).units("bytes").configurability(RUNTIME).change_listener([this]() {
        m_flush_queue = true;
        return true;
    });
    add_property("num_msgs", &m_num_msgs).units("per recvmmsg call").configurability(RUNTIME).change_listener([this]() {
        m_flush_queue = true;
        return true;
    });
}

udp_source::~udp_source() {
    close(m_socket);
}

auto udp_source::property_change_handler() -> void {
    logger()->trace(std::source_location::current().function_name());
    if (m_new_socket_required) {
        logger()->trace("property changes indicate new socket is required; closing socket");
        // Close socket
        close(m_socket);
        m_socket = -1;
        // Open socket
        logger()->trace("opening socket");
        m_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        // Setup poll
        m_pfds.at(0).fd = m_socket;
        m_pfds.at(0).events = POLLIN;
        // Set non-blocking
        logger()->trace("setting socket to non-blocking");
        fcntl(m_socket, F_SETFL, O_NONBLOCK);
        // Determine multicast from address
        auto multi_addr_start = htonl(inet_addr("224.0.0.0"));
        auto multi_addr_end = htonl(inet_addr("239.255.255.255"));
        auto bind_addr = htonl(inet_addr(m_ip_addr.c_str()));
        auto is_multicast = (bind_addr >= multi_addr_start) && (bind_addr <= multi_addr_end);
        if (is_multicast) {
            bind_addr = INADDR_ANY;
        }
        // Bind the socket
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = ntohl(bind_addr);
        addr.sin_port = htons(m_port);
        if (bind_addr == INADDR_ANY) {
            logger()->trace("binding socket to 0.0.0.0:{}", m_port);
        } else {
            logger()->trace("binding socket to {}:{}", m_ip_addr, m_port);
        }
        bind(m_socket, (struct  sockaddr*)&addr, sizeof(addr));
        if (is_multicast) {
            // Multicast group
            struct ip_mreq group{};
            auto bind_address = udpsrc::net::get_interface_ip(m_socket, m_interface);
            group.imr_interface.s_addr = inet_addr(bind_address.c_str());
            group.imr_multiaddr.s_addr = inet_addr(m_ip_addr.c_str());
            logger()->trace("subscribing to multicast group {} on interface {} ({})", m_ip_addr, m_interface, bind_address);
            setsockopt(m_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&group, sizeof(group));
        }
    }
    // Check if properties related to queue were adjusted
    if (m_flush_queue) {
        logger()->trace("property changes indicate queue flush is required; rebuilding buffer pool");
        m_pool.reset();
        // Setup buffer pool
        m_pool = std::make_unique<udpsrc::buffer_pool>(m_num_msgs * 32, m_msg_size);
        m_msgs.resize(m_num_msgs);
        m_iovecs.resize(m_num_msgs);
        m_buffers.resize(m_num_msgs);
    }
    // Set receive buffer size
    if (m_recv_buf_size > 0) {
        logger()->trace("setting socket receive buffer size to {}", m_recv_buf_size);
        setsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, (char*)&m_recv_buf_size, sizeof(m_recv_buf_size));
    }
}

auto udp_source::process() -> composite::retval {
    using enum composite::retval;

    // Populate iovecs with buffers from pool
    for (auto i=size_t{}; i < m_msgs.size(); ++i) {
        m_buffers.at(i) = m_pool->acquire();
        m_iovecs.at(i).iov_base = m_buffers.at(i)->data();
        m_iovecs.at(i).iov_len = m_buffers.at(i)->size();
        m_msgs.at(i).msg_hdr.msg_iov = &m_iovecs.at(i);
        m_msgs.at(i).msg_hdr.msg_iovlen = 1;
    }

    // Replenish if needed in async
    auto fut = std::async(std::launch::async, [this]{ m_pool->replenish(); });

    // Receive messages
    using timespec_t = struct timespec;
    auto timeout = timespec_t{.tv_sec = 1, .tv_nsec = 0};
    if (auto num_events = poll(m_pfds.data(), 1, 100/*ms*/)) [[likely]] {
        // check socket is ready to read
        if (m_pfds.at(0).revents & POLLIN) [[likely]] {
            if (auto recvd = recvmmsg(m_socket, m_msgs.data(), m_msgs.size(), 0, &timeout); recvd > 0) {
                for (auto i=0; i < recvd; ++i) {
                    m_out_port->send_data(std::move(m_buffers.at(i)), {});
                }
            }
        }
    }

    return NO_YIELD;
}

extern "C" {
    auto create() -> std::shared_ptr<composite::component> {
        return std::make_shared<udp_source>();
    }
}
