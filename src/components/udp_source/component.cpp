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
#include <netinet/in.h>
#include <net/if.h>
#include <mutex>
#include <poll.h>
#include <queue>
#include <ranges>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace udpsrc::net {

mmsgs::mmsgs(size_t num_msgs, size_t msg_size) : msgs(num_msgs), iovecs(num_msgs) {
    for (auto i=0u; i<msgs.size(); ++i) {
        buffers->emplace_back(msg_size, 0xFF);
        iovecs.at(i).iov_base = buffers->at(i).data();
        iovecs.at(i).iov_len = buffers->at(i).size();
        msgs.at(i).msg_hdr.msg_iov = &iovecs.at(i);
        msgs.at(i).msg_hdr.msg_iovlen = 1;
    }
}

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


udp_source::udp_source() :
  composite::component("udp_source"),
  m_socket(socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) {
    add_port(m_out_port.get());
    add_property("interface", &m_interface);
    add_property("ip_addr", &m_ip_addr);
    add_property("port", &m_port);
    add_property("recv_buf_size", &m_recv_buf_size);
    add_property("msg_size", &m_msg_size);
    add_property("num_msgs", &m_num_msgs);
}

udp_source::~udp_source() {
    close(m_socket);
}

auto udp_source::initialize() -> void {
    // Setup poll
    m_pfds.at(0).fd = m_socket;
    m_pfds.at(0).events = POLLIN;
    // Set receive buffer size
    setsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, (char*)&m_recv_buf_size, sizeof(m_recv_buf_size));
    // Set receive timeout
    using timeval_t = struct timeval;
    auto tv = timeval_t{.tv_sec = 1, .tv_usec = 0};
    setsockopt(m_socket, SOL_SOCKET, SO_RCVTIMEO, (char*)&tv, sizeof(tv));
    // Set non-blocking
    fcntl(m_socket, F_SETFL, O_NONBLOCK);
    // Setup queue of mmsg headers with iovec buffers
    m_queue_size = m_num_msgs / 2;
    for (auto _ : std::views::iota(size_t{0}, m_queue_size)) {
        m_queue.emplace(std::make_unique<udpsrc::net::mmsgs>(m_num_msgs, m_msg_size));
    }
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
    bind(m_socket, (struct  sockaddr*)&addr, sizeof(addr));
    if (is_multicast) {
        // Multicast group
        struct ip_mreq group{};
        auto bind_address = udpsrc::net::get_interface_ip(m_socket, m_interface);
        group.imr_interface.s_addr = inet_addr(bind_address.c_str());
        group.imr_multiaddr.s_addr = inet_addr(m_ip_addr.c_str());
        setsockopt(m_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, (char*)&group, sizeof(group));
    }
}

auto udp_source::start() -> void {
    m_filler = std::jthread(&udp_source::keep_full, this);
    composite::component::start();
}

auto udp_source::stop() -> void {
    m_filler.request_stop();
    if (m_filler.joinable()) {
        m_filler.join();
    }
    composite::component::stop();
}

auto udp_source::process() -> composite::retval {
    using enum composite::retval;
    auto data = std::unique_ptr<udpsrc::net::mmsgs>{nullptr};
    {
        auto lk = std::scoped_lock{m_mtx};
        if (m_queue.empty()) {
            return NO_YIELD;
        }
        data = std::move(m_queue.front());
        m_queue.pop();
        m_cv.notify_one();
    }
    if (data == nullptr) {
        return NO_YIELD;
    }
    using timespec_t = struct timespec;
    auto timeout = timespec_t{.tv_sec = 1, .tv_nsec = 0};
    if (auto num_events = poll(m_pfds.data(), 1, 1000/*1s*/)) [[likely]] {
        // check socket is ready to read
        if (m_pfds.at(0).revents & POLLIN) [[likely]] {
            if (auto recvd = recvmmsg(m_socket, data->msgs.data(), data->msgs.size(), 0, &timeout); recvd != -1) {
                data->buffers->resize(recvd);
                auto now = std::chrono::system_clock::now();
                auto secs = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch());
                auto nsecs = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
                m_out_port->send_data(std::move(data->buffers), {static_cast<uint32_t>(secs.count()), static_cast<uint64_t>((nsecs - secs).count())});
            }
        }
    }
    return NO_YIELD;
}

auto udp_source::keep_full(std::stop_token token) -> void {
    while (!token.stop_requested()) {
        auto lk = std::unique_lock{m_mtx};
        m_cv.wait_for(lk, std::chrono::seconds(1), [this]{ return m_queue.size() < m_queue_size; });
        if (m_queue.size() < m_queue_size) {
            m_queue.emplace(std::make_unique<udpsrc::net::mmsgs>(m_num_msgs, m_msg_size));
        }
    }
}

extern "C" {
    auto create() -> std::shared_ptr<composite::component> {
        return std::make_shared<udp_source>();
    }
}
