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
#include "overlay.hpp"

#include <arpa/inet.h>
#include <array>
#include <complex>
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
    add_property("transport", &m_transport).configurability(RUNTIME).change_listener([this]() {
        m_flush_queue = true;
        return (m_transport == "sdds") || (m_transport == "vita49");
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
    add_property("buffer_pool_size", &m_pool_size).units("buffer pool depth").configurability(RUNTIME).change_listener([this]() {
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
        // Setup epoll
        m_epoll_fd = epoll_create1(0);
        m_event.data.fd = m_socket;
        m_event.events = EPOLLIN;
        epoll_ctl(m_epoll_fd, EPOLL_CTL_ADD, m_socket, &m_event);
        // Set non-blocking
        // logger()->trace("setting socket to non-blocking");
        // fcntl(m_socket, F_SETFL, O_NONBLOCK);
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
        logger()->trace("property changes indicate queue flush is required; rebuilding buffer pool and processing queue");
        m_pool.reset();
        // Setup buffer pool
        m_pool = std::make_unique<udpsrc::buffer_pool>(m_pool_size, m_num_msgs, m_msg_size);
        // Empty processing queue
        m_queue.clear();
    }
    // Set receive buffer size
    if (m_recv_buf_size > 0) {
        logger()->trace("setting socket receive buffer size to {}", m_recv_buf_size);
        setsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, (char*)&m_recv_buf_size, sizeof(m_recv_buf_size));
        int optval;
        socklen_t optlen = sizeof(optval);
        if (getsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, &optval, &optlen) != -1) {
            logger()->trace("socket receive buffer size set to {}", optval);
        }
    }
}

auto udp_source::start() -> void {
    m_processing_thread = std::jthread(&udp_source::process_msgs, this);
    composite::component::start();
}

auto udp_source::stop() -> void {
    m_processing_thread.request_stop();
    if (m_processing_thread.joinable()) {
        m_processing_thread.join();
    }
    composite::component::stop();
}

auto udp_source::process() -> composite::retval {
    using enum composite::retval;

    if (m_pool == nullptr) {
        return NOOP;
    }

    // Get data from pool
    auto data = m_pool->acquire();

    // Replenish if needed in async
    auto fut = std::async(std::launch::async, [this]{ m_pool->replenish(); });

    // Receive messages
    struct timespec timeout = {
        .tv_sec = 0,
        .tv_nsec = 1000000,  // 1000µs timeout
    };
    data->recvd = 0;
    auto recvd = recvmmsg(m_socket, data->msgs.data(), data->msgs.size(), 0, &timeout);
    if (recvd <= 0) {
        return NORMAL;
    }
    data->recvd += recvd;

    // Move received messages to processing queue
    m_queue.push(std::move(data));
    
    // Fast return
    return NO_YIELD;
}

auto udp_source::process_msgs(std::stop_token token) -> void {
    while (!token.stop_requested()) {
        auto msgs = m_queue.pop();
        if (msgs == nullptr) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }
        // Iterate over data buffer
        for (auto i = std::size_t{}; i < msgs->recvd; ++i) {
            auto idx = i * m_msg_size;
            auto payload = std::shared_ptr<std::vector<std::byte>>{nullptr};
            auto ts = composite::timestamp{};
            // Parse packets based on protocol
            if (m_transport == "sdds") {
                auto packet = overlay::sdds::overlay({msgs->buffer->data() + idx, m_msg_size});
                auto seq_num = packet.seq_num();
                if (packet.pp_id() && ((seq_num % 32) != 31)) {
                    logger()->error("invalid SDDS packet received, pp_id=true, seq_num={}", seq_num);
                } else if (!packet.pp_id() && ((seq_num % 32) == 31)) {
                    logger()->error("invalid SDDS packet received pp_id=false, seq_num={}", seq_num);
                }
                auto expected_seq_num = static_cast<uint16_t>(m_pkt_count + 1);
                if ((expected_seq_num % 32) == 31) {
                    ++expected_seq_num;
                }
                if (seq_num != expected_seq_num) {
                    logger()->warn("dropped pkt(s) expected={}, got={}", expected_seq_num, seq_num);
                }
                m_pkt_count = seq_num;
                // if (!packet.ttv()) {
                //     spdlog::info("invalid time tag, seq_num={}", seq_num);
                // }
                // if (packet.is_parity()) {
                //     spdlog::info("parity packet received");
                // }
                ts = composite::timestamp{packet.secs(), packet.psecs()};
                auto span = packet.payload<std::byte>();
                payload = std::make_shared<std::vector<std::byte>>(span.begin(), span.end());
            } else if (m_transport == "vita49") {
                auto packet = overlay::v49::overlay({msgs->buffer->data() + idx, m_msg_size});
                auto& header = packet.header();
                if (!overlay::v49::is_data(header)) {
                    continue;
                }
                if (auto expected_count = ((m_pkt_count + 1) % 16); header.packet_count() != expected_count) {
                    logger()->warn("dropped pkt(s) expected={}, got={}", expected_count, header.packet_count());   
                }
                m_pkt_count = header.packet_count();
                if (auto int_ts = packet.integer_timestamp()) {
                    ts.seconds = int_ts.value();
                }
                if (auto frac_ts = packet.fractional_timestamp()) {
                    ts.picoseconds = frac_ts.value();
                }
                auto span = packet.payload<std::byte>();
                payload = std::make_shared<std::vector<std::byte>>(span.begin(), span.end());
            }
            // Send data
            if (payload != nullptr) {
                m_out_port->send_data(std::move(payload), ts);
            }
        }
    }
}

extern "C" {
    auto create() -> std::shared_ptr<composite::component> {
        return std::make_shared<udp_source>();
    }
}
