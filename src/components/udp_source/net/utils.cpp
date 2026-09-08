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

#include "utils.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cstring>
#include <format>
#include <limits>
#include <net/if.h>
#include <netinet/ether.h>
#include <string>
#include <stdexcept>
#include <chrono>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace net {

auto is_ipv4_multicast(std::string_view ip_addr) -> bool {
    in_addr addr{};
    if (inet_pton(AF_INET, ip_addr.data(), &addr) != 1) {
        return false;
    }
    auto ip = ntohl(addr.s_addr);
    return (ip & 0xF0000000) == 0xE0000000;
}

auto create_packet_mreq(std::string_view interface, std::string_view ip_addr) -> packet_mreq {
    // Validate address
    struct in_addr addr{};
    if (inet_pton(AF_INET, ip_addr.data(), &addr) != 1) {
        throw std::invalid_argument(std::format("invalid IPv4 address: {}", ip_addr));
    }
    auto ip = ntohl(addr.s_addr);
    if ((ip & 0xF0000000) != 0xE0000000) {
        throw std::invalid_argument(std::format("invalid multicast IPv4 address: {}", ip_addr));
    }

    // Build multicast MAC address
    auto mac_addr = create_multicast_mac(ip_addr);

    // Fill packet_mreq
    struct packet_mreq mreq{};
    mreq.mr_ifindex = if_nametoindex(interface.data());
    if (mreq.mr_ifindex == 0) {
        throw std::runtime_error(std::format("invalid interface name: {}", interface));
    }
    mreq.mr_type = PACKET_MR_MULTICAST;
    mreq.mr_alen = 6;
    std::memcpy(mreq.mr_address, mac_addr.data(), mac_addr.size());

    return mreq;
}

auto create_multicast_mac(std::string_view ip_addr) -> std::array<uint8_t, 6> {
    // Validate address
    struct in_addr addr{};
    if (inet_pton(AF_INET, ip_addr.data(), &addr) != 1) {
        throw std::invalid_argument(std::format("invalid IPv4 address: {}", ip_addr));
    }
    auto ip = ntohl(addr.s_addr);

    // Populate mac_addr array
    auto mac_addr = std::array<uint8_t, 6>{
        0x01,
        0x00,
        0x5e,
        static_cast<uint8_t>((ip >> 16) & 0x7F),  // only lower 7 bits
        static_cast<uint8_t>((ip >> 8) & 0xFF),
        static_cast<uint8_t>(ip & 0xFF)
    };
    return mac_addr;
}

auto get_interface_ip(int fd, std::string_view interface) -> std::string {
    struct ifreq ifr{};
    ifr.ifr_addr.sa_family = AF_INET;
    auto copy_len = std::min(interface.size(), static_cast<std::size_t>(IFNAMSIZ - 1));
    std::strncpy(ifr.ifr_name, interface.data(), copy_len);
    ifr.ifr_name[copy_len] = '\0'; // Ensure null termination
    if (auto res = ioctl(fd, SIOCGIFADDR, &ifr); res != -1) {
        return std::string{inet_ntoa(((struct sockaddr_in*)&ifr.ifr_addr)->sin_addr)};
    }
    return {};
}

auto create_ip_mreq(int fd, std::string_view interface, std::string_view ip_addr) -> ip_mreq {
    struct ip_mreq mreq{};
    auto bind_address = get_interface_ip(fd, interface);
    mreq.imr_interface.s_addr = inet_addr(bind_address.c_str());
    mreq.imr_multiaddr.s_addr = inet_addr(ip_addr.data());
    return mreq;
}

auto set_socket_recv_buffer(int fd, std::size_t size) -> void {
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        ::close(fd);
        throw std::invalid_argument("receive buffer size exceeds SO_RCVBUF integer range");
    }
    const auto requested = static_cast<int>(size);
    if (::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &requested, sizeof(requested)) < 0) {
        ::close(fd);
        throw std::runtime_error(
            std::format("failed to set receive buffer size: {}", std::strerror(errno))
        );
    }
}

auto get_socket_recv_buffer(int fd) -> std::size_t {
    int effective{};
    socklen_t len = sizeof(effective);
    if (::getsockopt(fd, SOL_SOCKET, SO_RCVBUF, &effective, &len) < 0) {
        ::close(fd);
        throw std::runtime_error(
            std::format("failed to read effective receive buffer size: {}", std::strerror(errno))
        );
    }
    return effective > 0 ? static_cast<std::size_t>(effective) : 0;
}

auto set_socket_reuse_addr(int fd, bool enable) -> void {
    int opt = enable ? 1 : 0;
    if (::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        ::close(fd);
        throw std::runtime_error(
            std::format("failed to set SO_REUSEADDR: {}", std::strerror(errno))
        );
    }
}

auto set_socket_reuse_port(int fd, bool enable) -> void {
    int opt = enable ? 1 : 0;
    if (::setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt)) < 0) {
        ::close(fd);
        throw std::runtime_error(
            std::format("failed to set SO_REUSEPORT: {}", std::strerror(errno))
        );
    }
}

auto get_interface_index(std::string_view interface) -> std::size_t {
    auto idx = ::if_nametoindex(interface.data());
    if (idx == 0) {
        throw std::runtime_error(
            std::format("invalid interface name: {}", interface)
        );
    }
    return idx;
}

} // namespace net

namespace net {

auto receive_fd(const std::string& path, int timeout_ms) -> int {
    const int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock < 0) {
        throw std::runtime_error(std::format("receive_fd: socket: {}", std::string{strerror(errno)}));
    }
    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (path.size() >= sizeof(addr.sun_path)) {
        ::close(sock);
        throw std::runtime_error("receive_fd: socket path too long");
    }
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);

    // Bounded connect: the helper may still be coming up.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (::connect(sock, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            const auto err = errno;
            ::close(sock);
            throw std::runtime_error(std::format("receive_fd: connect to '{}' timed out: {}",
                                                 path, std::string{strerror(err)}));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    struct timeval tv{.tv_sec = timeout_ms / 1000, .tv_usec = (timeout_ms % 1000) * 1000};
    (void)::setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    char data = 0;
    struct iovec iov{.iov_base = &data, .iov_len = 1};
    alignas(struct cmsghdr) char ctrl[CMSG_SPACE(sizeof(int))]{};
    struct msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = ctrl;
    msg.msg_controllen = sizeof(ctrl);
    const auto n = ::recvmsg(sock, &msg, MSG_CMSG_CLOEXEC);
    ::close(sock);
    if (n < 0) {
        throw std::runtime_error(std::format("receive_fd: recvmsg: {}", std::string{strerror(errno)}));
    }
    for (auto* cmsg = CMSG_FIRSTHDR(&msg); cmsg != nullptr; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS &&
            cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
            int fd = -1;
            std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
            return fd;
        }
    }
    throw std::runtime_error("receive_fd: no SCM_RIGHTS control message received");
}

} // namespace net
