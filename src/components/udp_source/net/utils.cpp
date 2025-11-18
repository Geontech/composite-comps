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
#include <net/if.h>
#include <netinet/ether.h>
#include <string>
#include <stdexcept>
#include <sys/ioctl.h>
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
    if (::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size)) < 0) {
        ::close(fd);
        throw std::runtime_error(
            std::format("failed to set receive buffer size: {}", std::strerror(errno))
        );
    }
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
