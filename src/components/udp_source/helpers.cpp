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

#include "helpers.hpp"

#include <arpa/inet.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <net/if.h>
#include <netinet/ether.h>
#include <arpa/inet.h>
#include <cstring>
#include <string>
#include <stdexcept>
#include <sys/ioctl.h>

namespace net {

auto create_ip_mreq(int fd, std::string_view interface, std::string_view ip_addr) -> ip_mreq {
    struct ip_mreq mreq{};
    auto bind_address = get_interface_ip(fd, interface);
    mreq.imr_interface.s_addr = inet_addr(bind_address.c_str());
    mreq.imr_multiaddr.s_addr = inet_addr(ip_addr.data());
    return mreq;
}

auto create_packet_mreq(std::string_view interface, std::string_view ip_addr) -> packet_mreq {
    // Validate address
    struct in_addr addr{};
    if (inet_pton(AF_INET, ip_addr.data(), &addr) != 1) {
        throw std::invalid_argument(std::format("invalid IPv4 address: {}", ip_addr));
    }
    auto ip = ntohl(addr.s_addr);
    if ((ip & 0xF0000000) != 0xE0000000) {
        throw std::invalid_argument(std::format("invalid multicast IPv4 address: ", ip_addr));
    }

    // Build multicast MAC address
    auto mac_addr = std::array<uint8_t, 6>{
        0x01,
        0x00,
        0x5e,
        static_cast<uint8_t>((ip >> 16) & 0x7F),  // only lower 7 bits
        static_cast<uint8_t>((ip >> 8) & 0xFF),
        static_cast<uint8_t>(ip & 0xFF)
    };

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

auto create_rte_ether_addr(std::string_view ip_addr) -> rte_ether_addr {
    // Validate address
    struct in_addr addr{};
    if (inet_pton(AF_INET, ip_addr.data(), &addr) != 1) {
        throw std::invalid_argument(std::format("invalid IPv4 address: {}", ip_addr));
    }
    auto ip = ntohl(addr.s_addr);
    if ((ip & 0xF0000000) != 0xE0000000) {
        throw std::invalid_argument(std::format("invalid multicast IPv4 address: ", ip_addr));
    }

    // Build multicast MAC address
    auto mac_addr = rte_ether_addr{};
    mac_addr.addr_bytes[0] = 0x01;
    mac_addr.addr_bytes[1] = 0x00;
    mac_addr.addr_bytes[2] = 0x5e;
    mac_addr.addr_bytes[3] = static_cast<uint8_t>((ip >> 16) & 0x7F);  // only lower 7 bits
    mac_addr.addr_bytes[4] = static_cast<uint8_t>((ip >> 8) & 0xFF);
    mac_addr.addr_bytes[5] =static_cast<uint8_t>(ip & 0xFF);
    return mac_addr;
}

auto detect_numa_node(std::string_view pci_addr) -> std::optional<std::size_t> {
    auto node_path = std::format("/sys/bus/pci/devices/{}/numa_node", pci_addr);
    auto node_file = std::ifstream(node_path);
    if (!node_file.is_open()) {
        return std::nullopt;
    }

    auto dev_numa = std::size_t{};
    node_file >> dev_numa;
    node_file.close();

    return {dev_numa};
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

auto is_ipv4_multicast(std::string_view ip_addr) -> bool {
    in_addr addr{};
    if (inet_pton(AF_INET, ip_addr.data(), &addr) != 1) {
        return false;
    }
    auto ip = ntohl(addr.s_addr);
    return (ip & 0xF0000000) == 0xE0000000;
}

} // namespace net
