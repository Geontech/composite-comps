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

#pragma once

#include <array>
#include <linux/if_packet.h>
#include <netinet/in.h>
#include <string_view>

namespace net {

auto is_ipv4_multicast(std::string_view ip_addr) -> bool;
auto create_packet_mreq(std::string_view interface, std::string_view ip_addr) -> packet_mreq;
auto create_multicast_mac(std::string_view mcast_ip) -> std::array<uint8_t, 6>;
auto get_interface_ip(int fd, std::string_view interface) -> std::string;
auto create_ip_mreq(int fd, std::string_view interface, std::string_view ip_addr) -> ip_mreq;
auto set_socket_recv_buffer(int fd, std::size_t size) -> void;
auto set_socket_reuse_addr(int fd, bool enable=true) -> void;
auto get_interface_index(std::string_view interface) -> std::size_t;

} // namespace net