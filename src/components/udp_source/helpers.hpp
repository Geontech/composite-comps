#pragma once

#include <linux/if_packet.h>
#include <netinet/in.h>
#include <string_view>

namespace net {

auto is_ipv4_multicast(std::string_view ip_addr) -> bool;
auto create_packet_mreq(std::string_view interface, std::string_view ip_addr) -> packet_mreq;
auto get_interface_ip(int fd, std::string_view interface) -> std::string;
auto create_ip_mreq(int fd, std::string_view interface, std::string_view ip_addr) -> ip_mreq;

} // namespace net