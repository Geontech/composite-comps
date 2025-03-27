#pragma once

#include <cstdint>

namespace udp {

struct statistics {

uint32_t pkts_recvd_user{};
uint32_t pkts_dropped_user{};
uint32_t pkts_recvd_kernel{};
uint32_t pkts_dropped_kernel{};
uint32_t pkts_dropped_nic{};

}; //struct statistics

} // namespace udp