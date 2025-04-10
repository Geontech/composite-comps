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

#pragma once

#include <cstdint>

namespace udp {

struct statistics {

uint32_t pkts_recvd_user{};
uint32_t pkts_dropped_user{};
uint32_t pkts_recvd_kernel{};
uint32_t pkts_dropped_kernel{};
uint32_t pkts_dropped_nic{};
uint32_t pkts_processed{};

}; //struct statistics

} // namespace udp
