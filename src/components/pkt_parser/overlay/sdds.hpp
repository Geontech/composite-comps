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

#include <bit>
#include <cstdint>
#include <map>
#include <optional>
#include <span>

namespace overlay {

static constexpr double TIME_TIC = 250e-12;
static constexpr double TIME_TWO32 = 4294967296.0;
static constexpr uint64_t PS250_PER_SEC = 4000000000;
static constexpr double FREQ_MULT = 1.3552527156068805e-11; // 125 MHz / 2^63

class sdds {
    static constexpr std::size_t DATA_IDX = 56;
    static constexpr std::size_t DATA_LEN = 1024;
public:
    explicit sdds(std::span<const uint8_t> data);

    auto standard_format() const -> bool;
    auto pp_id() const -> bool;
    auto is_parity() const -> bool;
    auto data_mode() const -> uint8_t;
    auto bps() const -> uint8_t;
    auto complex() const -> bool;
    auto seq_num() const -> uint16_t;
    auto ttv() const -> bool;
    auto ttag() const -> uint64_t;
    auto ttage() const -> uint32_t;
    auto dfdt() const -> int32_t;
    auto frequency() const -> uint64_t;
    auto sample_rate() const -> double;
    auto secs() const -> uint32_t;
    auto psecs() const -> uint64_t;
    template<typename T>
    auto payload() const -> std::span<const T>;

private:
    std::span<const uint8_t> m_data;

}; // class sdds

} // namespace overlay
