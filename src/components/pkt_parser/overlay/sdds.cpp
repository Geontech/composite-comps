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

#include "sdds.hpp"

#include <bit>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>

namespace overlay {

namespace {
// Big-endian field read via memcpy (not reinterpret_cast) to avoid misaligned-
// load / strict-aliasing UB, with a defensive bounds check. SDDS is fixed at
// 1080 bytes and the parser re-validates that before constructing this overlay,
// so for valid traffic these reads are always in-bounds; the check only fires if
// the overlay is ever built on a short buffer (then caught by the component).
template <typename U>
auto read_be(std::span<const uint8_t> d, std::size_t off) -> U {
    if (sizeof(U) > d.size() || off > d.size() - sizeof(U)) {
        throw std::out_of_range("sdds: field read past end of packet");
    }
    U v{};
    std::memcpy(&v, d.data() + off, sizeof(U));
    return std::byteswap(v);
}
} // namespace

sdds::sdds(std::span<const uint8_t> data) : m_data(data) {}

auto sdds::standard_format() const -> bool {
    return m_data[0] & 0x80;
}

auto sdds::pp_id() const -> bool {
    return m_data[0] & 0x20;
}

auto sdds::is_parity() const -> bool {
    return pp_id() && ((seq_num() % 32) == 31);
}

auto sdds::data_mode() const -> uint8_t {
    return m_data[0] & 0x07;
}

auto sdds::bps() const -> uint8_t {
    return m_data[1] & 0x1F;
}

auto sdds::complex() const -> bool {
    return m_data[1] & 0x80;
}

auto sdds::seq_num() const -> uint16_t {
    return read_be<uint16_t>(m_data, 2);
}

auto sdds::ttv() const -> bool {
    return m_data[4] & 0x40;
}

auto sdds::ttag() const -> uint64_t {
    return read_be<uint64_t>(m_data, 8);
}

auto sdds::ttage() const -> uint32_t {
    return read_be<uint32_t>(m_data, 16);
}

auto sdds::dfdt() const -> int32_t {
    return read_be<int32_t>(m_data, 20);
}

auto sdds::frequency() const -> uint64_t {
    return read_be<uint64_t>(m_data, 24);
}

auto sdds::sample_rate() const -> double {
    auto rate = static_cast<double>(frequency()) * FREQ_MULT;
    if (complex()) {
        return rate / 2;
    }
    return rate;
}

auto sdds::secs() const -> uint32_t {
    return static_cast<uint32_t>(ttag() / PS250_PER_SEC);
}

auto sdds::psecs() const -> uint64_t {
    return (ttag() * uint64_t{250u}) - (static_cast<uint64_t>(secs()) * uint64_t{1'000'000'000'000u});
}

template<typename T>
auto sdds::payload() const -> std::span<const T> {
    auto data = reinterpret_cast<const T*>(m_data.data() + DATA_IDX);
    return std::span<const T>(data, DATA_LEN / sizeof(T));
}

} // namespace overlay
