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

#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <vrtgen/vrtgen.hpp>

namespace overlay {

namespace sdds {

static constexpr double TIME_TIC = 250e-12;
static constexpr double TIME_TWO32 = 4294967296.0;
static constexpr uint64_t PS250_PER_SEC = 4000000000;

class overlay {
    static constexpr std::size_t DATA_IDX = 56;
    static constexpr std::size_t DATA_LEN = 1024;
public:
    explicit overlay(std::span<const uint8_t> data);

    auto pp_id() const -> bool;

    auto is_parity() const -> bool;

    auto bps() const -> uint8_t;

    auto seq_num() const -> uint16_t;

    auto ttv() const -> bool;

    auto ttag() const -> uint64_t;

    auto ttage() const -> uint32_t;

    auto secs() const -> uint32_t;

    auto psecs() const -> uint64_t;

    template<typename T>
    auto payload() const -> std::span<const T>;

private:
    std::span<const uint8_t> m_data;

}; // class overlay

} // namespace sdds

namespace v49 {

class overlay {
public:
    explicit overlay(std::span<const uint8_t> data);

    auto is_data() const -> bool;
    auto is_context() const -> bool;
    auto header() const -> const vrtgen::packing::Header&;
    auto stream_id() const -> std::optional<uint32_t>;
    auto class_id() const -> const std::optional<vrtgen::packing::ClassIdentifier>&;
    auto integer_timestamp() const -> std::optional<uint32_t>;
    auto fractional_timestamp() const -> std::optional<uint64_t>;
    template<typename T>
    auto payload() const -> std::span<const T>;
    auto payload_size() const -> size_t;
    auto payload_start() const -> size_t;

private:
    std::span<const uint8_t> m_data;
    std::map<std::string, std::size_t> m_positions;
    vrtgen::packing::Header m_header;
    std::optional<uint32_t> m_stream_id;
    std::optional<vrtgen::packing::ClassIdentifier> m_class_id;
    std::optional<uint32_t> m_int_ts;
    std::optional<uint64_t> m_frac_ts;
    std::optional<vrtgen::packing::Trailer> m_trailer;

}; // class overlay

} // namespace v49
} // namespace overlay
