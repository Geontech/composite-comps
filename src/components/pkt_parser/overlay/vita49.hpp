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
#include <vrtgen/vrtgen.hpp>

namespace overlay {

class v49 {
public:
    explicit v49(std::span<const uint8_t> data);

    auto is_data() const -> bool;
    auto is_ext_data() const -> bool;
    auto is_context() const -> bool;
    auto header() const -> const vrtgen::packing::Header&;
    auto stream_id() const -> std::optional<uint32_t>;
    auto class_id() const -> const std::optional<vrtgen::packing::ClassIdentifier>&;
    auto integer_timestamp() const -> std::optional<uint32_t>;
    auto fractional_timestamp() const -> std::optional<uint64_t>;
    template<typename T>
    auto payload() -> std::span<const T>;
    auto swap_iq(std::span<uint8_t>) -> void;
    auto payload_size() const -> size_t;
    auto payload_start() const -> size_t;

    // Metadata functions for info from context packets
    auto endianness() const -> std::endian;
    auto bandwidth() const -> std::optional<double>;
    auto rf_frequency() const -> std::optional<double>;
    auto sample_rate() const -> std::optional<double>;
    auto signal_data_format() const -> const std::optional<vrtgen::packing::PayloadFormat>&;

    // Static 32-bit word byteswap utilities (for PLRV endianness conversion)
    static auto byteswap_u32_words(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void;
    static auto byteswap_u32_words_scalar(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void;
    static auto byteswap_u32_words_avx2(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void;
    static auto byteswap_u32_words_avx512(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void;

private:
    std::span<const uint8_t> m_data;
    std::map<std::string, std::size_t> m_positions;
    vrtgen::packing::Header m_header;
    std::optional<uint32_t> m_stream_id;
    std::optional<vrtgen::packing::ClassIdentifier> m_class_id;
    std::optional<uint32_t> m_int_ts;
    std::optional<uint64_t> m_frac_ts;
    std::optional<vrtgen::packing::CIF0> m_cif0;
    std::optional<double> m_bandwidth;
    std::optional<double> m_rf_frequency;
    std::optional<double> m_sample_rate;
    std::optional<vrtgen::packing::PayloadFormat> m_signal_data_format;
    std::optional<vrtgen::packing::Trailer> m_trailer;

}; // class v49

} // namespace overlay
