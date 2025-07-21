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

#include <bit>
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
static constexpr double FREQ_MULT = 1.3552527156068805e-11; // 125 MHz / 2^63

class overlay {
    static constexpr std::size_t DATA_IDX = 56;
    static constexpr std::size_t DATA_LEN = 1024;
public:
    explicit overlay(std::span<const uint8_t> data);

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

}; // class overlay

} // namespace sdds

namespace v49 {

class overlay {
public:
    explicit overlay(std::span<const uint8_t> data);

    auto is_vrl() const -> bool;
    auto is_data() const -> bool;
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

private:
    auto swap_iq_scalar(std::span<uint8_t>) -> void;
    auto swap_iq_avx2(std::span<uint8_t>) -> void;
    auto swap_iq_avx512(std::span<uint8_t>) -> void;

    std::span<const uint8_t> m_data;
    std::map<std::string, std::size_t> m_positions;
    bool m_is_vrl{false};
    bool m_little_endian{false};
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

}; // class overlay

} // namespace v49
} // namespace overlay
