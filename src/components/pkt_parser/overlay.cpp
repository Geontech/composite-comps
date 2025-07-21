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

#include "overlay.hpp"

#include <bit>
#include <immintrin.h>

namespace overlay {

namespace sdds {

overlay:: overlay(std::span<const uint8_t> data) : m_data(data) {}

auto overlay::standard_format() const -> bool {
    return m_data[0] & 0x80;
}

auto overlay::pp_id() const -> bool {
    return m_data[0] & 0x20;
}

auto overlay::is_parity() const -> bool {
    return pp_id() && ((seq_num() % 32) == 31);
}

auto overlay::data_mode() const -> uint8_t {
    return m_data[0] & 0x07;
}

auto overlay::bps() const -> uint8_t {
    return m_data[1] & 0x1F;
}

auto overlay::complex() const -> bool {
    return m_data[1] & 0x80;
}

auto overlay::seq_num() const -> uint16_t {
    return std::byteswap(*reinterpret_cast<const uint16_t*>(m_data.data() + 2));
}

auto overlay::ttv() const -> bool {
    return m_data[4] & 0x40;
}

auto overlay::ttag() const -> uint64_t {
    return std::byteswap(*reinterpret_cast<const uint64_t*>(m_data.data() + 8));
}

auto overlay::ttage() const -> uint32_t {
    return std::byteswap(*reinterpret_cast<const uint32_t*>(m_data.data() + 16));
}

auto overlay::dfdt() const -> int32_t {
    return std::byteswap(*reinterpret_cast<const int32_t*>(m_data.data() + 20));
}

auto overlay::frequency() const -> uint64_t {
    return std::byteswap(*reinterpret_cast<const uint64_t*>(m_data.data() + 24));
}

auto overlay::sample_rate() const -> double {
    auto rate = static_cast<double>(frequency()) * FREQ_MULT;
    if (complex()) {
        return rate / 2;
    }
    return rate;
}

auto overlay::secs() const -> uint32_t {
    return static_cast<uint32_t>(ttag() / PS250_PER_SEC);
}

auto overlay::psecs() const -> uint64_t {
    return (ttag() * uint64_t{250u}) - (static_cast<uint64_t>(secs()) * uint64_t{1'000'000'000'000u});
}

template<typename T>
auto overlay::payload() const -> std::span<const T> {
    auto data = reinterpret_cast<const T*>(m_data.data() + DATA_IDX);
    return std::span<const T>(data, DATA_LEN / sizeof(T));
}

} // namespace sdds

namespace v49 {

overlay::overlay(std::span<const uint8_t> data) : m_data(data) {
    auto curr_idx = std::size_t{};
    auto v49_header_pos = curr_idx;
    // Check for VRL frame
    auto first_word = *reinterpret_cast<const uint32_t*>(m_data.data());
    constexpr auto VRLP = uint32_t{0x56524C50};
    if (first_word == VRLP) {
        m_is_vrl = true;
        m_little_endian = true;
        curr_idx += sizeof(VRLP);
        // TODO: parse frame word
        curr_idx += sizeof(uint32_t); // frame word
    } else if (first_word == std::byteswap(VRLP)) {
        m_is_vrl = true;
        curr_idx += sizeof(VRLP);
        // TODO: parse frame word
        curr_idx += sizeof(uint32_t); // frame word
    }
    // Parse VITA49 header
    v49_header_pos = curr_idx;
    if (m_little_endian) {
        auto header_word = std::byteswap(*reinterpret_cast<const uint32_t*>(m_data.data() + curr_idx));
        m_header.unpack_from(reinterpret_cast<uint8_t*>(&header_word));
    } else {
        m_header.unpack_from(m_data.data() + curr_idx);
    }
    curr_idx += m_header.size();
    // Check for stream id
    if (m_header.packet_type() != vrtgen::packing::PacketType::SIGNAL_DATA) {
        auto stream_id = *reinterpret_cast<const uint32_t*>(m_data.data() + curr_idx);
        if (m_little_endian) {
            m_stream_id = stream_id;
        } else {
            m_stream_id = std::byteswap(stream_id);
        }
        curr_idx += sizeof(stream_id);
    }
    // Check for class id
    if (m_header.class_id_enable()) {
        m_class_id = vrtgen::packing::ClassIdentifier{};
        if (m_little_endian) {
            auto class_id_words = std::byteswap(*reinterpret_cast<const uint64_t*>(m_data.data() + curr_idx));
            m_class_id->unpack_from(reinterpret_cast<uint8_t*>(&class_id_words));
        } else {
            m_class_id->unpack_from(m_data.data() + curr_idx);
        }
        curr_idx += m_class_id->size();
    }
    // Check and get integer timestamp
    if (m_header.tsi() != vrtgen::packing::TSI::NONE) {
        auto ts = *reinterpret_cast<const uint32_t*>(m_data.data() + curr_idx);
        if (m_little_endian) {
            m_int_ts = ts;
        } else {
            m_int_ts = std::byteswap(ts);
        }
        curr_idx += sizeof(ts);
    }
    // Check and get fractional timestamp
    if (m_header.tsf() != vrtgen::packing::TSF::NONE) {
        auto ts = *reinterpret_cast<const uint64_t*>(m_data.data() + curr_idx);
        if (m_little_endian) {
            m_frac_ts = ts;
        } else {
            m_frac_ts = std::byteswap(ts);
        }
        curr_idx += sizeof(ts);
    }
    if (is_data()) {
        m_positions["payload"] = curr_idx;
        auto data_header = vrtgen::packing::DataHeader{};
        data_header.unpack_from(m_data.data() + v49_header_pos);
        if (data_header.trailer_included()) {
            m_trailer = vrtgen::packing::Trailer{};
            auto pos = (m_header.packet_size() - 1) * sizeof(uint32_t)/*word size*/;
            if (m_is_vrl) {
                pos += sizeof(VRLP) + sizeof(uint32_t)/*frame word*/;
            }
            if (m_little_endian) {
                auto trailer_word = std::byteswap(*reinterpret_cast<const uint32_t*>(m_data.data() + pos));
                m_trailer->unpack_from(reinterpret_cast<uint8_t*>(&trailer_word));
            } else {
                m_trailer->unpack_from(m_data.data() + pos);
            }
        }

        // Check for needed q/i to i/q swap
        auto pos = m_positions.at("payload");
        auto len = payload_size();
        // swap_iq({m_data.data() + pos, len});
    } else if (is_context()) {
        m_cif0 = vrtgen::packing::CIF0{};
        if (m_little_endian) {
            auto cif0_word = std::byteswap(*reinterpret_cast<const uint32_t*>(m_data.data() + curr_idx));
            m_cif0->unpack_from(reinterpret_cast<uint8_t*>(&cif0_word));
        } else {
            m_cif0->unpack_from(m_data.data() + curr_idx);
        }
        curr_idx += m_cif0->size();
        if (m_cif0->reference_point_id()) {
            curr_idx += sizeof(uint32_t); // advance past
        }
        if (m_cif0->bandwidth()) {
            auto bw = *reinterpret_cast<const uint64_t*>(m_data.data() + curr_idx);
            if (m_little_endian) {
                m_bandwidth = vrtgen::fixed::to_fp<44,20>(bw);
            } else {
                m_bandwidth = vrtgen::fixed::to_fp<44,20>(std::byteswap(bw));
            }
            curr_idx += sizeof(bw);
        }
        if (m_cif0->if_ref_frequency()) {
            curr_idx += sizeof(uint64_t); // advance past
        }
        if (m_cif0->rf_ref_frequency()) {
            auto freq = *reinterpret_cast<const uint64_t*>(m_data.data() + curr_idx);
            if (m_little_endian) {
                m_rf_frequency = vrtgen::fixed::to_fp<44,20>(freq);
            } else {
                m_rf_frequency = vrtgen::fixed::to_fp<44,20>(std::byteswap(freq));
            }
            curr_idx += sizeof(freq);
        }
        if (m_cif0->rf_ref_frequency_offset()) {
            curr_idx += sizeof(uint64_t); // advance past
        }
        if (m_cif0->if_band_offset()) {
            curr_idx += sizeof(uint64_t); // advance past
        }
        if (m_cif0->reference_level()) {
            curr_idx += sizeof(uint32_t); // advance past
        }
        if (m_cif0->gain()) {
            curr_idx += sizeof(uint32_t); // advance past
        }
        if (m_cif0->over_range_count()) {
            curr_idx += sizeof(uint32_t); // advance past
        }
        if (m_cif0->sample_rate()) {
            auto sr = *reinterpret_cast<const uint64_t*>(m_data.data() + curr_idx);
            if (m_little_endian) {
                m_sample_rate = vrtgen::fixed::to_fp<44,20>(sr);
            } else {
                m_sample_rate = vrtgen::fixed::to_fp<44,20>(std::byteswap(sr));
            }
            curr_idx += sizeof(sr);
        }
        if (m_cif0->timestamp_adjustment()) {
            curr_idx += sizeof(uint32_t); // advance past
        }
        if (m_cif0->timestamp_calibration_time()) {
            curr_idx += sizeof(uint32_t); // advance past
        }
        if (m_cif0->temperature()) {
            curr_idx += sizeof(uint32_t); // advance past
        }
        if (m_cif0->device_id()) {
            curr_idx += sizeof(uint32_t); // advance past
        }
        if (m_cif0->state_event_indicators()) {
            curr_idx += sizeof(uint32_t); // advance past
        }
        if (m_cif0->signal_data_format()) {
            m_signal_data_format = vrtgen::packing::PayloadFormat{};
            if (m_little_endian) {
                auto sdf_words = std::byteswap(*reinterpret_cast<const uint64_t*>(m_data.data() + curr_idx));
                m_signal_data_format->unpack_from(reinterpret_cast<uint8_t*>(&sdf_words));
            } else {
                m_signal_data_format->unpack_from(m_data.data() + curr_idx);
            }
            curr_idx += m_signal_data_format->size();
        }
        // TODO: more context fields
    }
}

auto overlay::is_vrl() const -> bool {
    return m_is_vrl;
}

auto overlay::is_data() const -> bool {
    using enum vrtgen::packing::PacketType;
    return m_header.packet_type() == SIGNAL_DATA || m_header.packet_type() == SIGNAL_DATA_STREAM_ID;
}

auto overlay::is_context() const -> bool {
    using enum vrtgen::packing::PacketType;
    return m_header.packet_type() == CONTEXT;
}

auto overlay::header() const -> const vrtgen::packing::Header& {
    return m_header;
}

auto overlay::stream_id() const -> std::optional<uint32_t> {
    return m_stream_id;
}

auto overlay::class_id() const -> const std::optional<vrtgen::packing::ClassIdentifier>& {
    return m_class_id;
}

auto overlay::integer_timestamp() const -> std::optional<uint32_t> {
    return m_int_ts;
}

auto overlay::fractional_timestamp() const -> std::optional<uint64_t> {
    return m_frac_ts;
}

template<typename T>
auto overlay::payload() -> std::span<const T> {
    if (!m_positions.contains("payload")) {
        return {};
    }
    auto pos = m_positions.at("payload");
    auto data = reinterpret_cast<const T*>(m_data.data() + pos);
    return std::span<const T>(data, payload_size() / sizeof(T));
}

auto overlay::payload_size() const -> size_t {
    if (!m_positions.contains("payload")) {
        return {};
    }
    auto pos = m_positions.at("payload");
    if (m_is_vrl) {
        pos -= sizeof(uint32_t)/*VRLP*/ + sizeof(uint32_t)/*frame word*/;
    }
    auto size = (m_header.packet_size() * sizeof(uint32_t)/*word size*/) - pos;
    if (m_trailer.has_value()) {
        size -= sizeof(uint32_t);
    }
    return size;
}

auto overlay::payload_start() const -> size_t {
    if (!m_positions.contains("payload")) {
        return {};
    }
    return m_positions.at("payload");
}

auto overlay::endianness() const -> std::endian {
    return m_little_endian ? std::endian::little : std::endian::big;
}

auto overlay::bandwidth() const -> std::optional<double> {
    return m_bandwidth;
}

auto overlay::rf_frequency() const -> std::optional<double> {
    return m_rf_frequency;
}

auto overlay::sample_rate() const -> std::optional<double> {
    return m_sample_rate;
}

auto overlay::signal_data_format() const -> const std::optional<vrtgen::packing::PayloadFormat>& {
    return m_signal_data_format;
}

auto overlay::swap_iq(std::span<uint8_t> data) -> void {
    if (m_is_vrl && m_little_endian) {
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
            swap_iq_avx512(data);
        } else if (__builtin_cpu_supports("avx2")) {
            swap_iq_avx2(data);
        } else {
            swap_iq_scalar(data);
        }
    }
}

auto overlay::swap_iq_scalar(std::span<uint8_t> data) -> void {
    for (size_t i = 0; i + 3 < data.size(); i += 4) {
        std::swap(data[i], data[i + 2]);
        std::swap(data[i + 1], data[i + 3]);
    }
}

[[gnu::target("avx2")]]
auto overlay::swap_iq_avx2(std::span<uint8_t> data) -> void {
    const size_t stride = 32;
    auto shuffle_mask = _mm256_set_epi8(
        13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2,
        13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2
    );
    size_t i = 0;
    for (; i + stride <= data.size(); i += stride) {
        auto data_256 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(data.data() + i));
        data_256 = _mm256_shuffle_epi8(data_256, shuffle_mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(data.data() + i), data_256);
    }
    for (; i + 3 < data.size(); i += 4) {
        std::swap(data[i], data[i + 2]);
        std::swap(data[i + 1], data[i + 3]);
    }
}

[[gnu::target("avx512f,avx512bw")]]
auto overlay::swap_iq_avx512(std::span<uint8_t> data) -> void {
    const size_t stride = 64;
    auto shuffle_mask = _mm512_set_epi8(
        13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2,
        13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2,
        13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2,
        13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2
    );
    size_t i = 0;
    for (; i + stride <= data.size(); i += stride) {
        auto data_512 = _mm512_loadu_si512(data.data() + i);
        data_512 = _mm512_shuffle_epi8(data_512, shuffle_mask);
        _mm512_storeu_si512(data.data() + i, data_512);
    }
    for (; i + 3 < data.size(); i += 4) {
        std::swap(data[i], data[i + 2]);
        std::swap(data[i + 1], data[i + 3]);
    }
}

} // namespace v49
} // namespace overlay
