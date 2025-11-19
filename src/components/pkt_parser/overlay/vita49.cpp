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

#include "vita49.hpp"

#include <bit>
#include <immintrin.h>

namespace overlay {

v49::v49(std::span<const uint8_t> data) : m_data(data) {
    auto curr_idx = std::size_t{};

    // Parse VITA49 header (assumes big-endian packet)
    auto v49_header_pos = curr_idx;
    m_header.unpack_from(m_data.data() + curr_idx);
    curr_idx += m_header.size();

    // Check for stream id
    if (m_header.packet_type() != vrtgen::packing::PacketType::SIGNAL_DATA) {
        auto stream_id = *reinterpret_cast<const uint32_t*>(m_data.data() + curr_idx);
        m_stream_id = std::byteswap(stream_id);
        curr_idx += sizeof(stream_id);
    }
    // Check for class id
    if (m_header.class_id_enable()) {
        m_class_id = vrtgen::packing::ClassIdentifier{};
        m_class_id->unpack_from(m_data.data() + curr_idx);
        curr_idx += m_class_id->size();
    }

    // Check and get integer timestamp
    if (m_header.tsi() != vrtgen::packing::TSI::NONE) {
        auto ts = *reinterpret_cast<const uint32_t*>(m_data.data() + curr_idx);
        m_int_ts = std::byteswap(ts);
        curr_idx += sizeof(ts);
    }

    // Check and get fractional timestamp
    if (m_header.tsf() != vrtgen::packing::TSF::NONE) {
        auto ts = *reinterpret_cast<const uint64_t*>(m_data.data() + curr_idx);
        m_frac_ts = std::byteswap(ts);
        curr_idx += sizeof(ts);
    }
    if (is_data()) {
        m_positions["payload"] = curr_idx;
        auto data_header = vrtgen::packing::DataHeader{};
        data_header.unpack_from(m_data.data() + v49_header_pos);
        if (data_header.trailer_included()) {
            m_trailer = vrtgen::packing::Trailer{};
            auto pos = (m_header.packet_size() - 1) * sizeof(uint32_t)/*word size*/;
            m_trailer->unpack_from(m_data.data() + pos);
        }
    } else if (is_context()) {
        m_cif0 = vrtgen::packing::CIF0{};
        m_cif0->unpack_from(m_data.data() + curr_idx);
        curr_idx += m_cif0->size();

        if (m_cif0->reference_point_id()) {
            curr_idx += sizeof(uint32_t); // advance past
        }

        if (m_cif0->bandwidth()) {
            auto bw = *reinterpret_cast<const uint64_t*>(m_data.data() + curr_idx);
            m_bandwidth = vrtgen::fixed::to_fp<44,20>(std::byteswap(bw));
            curr_idx += sizeof(bw);
        }
        if (m_cif0->if_ref_frequency()) {
            curr_idx += sizeof(uint64_t); // advance past
        }

        if (m_cif0->rf_ref_frequency()) {
            auto freq = *reinterpret_cast<const uint64_t*>(m_data.data() + curr_idx);
            m_rf_frequency = vrtgen::fixed::to_fp<44,20>(std::byteswap(freq));
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
            m_sample_rate = vrtgen::fixed::to_fp<44,20>(std::byteswap(sr));
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
            m_signal_data_format->unpack_from(m_data.data() + curr_idx);
            curr_idx += m_signal_data_format->size();
        }
        // TODO: more context fields
    }
}

auto v49::is_data() const -> bool {
    using enum vrtgen::packing::PacketType;
    return m_header.packet_type() == SIGNAL_DATA || m_header.packet_type() == SIGNAL_DATA_STREAM_ID;
}

auto v49::is_ext_data() const -> bool {
    using enum vrtgen::packing::PacketType;
    return m_header.packet_type() == EXTENSION_DATA || m_header.packet_type() == EXTENSION_DATA_STREAM_ID;
}

auto v49::is_context() const -> bool {
    using enum vrtgen::packing::PacketType;
    return m_header.packet_type() == CONTEXT;
}

auto v49::header() const -> const vrtgen::packing::Header& {
    return m_header;
}

auto v49::stream_id() const -> std::optional<uint32_t> {
    return m_stream_id;
}

auto v49::class_id() const -> const std::optional<vrtgen::packing::ClassIdentifier>& {
    return m_class_id;
}

auto v49::integer_timestamp() const -> std::optional<uint32_t> {
    return m_int_ts;
}

auto v49::fractional_timestamp() const -> std::optional<uint64_t> {
    return m_frac_ts;
}

template<typename T>
auto v49::payload() -> std::span<const T> {
    if (!m_positions.contains("payload")) {
        return {};
    }
    auto pos = m_positions.at("payload");
    auto data = reinterpret_cast<const T*>(m_data.data() + pos);
    return std::span<const T>(data, payload_size() / sizeof(T));
}

auto v49::payload_size() const -> size_t {
    if (!m_positions.contains("payload")) {
        return {};
    }
    auto pos = m_positions.at("payload");
    auto size = (m_header.packet_size() * sizeof(uint32_t)/*word size*/) - pos;
    if (m_trailer.has_value()) {
        size -= sizeof(uint32_t);
    }
    return size;
}

auto v49::payload_start() const -> size_t {
    if (!m_positions.contains("payload")) {
        return {};
    }
    return m_positions.at("payload");
}

auto v49::endianness() const -> std::endian {
    // Overlay now only handles big-endian V49 packets
    return std::endian::big;
}

auto v49::bandwidth() const -> std::optional<double> {
    return m_bandwidth;
}

auto v49::rf_frequency() const -> std::optional<double> {
    return m_rf_frequency;
}

auto v49::sample_rate() const -> std::optional<double> {
    return m_sample_rate;
}

auto v49::signal_data_format() const -> const std::optional<vrtgen::packing::PayloadFormat>& {
    return m_signal_data_format;
}

// 32-bit word byteswap functions (for PLRV endianness conversion)

auto v49::byteswap_u32_words(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void {
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")) {
        byteswap_u32_words_avx512(src, dst);
    } else if (__builtin_cpu_supports("avx2")) {
        byteswap_u32_words_avx2(src, dst);
    } else {
        byteswap_u32_words_scalar(src, dst);
    }
}

auto v49::byteswap_u32_words_scalar(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void {
    auto num_words = src.size() / sizeof(uint32_t);
    auto src_words = reinterpret_cast<const uint32_t*>(src.data());
    auto dst_words = reinterpret_cast<uint32_t*>(dst.data());

    for (size_t i = 0; i < num_words; ++i) {
        dst_words[i] = std::byteswap(src_words[i]);
    }
}

[[gnu::target("avx2")]]
auto v49::byteswap_u32_words_avx2(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void {
    const size_t stride = 32;
    // Shuffle mask to reverse bytes within each 32-bit word
    // For each 4-byte word: [0,1,2,3] -> [3,2,1,0]
    auto shuffle_mask = _mm256_set_epi8(
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3,
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3
    );

    size_t i = 0;
    for (; i + stride <= src.size(); i += stride) {
        auto src_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src.data() + i));
        auto dst_256 = _mm256_shuffle_epi8(src_256, shuffle_mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data() + i), dst_256);
    }

    // Handle remaining words with scalar
    auto remaining_words = (src.size() - i) / sizeof(uint32_t);
    auto src_words = reinterpret_cast<const uint32_t*>(src.data() + i);
    auto dst_words = reinterpret_cast<uint32_t*>(dst.data() + i);
    for (size_t j = 0; j < remaining_words; ++j) {
        dst_words[j] = std::byteswap(src_words[j]);
    }
}

[[gnu::target("avx512f,avx512bw")]]
auto v49::byteswap_u32_words_avx512(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void {
    const size_t stride = 64;
    // Shuffle mask to reverse bytes within each 32-bit word
    auto shuffle_mask = _mm512_set_epi8(
        60,61,62,63, 56,57,58,59, 52,53,54,55, 48,49,50,51,
        44,45,46,47, 40,41,42,43, 36,37,38,39, 32,33,34,35,
        28,29,30,31, 24,25,26,27, 20,21,22,23, 16,17,18,19,
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3
    );

    size_t i = 0;
    for (; i + stride <= src.size(); i += stride) {
        auto src_512 = _mm512_loadu_si512(src.data() + i);
        auto dst_512 = _mm512_shuffle_epi8(src_512, shuffle_mask);
        _mm512_storeu_si512(dst.data() + i, dst_512);
    }

    // Handle remaining words with scalar
    auto remaining_words = (src.size() - i) / sizeof(uint32_t);
    auto src_words = reinterpret_cast<const uint32_t*>(src.data() + i);
    auto dst_words = reinterpret_cast<uint32_t*>(dst.data() + i);
    for (size_t j = 0; j < remaining_words; ++j) {
        dst_words[j] = std::byteswap(src_words[j]);
    }
}

} // namespace overlay
