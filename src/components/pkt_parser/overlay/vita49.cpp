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

#include "simd_fmv.hpp"

#include <bit>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>

namespace overlay {

namespace {
// The packet bytes are UNTRUSTED (raw UDP). Every read is bounds-checked: a
// field that would extend past the datagram throws std::out_of_range, which the
// pkt_parser component catches and turns into a counted drop (instead of an OOB
// read / crash). require_bytes guards a fixed-size unpack at an offset; read_be
// also fixes the prior strict-aliasing/misaligned-load UB by using memcpy.
auto require_bytes(std::span<const uint8_t> d, std::size_t off, std::size_t n) -> void {
    if (n > d.size() || off > d.size() - n) {
        throw std::out_of_range("vita49: field read past end of packet");
    }
}
template <typename U>
auto read_be(std::span<const uint8_t> d, std::size_t off) -> U {
    require_bytes(d, off, sizeof(U));
    U v{};
    std::memcpy(&v, d.data() + off, sizeof(U));
    return std::byteswap(v);
}
} // namespace

v49::v49(std::span<const uint8_t> data) : m_data(data) {
    auto curr_idx = std::size_t{};

    // Parse VITA49 header (assumes big-endian packet)
    require_bytes(m_data, curr_idx, m_header.size());
    m_header.unpack_from(m_data.data() + curr_idx);
    curr_idx += m_header.size();

    // Check for stream id
    if (m_header.packet_type() != vrtgen::packing::PacketType::SIGNAL_DATA) {
        m_stream_id = read_be<uint32_t>(m_data, curr_idx);
        curr_idx += sizeof(uint32_t);
    }
    // Check for class id
    if (m_header.class_id_enable()) {
        m_class_id = vrtgen::packing::ClassIdentifier{};
        require_bytes(m_data, curr_idx, m_class_id->size());
        m_class_id->unpack_from(m_data.data() + curr_idx);
        curr_idx += m_class_id->size();
    }

    // Check and get integer timestamp
    if (m_header.tsi() != vrtgen::packing::TSI::NONE) {
        m_int_ts = read_be<uint32_t>(m_data, curr_idx);
        curr_idx += sizeof(uint32_t);
    }

    // Check and get fractional timestamp
    if (m_header.tsf() != vrtgen::packing::TSF::NONE) {
        m_frac_ts = read_be<uint64_t>(m_data, curr_idx);
        curr_idx += sizeof(uint64_t);
    }
    if (is_data()) {
        m_payload_pos = curr_idx;
        m_has_payload = true;
        // m_header is a DataHeader already decoded from the one header word above, so the
        // trailer-included bit is available directly — no second unpack of the same bytes.
        if (m_header.trailer_included()) {
            // packet_size() is an untrusted uint16_t word count; packet_size()==0
            // would underflow (0-1)*4 to a wild offset. Compute in size_t and
            // bounds-check the trailer before unpacking.
            if (m_header.packet_size() != 0) {
                m_trailer = vrtgen::packing::Trailer{};
                auto pos = static_cast<std::size_t>(m_header.packet_size() - 1) * sizeof(uint32_t)/*word size*/;
                require_bytes(m_data, pos, m_trailer->size());
                m_trailer->unpack_from(m_data.data() + pos);
            }
        }
    } else if (is_context()) {
        m_cif0 = vrtgen::packing::CIF0{};
        require_bytes(m_data, curr_idx, m_cif0->size());
        m_cif0->unpack_from(m_data.data() + curr_idx);
        curr_idx += m_cif0->size();

        if (m_cif0->reference_point_id()) {
            curr_idx += sizeof(uint32_t); // advance past
        }

        if (m_cif0->bandwidth()) {
            m_bandwidth = vrtgen::fixed::to_fp<44,20>(read_be<uint64_t>(m_data, curr_idx));
            curr_idx += sizeof(uint64_t);
        }
        if (m_cif0->if_ref_frequency()) {
            curr_idx += sizeof(uint64_t); // advance past
        }

        if (m_cif0->rf_ref_frequency()) {
            m_rf_frequency = vrtgen::fixed::to_fp<44,20>(read_be<uint64_t>(m_data, curr_idx));
            curr_idx += sizeof(uint64_t);
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
            m_sample_rate = vrtgen::fixed::to_fp<44,20>(read_be<uint64_t>(m_data, curr_idx));
            curr_idx += sizeof(uint64_t);
        }
        if (m_cif0->timestamp_adjustment()) {
            // 64-bit field (VITA 49.2 section 9.7: fractional-time adjustment, two words).
            // Advancing one word here shifted every later field — including the payload
            // format — by 4 bytes, corrupting the published stream format.
            curr_idx += sizeof(uint64_t); // advance past
        }
        if (m_cif0->timestamp_calibration_time()) {
            curr_idx += sizeof(uint32_t); // advance past
        }
        if (m_cif0->temperature()) {
            curr_idx += sizeof(uint32_t); // advance past
        }
        if (m_cif0->device_id()) {
            // 64-bit field (VITA 49.2 section 9.10.1: OUI word + device-code word); vrtgen's
            // DeviceIdentifier packs a uint64. Same 4-byte-shift consequence as above.
            curr_idx += sizeof(uint64_t); // advance past
        }
        if (m_cif0->state_event_indicators()) {
            curr_idx += sizeof(uint32_t); // advance past
        }

        if (m_cif0->signal_data_format()) {
            m_signal_data_format = vrtgen::packing::PayloadFormat{};
            require_bytes(m_data, curr_idx, m_signal_data_format->size());
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
    if (!m_has_payload) {
        return {};
    }
    auto pos = m_payload_pos;
    auto data = reinterpret_cast<const T*>(m_data.data() + pos);
    return std::span<const T>(data, payload_size() / sizeof(T));
}

auto v49::payload_size() const -> size_t {
    if (!m_has_payload) {
        return {};
    }
    auto pos = m_payload_pos;
    // packet_size() is an untrusted uint16_t word count. Compute the total claimed
    // byte length in size_t and validate it against pos AND the actual datagram
    // before subtracting — otherwise total < pos underflows to ~SIZE_MAX, which the
    // caller would feed to slice() as a giant out-of-bounds count.
    auto total = static_cast<std::size_t>(m_header.packet_size()) * sizeof(uint32_t)/*word size*/;
    if (total <= pos || total > m_data.size()) {
        return 0;
    }
    auto size = total - pos;
    if (m_trailer.has_value()) {
        if (size < sizeof(uint32_t)) {
            return 0;
        }
        size -= sizeof(uint32_t);
    }
    return size;
}

auto v49::payload_start() const -> size_t {
    if (!m_has_payload) {
        return {};
    }
    return m_payload_pos;
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

// 32-bit word byteswap (for PLRV endianness conversion). GCC native function
// multiversioning selects the SIMD tier once at load time (same-name overloads with
// [[gnu::target]]; see simd_fmv.hpp) — no per-call __builtin_cpu_supports probes.

namespace {

// Scalar tail shared by every tier. memcpy loads/stores: the spans point into packet
// bytes at arbitrary offsets, and a misaligned uint32_t lvalue access is UB.
auto byteswap_words_tail(std::span<const uint8_t> src, std::span<uint8_t> dst, std::size_t i) -> void {
    for (; i + sizeof(uint32_t) <= src.size(); i += sizeof(uint32_t)) {
        uint32_t w{};
        std::memcpy(&w, src.data() + i, sizeof(w));
        w = std::byteswap(w);
        std::memcpy(dst.data() + i, &w, sizeof(w));
    }
}

COMPS_FMV_DEFAULT
auto byteswap_words_impl(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void {
    byteswap_words_tail(src, dst, 0);
}

#if COMPS_FMV_ENABLED
[[gnu::target("avx2")]]
auto byteswap_words_impl(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void {
    constexpr std::size_t stride = 32;
    // Reverse bytes within each 32-bit word: [0,1,2,3] -> [3,2,1,0]
    const auto shuffle_mask = _mm256_set_epi8(
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3,
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3
    );

    std::size_t i = 0;
    for (; i + stride <= src.size(); i += stride) {
        auto src_256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src.data() + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst.data() + i),
                            _mm256_shuffle_epi8(src_256, shuffle_mask));
    }
    byteswap_words_tail(src, dst, i);
}

[[gnu::target("avx512f,avx512bw")]]
auto byteswap_words_impl(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void {
    constexpr std::size_t stride = 64;
    const auto shuffle_mask = _mm512_set_epi8(
        60,61,62,63, 56,57,58,59, 52,53,54,55, 48,49,50,51,
        44,45,46,47, 40,41,42,43, 36,37,38,39, 32,33,34,35,
        28,29,30,31, 24,25,26,27, 20,21,22,23, 16,17,18,19,
        12,13,14,15, 8,9,10,11, 4,5,6,7, 0,1,2,3
    );

    std::size_t i = 0;
    for (; i + stride <= src.size(); i += stride) {
        auto src_512 = _mm512_loadu_si512(src.data() + i);
        _mm512_storeu_si512(dst.data() + i, _mm512_shuffle_epi8(src_512, shuffle_mask));
    }
    byteswap_words_tail(src, dst, i);
}
#endif

} // anonymous namespace

auto v49::byteswap_u32_words(std::span<const uint8_t> src, std::span<uint8_t> dst) -> void {
    byteswap_words_impl(src, dst);
}

} // namespace overlay
