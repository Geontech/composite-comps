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

#include "vita49_parser.hpp"
#include "overlay/vita49.hpp"

#include <algorithm>
#include <bit>
#include <format>

namespace parsers {

namespace {
    // VITA 49 Data Item Format ranges (per spec section 9.5.7)
    constexpr uint32_t MAX_SIGNED_FORMAT = 0x07;    // Formats 0x00-0x07 are signed integer
    constexpr uint32_t MIN_UNSIGNED_FORMAT = 0x10;  // Formats 0x10-0x17 are unsigned integer
                                                     // Formats 0x08-0x0F are floating point
} // anonymous namespace

vita49_parser::vita49_parser(const struct_props::signal_overrides& overrides)
    : m_overrides(overrides) {}

auto vita49_parser::can_parse(const composite::immutable_buffer<uint8_t>& data) const -> bool {
    // Minimum VITA 49 packet size (header only)
    if (data.size() < 4) {
        return false;
    }

    // Overlay and check packet type
    auto packet = overlay::v49(std::span{data.data(), data.size()});
    return packet.is_data() || packet.is_ext_data() || packet.is_context();
}

auto vita49_parser::parse(
    const composite::immutable_buffer<uint8_t>& data,
    const composite::metadata& current_metadata
) -> parse_result {
    parse_result result;
    result.metadata = current_metadata;

    // Overlay VITA 49 packet (read-only, no IQ swap)
    auto packet = overlay::v49(std::span{data.data(), data.size()});

    auto is_tsf_sc = false;

    if (packet.is_data()) [[likely]] {
        auto& header = packet.header();

        // Track packet sequence
        auto expected_count = ((m_pkt_count + 1) % 16);
        if (header.packet_count() != expected_count && m_pkt_count != 0) {
            result.warning = std::format("dropped pkt(s) expected={}, got={}", expected_count, header.packet_count());
        }
        m_pkt_count = header.packet_count();

        // Extract timestamps
        if (auto int_ts = packet.integer_timestamp()) {
            result.timestamp.seconds = int_ts.value();
        }
        if (auto frac_ts = packet.fractional_timestamp()) {
            result.timestamp.picoseconds = frac_ts.value();
            is_tsf_sc = (header.tsf() == vrtgen::packing::TSF::SAMPLE_COUNT);
        }

        // Extract payload (zero-copy slice)
        auto payload_start = packet.payload_start();
        auto payload_size = packet.payload_size();
        result.payload = data.slice(payload_start, payload_size);

        result.should_send = true;
    } else if (packet.is_context()) {
        // Extract metadata from context packet
        if (auto format = packet.signal_data_format()) {
            result.metadata.format.is_complex = format->real_complex_type() != vrtgen::packing::DataSampleType::REAL;

            // Determine data type from VITA 49 format encoding
            auto format_code = std::to_underlying(format->data_item_format());
            if (format_code <= MAX_SIGNED_FORMAT) {
                result.metadata.format.type = composite::data_type::signed_integer;
            } else if (format_code >= MIN_UNSIGNED_FORMAT) {
                result.metadata.format.type = composite::data_type::unsigned_integer;
            } else {
                result.metadata.format.type = composite::data_type::floating_point;
            }
            result.metadata.format.bit_width = format->data_item_size();
            result.metadata.format.endianness = packet.endianness();
        }

        result.metadata.center_frequency = packet.rf_frequency().value_or(0);
        result.metadata.bandwidth = packet.bandwidth().value_or(0);
        result.metadata.sample_rate = packet.sample_rate().value_or(0);

        result.should_send = false;  // Context packets don't carry data
    }

    // Apply overrides
    if (m_overrides.data_format.is_complex.has_value()) {
        result.metadata.format.is_complex = m_overrides.data_format.is_complex.value();
    }
    if (!m_overrides.data_format.type.empty()) {
        if (m_overrides.data_format.type == "signed_integer") {
            result.metadata.format.type = composite::data_type::signed_integer;
        } else if (m_overrides.data_format.type == "unsigned_integer") {
            result.metadata.format.type = composite::data_type::unsigned_integer;
        } else if (m_overrides.data_format.type == "floating_point") {
            result.metadata.format.type = composite::data_type::floating_point;
        }
    }
    if (m_overrides.data_format.bit_width > 0) {
        result.metadata.format.bit_width = m_overrides.data_format.bit_width;
    }
    if (!m_overrides.data_format.endianness.empty()) {
        if (m_overrides.data_format.endianness == "big") {
            result.metadata.format.endianness = std::endian::big;
        } else if (m_overrides.data_format.endianness == "little") {
            result.metadata.format.endianness = std::endian::little;
        }
    }
    if (m_overrides.center_frequency.has_value()) {
        result.metadata.center_frequency = m_overrides.center_frequency.value();
    }
    if (m_overrides.bandwidth.has_value()) {
        result.metadata.bandwidth = m_overrides.bandwidth.value();
    }
    if (m_overrides.sample_rate.has_value()) {
        result.metadata.sample_rate = m_overrides.sample_rate.value();
    }

    result.metadata.annotations["protocol"] = "v49";

    // Adjust fractional timestamp if in sample count mode
    if (is_tsf_sc) {
        if (result.metadata.sample_rate == 0.0) {
            if (!m_tsf_warn) {
                result.warning = "unable to set fractional timestamp: unknown sample rate in SAMPLE_COUNT mode; dropping data until sample rate discovered";
                m_tsf_warn = true;
            }
            result.should_send = false;
        } else {
            // Convert sample count to picoseconds: samples / sample_rate * 1e12
            auto samples = static_cast<double>(result.timestamp.picoseconds);
            auto picoseconds_per_sample = 1e12 / result.metadata.sample_rate;
            auto picoseconds = samples * picoseconds_per_sample;

            // Clamp to uint64_t range to prevent overflow
            constexpr auto max_uint64 = static_cast<double>(UINT64_MAX);
            result.timestamp.picoseconds = static_cast<uint64_t>(std::min(picoseconds, max_uint64));
        }
    }

    return result;
}

} // namespace parsers
