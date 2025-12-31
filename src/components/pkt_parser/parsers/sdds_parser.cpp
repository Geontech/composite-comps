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

#include "overlay/sdds.hpp"
#include "sdds_parser.hpp"

#include <algorithm>
#include <bit>
#include <format>

namespace parsers {

namespace {
    // SDDS Protocol Constants (per spec)
    constexpr std::size_t SDDS_PACKET_SIZE = 1080;  // Total packet size
    constexpr std::size_t SDDS_HEADER_SIZE = 56;     // Header size
    constexpr std::size_t SDDS_PAYLOAD_SIZE = 1024;  // Data payload size

    // SDDS data modes (encoding types)
    constexpr uint8_t DM_4BIT_SIGNED = 0;      // 4-bit signed integer
    constexpr uint8_t DM_8BIT_SIGNED = 1;      // 8-bit signed integer
    constexpr uint8_t DM_16BIT_SIGNED = 2;     // 16-bit signed integer
    constexpr uint8_t DM_8BIT_UNSIGNED = 5;    // 8-bit unsigned integer
    constexpr uint8_t DM_16BIT_UNSIGNED = 6;   // 16-bit unsigned integer
} // anonymous namespace

sdds_parser::sdds_parser(const struct_props::signal_overrides& overrides)
    : m_overrides(overrides) {}

auto sdds_parser::can_parse(const composite::immutable_buffer<uint8_t>& data) const -> bool {
    // SDDS packets are exactly 1080 bytes (56 header + 1024 payload)
    if (data.size() != SDDS_PACKET_SIZE) {
        return false;
    }

    // Overlay and validate SDDS header (read-only)
    auto packet = overlay::sdds(std::span{data.data(), data.size()});
    auto dm = packet.data_mode();
    auto bps = packet.bps();

    // Valid data mode and bits-per-sample combinations per SDDS spec
    bool valid_dm = (dm == DM_4BIT_SIGNED && bps == 4) ||
                    (dm == DM_8BIT_SIGNED && bps == 8) ||
                    (dm == DM_16BIT_SIGNED && bps == 16) ||
                    (dm == DM_8BIT_UNSIGNED && bps == 8) ||
                    (dm == DM_16BIT_UNSIGNED && bps == 16);

    return valid_dm;
}

auto sdds_parser::parse(
    const composite::immutable_buffer<uint8_t>& data,
    const composite::metadata& current_metadata
) -> parse_result {
    parse_result result;
    result.metadata = current_metadata;

    // Overlay SDDS packet (read-only)
    auto packet = overlay::sdds(std::span{data.data(), data.size()});

    // Validate and track sequence number
    auto seq_num = packet.seq_num();
    if (packet.pp_id() && ((seq_num % 32) != 31)) [[unlikely]] {
        result.warning = std::format("invalid SDDS packet received, pp_id=true, seq_num={}", seq_num);
    } else if (!packet.pp_id() && ((seq_num % 32) == 31)) [[unlikely]] {
        result.warning = std::format("invalid SDDS packet received pp_id=false, seq_num={}", seq_num);
    }

    auto expected_seq_num = static_cast<uint16_t>(m_pkt_count + 1);
    if ((expected_seq_num % 32) == 31) {
        ++expected_seq_num;
    }
    if (seq_num != expected_seq_num && m_pkt_count != 0) [[unlikely]] {
        result.warning = std::format("dropped pkt(s) expected={}, got={}", expected_seq_num, seq_num);
    }
    m_pkt_count = seq_num;

    // Extract metadata from packet
    result.metadata.format.is_complex = packet.complex();
    result.metadata.format.type = composite::data_type::signed_integer;
    result.metadata.format.endianness = std::endian::big;
    result.metadata.format.bit_width = packet.bps();
    result.metadata.sample_rate = packet.sample_rate();

    // Apply overrides
    if (m_overrides.data_format.is_complex.has_value()) {
        result.metadata.format.is_complex = m_overrides.data_format.is_complex.value();
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

    result.metadata.annotations["protocol"] = "sdds";

    // Extract timestamp
    result.timestamp = composite::timestamp{packet.secs(), packet.psecs()};

    // Extract payload (zero-copy slice)
    result.payload = data.slice(SDDS_HEADER_SIZE, SDDS_PAYLOAD_SIZE);

    result.should_send = true;

    return result;
}

} // namespace parsers
