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
#include <stdexcept>

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
    : m_overrides(overrides) {
    // SDDS is big-endian per spec, but honor an explicit endianness override for receivers
    // that emit little-endian samples (resolved once, not re-parsed per packet).
    if (m_overrides.data_format.endianness == "little") {
        m_ov_endianness = std::endian::little;
    } else if (m_overrides.data_format.endianness == "big") {
        m_ov_endianness = std::endian::big;
    }
}

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

    // SDDS is a fixed 1080-byte protocol. Protocol lock-in does NOT trust later
    // packets, so re-validate the size before the overlay reads fixed header
    // offsets and slices [56, 56+1024) — a short/oversized packet would otherwise
    // read out of bounds. The pkt_parser component catches this and drops + counts.
    if (data.size() != SDDS_PACKET_SIZE) {
        throw std::out_of_range("sdds_parser: packet size != SDDS_PACKET_SIZE (1080)");
    }

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

    // Effective metadata this packet carries (extraction + the constant overrides). SDDS is
    // signed and big-endian per spec (endianness overridable for non-conforming receivers);
    // center_frequency and bandwidth come only from overrides (otherwise carried from the
    // current metadata).
    const bool eff_complex = m_overrides.data_format.is_complex.value_or(packet.complex());
    const uint32_t eff_bps = packet.bps();
    const double eff_sr = m_overrides.sample_rate.value_or(packet.sample_rate());
    const std::endian eff_endianness = m_ov_endianness.value_or(std::endian::big);

    // Rebuild metadata ONLY when it actually differs from the last published value
    // (current_metadata), or on the first packet after (re)activation. Steady state does no
    // copy, no map write, and no full compare — the component then reuses the shared instance.
    bool changed = !m_emitted
        || current_metadata.format.is_complex != eff_complex
        || current_metadata.format.type != composite::data_type::signed_integer
        || current_metadata.format.bit_width != eff_bps
        || current_metadata.format.endianness != eff_endianness
        || current_metadata.sample_rate != eff_sr;
    if (m_overrides.center_frequency.has_value()) {
        changed = changed || current_metadata.center_frequency != *m_overrides.center_frequency;
    }
    if (m_overrides.bandwidth.has_value()) {
        changed = changed || current_metadata.bandwidth != *m_overrides.bandwidth;
    }
    if (changed) {
        result.metadata = current_metadata;
        result.metadata.format.is_complex = eff_complex;
        result.metadata.format.type = composite::data_type::signed_integer;
        result.metadata.format.endianness = eff_endianness;
        result.metadata.format.bit_width = eff_bps;
        result.metadata.sample_rate = eff_sr;
        if (m_overrides.center_frequency.has_value()) {
            result.metadata.center_frequency = *m_overrides.center_frequency;
        }
        if (m_overrides.bandwidth.has_value()) {
            result.metadata.bandwidth = *m_overrides.bandwidth;
        }
        result.metadata.annotations["protocol"] = "sdds";
        result.metadata_changed = true;
        m_emitted = true;
    }

    // Extract timestamp
    result.timestamp = composite::timestamp{packet.secs(), packet.psecs()};

    // Extract payload (zero-copy slice)
    result.payload = data.slice(SDDS_HEADER_SIZE, SDDS_PAYLOAD_SIZE);

    result.should_send = true;

    return result;
}

} // namespace parsers
