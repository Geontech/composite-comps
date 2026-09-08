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

namespace {
// Valid data mode and bits-per-sample combinations per SDDS spec. Checked at detection AND
// on every locked-in parse: protocol lock-in does not trust later packets.
constexpr auto valid_data_mode(uint8_t dm, uint8_t bps) -> bool {
    return (dm == DM_4BIT_SIGNED && bps == 4) ||
           (dm == DM_8BIT_SIGNED && bps == 8) ||
           (dm == DM_16BIT_SIGNED && bps == 16) ||
           (dm == DM_8BIT_UNSIGNED && bps == 8) ||
           (dm == DM_16BIT_UNSIGNED && bps == 16);
}
} // anonymous namespace

auto sdds_parser::can_parse(const composite::immutable_buffer<uint8_t>& data) const -> bool {
    // SDDS packets are exactly 1080 bytes (56 header + 1024 payload)
    if (data.size() != SDDS_PACKET_SIZE) {
        return false;
    }

    // Overlay and validate SDDS header (read-only)
    auto packet = overlay::sdds(std::span{data.data(), data.size()});
    return valid_data_mode(packet.data_mode(), packet.bps());
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

    // Validate packet size before processing
    if (data.size() < SDDS_PACKET_SIZE) {
        result.warning = std::format(
            "truncated SDDS packet: expected {} bytes, got {} (possible UDP fragmentation or MTU issue)",
            SDDS_PACKET_SIZE, data.size());
        result.should_send = false;
        return result;
    }

    // Overlay SDDS packet (read-only)
    auto packet = overlay::sdds(std::span{data.data(), data.size()});

    // Protocol lock-in does NOT trust later packets: re-validate the data-mode/bps
    // combination too, not just the size — a malformed 1080-byte packet must become a
    // counted drop (and re-detection pressure), not get forwarded with invalid metadata.
    if (!valid_data_mode(packet.data_mode(), packet.bps())) {
        throw std::out_of_range("sdds_parser: invalid data_mode/bps combination");
    }

    // Validate and track sequence number
    auto seq_num = packet.seq_num();
    const bool is_parity = packet.is_parity();

    // Gap check runs for EVERY packet kind. Parity packets occupy the seq % 32 == 31 slots,
    // so a parity packet arriving when a data sequence number was expected still reveals the
    // loss of the data packets before it (checking after the parity return used to hide
    // that). A dedicated initialized flag (not `m_pkt_count != 0`) keeps the check armed
    // across the 65535 -> 0 wrap and after a packet with sequence number 0.
    if (m_seq_initialized) [[likely]] {
        auto expected_seq_num = static_cast<uint16_t>(m_pkt_count + 1);
        if (!is_parity && (expected_seq_num % 32) == 31) {
            ++expected_seq_num;  // data packets never occupy a parity slot
        }
        if (seq_num != expected_seq_num) [[unlikely]] {
            // seq_gap feeds the component's counter on EVERY gap; the warning is one-shot.
            result.seq_gap = true;
            if (!m_gap_warn) {
                m_gap_warn = true;
                result.warning = std::format("dropped pkt(s) expected={}, got={} (warning once; see the sequence-gap counter)",
                                             expected_seq_num, seq_num);
            }
        }
    }
    m_seq_initialized = true;
    m_pkt_count = seq_num;

    // Parity packets (pp_id set, on a seq % 32 == 31 slot) carry FEC parity bytes, NOT
    // samples: forwarding their payload as data corrupted the sample stream once per 32
    // packets on a parity-enabled stream. The sequence slot was consumed above; drop the
    // payload.
    if (is_parity) [[unlikely]] {
        result.should_send = false;
        return result;
    }

    // Inconsistent parity flags: one-shot warning (a persistently misbehaving sender
    // otherwise costs a format + warn line per packet).
    if (packet.pp_id() != ((seq_num % 32) == 31)) [[unlikely]] {
        if (!m_anomaly_warn) {
            m_anomaly_warn = true;
            result.warning = std::format("invalid SDDS packet received, pp_id={}, seq_num={} (warning once)",
                                         packet.pp_id(), seq_num);
        }
    }

    // Effective metadata this packet carries (extraction + the constant overrides). SDDS is
    // big-endian per spec (endianness overridable for non-conforming receivers) and the sample
    // type follows the packet's data mode (signed or unsigned);
    // center_frequency and bandwidth come only from overrides (otherwise carried from the
    // current metadata).
    const bool eff_complex = m_overrides.data_format.is_complex.value_or(packet.complex());
    const uint32_t eff_bps = packet.bps();
    // The published sample type must FOLLOW the packet's data mode. can_parse() accepts the
    // unsigned 8/16-bit modes, and publishing them as signed_integer (as this parser used to)
    // put every downstream sample interpretation off by half full-scale.
    const auto dm = packet.data_mode();
    const auto eff_type = (dm == DM_8BIT_UNSIGNED || dm == DM_16BIT_UNSIGNED)
                              ? composite::data_type::unsigned_integer
                              : composite::data_type::signed_integer;
    const double eff_sr = m_overrides.sample_rate.value_or(packet.sample_rate());
    const std::endian eff_endianness = m_ov_endianness.value_or(std::endian::big);

    // Rebuild metadata ONLY when it actually differs from the last published value
    // (current_metadata), or on the first packet after (re)activation. Steady state does no
    // copy, no map write, and no full compare — the component then reuses the shared instance.
    bool changed = !m_emitted
        || current_metadata.format.is_complex != eff_complex
        || current_metadata.format.type != eff_type
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
        result.metadata.format.type = eff_type;
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
