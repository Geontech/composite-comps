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
#include <type_traits>
#include <utility>

namespace parsers {

namespace {
    // VITA 49 Data Item Format ranges (per spec section 9.5.7).
    //
    // Typed to DataItemFormat's underlying type rather than a fixed uint32_t: these are
    // compared against std::to_underlying(format->data_item_format()), and hardcoding an
    // unsigned type made both comparisons signed/unsigned mismatches (-Wsign-compare, the
    // only two warnings in the fleet once -Wall -Wextra -Wpedantic actually reached it).
    // Deriving the type keeps that true if vrtgen ever changes the enum's base.
    using format_code_t = std::underlying_type_t<vrtgen::packing::DataItemFormat>;
    constexpr format_code_t MAX_SIGNED_FORMAT = 0x07;    // 0x00-0x07 signed integer
    constexpr format_code_t MIN_UNSIGNED_FORMAT = 0x10;  // 0x10-0x17 unsigned integer
                                                         // 0x08-0x0F floating point
} // anonymous namespace

vita49_parser::vita49_parser(const struct_props::signal_overrides& overrides,
                             std::string_view transport_annotation)
    : m_overrides(overrides), m_transport(transport_annotation) {
    // Resolve the string-valued overrides to enums here, once, off the per-packet path.
    // parse() then applies them with a plain branch + assign instead of re-parsing these
    // strings (== "signed_integer", == "big", ...) on every packet.
    if (!m_overrides.data_format.type.empty()) {
        if (m_overrides.data_format.type == "signed_integer") {
            m_ov_type = composite::data_type::signed_integer;
        } else if (m_overrides.data_format.type == "unsigned_integer") {
            m_ov_type = composite::data_type::unsigned_integer;
        } else if (m_overrides.data_format.type == "floating_point") {
            m_ov_type = composite::data_type::floating_point;
        }
    }
    if (!m_overrides.data_format.endianness.empty()) {
        if (m_overrides.data_format.endianness == "big") {
            m_ov_endianness = std::endian::big;
        } else if (m_overrides.data_format.endianness == "little") {
            m_ov_endianness = std::endian::little;
        }
    }
}

auto vita49_parser::can_parse(const composite::immutable_buffer<uint8_t>& data) const -> bool {
    // Minimum VITA 49 packet size (header only)
    if (data.size() < 4) {
        return false;
    }

    // Overlay and check packet type
    auto packet = overlay::v49(std::span{data.data(), data.size()});
    // Deliberately NOT is_ext_data(): parse() has no extension-data handling, so claiming those
    // packets here made this parser win detection for a stream it cannot decode. Leaving them
    // unclaimed lets the component report an unknown protocol -- and lets a downstream parser that
    // DOES implement the extension format claim them (see parsers/parser_table.hpp).
    return packet.is_data() || packet.is_context();
}

auto vita49_parser::parse(
    const composite::immutable_buffer<uint8_t>& data,
    const composite::metadata& current_metadata
) -> parse_result {
    parse_result result;

    // Apply the (constant) signal overrides onto a metadata value. Called only when we
    // (re)build metadata, never on the steady-state data path. The string-valued overrides
    // (type, endianness) were resolved to enums once at construction (m_ov_type/m_ov_endianness).
    auto apply_overrides = [this](composite::metadata& m) {
        if (m_overrides.data_format.is_complex.has_value()) {
            m.format.is_complex = *m_overrides.data_format.is_complex;
        }
        if (m_ov_type.has_value()) { m.format.type = *m_ov_type; }
        if (m_overrides.data_format.bit_width > 0) { m.format.bit_width = m_overrides.data_format.bit_width; }
        if (m_ov_endianness.has_value()) { m.format.endianness = *m_ov_endianness; }
        if (m_overrides.center_frequency.has_value()) { m.center_frequency = *m_overrides.center_frequency; }
        if (m_overrides.bandwidth.has_value()) { m.bandwidth = *m_overrides.bandwidth; }
        if (m_overrides.sample_rate.has_value()) { m.sample_rate = *m_overrides.sample_rate; }
    };

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

        // Data packets carry the current signal metadata unchanged (format / center_frequency
        // / bandwidth / sample_rate come from context packets). Republish only on the first
        // packet after (re)activation; steady-state data packets do no metadata work here.
        if (!m_emitted) [[unlikely]] {
            result.metadata = current_metadata;
            apply_overrides(result.metadata);
            result.metadata.annotations["protocol"] = m_transport;
            result.metadata_changed = true;
            m_emitted = true;
        }
    } else if (packet.is_context()) {
        // Build the metadata this context packet implies, then publish it only if it actually
        // differs from the current value (context packets are rare, so a full compare is fine).
        auto candidate = current_metadata;
        if (auto format = packet.signal_data_format()) {
            candidate.format.is_complex = format->real_complex_type() != vrtgen::packing::DataSampleType::REAL;

            // Determine data type from VITA 49 format encoding
            auto format_code = std::to_underlying(format->data_item_format());
            if (format_code <= MAX_SIGNED_FORMAT) {
                candidate.format.type = composite::data_type::signed_integer;
            } else if (format_code >= MIN_UNSIGNED_FORMAT) {
                candidate.format.type = composite::data_type::unsigned_integer;
            } else {
                candidate.format.type = composite::data_type::floating_point;
            }
            candidate.format.bit_width = format->data_item_size();
            candidate.format.endianness = packet.endianness();
        }

        candidate.center_frequency = packet.rf_frequency().value_or(0);
        candidate.bandwidth = packet.bandwidth().value_or(0);
        candidate.sample_rate = packet.sample_rate().value_or(0);
        apply_overrides(candidate);
        candidate.annotations["protocol"] = m_transport;

        if (!m_emitted || candidate != current_metadata) {
            result.metadata = std::move(candidate);
            result.metadata_changed = true;
            m_emitted = true;
        }

        result.should_send = false;  // Context packets don't carry data
    } else {
        // Neither a data nor a context packet. Reachable even with can_parse() narrowed, because
        // protocol lock-in keeps calling parse() for every packet on the stream once vita49 is the
        // active parser -- so an extension-data (or command) packet can still arrive here.
        //
        // Previously this fell through to a default-constructed parse_result, whose should_send
        // defaults to TRUE: the component forwarded an EMPTY payload with a zero timestamp
        // downstream, silently, uncounted, as though it were valid data. Drop it and say so once.
        result.should_send = false;
        if (!m_ext_warn) {
            result.warning = std::format(
                "unsupported VITA 49 packet type {} (extension/command); dropping these packets",
                static_cast<unsigned>(std::to_underlying(packet.header().packet_type())));
            m_ext_warn = true;
        }
    }

    // Adjust fractional timestamp if in sample count mode. Reads the effective current sample
    // rate (result.metadata is only populated when metadata changed on this packet).
    if (is_tsf_sc) {
        const double eff_sample_rate = m_overrides.sample_rate.value_or(current_metadata.sample_rate);
        if (eff_sample_rate == 0.0) {
            if (!m_tsf_warn) {
                result.warning = "unable to set fractional timestamp: unknown sample rate in SAMPLE_COUNT mode; dropping data until sample rate discovered";
                m_tsf_warn = true;
            }
            result.should_send = false;
        } else {
            // Convert sample count to picoseconds: samples / sample_rate * 1e12
            auto samples = static_cast<double>(result.timestamp.picoseconds);
            auto picoseconds_per_sample = 1e12 / eff_sample_rate;
            auto picoseconds = samples * picoseconds_per_sample;

            // Clamp to uint64_t range to prevent overflow
            constexpr auto max_uint64 = static_cast<double>(UINT64_MAX);
            result.timestamp.picoseconds = static_cast<uint64_t>(std::min(picoseconds, max_uint64));
        }
    }

    return result;
}

} // namespace parsers
