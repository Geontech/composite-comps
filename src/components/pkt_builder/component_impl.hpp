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

#include "component.hpp"

#include <algorithm>
#include <bit>
#include <cstring>

namespace {
    // VITA 49 constants
    constexpr uint32_t VITA49_HEADER_SIZE = 4;      // 32-bit header
    constexpr uint32_t STREAM_ID_SIZE = 4;          // 32-bit stream ID
    constexpr uint32_t CLASS_ID_SIZE = 8;           // 64-bit class ID
    constexpr uint32_t INTEGER_TS_SIZE = 4;         // 32-bit integer timestamp
    constexpr uint32_t FRACTIONAL_TS_SIZE = 8;      // 64-bit fractional timestamp
    constexpr uint32_t CIF0_SIZE = 4;               // 32-bit Context Indicator Field

    // VITA 49 packet type codes
    constexpr uint8_t PKT_TYPE_DATA = 0x00;
    constexpr uint8_t PKT_TYPE_CONTEXT = 0x04;

    // Helper to write 32-bit value in big-endian
    inline auto write_u32_be(uint8_t* dest, uint32_t value) -> void {
        if constexpr (std::endian::native == std::endian::little) {
            value = std::byteswap(value);
        }
        std::memcpy(dest, &value, sizeof(value));
    }

    // Helper to write 64-bit value in big-endian
    inline auto write_u64_be(uint8_t* dest, uint64_t value) -> void {
        if constexpr (std::endian::native == std::endian::little) {
            value = std::byteswap(value);
        }
        std::memcpy(dest, &value, sizeof(value));
    }

    // Convert composite data type to VITA 49 data item format code
    inline auto get_vita49_format_code(composite::data_type type) -> uint32_t {
        switch (type) {
            case composite::data_type::signed_integer:
                return 0x01;  // Signed fixed-point
            case composite::data_type::unsigned_integer:
                return 0x11;  // Unsigned fixed-point
            case composite::data_type::floating_point:
                return 0x08;  // Single-precision float
            default:
                return 0x01;
        }
    }

    // Calculate bytes per sample from format metadata
    inline auto bytes_per_sample(const composite::data_format& fmt) -> size_t {
        size_t bytes = fmt.bit_width / 8;
        return fmt.is_complex ? bytes * 2 : bytes;
    }

    // Byte-swap payload in-place from source endianness to big-endian (VITA 49 wire format).
    // Only swaps when the source data is not already big-endian and elements are > 8 bits.
    inline auto swap_payload_to_be(uint8_t* data, size_t len, const composite::data_format& fmt) -> void {
        if (fmt.endianness == std::endian::big || fmt.bit_width <= 8) {
            return;
        }

        if (fmt.bit_width == 16) {
            for (size_t i = 0; i + 1 < len; i += 2) {
                std::swap(data[i], data[i + 1]);
            }
        } else if (fmt.bit_width == 32) {
            for (size_t i = 0; i + 3 < len; i += 4) {
                std::swap(data[i], data[i + 3]);
                std::swap(data[i + 1], data[i + 2]);
            }
        } else if (fmt.bit_width == 64) {
            for (size_t i = 0; i + 7 < len; i += 8) {
                std::swap(data[i], data[i + 7]);
                std::swap(data[i + 1], data[i + 6]);
                std::swap(data[i + 2], data[i + 5]);
                std::swap(data[i + 3], data[i + 4]);
            }
        }
    }
} // anonymous namespace

pkt_builder::pkt_builder(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_port(&m_out_port);
    using enum composite::properties::config_type;

    // Packet configuration properties
    add_property("context_interval_ms", m_context_interval_ms, RUNTIME).units("ms");
    add_property("max_payload_size", m_max_payload_size, RUNTIME).units("samples");
    add_property("stream_id_key", m_stream_id_key, RUNTIME);
    add_property("default_stream_id", m_default_stream_id, RUNTIME);
    add_property("oui", m_oui, RUNTIME);
    add_property("packet_class_code", m_packet_class_code, RUNTIME);

    // Default metadata properties (used when upstream doesn't provide metadata)
    add_property("default_center_frequency", m_default_center_frequency, RUNTIME).units("Hz");
    add_property("default_bandwidth", m_default_bandwidth, RUNTIME).units("Hz");
    add_property("default_sample_rate", m_default_sample_rate, RUNTIME).units("Hz");

    // Optional feature flags
    add_property("include_class_id", m_include_class_id, RUNTIME);
    add_property("include_timestamp", m_include_timestamp, RUNTIME);
    add_property("warn_on_missing_metadata", m_warn_on_missing_metadata, RUNTIME);

    logger()->info("pkt_builder initialized (datatype determined from incoming metadata)");
}

auto pkt_builder::process() -> composite::retval {
    using enum composite::retval;

    // Get input data (as raw bytes with format metadata)
    auto [data, timestamp, metadata_opt] = m_in_port.get_data();
    if (!data) {
        return NORMAL;
    }

    // Get stream ID first (using metadata if available, or default)
    auto temp_metadata = metadata_opt.value_or(composite::metadata{});
    auto stream_id = get_stream_id(temp_metadata);
    auto& state = get_or_create_stream_state(stream_id);

    // Use new metadata if available, otherwise use last known metadata for this stream
    composite::metadata metadata;
    if (metadata_opt.has_value()) {
        metadata = metadata_opt.value();
    } else if (!state.first_packet) {
        // Use previously latched metadata for this stream
        metadata = state.last_metadata;
    }

    // Apply default RF values (format comes from upstream metadata)
    apply_defaults(metadata);

    // Validate metadata and warn if necessary
    validate_metadata(state, metadata);

    // Check if we should send a context packet
    if (should_send_context(state, metadata)) {
        auto context_vec = build_context_packet(state, metadata);
        auto context_buf = composite::immutable_buffer<uint8_t>(std::move(context_vec));
        // Forward input metadata to output for downstream routing (e.g., udp_sink)
        m_out_port.send_metadata(metadata);
        m_out_port.send_data(std::move(context_buf), timestamp);
        state.last_context_time = std::chrono::steady_clock::now();
        state.context_packet_count = (state.context_packet_count + 1) % 16;
    }

    // Calculate bytes per sample from format metadata
    size_t sample_size = bytes_per_sample(metadata.format);
    if (sample_size == 0) {
        logger()->warn("pkt_builder: invalid format metadata (bytes_per_sample=0), skipping packet");
        return NORMAL;
    }

    // max_payload_size is in samples, convert to bytes for chunking
    size_t max_chunk_bytes = m_max_payload_size * sample_size;

    // Build and send data packet(s)
    // Split large payloads into multiple packets if needed
    size_t offset = 0;
    while (offset < data.size()) {
        auto chunk_bytes = std::min(max_chunk_bytes, data.size() - offset);
        auto chunk = data.slice(offset, chunk_bytes);

        auto data_vec = build_data_packet(state, chunk, timestamp, metadata.format);
        auto data_buf = composite::immutable_buffer<uint8_t>(std::move(data_vec));
        // Forward input metadata to output for downstream routing (e.g., udp_sink)
        m_out_port.send_metadata(metadata);
        m_out_port.send_data(std::move(data_buf), timestamp);

        state.data_packet_count = (state.data_packet_count + 1) % 16;

        offset += chunk_bytes;
    }

    // Update stream state
    state.last_metadata = metadata;
    state.first_packet = false;

    return NORMAL;
}

auto pkt_builder::get_stream_id(const composite::metadata& metadata) -> uint32_t {
    if (auto it = metadata.annotations.find(m_stream_id_key); it != metadata.annotations.end()) {
        try {
            return static_cast<uint32_t>(std::stoul(it->second));
        } catch (const std::exception& e) {
            logger()->warn("Invalid stream_id value in metadata: '{}', using default ({})", it->second, e.what());
        }
    }
    return m_default_stream_id;
}

auto pkt_builder::get_or_create_stream_state(uint32_t stream_id) -> stream_state& {
    if (auto it = m_stream_states.find(stream_id); it != m_stream_states.end()) {
        return it->second;
    }

    // Create new stream state
    auto& state = m_stream_states[stream_id];
    state.stream_id = stream_id;
    logger()->info("Created new stream state for stream_id={}", stream_id);
    return state;
}

auto pkt_builder::apply_defaults(composite::metadata& metadata) -> void {
    // Format comes from upstream metadata (e.g., sigmf_source)
    // Only apply default RF parameters if not set (zero indicates not set)
    if (metadata.center_frequency == 0.0 && m_default_center_frequency != 0.0) {
        metadata.center_frequency = m_default_center_frequency;
    }
    if (metadata.bandwidth == 0.0 && m_default_bandwidth != 0.0) {
        metadata.bandwidth = m_default_bandwidth;
    }
    if (metadata.sample_rate == 0.0 && m_default_sample_rate != 0.0) {
        metadata.sample_rate = m_default_sample_rate;
    }
}

auto pkt_builder::validate_metadata(stream_state& state, const composite::metadata& metadata) -> void {
    if (!m_warn_on_missing_metadata || state.warned_missing_metadata) {
        return;
    }

    bool has_warnings = false;
    std::string warnings;

    if (metadata.sample_rate == 0.0) {
        warnings += "sample_rate=0 ";
        has_warnings = true;
    }
    if (metadata.bandwidth == 0.0) {
        warnings += "bandwidth=0 ";
        has_warnings = true;
    }
    if (metadata.center_frequency == 0.0) {
        warnings += "center_frequency=0 ";
        has_warnings = true;
    }
    if (metadata.format.bit_width == 0) {
        warnings += "format.bit_width=0 ";
        has_warnings = true;
    }

    if (has_warnings) {
        logger()->warn("stream_id={}: Context packet will have incomplete metadata: {}. "
                       "Set default_* properties or provide metadata from upstream.",
                       state.stream_id, warnings);
        state.warned_missing_metadata = true;
    }
}

auto pkt_builder::metadata_changed(const composite::metadata& current, const composite::metadata& previous) -> bool {
    return current.center_frequency != previous.center_frequency ||
           current.bandwidth != previous.bandwidth ||
           current.sample_rate != previous.sample_rate ||
           current.format.is_complex != previous.format.is_complex ||
           current.format.type != previous.format.type ||
           current.format.bit_width != previous.format.bit_width ||
           current.format.endianness != previous.format.endianness;
}

auto pkt_builder::should_send_context(stream_state& state, const composite::metadata& metadata) -> bool {
    // Always send on first packet
    if (state.first_packet) {
        return true;
    }

    // Send if metadata changed
    if (metadata_changed(metadata, state.last_metadata)) {
        logger()->debug("Metadata changed for stream_id={}, sending context packet", state.stream_id);
        return true;
    }

    // Send periodically based on time interval
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - state.last_context_time);
    if (elapsed.count() >= m_context_interval_ms) {
        logger()->trace("Periodic context packet for stream_id={} ({}ms elapsed)", state.stream_id, elapsed.count());
        return true;
    }

    return false;
}

auto pkt_builder::build_context_packet(const stream_state& state, const composite::metadata& metadata) -> std::shared_ptr<std::vector<uint8_t>> {
    // Calculate packet size
    size_t packet_size = VITA49_HEADER_SIZE;
    packet_size += STREAM_ID_SIZE;
    if (m_include_class_id) {
        packet_size += CLASS_ID_SIZE;
    }
    if (m_include_timestamp) {
        packet_size += INTEGER_TS_SIZE + FRACTIONAL_TS_SIZE;
    }

    // Context fields size (CIF0 + field values)
    constexpr size_t CONTEXT_FIELDS_SIZE =
        CIF0_SIZE +                    // CIF0 field
        8 +                            // RF Reference Frequency (64-bit)
        8 +                            // Bandwidth (64-bit)
        8 +                            // Sample Rate (64-bit)
        8;                             // Data Packet Payload Format (64-bit)

    packet_size += CONTEXT_FIELDS_SIZE;

    // Round up to 32-bit word boundary
    packet_size = (packet_size + 3) & ~3;

    // Create buffer
    auto vec = std::make_shared<std::vector<uint8_t>>(packet_size);
    auto* dest = vec->data();
    size_t offset = 0;

    // Write header
    uint16_t packet_size_words = static_cast<uint16_t>(packet_size / 4);
    offset += write_header(dest + offset, true, true, m_include_class_id, m_include_timestamp,
                           packet_size_words, state.context_packet_count);

    // Write stream ID
    offset += write_stream_id(dest + offset, state.stream_id);

    // Write class ID (if enabled)
    if (m_include_class_id) {
        offset += write_class_id(dest + offset);
    }

    // Write timestamp (if enabled)
    if (m_include_timestamp) {
        composite::timestamp ts{0, 0};
        offset += write_timestamp(dest + offset, ts);
    }

    // Write context fields
    offset += write_context_fields(dest + offset, metadata);

    return vec;
}

auto pkt_builder::build_data_packet(stream_state& state, const composite::immutable_buffer<std::byte>& payload,
                                        const composite::timestamp& ts, const composite::data_format& fmt) -> std::shared_ptr<std::vector<uint8_t>> {
    // Payload is already in bytes (raw data from upstream)
    size_t payload_bytes = payload.size();
    size_t packet_size = VITA49_HEADER_SIZE;
    packet_size += STREAM_ID_SIZE;
    if (m_include_class_id) {
        packet_size += CLASS_ID_SIZE;
    }
    if (m_include_timestamp) {
        packet_size += INTEGER_TS_SIZE + FRACTIONAL_TS_SIZE;
    }
    packet_size += payload_bytes;

    // Round up to 32-bit word boundary
    packet_size = (packet_size + 3) & ~3;

    // Create buffer
    auto vec = std::make_shared<std::vector<uint8_t>>(packet_size);
    auto* dest = vec->data();
    size_t offset = 0;

    // Write header
    uint16_t packet_size_words = static_cast<uint16_t>(packet_size / 4);
    offset += write_header(dest + offset, false, true, m_include_class_id, m_include_timestamp,
                           packet_size_words, state.data_packet_count);

    // Write stream ID
    offset += write_stream_id(dest + offset, state.stream_id);

    // Write class ID (if enabled)
    if (m_include_class_id) {
        offset += write_class_id(dest + offset);
    }

    // Write timestamp (if enabled)
    if (m_include_timestamp) {
        offset += write_timestamp(dest + offset, ts);
    }

    // Write payload and convert to big-endian (VITA 49 wire format)
    std::memcpy(dest + offset, payload.data(), payload_bytes);
    swap_payload_to_be(dest + offset, payload_bytes, fmt);
    offset += payload_bytes;

    // Zero-pad to word boundary if needed
    while (offset < packet_size) {
        dest[offset++] = 0;
    }

    return vec;
}

auto pkt_builder::write_header(uint8_t* dest, bool is_context, bool has_stream_id, bool has_class_id,
                                   bool has_timestamp, uint16_t packet_size_words, uint16_t packet_count) -> size_t {
    uint32_t header = 0;

    // Packet type (bits 31-28)
    uint8_t pkt_type;
    if (is_context) {
        pkt_type = PKT_TYPE_CONTEXT;  // 0x4 - context always has stream ID
    } else {
        pkt_type = has_stream_id ? 0x01 : PKT_TYPE_DATA;
    }
    header |= static_cast<uint32_t>(pkt_type) << 28;

    // Class ID present (bit 27)
    if (has_class_id) {
        header |= (1U << 27);
    }

    // TSM (Time Stamp Mode) (bit 24) - precise timestamp
    if (has_timestamp) {
        header |= (1U << 24);
    }

    // TSI (Integer timestamp) (bits 23-22) - UTC
    if (has_timestamp) {
        header |= (0x01U << 22);
    }

    // TSF (Fractional timestamp) (bits 21-20) - Real-time picoseconds
    if (has_timestamp) {
        header |= (0x02U << 20);
    }

    // Packet count (bits 19-16)
    header |= (static_cast<uint32_t>(packet_count & 0x0F) << 16);

    // Packet size in words (bits 15-0)
    header |= (packet_size_words & 0xFFFF);

    write_u32_be(dest, header);
    return VITA49_HEADER_SIZE;
}

auto pkt_builder::write_stream_id(uint8_t* dest, uint32_t stream_id) -> size_t {
    write_u32_be(dest, stream_id);
    return STREAM_ID_SIZE;
}

auto pkt_builder::write_class_id(uint8_t* dest) -> size_t {
    // Write OUI (upper 24 bits of first word) + reserved (lower 8 bits)
    uint32_t class_id_upper = (m_oui << 8);
    write_u32_be(dest, class_id_upper);

    // Write ICC (16 bits) + PCC (16 bits)
    uint32_t class_id_lower = (static_cast<uint32_t>(0) << 16) | m_packet_class_code;
    write_u32_be(dest + 4, class_id_lower);

    return CLASS_ID_SIZE;
}

auto pkt_builder::write_timestamp(uint8_t* dest, const composite::timestamp& ts) -> size_t {
    // Integer timestamp (seconds since epoch)
    write_u32_be(dest, static_cast<uint32_t>(ts.seconds));

    // Fractional timestamp (picoseconds)
    write_u64_be(dest + INTEGER_TS_SIZE, ts.picoseconds);

    return INTEGER_TS_SIZE + FRACTIONAL_TS_SIZE;
}

auto pkt_builder::write_context_fields(uint8_t* dest, const composite::metadata& metadata) -> size_t {
    size_t offset = 0;

    // CIF0 (Context Indicator Field 0)
    uint32_t cif0 = 0;
    cif0 |= (1U << 31);  // Bandwidth present
    cif0 |= (1U << 27);  // RF Reference Frequency present
    cif0 |= (1U << 23);  // Sample Rate present
    cif0 |= (1U << 7);   // Data Packet Payload Format present

    write_u32_be(dest + offset, cif0);
    offset += CIF0_SIZE;

    // Bandwidth (64-bit, fixed-point in Hz with radix point at bit 20)
    auto bandwidth_fixed = static_cast<uint64_t>(metadata.bandwidth * (1ULL << 20));
    write_u64_be(dest + offset, bandwidth_fixed);
    offset += 8;

    // RF Reference Frequency (64-bit, fixed-point in Hz with radix point at bit 20)
    auto rf_freq_fixed = static_cast<uint64_t>(metadata.center_frequency * (1ULL << 20));
    write_u64_be(dest + offset, rf_freq_fixed);
    offset += 8;

    // Sample Rate (64-bit, fixed-point in Hz with radix point at bit 20)
    auto sample_rate_fixed = static_cast<uint64_t>(metadata.sample_rate * (1ULL << 20));
    write_u64_be(dest + offset, sample_rate_fixed);
    offset += 8;

    // Data Packet Payload Format (64-bit)
    uint64_t format = 0;

    // Real/Complex (bit 63)
    if (metadata.format.is_complex) {
        format |= (1ULL << 63);
    }

    // Data Item Format (bits 62-56)
    auto format_code = get_vita49_format_code(metadata.format.type);
    format |= (static_cast<uint64_t>(format_code) << 56);

    // Item Packing Field Size (bits 37-32) - sample size in bits minus 1
    format |= (static_cast<uint64_t>(metadata.format.bit_width - 1) << 32);

    // Data Item Size (bits 20-16) - sample size in bits minus 1
    format |= (static_cast<uint64_t>(metadata.format.bit_width - 1) << 16);

    write_u64_be(dest + offset, format);
    offset += 8;

    return offset;
}