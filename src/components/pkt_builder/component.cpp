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

#include "component.hpp"

#include <composite/core/register.hpp>

#include <vrtgen/vrtgen.hpp>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cstring>
#include <optional>

namespace {

// Sizes of the raw prologue words this builder writes around the vrtgen-packed pieces.
constexpr std::size_t STREAM_ID_SIZE = 4;     // 32-bit stream ID
constexpr std::size_t INTEGER_TS_SIZE = 4;    // 32-bit integer timestamp
constexpr std::size_t FRACTIONAL_TS_SIZE = 8; // 64-bit fractional timestamp

inline auto write_u32_be(uint8_t* dest, uint32_t value) -> void {
    if constexpr (std::endian::native == std::endian::little) {
        value = std::byteswap(value);
    }
    std::memcpy(dest, &value, sizeof(value));
}

inline auto write_u64_be(uint8_t* dest, uint64_t value) -> void {
    if constexpr (std::endian::native == std::endian::little) {
        value = std::byteswap(value);
    }
    std::memcpy(dest, &value, sizeof(value));
}

// Map the composite sample type onto the VITA 49.2 data item format the parser maps back.
inline auto to_data_item_format(composite::data_type type) -> vrtgen::packing::DataItemFormat {
    using enum vrtgen::packing::DataItemFormat;
    switch (type) {
    case composite::data_type::signed_integer:
        return SIGNED_FIXED;
    case composite::data_type::unsigned_integer:
        return UNSIGNED_FIXED;
    case composite::data_type::floating_point:
        return IEEE754_SINGLE_PRECISION;
    }
    return SIGNED_FIXED;
}

// Calculate bytes per sample from format metadata
inline auto bytes_per_sample(const composite::data_format& fmt) -> std::size_t {
    std::size_t bytes = fmt.bit_width / 8;
    return fmt.is_complex ? bytes * 2 : bytes;
}

// Byte-swap payload in-place from source endianness to big-endian (VITA 49 wire format).
// Only swaps when the source data is not already big-endian and elements are > 8 bits.
inline auto swap_payload_to_be(uint8_t* data, std::size_t len, const composite::data_format& fmt) -> void {
    if (fmt.endianness == std::endian::big || fmt.bit_width <= 8) {
        return;
    }

    if (fmt.bit_width == 16) {
        for (std::size_t i = 0; i + 1 < len; i += 2) {
            std::swap(data[i], data[i + 1]);
        }
    } else if (fmt.bit_width == 32) {
        for (std::size_t i = 0; i + 3 < len; i += 4) {
            std::swap(data[i], data[i + 3]);
            std::swap(data[i + 1], data[i + 2]);
        }
    } else if (fmt.bit_width == 64) {
        for (std::size_t i = 0; i + 7 < len; i += 8) {
            std::swap(data[i], data[i + 7]);
            std::swap(data[i + 1], data[i + 6]);
            std::swap(data[i + 2], data[i + 5]);
            std::swap(data[i + 3], data[i + 4]);
        }
    }
}

// Read an annotation as an unsigned integer: typed integers directly, strings parsed.
auto annotation_as_uint(const composite::annotation_value& v) -> std::optional<uint64_t> {
    if (v.holds<std::int64_t>()) {
        const auto i = v.get<std::int64_t>();
        return i >= 0 ? std::optional<uint64_t>{static_cast<uint64_t>(i)} : std::nullopt;
    }
    const auto s = v.to_string();
    uint64_t out{};
    const auto* end = s.data() + s.size();
    if (auto [p, ec] = std::from_chars(s.data(), end, out); ec == std::errc{} && p == end) {
        return out;
    }
    return std::nullopt;
}

} // namespace

pkt_builder::pkt_builder(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_port(&m_out_port);
    using enum composite::properties::config_type;

    // Packet configuration properties
    add_property("context_interval_ms", m_context_interval_ms, RUNTIME).units("ms");
    add_property("max_payload_size", m_max_payload_size, RUNTIME)
        .units("samples")
        .validate([](const uint32_t& v) { return v > 0; });
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
}

auto pkt_builder::property_change_handler(const composite::properties::json& diff) -> void {
    (void)diff;
    // Defaults / keys / packet options may have changed: drop the per-stream effective-metadata
    // caches so the next packet on each stream rebuilds (and re-sends context) with the new
    // configuration, even when the incoming metadata instance is unchanged.
    for (auto& [_, state] : m_stream_states) {
        state.last_in_meta = nullptr;
        state.out_meta = nullptr;
    }
}

auto pkt_builder::process() -> composite::retval {
    using enum composite::retval;

    auto pkt = m_in_port.try_get();
    if (!pkt) {
        return NOOP;
    }
    auto& [data, ts, meta] = *pkt;

    auto stream_id = meta ? get_stream_id(*meta) : m_default_stream_id;
    auto& state = get_or_create_stream_state(stream_id);

    // Rebuild the effective metadata (defaults applied) only when the incoming shared
    // instance changed. When the values come out equal, the OLD instance is kept, so
    // downstream consumers keep their pointer-identity fast path.
    bool signal_changed = false;
    if (state.out_meta == nullptr || meta != state.last_in_meta) {
        auto effective = meta        ? *meta
                         : state.out_meta ? *state.out_meta
                                          : composite::metadata{};
        apply_defaults(effective);
        validate_metadata(state, effective);
        signal_changed = state.out_meta == nullptr || effective != *state.out_meta;
        state.last_in_meta = meta;
        if (signal_changed) {
            state.out_meta = composite::make_metadata(std::move(effective));
        }
    }
    const auto& metadata = *state.out_meta;

    // Send a context packet on the first packet, on a signal change, or periodically.
    auto now = std::chrono::steady_clock::now();
    const bool interval_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - state.last_context_time).count() >=
        m_context_interval_ms;
    if (state.first_packet || signal_changed || interval_elapsed) {
        auto context_vec = build_context_packet(state, metadata);
        m_out_port.send_data(composite::immutable_buffer<uint8_t>(std::move(context_vec)), ts,
                             state.out_meta);
        state.last_context_time = now;
        state.context_packet_count = (state.context_packet_count + 1) % 16;
    }

    // Calculate bytes per sample from format metadata
    std::size_t sample_size = bytes_per_sample(metadata.format);
    if (sample_size == 0) {
        logger()->warn("pkt_builder: invalid format metadata (bytes_per_sample=0), skipping packet");
        state.first_packet = false;
        return NORMAL;
    }

    // max_payload_size is in samples; chunk large inputs into multiple packets.
    std::size_t max_chunk_bytes = static_cast<std::size_t>(m_max_payload_size) * sample_size;
    std::size_t offset = 0;
    while (offset < data.size()) {
        auto chunk_bytes = std::min(max_chunk_bytes, data.size() - offset);
        auto data_vec = build_data_packet(state, std::span{data.data() + offset, chunk_bytes}, ts,
                                          metadata.format);
        m_out_port.send_data(composite::immutable_buffer<uint8_t>(std::move(data_vec)), ts,
                             state.out_meta);
        state.data_packet_count = (state.data_packet_count + 1) % 16;
        offset += chunk_bytes;
    }

    state.first_packet = false;
    return NORMAL;
}

auto pkt_builder::get_stream_id(const composite::metadata& metadata) -> uint32_t {
    if (auto it = metadata.annotations.find(m_stream_id_key); it != metadata.annotations.end()) {
        if (auto v = annotation_as_uint(it->second)) {
            return static_cast<uint32_t>(*v);
        }
        logger()->warn("pkt_builder: invalid stream_id value in metadata: '{}', using default",
                       it->second.to_string());
    }
    return m_default_stream_id;
}

auto pkt_builder::get_or_create_stream_state(uint32_t stream_id) -> stream_state& {
    if (auto it = m_stream_states.find(stream_id); it != m_stream_states.end()) {
        return it->second;
    }
    auto& state = m_stream_states[stream_id];
    state.stream_id = stream_id;
    logger()->info("pkt_builder: created stream state for stream_id={}", stream_id);
    return state;
}

auto pkt_builder::apply_defaults(composite::metadata& metadata) -> void {
    // Format comes from upstream metadata; only fill RF parameters left unset (zero).
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

    std::string warnings;
    if (metadata.sample_rate == 0.0) {
        warnings += "sample_rate=0 ";
    }
    if (metadata.bandwidth == 0.0) {
        warnings += "bandwidth=0 ";
    }
    if (metadata.center_frequency == 0.0) {
        warnings += "center_frequency=0 ";
    }
    if (metadata.format.bit_width == 0) {
        warnings += "format.bit_width=0 ";
    }

    if (!warnings.empty()) {
        logger()->warn("pkt_builder: stream_id={}: context packet will have incomplete metadata: {}. "
                       "Set default_* properties or provide metadata from upstream.",
                       state.stream_id, warnings);
        state.warned_missing_metadata = true;
    }
}

auto pkt_builder::build_context_packet(stream_state& state, const composite::metadata& metadata)
    -> std::shared_ptr<std::vector<uint8_t>> {
    using namespace vrtgen::packing;

    Header hdr;
    hdr.packet_type(PacketType::CONTEXT);
    hdr.class_id_enable(m_include_class_id);
    if (m_include_timestamp) {
        hdr.tsi(TSI::UTC);
        hdr.tsf(TSF::REAL_TIME);
    }
    hdr.packet_count(static_cast<uint8_t>(state.context_packet_count));

    CIF0 cif;
    cif.bandwidth(true);
    cif.rf_ref_frequency(true);
    cif.sample_rate(true);
    cif.signal_data_format(true);

    PayloadFormat pf;
    pf.real_complex_type(metadata.format.is_complex ? DataSampleType::COMPLEX_CARTESIAN
                                                    : DataSampleType::REAL);
    pf.data_item_format(to_data_item_format(metadata.format.type));
    if (metadata.format.bit_width > 0) {
        pf.data_item_size(static_cast<uint8_t>(metadata.format.bit_width));
        pf.item_packing_field_size(static_cast<uint8_t>(metadata.format.bit_width));
    }

    ClassIdentifier cid;
    cid.oui(m_oui);
    cid.packet_code(m_packet_class_code);

    // header | stream id | [class id] | [timestamps] | CIF0 | bandwidth | rf | sample rate | payload fmt
    std::size_t packet_size = hdr.size() + STREAM_ID_SIZE;
    if (m_include_class_id) {
        packet_size += cid.size();
    }
    if (m_include_timestamp) {
        packet_size += INTEGER_TS_SIZE + FRACTIONAL_TS_SIZE;
    }
    packet_size += cif.size() + 8 + 8 + 8 + pf.size();
    hdr.packet_size(static_cast<uint16_t>(packet_size / 4));

    auto vec = std::make_shared<std::vector<uint8_t>>(packet_size);
    auto* dest = vec->data();
    std::size_t offset = 0;

    hdr.pack_into(dest + offset);
    offset += hdr.size();
    write_u32_be(dest + offset, state.stream_id);
    offset += STREAM_ID_SIZE;
    if (m_include_class_id) {
        cid.pack_into(dest + offset);
        offset += cid.size();
    }
    if (m_include_timestamp) {
        // Context timestamps convey "as of": zero here (data packets carry the sample time).
        write_u32_be(dest + offset, 0);
        offset += INTEGER_TS_SIZE;
        write_u64_be(dest + offset, 0);
        offset += FRACTIONAL_TS_SIZE;
    }
    cif.pack_into(dest + offset);
    offset += cif.size();
    // Fixed-point 44.20 values, in the descending-CIF0-bit order the fields appear on the wire.
    write_u64_be(dest + offset, static_cast<uint64_t>(vrtgen::fixed::to_int<44, 20>(metadata.bandwidth)));
    offset += 8;
    write_u64_be(dest + offset,
                 static_cast<uint64_t>(vrtgen::fixed::to_int<44, 20>(metadata.center_frequency)));
    offset += 8;
    write_u64_be(dest + offset, static_cast<uint64_t>(vrtgen::fixed::to_int<44, 20>(metadata.sample_rate)));
    offset += 8;
    pf.pack_into(dest + offset);

    return vec;
}

auto pkt_builder::build_data_packet(stream_state& state, std::span<const std::byte> payload,
                                    const composite::timestamp& ts, const composite::data_format& fmt)
    -> std::shared_ptr<std::vector<uint8_t>> {
    using namespace vrtgen::packing;

    Header hdr;
    hdr.packet_type(PacketType::SIGNAL_DATA_STREAM_ID);
    hdr.class_id_enable(m_include_class_id);
    if (m_include_timestamp) {
        hdr.tsi(TSI::UTC);
        hdr.tsf(TSF::REAL_TIME);
    }
    hdr.packet_count(static_cast<uint8_t>(state.data_packet_count));

    ClassIdentifier cid;
    cid.oui(m_oui);
    cid.packet_code(m_packet_class_code);

    std::size_t packet_size = hdr.size() + STREAM_ID_SIZE;
    if (m_include_class_id) {
        packet_size += cid.size();
    }
    if (m_include_timestamp) {
        packet_size += INTEGER_TS_SIZE + FRACTIONAL_TS_SIZE;
    }
    packet_size += payload.size();
    packet_size = (packet_size + 3) & ~std::size_t{3}; // round up to a 32-bit word boundary
    hdr.packet_size(static_cast<uint16_t>(packet_size / 4));

    auto vec = std::make_shared<std::vector<uint8_t>>(packet_size); // zero-filled: pad bytes stay 0
    auto* dest = vec->data();
    std::size_t offset = 0;

    hdr.pack_into(dest + offset);
    offset += hdr.size();
    write_u32_be(dest + offset, state.stream_id);
    offset += STREAM_ID_SIZE;
    if (m_include_class_id) {
        cid.pack_into(dest + offset);
        offset += cid.size();
    }
    if (m_include_timestamp) {
        write_u32_be(dest + offset, static_cast<uint32_t>(ts.seconds));
        offset += INTEGER_TS_SIZE;
        write_u64_be(dest + offset, ts.picoseconds);
        offset += FRACTIONAL_TS_SIZE;
    }
    std::memcpy(dest + offset, payload.data(), payload.size());
    swap_payload_to_be(dest + offset, payload.size(), fmt);

    return vec;
}

COMPOSITE_REGISTER_SIMPLE(pkt_builder)
