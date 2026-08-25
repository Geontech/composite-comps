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
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <optional>
#include <thread>
#include "simd_fmv.hpp"

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
inline auto to_data_item_format(composite::data_type type, uint32_t bit_width)
    -> vrtgen::packing::DataItemFormat {
    using enum vrtgen::packing::DataItemFormat;
    switch (type) {
    case composite::data_type::signed_integer:
        return SIGNED_FIXED;
    case composite::data_type::unsigned_integer:
        return UNSIGNED_FIXED;
    case composite::data_type::floating_point:
        if (bit_width == 16) { return IEEE754_HALF_PRECISION; }
        if (bit_width == 64) { return IEEE754_DOUBLE_PRECISION; }
        return IEEE754_SINGLE_PRECISION;
    }
    return SIGNED_FIXED;
}

// Calculate bytes per sample from format metadata
inline auto bytes_per_sample(const composite::data_format& fmt) -> std::size_t {
    std::size_t bytes = fmt.bit_width / 8;
    return fmt.is_complex ? bytes * 2 : bytes;
}

// FUSED copy + byte-swap of the payload into the packet (source endianness -> big-endian
// wire format): the memcpy-then-swap-in-place this replaces walked the payload twice. The
// pshufb control bytes repeat per 16-byte lane, so one 64-byte constant per element width
// serves the 128/256/512-bit loads as prefixes; SIMD selection is GCC native function
// multiversioning (simd_fmv.hpp), one resolved call per payload.
namespace swap_masks {
alignas(64) constexpr std::int8_t U16[64] = {
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
};
alignas(64) constexpr std::int8_t U32[64] = {
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
};
alignas(64) constexpr std::int8_t U64[64] = {
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
};
} // namespace swap_masks

template <typename U>
auto copy_swap_tail(uint8_t* dst, const uint8_t* src, std::size_t i, std::size_t len) -> void {
    for (; i + sizeof(U) <= len; i += sizeof(U)) {
        U v{};
        std::memcpy(&v, src + i, sizeof(U));
        v = std::byteswap(v);
        std::memcpy(dst + i, &v, sizeof(U));
    }
    // Trailing sub-element bytes (a non-element-aligned payload) copy unswapped; callers
    // validate the format so this is unreachable for conforming input.
    for (; i < len; ++i) {
        dst[i] = src[i];
    }
}

inline auto copy_swap_scalar(uint8_t* dst, const uint8_t* src, std::size_t len, std::size_t width_bytes) -> void {
    switch (width_bytes) {
    case 2: copy_swap_tail<uint16_t>(dst, src, 0, len); break;
    case 4: copy_swap_tail<uint32_t>(dst, src, 0, len); break;
    case 8: copy_swap_tail<uint64_t>(dst, src, 0, len); break;
    default: std::memcpy(dst, src, len); break;
    }
}

COMPS_FMV_DEFAULT
auto copy_swap(uint8_t* dst, const uint8_t* src, std::size_t len, std::size_t width_bytes,
               const std::int8_t* /*mask*/) -> void {
    copy_swap_scalar(dst, src, len, width_bytes);
}

#if COMPS_FMV_ENABLED
[[gnu::target("avx2")]]
auto copy_swap(uint8_t* dst, const uint8_t* src, std::size_t len, std::size_t width_bytes,
               const std::int8_t* mask) -> void {
    const auto m = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(mask));
    std::size_t i = 0;
    for (; i + 32 <= len; i += 32) {  // 32 bytes is a whole number of 2/4/8-byte elements
        const auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_shuffle_epi8(v, m));
    }
    switch (width_bytes) {
    case 2: copy_swap_tail<uint16_t>(dst, src, i, len); break;
    case 4: copy_swap_tail<uint32_t>(dst, src, i, len); break;
    default: copy_swap_tail<uint64_t>(dst, src, i, len); break;
    }
}

[[gnu::target("avx512f,avx512bw")]]
auto copy_swap(uint8_t* dst, const uint8_t* src, std::size_t len, std::size_t width_bytes,
               const std::int8_t* mask) -> void {
    const auto m = _mm512_loadu_si512(mask);
    std::size_t i = 0;
    for (; i + 64 <= len; i += 64) {
        const auto v = _mm512_loadu_si512(src + i);
        _mm512_storeu_si512(dst + i, _mm512_shuffle_epi8(v, m));
    }
    switch (width_bytes) {
    case 2: copy_swap_tail<uint16_t>(dst, src, i, len); break;
    case 4: copy_swap_tail<uint32_t>(dst, src, i, len); break;
    default: copy_swap_tail<uint64_t>(dst, src, i, len); break;
    }
}
#endif

/// Copy the payload into the packet, converting to big-endian wire order in the same pass.
inline auto copy_payload_to_be(uint8_t* dst, const std::byte* src, std::size_t len,
                               const composite::data_format& fmt) -> void {
    const auto* bytes = reinterpret_cast<const uint8_t*>(src);
    if (fmt.endianness == std::endian::big || fmt.bit_width <= 8) {
        std::memcpy(dst, bytes, len);
        return;
    }
    switch (fmt.bit_width) {
    case 16: copy_swap(dst, bytes, len, 2, swap_masks::U16); break;
    case 32: copy_swap(dst, bytes, len, 4, swap_masks::U32); break;
    case 64: copy_swap(dst, bytes, len, 8, swap_masks::U64); break;
    default: std::memcpy(dst, bytes, len); break;  // unreachable: process() validates widths
    }
}

// VITA context RF fields are 64-bit fixed-point with 20 fractional bits: 44 integer bits,
// so |v| < 2^43 Hz (~8.8 THz). vrtgen's to_int<44,20> casts double -> int64 with NO range
// or finiteness check — NaN/inf/oversized values are UNDEFINED BEHAVIOR at the cast, not
// just wrong wire values. Beyond UB, the fields have semantics: bandwidth cannot be
// negative, and a sample rate must be positive AND representable at 20 fractional bits
// (below 2^-20 Hz the wire value rounds to zero while the output metadata stayed nonzero —
// downstream would disagree with the wire). 0 = unset/invalid throughout.
constexpr double k_max_44_20 = 8.7e12;      // < 2^43, with margin for the +0.5 rounding
constexpr double k_min_rate = 1.0 / (1u << 20);  // one fixed-point LSB

inline auto sanitize_frequency(double v) -> double {  // signed: baseband offsets are legal
    return (std::isfinite(v) && std::abs(v) < k_max_44_20) ? v : 0.0;
}
inline auto sanitize_bandwidth(double v) -> double {
    return (std::isfinite(v) && v >= 0.0 && v < k_max_44_20) ? v : 0.0;
}
inline auto sanitize_sample_rate(double v) -> double {
    return (std::isfinite(v) && v >= k_min_rate && v < k_max_44_20) ? v : 0.0;
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
    m_packets_dropped = &create_counter(
        "pkt_builder.packets_dropped",
        "Input buffers dropped (unsupported sample format, or distinct-stream cap overflow)");
    m_bytes_dropped = &create_counter(
        "pkt_builder.bytes_dropped",
        "Trailing bytes dropped to keep packets whole-sample and 32-bit-word aligned");

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
    // stream_id_key / default_stream_id may have changed: the instance-keyed lookup cache
    // would otherwise keep routing an unchanged metadata instance under the old config.
    m_sid_cache_meta = nullptr;
}

auto pkt_builder::process() -> composite::retval {
    using enum composite::retval;

    // Resume an interrupted packetization first: a full downstream must PAUSE the chunk loop
    // (AWAIT_OUTPUT parks the worker on the reverse doorbell), never silently drop the rest
    // of the buffer at the output ring the way an unpaced send loop did.
    if (m_pending.has_value()) {
        // Whatever emit_pending() achieved, this call did (or attempted) work: return its
        // verdict rather than falling through to try_get(), whose empty-ring NOOP would
        // misreport a productive call as idle. Fresh input is ingested on the next call.
        return emit_pending();
    }

    auto pkt = m_in_port.try_get();
    if (!pkt) {
        return NOOP;
    }
    auto& [data, ts, meta] = *pkt;

    // Stream-id lookup cached by metadata instance: steady state (the same shared instance
    // on every packet) skips the annotation-map find + parse entirely. The cache is cleared
    // by property_change_handler (stream_id_key / default_stream_id may have changed).
    uint32_t stream_id{};
    if (meta == nullptr) {
        stream_id = m_default_stream_id;
    } else if (meta == m_sid_cache_meta) {
        stream_id = m_sid_cache_id;
    } else {
        stream_id = get_stream_id(*meta);
        m_sid_cache_meta = meta;
        m_sid_cache_id = stream_id;
    }
    auto* state = find_or_create_stream_state(stream_id);

    // Rebuild the effective metadata (defaults applied) only when the incoming shared
    // instance changed. When the values come out equal, the OLD instance is kept, so
    // downstream consumers keep their pointer-identity fast path. A BARE packet (per the
    // port contract, nullptr = no metadata) builds from the component defaults alone —
    // inheriting the stream's previous metadata would serialize unknown bytes under a
    // stale format declaration.
    bool signal_changed = false;
    if (state->out_meta == nullptr || meta != state->last_in_meta) {
        auto effective = meta ? *meta : composite::metadata{};
        apply_defaults(effective);
        // Sanitize the RF fields BEFORE they can reach the fixed-point casts (see
        // sanitize_rf_value: NaN/inf/out-of-range doubles are UB there, not just garbage).
        const auto bw = sanitize_bandwidth(effective.bandwidth);
        const auto cf = sanitize_frequency(effective.center_frequency);
        const auto sr = sanitize_sample_rate(effective.sample_rate);
        if ((bw != effective.bandwidth || cf != effective.center_frequency ||
             sr != effective.sample_rate) && !state->warned_invalid_rf) {
            state->warned_invalid_rf = true;
            logger()->warn("pkt_builder: stream_id={}: non-finite or out-of-range RF metadata "
                           "(bandwidth/center_frequency/sample_rate) reset to 0 (warning once per stream)",
                           state->stream_id);
        }
        effective.bandwidth = bw;
        effective.center_frequency = cf;
        effective.sample_rate = sr;
        validate_metadata(*state, effective);
        signal_changed = state->out_meta == nullptr || effective != *state->out_meta;
        state->last_in_meta = meta;
        if (signal_changed) {
            state->out_meta = composite::make_metadata(std::move(effective));
            // Residue was captured under the previous signal declaration; its format may no
            // longer apply. Drop + count rather than splice old-format bytes into new frames.
            if (!state->residue.empty()) {
                m_bytes_dropped->add(state->residue.size());
                state->residue.clear();
            }
        }
    }
    const auto& metadata = *state->out_meta;

    // Only formats this builder can correctly serialize big-endian: 8/16/32/64-bit integers
    // and 16/32/64-bit IEEE floats (there is no 8-bit IEEE format to declare). Anything else
    // would go out mislabeled or misframed: drop and count, BEFORE emitting a context packet
    // that would describe a stream carrying no data. first_packet stays set so the first
    // VALID packet still opens with context.
    const auto width = metadata.format.bit_width;
    const bool is_fp = metadata.format.type == composite::data_type::floating_point;
    const bool width_ok = is_fp ? (width == 16 || width == 32 || width == 64)
                                : (width == 8 || width == 16 || width == 32 || width == 64);
    const std::size_t sample_size = bytes_per_sample(metadata.format);
    if (sample_size == 0 || !width_ok) {
        m_packets_dropped->inc();
        if (!state->warned_bad_format) {
            state->warned_bad_format = true;
            logger()->warn("pkt_builder: stream_id={}: unsupported format (type={}, bit_width={}), dropping data "
                           "(counted in pkt_builder.packets_dropped; warning once per stream)",
                           state->stream_id, static_cast<int>(metadata.format.type), width);
        }
        return NORMAL;
    }

    // max_payload_size is in samples; chunk large inputs into multiple packets.
    // VITA 49's packet_size field is a 16-bit WORD count. The header write casts words to
    // uint16_t, so a chunk that would exceed 65535 words used to WRAP modulo 65536 and mis-frame
    // the stream for every downstream parser. Clamp the chunk (whole samples) so the built
    // packet always fits; an oversized max_payload_size then just chunks smaller.
    constexpr std::size_t MAX_VITA_PACKET_BYTES = std::size_t{0xFFFF} * 4;
    constexpr std::size_t VITA_HEADER_ALLOWANCE = 64; // hdr + stream id + class id + timestamps + pad, generously
    const std::size_t max_samples_per_packet =
        std::min<std::size_t>(m_max_payload_size, (MAX_VITA_PACKET_BYTES - VITA_HEADER_ALLOWANCE) / sample_size);
    if (max_samples_per_packet < m_max_payload_size && !m_payload_clamp_warned) {
        m_payload_clamp_warned = true;
        logger()->warn("pkt_builder: max_payload_size {} samples x {} bytes/sample exceeds the VITA-49 16-bit "
                       "packet_size limit; clamping to {} samples per packet",
                       m_max_payload_size, sample_size, max_samples_per_packet);
    }
    // Chunks must be WORD-multiples of whole samples: VITA payloads are 32-bit-word granular
    // and this builder uses no trailer pad-bit encoding, so a padded packet would fabricate
    // zero samples in every downstream parser. Round the standing chunk size down to a word
    // multiple (floored at one word's worth of samples); the final sub-word remainder of a
    // buffer is CARRIED into the next buffer's first packet (see emit_pending), so no valid
    // sample is ever dropped at a buffer boundary.
    std::size_t max_chunk_bytes = max_samples_per_packet * sample_size;
    if (sample_size < 4) {
        max_chunk_bytes = std::max<std::size_t>(max_chunk_bytes & ~std::size_t{3}, 4);
    }
    // Whole samples only: a trailing partial element would be copied unswapped and parsed as
    // data downstream. Dropped + counted (warn once per stream).
    const std::size_t payload_end = data.size() - (data.size() % sample_size);
    if (payload_end != data.size()) {
        m_bytes_dropped->add(data.size() - payload_end);
        if (!state->warned_unaligned) {
            state->warned_unaligned = true;
            logger()->warn("pkt_builder: stream_id={}: input not a whole number of {}-byte samples; "
                           "trailing bytes dropped (counted in pkt_builder.bytes_dropped)",
                           state->stream_id, sample_size);
        }
    }

    // Send a context packet on the first packet, on a signal change, or periodically.
    const auto now = std::chrono::steady_clock::now();
    const bool interval_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - state->last_context_time).count() >=
        m_context_interval_ms;

    m_pending.emplace();
    m_pending->data = std::move(data);
    m_pending->base_ts = ts;
    m_pending->stream_id = stream_id;
    m_pending->lead = std::move(state->residue);  // carried sub-word remainder, if any
    state->residue.clear();
    m_pending->payload_end = payload_end;
    m_pending->sample_size = sample_size;
    m_pending->max_chunk_bytes = max_chunk_bytes;
    m_pending->rate = metadata.sample_rate;  // sanitized above; 0 = no per-chunk advance
    m_pending->out_meta = state->out_meta;
    m_pending->fmt = metadata.format;
    if (state->first_packet || signal_changed || interval_elapsed) {
        m_pending->context = build_context_packet(*state, metadata, ts);
    }
    state->first_packet = false;

    return emit_pending();
}

auto pkt_builder::emit_pending() -> composite::retval {
    using enum composite::retval;
    auto& p = *m_pending;
    auto& state = m_stream_states.at(p.stream_id);  // created before the pending was staged

    const auto can_send = [this] {
        return !m_out_port.producer_is_connected() || m_out_port.producer_can_send();
    };

    if (p.context != nullptr) {
        if (!can_send()) {
            return AWAIT_OUTPUT;
        }
        m_out_port.send_data(composite::immutable_buffer<uint8_t>(std::move(p.context)), p.base_ts,
                             p.out_meta);
        p.context = nullptr;
        state.last_context_time = std::chrono::steady_clock::now();
        state.context_packet_count = (state.context_packet_count + 1) % 16;
    }

    // Bounded work per call: a writable output must not let one enormous input pin the
    // worker here (stop and property handling run between process() calls). NORMAL with the
    // pending retained means "progress made, call me again".
    std::size_t budget = EMIT_BUDGET_PER_CALL;

    for (;;) {
        const auto lead_len = p.lead.size();
        const auto remaining = p.payload_end - p.offset;
        const auto total = lead_len + remaining;
        if (total < 4) {
            break;  // sub-word remainder: carried as the stream's residue below
        }
        if (!can_send()) {
            return AWAIT_OUTPUT;
        }
        if (budget-- == 0) {
            return NORMAL;
        }
        // Word-multiple chunk over the logical stream (carried lead + this buffer).
        auto chunk_total = std::min(p.max_chunk_bytes, total) & ~std::size_t{3};
        const auto from_lead = std::min(lead_len, chunk_total);  // lead < 4 <= chunk_total
        const auto from_payload = chunk_total - from_lead;
        // Per-chunk timestamps derive from the TOTAL consumed-payload offset (no accumulated
        // rounding drift, trivially resumable), and advance whether or not wire timestamps
        // are enabled — the framework-level timestamp is not a wire-serialization option.
        // The carried lead predates this buffer's timestamp, so the first chunk keeps the
        // base timestamp (off by at most one sample period). Overflow-guarded against both
        // the increment and the base value.
        composite::timestamp chunk_ts = p.base_ts;
        if (p.rate > 0.0) {
            const auto ps = static_cast<double>(p.offset / p.sample_size) * 1e12 / p.rate;
            if (ps < 9.0e18) {
                const auto inc = static_cast<uint64_t>(ps + 0.5);
                if (chunk_ts.picoseconds <= std::numeric_limits<uint64_t>::max() - inc) {
                    chunk_ts.picoseconds += inc;
                    chunk_ts.normalize();
                }
            }
        }
        auto data_vec = build_data_packet(state, std::span{p.lead.data(), from_lead},
                                          std::span{p.data.data() + p.offset, from_payload},
                                          chunk_ts, p.fmt);
        m_out_port.send_data(composite::immutable_buffer<uint8_t>(std::move(data_vec)), chunk_ts,
                             p.out_meta);
        state.data_packet_count = (state.data_packet_count + 1) % 16;
        p.lead.erase(p.lead.begin(), p.lead.begin() + static_cast<std::ptrdiff_t>(from_lead));
        p.offset += from_payload;
    }

    // Whole-sample bytes short of a word: carry into this stream's next buffer.
    state.residue.assign(p.lead.begin(), p.lead.end());
    state.residue.insert(state.residue.end(), p.data.data() + p.offset,
                         p.data.data() + p.payload_end);

    m_pending.reset();
    return NORMAL;
}

auto pkt_builder::on_end_of_stream() -> void {
    // Best-effort flush of an in-flight packetization (bounded: the output may be gone).
    for (int i = 0; i < 100 && m_pending.has_value(); ++i) {
        if (emit_pending() == composite::retval::AWAIT_OUTPUT) {
            std::this_thread::yield();
        }
    }
    if (m_pending.has_value()) {
        logger()->warn("pkt_builder: end-of-stream with a packetization still blocked on the output");
    }
    // Per-stream residues (< 4 bytes each) cannot be emitted without fabricating pad
    // samples; count them as dropped so the loss is visible.
    for (auto& [_, state] : m_stream_states) {
        if (!state.residue.empty()) {
            m_bytes_dropped->add(state.residue.size());
            state.residue.clear();
        }
    }
}

auto pkt_builder::get_stream_id(const composite::metadata& metadata) -> uint32_t {
    if (auto it = metadata.annotations.find(m_stream_id_key); it != metadata.annotations.end()) {
        // Range-checked: the VITA stream id is 32 bits, and the silent uint32 truncation this
        // used to do would alias two distinct declared streams onto one wire id.
        if (auto v = annotation_as_uint(it->second); v && *v <= std::numeric_limits<uint32_t>::max()) {
            return static_cast<uint32_t>(*v);
        }
        logger()->warn("pkt_builder: invalid stream_id value in metadata: '{}', using default",
                       it->second.to_string());
    }
    return m_default_stream_id;
}

auto pkt_builder::find_or_create_stream_state(uint32_t stream_id) -> stream_state* {
    const auto now = std::chrono::steady_clock::now();
    if (auto it = m_stream_states.find(stream_id); it != m_stream_states.end()) {
        it->second.last_seen = now;
        return &it->second;
    }
    // Bound the per-stream state map: ids come from upstream metadata, so unbounded distinct
    // ids must not grow memory without limit — but a hard reject would let 64 disposable ids
    // permanently lock out every later legitimate stream (state-exhaustion DoS). Evict the
    // least-recently-seen stream instead: its next packet recreates state and resends
    // context (downstream sees a sequence restart, which parsers already tolerate). Safe
    // while a packetization is pending: the pending stream was touched this call, so it is
    // never the LRU victim, and eviction only runs from process() on the worker thread.
    if (m_stream_states.size() >= MAX_STREAM_STATES) {
        auto victim = m_stream_states.begin();
        for (auto it = m_stream_states.begin(); it != m_stream_states.end(); ++it) {
            if (it->second.last_seen < victim->second.last_seen) {
                victim = it;
            }
        }
        if (!victim->second.residue.empty()) {
            m_bytes_dropped->add(victim->second.residue.size());
        }
        if (!m_stream_cap_warned) {
            m_stream_cap_warned = true;
            logger()->warn("pkt_builder: more than {} distinct stream ids; evicting the least-"
                           "recently-seen (id={}) — sustained id churn degrades to context "
                           "resends, never a lockout (warning once)",
                           MAX_STREAM_STATES, victim->first);
        }
        m_stream_states.erase(victim);
    }
    auto& state = m_stream_states[stream_id];
    state.stream_id = stream_id;
    state.last_seen = now;
    logger()->info("pkt_builder: created stream state for stream_id={}", stream_id);
    return &state;
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

auto pkt_builder::build_context_packet(stream_state& state, const composite::metadata& metadata,
                                       const composite::timestamp& ts)
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
    pf.data_item_format(to_data_item_format(metadata.format.type, metadata.format.bit_width));
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
        // Context timestamps convey "as of": the triggering packet's time. (Writing zeros, as
        // this used to, declared TSI::UTC and then stamped the epoch — a receiver correlating
        // context to stream time saw 1970.)
        write_u32_be(dest + offset, static_cast<uint32_t>(ts.seconds));
        offset += INTEGER_TS_SIZE;
        write_u64_be(dest + offset, ts.picoseconds);
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

auto pkt_builder::build_data_packet(stream_state& state, std::span<const std::byte> lead,
                                    std::span<const std::byte> payload,
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
    packet_size += lead.size() + payload.size();
    packet_size = (packet_size + 3) & ~std::size_t{3}; // whole words by construction (see emit_pending)
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
    // Both pieces are element-aligned (the lead is whole samples), so each fused copy-swap
    // sees complete elements.
    copy_payload_to_be(dest + offset, lead.data(), lead.size(), fmt);
    offset += lead.size();
    copy_payload_to_be(dest + offset, payload.data(), payload.size(), fmt);

    return vec;
}

COMPOSITE_REGISTER_SIMPLE(pkt_builder)
