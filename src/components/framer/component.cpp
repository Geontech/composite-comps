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
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */

#include "component.hpp"

#include <composite/core/register.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <thread>

template <typename T>
framer<T>::framer(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_port(&m_out_port);

    // Cross-field invariants run on the proposed whole-struct candidate (so a single-field
    // PATCH that would make overlap >= frame_size is rejected against the combined state, not
    // a stale sibling). Set validators + reaction BEFORE add_config — the binding snapshots
    // the validators at registration.
    m_cfg.validate([](const framer_detail::framer_config& c) { return c.frame_size > 0; },
                   "frame_size must be > 0")
         .validate([](const framer_detail::framer_config& c) { return c.overlap < c.frame_size; },
                   "overlap must be < frame_size")
         .validate([](const framer_detail::framer_config& c) { return c.frame_count >= 2; },
                   "frame_count must be >= 2");
    m_cfg.on_apply([this](const framer_detail::framer_config&,
                          const composite::changes<framer_detail::framer_config>&) {
        // Any field change rebuilds the pool (frame geometry is structural). Runs at the
        // worker loop-top, or inline at INITIALIZE-time when there is no worker yet.
        initialize_pool();
    });
    add_config(m_cfg);

    // Drop statistics live in the shared metrics registry so operators can see sample loss
    // (gaps in downstream FFT/PSD integration) without scraping logs. The base helper
    // labels each series with this component's id and ~component removes them by label.
    m_samples_dropped = &create_counter(
        "framer.samples_dropped", "Input samples dropped instead of framed");
    m_drops_batch_too_large = &create_counter(
        "framer.drops_batch_too_large", "Write batches dropped because they exceed the ring size");
    m_drops_backpressure = &create_counter(
        "framer.drops_backpressure",
        "Write batches dropped because downstream held frames past the backpressure timeout");
    m_bytes_dropped_no_format = &create_counter(
        "framer.bytes_dropped_no_format",
        "Input bytes dropped while the input format was missing or unsupported");
    m_bytes_dropped_unaligned = &create_counter(
        "framer.bytes_dropped_unaligned",
        "Trailing bytes discarded from buffers not aligned to sample boundaries");
    m_samples_dropped_boundary = &create_counter(
        "framer.samples_dropped_metadata_boundary",
        "Buffered samples skipped at a metadata change so no emitted frame mixes streams");
    m_packets_dropped_reconfiguration = &create_counter(
        "framer.packets_dropped_reconfiguration",
        "Queued input packets discarded when frame geometry changes");
    m_samples_dropped_reconfiguration = &create_counter(
        "framer.samples_dropped_reconfiguration",
        "Buffered samples discarded when frame geometry changes");
}

template <typename T>
auto framer<T>::reset_state() -> void {
    m_in_port.clear();
    m_metadata_ready = false;
    m_last_input_meta = nullptr;  // force full metadata handling on the next packet
    m_out_metadata = nullptr;
    m_input_format = {};  // Reset to default
    m_input_stride = 0;
    m_next_frame_start = 0;
    m_anchors.clear();
    m_reanchor_pending = true;
    m_warned_no_format = false;
    m_warned_unaligned = false;
}

template <typename T>
auto framer<T>::initialize_pool() -> void {
    const auto frame_size = m_cfg->frame_size;
    const auto overlap = m_cfg->overlap;

    if (frame_size == 0) {
        logger()->error("framer: frame_size must be > 0");
        return;
    }

    if (overlap >= frame_size) {
        logger()->error("framer: overlap ({}) must be < frame_size ({})", overlap, frame_size);
        return;
    }

    // A geometry update is an explicit discontinuity: the old partial frame cannot be
    // interpreted under the new frame_size/overlap, and packets already queued may belong to
    // either side of the control-plane update. Drop both sets rather than ever mixing them into
    // a frame. Count the loss before reset_state() clears the queue and state. The queued-packet
    // count is a best-effort lower bound: an upstream producer can enqueue between pending() and
    // clear(), because parking this component's worker does not park its producer.
    const auto queued_packets = m_in_port.pending();
    const auto buffered_samples = m_pool != nullptr && m_pool->head() > m_next_frame_start
        ? m_pool->head() - m_next_frame_start
        : std::size_t{0};

    try {
        auto frames = std::max<uint32_t>(m_cfg->frame_count, 2);
        m_pool = std::make_shared<composite::overlap_ring<T>>(frame_size, overlap, frames);
        logger()->debug("framer: initialized pool with frame_size={}, overlap={}, frame_count={}, "
                       "hop_size={}, ring_size={}",
                       frame_size, overlap, frames,
                       frame_size - overlap, frames * (frame_size - overlap) + overlap);
    } catch (const std::exception& e) {
        logger()->error("framer: failed to initialize pool: {}", e.what());
        m_pool.reset();
        return;
    }

    if (queued_packets != 0 || buffered_samples != 0) {
        m_packets_dropped_reconfiguration->add(queued_packets);
        m_samples_dropped_reconfiguration->add(buffered_samples);
        logger()->info("framer: frame geometry changed; discarded {} queued packets and {} "
                       "buffered samples",
                       queued_packets, buffered_samples);
    }
    reset_state();
}

template <typename T>
auto framer<T>::configure_output_metadata() -> void {
    if constexpr (std::is_same_v<T, std::complex<float>>) {
        m_metadata.format.type = composite::data_type::floating_point;
        m_metadata.format.bit_width = 32; // per component
    } else {
        m_metadata.format.type = composite::data_type::signed_integer;
        m_metadata.format.bit_width = 16;
    }
    m_metadata.format.is_complex = true;
    m_metadata.format.endianness = std::endian::native;
}

template <typename T>
auto framer<T>::handle_metadata(const composite::metadata_ptr& meta) -> void {
    // Metadata rides on every packet as a shared instance the producer latches: the steady
    // state is the SAME pointer, so this early-out is one pointer compare — no copy, no
    // deep compare, no converter rebuild on the hot path.
    if (meta == m_last_input_meta) {
        return;
    }
    // Different instance but equal value (a producer that doesn't latch): adopt the new
    // instance so subsequent packets hit the pointer fast path, but re-process nothing.
    if (m_last_input_meta != nullptr && *meta == *m_last_input_meta) {
        m_last_input_meta = meta;
        return;
    }
    // FRAMES MUST NOT CROSS A METADATA BOUNDARY. A value change past the early-outs above
    // means the samples already buffered in the ring belong to the PREVIOUS stream shape:
    // letting a later packet complete that partial frame would emit a frame whose samples
    // straddle two streams but whose label (m_out_metadata, attached at emission) describes
    // only the new one — and whose timestamp could come from the old anchor. Skip the
    // partial residue instead (counted); the first frame of the new stream starts exactly
    // at its first sample.
    discard_partial_frame();

    // Re-anchor at EVERY value-changing boundary, residue or not: a new stream at the SAME
    // rate (session change, retune, format change) arrives with its own wall-clock, and
    // extrapolating the old stream's anchor across the transition would stamp its first
    // frames with the previous stream's timeline. Set here — after discard_partial_frame()
    // (whose clean-boundary early-out must not skip this) and before the unsupported-format
    // return below (a to-unsupported transition is a boundary too).
    m_reanchor_pending = true;

    m_last_input_meta = meta;

    // New metadata starts a new warn episode for the one-shot stream-shape warnings.
    m_warned_no_format = false;
    m_warned_unaligned = false;

    using scalar_t = typename T::value_type;
    const auto* support = find_input_format<scalar_t>(meta->format);
    if (support == nullptr) {
        // One warn per metadata change (the early-out above swallows repeats), so a
        // misconfigured flow is visible without flooding the log at packet rate.
        logger()->warn("framer: unsupported input format for {} output "
                       "(bit_width={}, type={}, complex={}); dropping data until a supported "
                       "format arrives",
                       std::is_same_v<scalar_t, float> ? "complex<float>" : "complex<int16_t>",
                       meta->format.bit_width, static_cast<int>(meta->format.type),
                       meta->format.is_complex);
        m_metadata_ready = false;
        return;
    }

    // Save input format before overwriting metadata
    m_input_format = meta->format;
    m_metadata = *meta;
    m_metadata_ready = true;
    m_input_stride = support->bytes_per_sample;

    configure_output_metadata();
    // Publish the output metadata as one shared instance; every emitted frame attaches it
    // with a refcount bump until the next input-metadata change.
    m_out_metadata = composite::make_metadata(m_metadata);

    // Byte swapping is determined by INPUT endianness vs native.
    const bool needs_swap = (m_input_format.endianness != std::endian::native);
    m_converter = support->make(needs_swap);

    logger()->trace("framer: updated metadata:\n{}", m_metadata.to_string());
}

template <typename T>
auto framer<T>::discard_partial_frame() -> void {
    if (!m_pool) {
        return;
    }
    const auto head = m_pool->head();
    if (head <= m_next_frame_start) {
        return;  // ring sits on a frame boundary: nothing buffered to skip
    }
    const auto residue = head - m_next_frame_start;
    // With overlap enabled, up to `overlap` of these samples were already delivered inside
    // the previous emitted frame, so this counter slightly over-states unseen-sample loss;
    // exact accounting would need per-segment emission state for a diagnostic nicety.
    m_samples_dropped_boundary->add(residue);
    logger()->debug("framer: skipping {} buffered samples at a metadata boundary so no frame "
                    "mixes streams (counted in framer.samples_dropped_metadata_boundary)",
                    residue);

    // try_emit_frame REQUIRES hop-aligned frame starts, so the new stream's first frame
    // begins at the next hop boundary. Zero-pad the ring up to it: the pad lives entirely
    // inside the skipped region (never emitted), and the new stream's first sample then
    // lands exactly on the boundary. If the pad write fails (full backpressure at the exact
    // moment of a metadata change), the alignment still holds — at most hop-1 of the new
    // stream's samples fall below the boundary and are skipped with the residue.
    const auto hop = m_pool->hop_size();
    const auto aligned = ((head + hop - 1) / hop) * hop;
    if (const auto pad = aligned - head; pad > 0) {
        const bool ok = m_pool->write(pad, [](T* dst, std::size_t /*src*/, std::size_t count) {
            for (std::size_t i = 0; i < count; ++i) { dst[i] = T{}; }
        });
        if (!ok) {
            logger()->debug("framer: could not pad {} samples to the hop boundary at a metadata "
                            "change; up to that many new-stream samples will be skipped", pad);
        }
    }
    m_next_frame_start = aligned;
    // (The re-anchor for the boundary — residue or not — is owned by handle_metadata,
    // immediately after this returns.)
}

template <typename T>
auto framer<T>::process_buffer(const composite::immutable_buffer<uint8_t>& buffer, composite::timestamp ts) -> void {
    if (!m_metadata_ready || m_input_stride == 0 || !m_converter || !m_pool) {
        return;
    }

    auto span = buffer.as_span();
    const uint8_t* bytes = span.data();
    std::size_t byte_count = span.size();
    const bool is_complex = m_input_format.is_complex;

    // Process complete samples in batch
    std::size_t complete_samples = byte_count / m_input_stride;

    if (complete_samples > 0) {
        using scalar_t = typename T::value_type;
        // Produce/convert samples directly into the ring's destination (zero-copy). The overlap_ring
        // hands us the contiguous destination and the source-sample offset of each (wrap-split)
        // chunk; we convert the raw input bytes into complex T in place.
        auto produce = [&](T* dst, std::size_t src_sample_index, std::size_t count) {
            const uint8_t* src = bytes + src_sample_index * m_input_stride;
            if (is_complex) {
                convert(m_converter.value(), src, reinterpret_cast<scalar_t*>(dst), count * 2);
            } else {
                // Real input: convert into the back half of the destination, then expand
                // forward in place to complex (imag = 0). The expansion kernel documents
                // why the in-place aliasing is safe; no scratch buffer is needed.
                auto* dst_scalars = reinterpret_cast<scalar_t*>(dst);
                convert(m_converter.value(), src, dst_scalars + count, count);
                expand_real_to_complex(dst_scalars, count);
            }
        };

        // The write start position keys the timestamp anchor: this buffer's timestamp is the
        // wall time of the first sample it lands at the current ring head.
        const auto write_start = m_pool->head();
        if (m_pool->write(complete_samples, produce)) {
            if (m_reanchor_pending) {
                // Anchor only AFTER a successful write: anchoring on arrival would bind this
                // buffer's timestamp to ring samples it never contributed (had the write
                // dropped, the head would not have advanced).
                push_anchor(ts, write_start);
            }
        } else {
            auto diag = m_pool->get_diagnostics();

            m_samples_dropped->add(complete_samples);

            // Dropped samples break the sample-index -> wall-time mapping for everything
            // written after them (the ring head did not advance but real time did); anchor a
            // new segment at the next successful write. Frames already buffered keep the old
            // segment's mapping.
            m_reanchor_pending = true;

            switch (diag.last_drop_reason) {
                case composite::overlap_ring<T>::drop_reason::BATCH_TOO_LARGE:
                    m_drops_batch_too_large->inc();
                    logger()->error("framer: failed to write {} samples (BATCH TOO LARGE) - "
                                   "batch size exceeds ring_size ({}). Dropping samples.",
                                   complete_samples, diag.ring_size);
                    break;
                case composite::overlap_ring<T>::drop_reason::BACKPRESSURE_TIMEOUT:
                    m_drops_backpressure->inc();
                    logger()->warn("framer: failed to write {} samples (BACKPRESSURE TIMEOUT) - "
                                  "downstream holding frames. Dropping samples. "
                                  "Slots in use: {}/{}, available space: {} samples, write_head: {}, "
                                  "oldest protected: {}",
                                  complete_samples, diag.slots_in_use, diag.total_slots,
                                  diag.available_space, diag.write_head, diag.oldest_protected_sample);
                    break;
                default:
                    logger()->warn("framer: failed to write {} samples (UNKNOWN), dropping", complete_samples);
                    break;
            }
        }
    }

    // If there are leftover bytes, it indicates a non-sample-aligned buffer, which is now considered
    // an error. Count every occurrence, but warn once per metadata episode: a persistently
    // misaligned upstream would otherwise emit this at packet rate.
    if (const auto leftover = byte_count % m_input_stride; leftover != 0) {
        m_bytes_dropped_unaligned->add(leftover);
        // The discarded fraction of a sample is lost TIME as well as lost bytes; without a
        // re-anchor it would accumulate as timestamp drift (a fraction of a sample per
        // misaligned buffer). Anchor the next buffer at its own timestamp instead.
        m_reanchor_pending = true;
        if (!m_warned_unaligned) {
            m_warned_unaligned = true;
            logger()->warn("framer: received a buffer that is not aligned to sample boundaries ({} bytes, stride {}). "
                           "Partial sample handling has been disabled, so leftover bytes will be discarded "
                           "(counted in framer.bytes_dropped_unaligned; warning once per metadata change).",
                           byte_count, m_input_stride);
        }
    }
}

template <typename T>
auto framer<T>::try_emit_frames() -> void {
    if (!m_pool) {
        return;
    }

    const std::size_t head = m_pool->head();
    const std::size_t hop = m_pool->hop_size();
    const std::size_t frame_size = m_pool->frame_size();

    while (head >= m_next_frame_start + frame_size) {
        auto buffer_opt = m_pool->try_emit_frame(m_next_frame_start);
        if (!buffer_opt.has_value()) {
            break;  // Slot busy, stop emitting
        }

        auto ts = compute_frame_timestamp(m_next_frame_start);
        m_out_port.send_data(std::move(buffer_opt.value()), ts, m_out_metadata);

        m_next_frame_start += hop;
    }

    // Emission is monotonic in m_next_frame_start, so an anchor is dead once the NEXT anchor
    // already covers every frame still to come. Steady state keeps exactly one anchor.
    while (m_anchors.size() > 1 && m_anchors[1].origin_sample <= m_next_frame_start) {
        m_anchors.erase(m_anchors.begin());
    }
}

template <typename T>
auto framer<T>::on_end_of_stream() -> void {
    // Flush the residue: a tail of fewer than frame_size samples past m_next_frame_start stays
    // buffered in the ring and would otherwise be dropped at end-of-stream. Zero-pad the ring up to
    // exactly one full frame, then emit it via the normal path. Nothing to do if no metadata was
    // ever seen (no ring / nothing buffered) or if the ring sits on a frame boundary (no residue).
    if (!m_pool || !m_metadata_ready) { return; }
    const std::size_t head = m_pool->head();
    if (head <= m_next_frame_start) { return; }              // no partial residue buffered
    const std::size_t frame_size = m_pool->frame_size();
    const std::size_t have = head - m_next_frame_start;      // samples in the final partial frame
    if (have >= frame_size) { return; }                      // defensive: a full frame already emitted
    const std::size_t pad = frame_size - have;
    // Append `pad` zero samples so the residue completes exactly one frame, then emit it.
    const bool ok = m_pool->write(pad, [](T* dst, std::size_t /*src_sample_index*/, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) { dst[i] = T{}; }
    });
    if (!ok) {
        logger()->warn("framer: could not zero-pad the final {}-sample partial frame at end-of-stream; "
                       "residue dropped", have);
        return;
    }
    try_emit_frames();  // emits the now-complete zero-padded final frame

    // This flush is one-shot (the worker will not call process() again), so if the frame's
    // slot is still held downstream, retry briefly rather than silently losing the final
    // frame — and say so if it still cannot be emitted.
    constexpr int MAX_EMIT_RETRIES = 100;
    for (int i = 0; i < MAX_EMIT_RETRIES && m_pool->head() >= m_next_frame_start + frame_size; ++i) {
        std::this_thread::yield();
        try_emit_frames();
    }
    if (m_pool->head() >= m_next_frame_start + frame_size) {
        logger()->warn("framer: final zero-padded frame could not be emitted at end-of-stream "
                       "(downstream still holding its frame slot); residue dropped");
    }
}

template <typename T>
auto framer<T>::push_anchor(composite::timestamp ts, std::size_t origin_sample) -> void {
    time_anchor anchor{.origin = ts, .origin_sample = origin_sample};
    // The rate comes off the wire (parser metadata / user overrides); a non-finite or
    // non-positive value must not poison the uint64 casts in compute_frame_timestamp
    // (casting a NaN/negative long double to uint64 is UB). rate == 0 in the anchor means
    // "no extrapolation": frames in this segment reuse the anchor time as-is.
    if (const auto sr = m_metadata.sample_rate; std::isfinite(sr) && sr > 0.0) {
        anchor.rate = static_cast<long double>(sr);
        anchor.inv_rate = 1.0L / anchor.rate;
        anchor.ps_per_sample = static_cast<long double>(framer_detail::PS_PER_SEC) / anchor.rate;
    }
    if (m_anchors.size() >= MAX_ANCHORS) {
        // Discontinuities are outpacing emission; shed the oldest segment (its frames get
        // nearest-anchor stamps) rather than grow without bound.
        m_anchors.erase(m_anchors.begin());
    }
    m_anchors.push_back(anchor);
    m_reanchor_pending = false;
}

template <typename T>
auto framer<T>::compute_frame_timestamp(std::size_t start_sample) const -> composite::timestamp {
    if (m_anchors.empty()) {
        // Unreachable in practice: every emittable frame's samples came from a successful
        // write, and the first successful write always pushes an anchor.
        return {};
    }

    // Newest anchor with origin_sample <= start_sample owns this frame's mapping (the list is
    // ordered and tiny; steady state is a single element).
    const auto* anchor = &m_anchors.front();
    for (const auto& candidate : m_anchors) {
        if (candidate.origin_sample > start_sample) {
            break;
        }
        anchor = &candidate;
    }

    if (anchor->rate <= 0.0L || start_sample < anchor->origin_sample) {
        // No usable rate for this segment, or the frame predates the oldest retained anchor
        // (possible only after MAX_ANCHORS eviction): the anchor time is the best available.
        return anchor->origin;
    }

    // Split the elapsed time into whole seconds plus a sub-second remainder BEFORE converting
    // to picoseconds: a single delta_samples * ps_per_sample product overflows uint64 after
    // ~213 days of continuous streaming, while the remainder term here stays under one second.
    // The divisions were precomputed when the anchor was established.
    const auto delta_samples = static_cast<long double>(start_sample - anchor->origin_sample);
    const auto whole_seconds = static_cast<uint64_t>(delta_samples * anchor->inv_rate);
    const auto remainder_samples =
        std::max(delta_samples - static_cast<long double>(whole_seconds) * anchor->rate, 0.0L);
    const auto ps_rem = static_cast<uint64_t>(remainder_samples * anchor->ps_per_sample);

    auto ts = anchor->origin;
    ts.seconds += whole_seconds + ps_rem / framer_detail::PS_PER_SEC;
    auto picoseconds = ts.picoseconds + (ps_rem % framer_detail::PS_PER_SEC);
    if (picoseconds >= framer_detail::PS_PER_SEC) {
        ts.seconds += 1;
        picoseconds -= framer_detail::PS_PER_SEC;
    }
    ts.picoseconds = picoseconds;
    return ts;
}


template <typename T>
auto framer<T>::process() -> composite::retval {
    using enum composite::retval;

    // Bounded batch drain: one ring-head publication per call instead of one per packet
    // (upstream sends per-packet). Each packet is still processed sequentially — metadata is
    // ordered stream state — and frames are emitted after each buffer so downstream latency
    // does not stretch with the batch.
    const auto count = m_in_port.get_batch(std::span{m_input_batch});
    if (count == 0) {
        // No input: NOOP so the worker parks on the read-doorbell until upstream delivers,
        // rather than busy-spinning process() while idle. At end-of-stream the base promotes this
        // NOOP to FINISH (see on_end_of_stream() for the residue flush).
        return NOOP;
    }

    for (std::size_t i = 0; i < count; ++i) {
        // Move the packet out so its buffer reference is released at scope end, not held in
        // the batch array until the next drain overwrites this slot.
        auto [buffer, ts, meta] = std::move(m_input_batch[i]);

        if (meta != nullptr) {
            handle_metadata(meta);
        }

        if (!m_metadata_ready) {
            // No usable input format (none seen yet, or the current one is unsupported):
            // the data cannot be framed. The metric carries the loss rate; warn once per
            // metadata episode so a stream with no metadata at all is still visible.
            m_bytes_dropped_no_format->add(buffer.size());
            if (!m_warned_no_format) {
                m_warned_no_format = true;
                logger()->warn("framer: dropping data ({} bytes) with no usable input format "
                               "(counted in framer.bytes_dropped_no_format)", buffer.size());
            }
            if (buffer.size() > 0) {
                // Real time passed while these bytes were lost, so the sample-index ->
                // wall-time mapping no longer holds for anything written later — even if a
                // supported format resumes at the SAME sample rate (which would not trip the
                // rate-change re-anchor in handle_metadata).
                m_reanchor_pending = true;
            }
            continue;
        }

        process_buffer(buffer, ts);
        try_emit_frames();
    }

    return NORMAL;
}

// Explicit instantiations so the out-of-line member definitions (e.g. the
// constructor) are emitted as linkable symbols for translation units that
// construct framer<T> directly (such as the integration tests).
template class framer<std::complex<float>>;
template class framer<std::complex<int16_t>>;

COMPOSITE_REGISTER_COMPONENT([](std::string_view id, const composite::create_args& args)
                                 -> std::shared_ptr<composite::component> {
    const auto type = args.type();
    if (type == "cf32") {
        return composite::make_component<framer<std::complex<float>>>(id);
    } else if (type == "ci16") {
        return composite::make_component<framer<std::complex<int16_t>>>(id);
    }
    throw std::runtime_error(std::format("unknown type '{}' for framer component", type));
})
