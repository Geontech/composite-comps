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
#include <complex>
#include <format>
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
    m_timestamp_initialized = false;
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
    const bool rate_changed =
        m_last_input_meta == nullptr || meta->sample_rate != m_last_input_meta->sample_rate;
    m_last_input_meta = meta;

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

    if (rate_changed) {
        // The sample-index -> wall-time mapping changed; re-anchor from the next buffer's
        // timestamp instead of extrapolating the new rate across the old origin.
        m_timestamp_initialized = false;
    }

    logger()->trace("framer: updated metadata:\n{}", m_metadata.to_string());
}

template <typename T>
auto framer<T>::process_buffer(const composite::immutable_buffer<uint8_t>& buffer, composite::timestamp ts) -> void {
    if (!m_metadata_ready || m_input_stride == 0 || !m_converter || !m_pool) {
        return;
    }

    if (!m_timestamp_initialized) {
        m_timestamp_origin = ts;
        m_timestamp_origin_sample = m_pool->head();
        m_timestamp_initialized = true;
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
                // forward in place to complex (imag = 0). Iteration i writes scalar slots
                // 2i and 2i+1 and reads slot count+i, which it has not yet overwritten
                // (2i+1 < count+i+1 for all i < count), so no scratch buffer is needed.
                auto* dst_scalars = reinterpret_cast<scalar_t*>(dst);
                convert(m_converter.value(), src, dst_scalars + count, count);
                for (std::size_t i = 0; i < count; ++i) {
                    const auto value = dst_scalars[count + i];
                    dst[i] = T{value, scalar_t{0}};
                }
            }
        };

        if (!m_pool->write(complete_samples, produce)) {
            auto diag = m_pool->get_diagnostics();

            m_samples_dropped->add(complete_samples);

            // Dropped samples break the sample-index -> wall-time mapping (the ring head did
            // not advance but real time did); re-anchor from the next buffer's timestamp so
            // subsequent frame timestamps stay correct.
            m_timestamp_initialized = false;

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

    // If there are leftover bytes, it indicates a non-sample-aligned buffer, which is now considered an error.
    if (byte_count % m_input_stride != 0) {
        logger()->warn("framer: received a buffer that is not aligned to sample boundaries ({} bytes, stride {}). "
                       "Partial sample handling has been disabled, so leftover bytes will be discarded.",
                       byte_count, m_input_stride);
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
auto framer<T>::compute_frame_timestamp(std::size_t start_sample) const -> composite::timestamp {
    if (!m_timestamp_initialized || m_metadata.sample_rate <= 0.0) {
        return m_timestamp_origin;
    }

    // Handle potential underflow by checking if start_sample is less than origin
    if (start_sample < m_timestamp_origin_sample) {
        logger()->warn("framer: start_sample ({}) < origin_sample ({}), using origin timestamp",
                       start_sample, m_timestamp_origin_sample);
        return m_timestamp_origin;
    }

    // Split the elapsed time into whole seconds plus a sub-second remainder BEFORE converting
    // to picoseconds: a single delta_samples * ps_per_sample product overflows uint64 after
    // ~213 days of continuous streaming, while the remainder term here stays under one second.
    const auto rate = static_cast<long double>(m_metadata.sample_rate);
    const auto delta_samples = static_cast<long double>(start_sample - m_timestamp_origin_sample);
    const auto whole_seconds = static_cast<uint64_t>(delta_samples / rate);
    const auto remainder_samples =
        std::max(delta_samples - static_cast<long double>(whole_seconds) * rate, 0.0L);
    const auto ps_per_sample = static_cast<long double>(framer_detail::PS_PER_SEC) / rate;
    const auto ps_rem = static_cast<uint64_t>(remainder_samples * ps_per_sample);

    auto ts = m_timestamp_origin;
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

    auto pkt = m_in_port.try_get();
    if (!pkt) {
        // No input: NOOP so the worker parks on the read-doorbell until upstream delivers,
        // rather than busy-spinning process() while idle. At end-of-stream the base promotes this
        // NOOP to FINISH (see on_end_of_stream() for the residue flush).
        return NOOP;
    }
    auto& [buffer, ts, meta] = *pkt;

    if (meta != nullptr) {
        handle_metadata(meta);
    }

    if (!m_metadata_ready) {
        return NORMAL;
    }

    process_buffer(buffer, ts);
    try_emit_frames();

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
        return std::make_shared<framer<std::complex<float>>>(id);
    } else if (type == "ci16") {
        return std::make_shared<framer<std::complex<int16_t>>>(id);
    }
    throw std::runtime_error(std::format("unknown type '{}' for framer component", type));
})
