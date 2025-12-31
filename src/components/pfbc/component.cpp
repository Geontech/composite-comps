// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include "component.hpp"
#include "coefficients.hpp"
#include "kernels.hpp"
#include "fft_plan.hpp"

#include <composite/buffers/external_buffer.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstring>
#include <format>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <vector>

polyphase_channelizer::polyphase_channelizer(std::string_view id) : composite::component(id) {
    add_port(&m_data_in);
    add_port(&m_data_out);

    using enum composite::properties::config_type;
    add_property("num_channels", m_num_channels).change_listener([this]() {
        return m_num_channels > 0 && std::has_single_bit(m_num_channels);
    });

    add_property("taps_per_phase", m_taps_per_phase).change_listener([this]() {
        return m_taps_per_phase > 0;
    });

    add_property("frame_size", m_frame_size)
        .units("samples")
        .change_listener([this]() {
            return m_frame_size > 0;
        });

    add_property("prototype_filter", m_prototype_filter);
}

auto polyphase_channelizer::validate_properties() const -> bool {
    if (m_num_channels == 0 || !std::has_single_bit(m_num_channels)) {
        logger()->error("pfbc: num_channels ({}) must be a power of 2", m_num_channels);
        return false;
    }

    if (m_taps_per_phase == 0) {
        logger()->error("pfbc: taps_per_phase must be > 0");
        return false;
    }

    if (m_frame_size == 0) {
        logger()->error("pfbc: frame_size must be > 0");
        return false;
    }

    // Validate prototype filter if provided
    if (!m_prototype_filter.empty()) {
        const auto expected_size = static_cast<std::size_t>(m_num_channels) * m_taps_per_phase;
        if (m_prototype_filter.size() != expected_size) {
            logger()->error("pfbc: prototype_filter size ({}) must equal num_channels * taps_per_phase ({})",
                          m_prototype_filter.size(), expected_size);
            return false;
        }
    }

    return true;
}

auto polyphase_channelizer::prepare_coefficients() -> void {
    const auto M = static_cast<std::size_t>(m_num_channels);
    const auto K = static_cast<std::size_t>(m_taps_per_phase);
    const auto N = M * K;

    // Generate or use provided prototype filter
    std::vector<float> prototype;
    if (!m_prototype_filter.empty()) {
        prototype = m_prototype_filter;
        logger()->info("pfbc: using custom prototype filter ({} taps)", prototype.size());
    } else {
        prototype = pfbc::generate_default_prototype(M, K);
        logger()->info("pfbc: using default sinc-windowed lowpass prototype ({} taps)", N);
    }

    auto phase_coeffs = pfbc::build_phase_coeffs(prototype, M, K);
    auto interleaved = pfbc::prepare_interleaved_coeffs(phase_coeffs, M, K);

    // Copy to aligned storage
    m_coeffs = composite::make_aligned<float>(64, interleaved.size());
    std::copy(interleaved.begin(), interleaved.end(), m_coeffs->begin());

    logger()->debug("pfbc: prepared interleaved coefficients ({} floats)", interleaved.size());
}

auto polyphase_channelizer::allocate_buffers() -> void {
    const auto M = static_cast<std::size_t>(m_num_channels);
    const auto K = static_cast<std::size_t>(m_taps_per_phase);
    const auto N = static_cast<std::size_t>(m_frame_size);

    // Prepare interleaved coefficients
    prepare_coefficients();

    // Calculate history requirements: (K-1) time steps, M samples each
    m_history_len = K > 0 ? K - 1 : 0;
    const std::size_t history_samples = m_history_len * M;

    // Allocate history buffer - exactly (K-1)*M samples
    m_history = composite::make_aligned<cf32_t>(64, history_samples);

    // Allocate boundary buffer - enough for boundary outputs plus a full K-row scratch.
    // Boundary outputs (first K-1) span history + new data, need assembled input.
    // The direct-output tail path also needs a full K*M scratch (K can be 1).
    const std::size_t boundary_rows = std::max<std::size_t>(2 * K - 2, K);
    const std::size_t boundary_samples = boundary_rows * M;
    m_boundary_buffer = composite::make_aligned<cf32_t>(64, boundary_samples);

    // Allocate tail buffer - up to M-1 samples for partial rows between calls
    m_tail.resize(M);
    m_tail_len = 0;

    // Allocate filter output buffer (64-byte aligned)
    // Size based on expected max input block size
    // Layout: [t0: p0,p1,...,pM-1][t1: ...][...]
    constexpr std::size_t INITIAL_OUTPUT_CAPACITY = 8192;
    m_filter_output_capacity = INITIAL_OUTPUT_CAPACITY;
    m_filter_output = composite::make_aligned<cf32_t>(64, M * m_filter_output_capacity);

    // Initialize FFT plan (single FFT, single-threaded FFTW)
    // FFT writes directly to frame buffer, no intermediate output buffer needed
    m_fft = std::make_unique<fft_plan<cf32_t>>(static_cast<uint32_t>(M));

    // Allocate time-major frame buffer (64-byte aligned)
    // Layout: [t0: ch0,ch1,...,ch(M-1)][t1: ...][...]
    // Size: M * N samples
    // Time-major enables contiguous writes from FFT output.
    m_frame_tm = composite::make_aligned<cf32_t>(64, M * N);

    // Allocate output buffer pool - M channels * 2 for pipeline headroom
    // Each buffer holds N samples for one channel's frame
    m_output_pool = composite::slab_pool<cf32_t>::create(N, M * 2);
    m_output_ptrs.resize(M, nullptr);
    m_output_buffers.resize(M);

    logger()->debug("pfbc: allocated buffers - M={}, K={}, N={}, history={} samples",
                   M, K, N, M * m_history_len);
}

auto polyphase_channelizer::reset_state() -> void {
    m_frame_idx = 0;

    // Zero out history buffer
    if (m_history) {
        std::fill(m_history->begin(), m_history->end(), cf32_t{0.0f, 0.0f});
    }

    // Reset tail
    m_tail_len = 0;

    // Zero out time-major frame buffer
    if (m_frame_tm) {
        std::fill(m_frame_tm->begin(), m_frame_tm->end(), cf32_t{0.0f, 0.0f});
    }

    // Reset timestamp tracking
    m_stream_sample_idx = 0;
    m_ts_initialized = false;
    m_ts_base = composite::timestamp{};
}

auto polyphase_channelizer::configure() -> bool {
    m_configured = false;

    if (!validate_properties()) {
        return false;
    }

    try {
        allocate_buffers();
        reset_state();
        m_configured = true;

        logger()->info("pfbc: configured with {} channels, {} taps/phase, {} frame size",
                      m_num_channels, m_taps_per_phase, m_frame_size);
    } catch (const std::exception& e) {
        logger()->error("pfbc: configuration failed: {}", e.what());
        return false;
    }

    return true;
}

auto polyphase_channelizer::property_change_handler() -> void {
    configure();
}

auto polyphase_channelizer::emit_frames(composite::timestamp frame_ts) -> void {
    const auto M = static_cast<std::size_t>(m_num_channels);
    const auto N = static_cast<std::size_t>(m_frame_size);

    // Acquire output buffers up front so transpose can write contiguously.
    for (std::size_t ch = 0; ch < M; ++ch) {
        // Acquire buffer from pool (non-blocking, returns nullopt if exhausted)
        auto pool_buf = m_output_pool->acquire();
        if (!pool_buf) {
            logger()->warn("pfbc: output pool exhausted, dropping channel {} frame", ch);
            m_output_ptrs[ch] = nullptr;
            m_output_buffers[ch].reset();
            continue;
        }

        m_output_buffers[ch] = std::move(*pool_buf);
        m_output_ptrs[ch] = m_output_buffers[ch]->data();
    }

    // Transpose time-major frame to channel-major output buffers using AVX-512.
    // Layout: frame_tm[t * M + ch] -> output[ch][t]
    kernels::transpose_to_channel_buffers(
        m_frame_tm->data(),
        m_output_ptrs.data(),
        M,
        N
    );

    for (std::size_t ch = 0; ch < M; ++ch) {
        if (!m_output_buffers[ch].has_value()) {
            continue;
        }

        auto buffer = std::move(m_output_buffers[ch].value());
        m_output_buffers[ch].reset();
        m_output_ptrs[ch] = nullptr;

        // Wrap pool buffer in immutable_buffer for port send
        auto out = composite::immutable_buffer<cf32_t>(
            std::make_shared<composite::external_buffer<cf32_t>>(
                std::move(buffer)
            )
        );
        m_data_out.send_data(out, frame_ts);
    }

    m_frame_idx = 0;
}

auto polyphase_channelizer::process() -> composite::retval {
    using enum composite::retval;

    if (!m_configured) {
        return NOOP;
    }

    auto [buffer, ts, meta] = m_data_in.get_data();
    if (!buffer) {
        return NOOP;
    }

    // Handle metadata updates
    if (meta.has_value()) {
        m_metadata = meta.value();
        auto out_metadata = m_metadata;
        if (out_metadata.sample_rate > 0.0) {
            out_metadata.sample_rate = out_metadata.sample_rate / static_cast<double>(m_num_channels);
        }
        m_data_out.send_metadata(out_metadata);
    }

    auto span = buffer.as_span();
    const auto M = static_cast<std::size_t>(m_num_channels);
    const auto K = static_cast<std::size_t>(m_taps_per_phase);
    const auto N = static_cast<std::size_t>(m_frame_size);

    // =========================================================================
    // Timestamp tracking
    // =========================================================================
    // The input timestamp corresponds to the first sample of the new batch,
    // which starts after any remainder samples from the previous call.
    const composite::timestamp zero_ts{};
    const bool ts_present = (ts != zero_ts);
    const auto sample_rate = m_metadata.sample_rate;
    const bool has_rate = (sample_rate > 0.0);

    // =========================================================================
    // Calculate available data and outputs
    // =========================================================================
    const std::size_t total_data_samples = m_tail_len + span.size();
    const std::size_t num_outputs = total_data_samples / M;

    if (num_outputs == 0) {
        // Not enough samples for a complete row - accumulate in tail
        std::memcpy(m_tail.data() + m_tail_len, span.data(), span.size() * sizeof(cf32_t));
        m_tail_len += span.size();
        return NORMAL;
    }

    // Timestamp tracking
    if (ts_present && has_rate) {
        const auto offset_seconds = static_cast<double>(m_stream_sample_idx + m_tail_len) / sample_rate;
        const auto offset_ns = std::chrono::nanoseconds(
            static_cast<int64_t>(offset_seconds * 1e9)
        );
        m_ts_base = ts - offset_ns;
        m_ts_initialized = true;
    } else if (ts_present) {
        m_ts_base = ts;
        m_ts_initialized = true;
    }

    // Ensure filter output buffer has sufficient capacity
    if (num_outputs > m_filter_output_capacity) {
        m_filter_output_capacity = num_outputs;
        m_filter_output = composite::make_aligned<cf32_t>(64, M * m_filter_output_capacity);
    }

    // =========================================================================
    // Stage 1: Polyphase filtering with direct span access
    // =========================================================================
    // Split processing into boundary outputs (need history) and direct outputs
    // (entirely from new data, can read directly from span).
    //
    // Boundary outputs: 0..(K-2) - span history and new data
    // Direct outputs: (K-1)..(num_outputs-1) - entirely from new data
    //
    const std::size_t boundary_outputs = std::min(m_history_len, num_outputs);
    const std::size_t direct_outputs = num_outputs - boundary_outputs;
    const std::size_t history_samples = m_history_len * M;

    // Process boundary outputs (first K-1) - need assembled buffer
    if (boundary_outputs > 0) {
        // Assemble boundary buffer: [history: (K-1)*M][new data: boundary_outputs*M]
        // Copy history
        std::memcpy(m_boundary_buffer->data(),
                    m_history->data(),
                    history_samples * sizeof(cf32_t));

        // Copy first boundary_outputs*M samples from new data (tail + part of span)
        cf32_t* new_data_dst = m_boundary_buffer->data() + history_samples;
        const std::size_t boundary_new_samples = boundary_outputs * M;
        const std::size_t from_tail = std::min(m_tail_len, boundary_new_samples);
        if (from_tail > 0) {
            std::memcpy(new_data_dst, m_tail.data(), from_tail * sizeof(cf32_t));
            new_data_dst += from_tail;
        }
        const std::size_t from_span = boundary_new_samples - from_tail;
        if (from_span > 0) {
            std::memcpy(new_data_dst, span.data(), from_span * sizeof(cf32_t));
        }

        // Process boundary outputs
        kernels::filter_interleaved(
            m_boundary_buffer->data(),
            m_coeffs->data(),
            m_filter_output->data(),
            M,
            boundary_outputs,
            K
        );
    }

    // Process direct outputs - read directly from span when possible (NO COPY!)
    // Direct outputs exist only when boundary_outputs == m_history_len (i.e., num_outputs >= K-1).
    if (direct_outputs > 0) {
        if (m_tail_len == 0) {
            // Simple case: new_data == span, process all direct outputs from row 0
            kernels::filter_interleaved(
                span.data(),
                m_coeffs->data(),
                m_filter_output->data() + boundary_outputs * M,
                M,
                direct_outputs,
                K
            );
        } else {
            // Tail present: output K-1 needs new_data row 0 which starts in tail.
            // Assemble new_data rows 0..K-1 (K*M samples) in boundary buffer and process 1 output.
            cf32_t* extra_buf = m_boundary_buffer->data();
            std::memcpy(extra_buf, m_tail.data(), m_tail_len * sizeof(cf32_t));
            std::memcpy(extra_buf + m_tail_len, span.data(), (K * M - m_tail_len) * sizeof(cf32_t));

            kernels::filter_interleaved(
                extra_buf,
                m_coeffs->data(),
                m_filter_output->data() + boundary_outputs * M,
                M,
                1,
                K
            );

            // Remaining direct outputs (t >= K) read new_data rows 1..,
            // which start at span offset (M - tail_len).
            if (direct_outputs > 1) {
                const cf32_t* remaining_input = span.data() + (M - m_tail_len);
                kernels::filter_interleaved(
                    remaining_input,
                    m_coeffs->data(),
                    m_filter_output->data() + (boundary_outputs + 1) * M,
                    M,
                    direct_outputs - 1,
                    K
                );
            }
        }
    }

    // =========================================================================
    // Stage 2: FFT each output time step and accumulate
    // =========================================================================
    // Filter output is already in correct layout for FFT: [t0: p0,p1,...,pM-1]
    //
    // Track the sample offset for timestamp calculation:
    // - Each output time step corresponds to M input samples
    // - Frame timestamp = base_ts + (samples_at_frame_start / sample_rate)
    // - Since we don't have sample_rate here, we store sample counts and let
    //   the downstream compute time if needed. For now, we compute timestamps
    //   assuming the base timestamp corresponds to sample 0.

    for (std::size_t t = 0; t < num_outputs; ++t) {
        // FFT input is already at m_filter_output[t * M]
        const cf32_t* fft_input = m_filter_output->data() + t * M;

        // Execute FFT directly to frame buffer (no intermediate copy)
        cf32_t* frame_row = m_frame_tm->data() + m_frame_idx * M;
        m_fft->execute(fft_input, frame_row);

        // Advance frame index
        ++m_frame_idx;

        // Track input samples consumed (M samples per output time step)
        m_stream_sample_idx += M;

        // Check if frame is complete and emit
        if (m_frame_idx >= N) {
            // Compute timestamp for this frame
            // The frame started N output time steps ago, each consuming M input samples
            // Frame start sample = m_stream_sample_idx - N * M
            const uint64_t frame_start_sample = m_stream_sample_idx - N * M;

            // Convert sample offset to nanoseconds using sample_rate from metadata
            // If sample_rate is 0 or not set, use the base timestamp directly
            composite::timestamp frame_ts = ts_present ? m_ts_base : composite::timestamp{};
            if (ts_present && has_rate) {
                // offset_ns = frame_start_sample / sample_rate * 1e9
                const double offset_seconds = static_cast<double>(frame_start_sample) / sample_rate;
                const auto offset_ns = std::chrono::nanoseconds(
                    static_cast<int64_t>(offset_seconds * 1e9)
                );
                frame_ts = m_ts_base + offset_ns;
            }

            emit_frames(frame_ts);
        }
    }

    // =========================================================================
    // Stage 3: Update history and tail for next call
    // =========================================================================
    // History: last (K-1) rows from the combined (old_history + new_data) stream
    // new_data = tail + span
    //
    // After processing num_outputs, new history starts at combined sample num_outputs*M
    // Combined layout: [old_history: (K-1)*M][new_data: tail + span]

    const std::size_t history_start_combined = num_outputs * M;

    if (history_start_combined >= history_samples) {
        // Common case: history entirely from new_data
        const std::size_t start_in_newdata = history_start_combined - history_samples;

        if (start_in_newdata >= m_tail_len) {
            // All from span (most common)
            std::memcpy(m_history->data(),
                        span.data() + (start_in_newdata - m_tail_len),
                        history_samples * sizeof(cf32_t));
        } else {
            // Spans tail and span
            const std::size_t from_tail = m_tail_len - start_in_newdata;
            std::memcpy(m_history->data(), m_tail.data() + start_in_newdata, from_tail * sizeof(cf32_t));
            std::memcpy(m_history->data() + from_tail, span.data(), (history_samples - from_tail) * sizeof(cf32_t));
        }
    } else {
        // Rare case: history spans old history and new_data (small input batch)
        const std::size_t from_old = history_samples - history_start_combined;
        const std::size_t from_new = history_samples - from_old;

        // Shift old history within m_history buffer
        std::memmove(m_history->data(), m_history->data() + history_start_combined, from_old * sizeof(cf32_t));

        // Append from new_data (tail + span)
        cf32_t* dst = m_history->data() + from_old;
        if (from_new <= m_tail_len) {
            std::memcpy(dst, m_tail.data(), from_new * sizeof(cf32_t));
        } else {
            if (m_tail_len > 0) {
                std::memcpy(dst, m_tail.data(), m_tail_len * sizeof(cf32_t));
            }
            std::memcpy(dst + m_tail_len, span.data(), (from_new - m_tail_len) * sizeof(cf32_t));
        }
    }

    // Update tail with leftover samples
    const std::size_t leftover = total_data_samples - num_outputs * M;
    if (leftover > 0) {
        // Leftover is at end of new_data = tail + span
        // leftover_start_in_newdata = num_outputs * M
        const std::size_t leftover_start = num_outputs * M;
        if (leftover_start >= m_tail_len) {
            // Leftover entirely in span
            std::memcpy(m_tail.data(),
                        span.data() + (leftover_start - m_tail_len),
                        leftover * sizeof(cf32_t));
        } else {
            // Leftover spans tail and span (shouldn't happen with M-aligned, but handle it)
            const std::size_t from_old_tail = m_tail_len - leftover_start;
            std::memmove(m_tail.data(), m_tail.data() + leftover_start, from_old_tail * sizeof(cf32_t));
            std::memcpy(m_tail.data() + from_old_tail, span.data(), (leftover - from_old_tail) * sizeof(cf32_t));
        }
    }
    m_tail_len = leftover;

    return NORMAL;
}

extern "C" {
    auto create(std::string_view id) -> std::shared_ptr<composite::component> {
        return std::make_shared<polyphase_channelizer>(id);
    }
}
