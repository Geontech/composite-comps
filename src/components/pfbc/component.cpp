// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include "component.hpp"
#include "coefficients.hpp"
#include "fft_plan.hpp"
#include "pfbc_engine.hpp"

#include <composite/buffers/external_buffer.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <format>
#include <stdexcept>
#include <string_view>

// =============================================================================
// Constructor / Destructor
// =============================================================================

polyphase_channelizer::polyphase_channelizer(std::string_view id)
    : composite::component(id)
{
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

polyphase_channelizer::~polyphase_channelizer() = default;

// =============================================================================
// Configuration
// =============================================================================

auto polyphase_channelizer::validate_properties() const -> bool {
    if (m_num_channels == 0 || !std::has_single_bit(m_num_channels)) {
        logger()->error("pfbc: num_channels ({}) must be a power of 2", m_num_channels);
        return false;
    }

    if (m_num_channels < 8) {
        logger()->error("pfbc: num_channels ({}) must be at least 8", m_num_channels);
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

    std::vector<float> prototype;
    if (!m_prototype_filter.empty()) {
        prototype = m_prototype_filter;
        logger()->info("pfbc: using custom prototype filter ({} taps)", prototype.size());
    } else {
        prototype = pfbc::generate_default_prototype(M, K);
        logger()->info("pfbc: using default sinc-windowed lowpass prototype ({} taps)", N);
    }

    auto phase_coeffs = pfbc::build_phase_coeffs(prototype, M, K);
    auto blocked = pfbc::prepare_blocked_coeffs(phase_coeffs, M, K);

    m_coeffs = composite::make_aligned<float>(64, blocked.size());
    std::copy(blocked.begin(), blocked.end(), m_coeffs->begin());

    logger()->debug("pfbc: prepared blocked coefficients ({} floats, {} blocks)",
                   blocked.size(), M / 8);
}

auto polyphase_channelizer::allocate_buffers() -> void {
    const auto M = static_cast<std::size_t>(m_num_channels);
    const auto K = static_cast<std::size_t>(m_taps_per_phase);
    const auto N = static_cast<std::size_t>(m_frame_size);

    // Coefficients
    prepare_coefficients();

    // Ring Buffer
    const std::size_t history_samples = (K - 1) * M;
    const std::size_t processing_samples = N * M * 4;
    const std::size_t min_capacity = history_samples + processing_samples;

    m_ring.allocate(min_capacity);
    m_ring.prefill_zeros(history_samples);

    logger()->debug("pfbc: ring buffer capacity {} samples ({} KB)",
                   m_ring.capacity(), m_ring.capacity() * sizeof(cf32_t) / 1024);

    // L1 Scratch Buffer
    const std::size_t scratch_samples = TILE_SIZE * M;
    m_l1_scratch = composite::make_aligned<cf32_t>(64, scratch_samples);

    logger()->debug("pfbc: L1 scratch {} samples ({} KB)",
                   scratch_samples, scratch_samples * sizeof(cf32_t) / 1024);

    // FFT plan (for partial tiles)
    m_fft = std::make_unique<fft_plan<cf32_t>>(static_cast<uint32_t>(M));

    // Output Buffer Pool
    m_output_pool = composite::slab_pool<cf32_t>::create(N, M * 8);
    m_output_ptrs.resize(M, nullptr);
    m_output_buffers.resize(M);

    logger()->debug("pfbc: allocated - M={}, K={}, N={}, tile_size={}",
                   M, K, N, TILE_SIZE);
}

auto polyphase_channelizer::reset_state() -> void {
    m_frame_idx = 0;
    m_samples_processed = 0;
    m_ts_initialized = false;
    m_ts_base = composite::timestamp{};

    if (m_ring.is_allocated()) {
        const auto M = static_cast<std::size_t>(m_num_channels);
        const auto K = static_cast<std::size_t>(m_taps_per_phase);
        m_ring.reset();
        m_ring.prefill_zeros((K - 1) * M);
    }

    if (m_l1_scratch) {
        std::fill(m_l1_scratch->begin(), m_l1_scratch->end(), cf32_t{0.0f, 0.0f});
    }

    for (auto& buf : m_output_buffers) {
        buf.reset();
    }
    std::fill(m_output_ptrs.begin(), m_output_ptrs.end(), nullptr);
}

auto polyphase_channelizer::configure() -> bool {
    m_configured = false;

    if (!validate_properties()) {
        return false;
    }

    try {
        const auto M = static_cast<std::size_t>(m_num_channels);
        const auto K = static_cast<std::size_t>(m_taps_per_phase);

        // Create processing engine (selects fused/tiled/batch based on M)
        m_engine = pfbc::make_engine(M, K);
        if (!m_engine) {
            logger()->error("pfbc: failed to create engine (M={}, K={})", M, K);
            return false;
        }

        allocate_buffers();
        reset_state();
        acquire_output_buffers();

        m_configured = true;

        logger()->info("pfbc: configured - {} channels, {} taps/phase, {} frame size, policy={}",
                      m_num_channels, m_taps_per_phase, m_frame_size, m_engine->name());

    } catch (const std::exception& e) {
        logger()->error("pfbc: configuration failed: {}", e.what());
        return false;
    }

    return true;
}

auto polyphase_channelizer::property_change_handler() -> void {
    configure();
}

// =============================================================================
// Output Buffer Management
// =============================================================================

auto polyphase_channelizer::acquire_output_buffers() -> void {
    const auto M = static_cast<std::size_t>(m_num_channels);

    for (std::size_t ch = 0; ch < M; ++ch) {
        auto pool_buf = m_output_pool->acquire();
        if (!pool_buf) {
            logger()->warn("pfbc: output pool exhausted, channel {} unavailable", ch);
            m_output_ptrs[ch] = nullptr;
            m_output_buffers[ch].reset();
            continue;
        }
        m_output_buffers[ch] = std::move(*pool_buf);
        m_output_ptrs[ch] = m_output_buffers[ch]->data();
    }
}

auto polyphase_channelizer::emit_frame() -> void {
    const auto M = static_cast<std::size_t>(m_num_channels);
    const auto N = static_cast<std::size_t>(m_frame_size);

    composite::timestamp frame_ts{};
    if (m_ts_initialized) {
        if (m_metadata.sample_rate > 0.0) {
            const uint64_t frame_start = m_samples_processed - N * M;
            const double offset_seconds = static_cast<double>(frame_start) / m_metadata.sample_rate;
            const auto offset_ns = std::chrono::nanoseconds(
                static_cast<int64_t>(offset_seconds * 1e9)
            );
            frame_ts = m_ts_base + offset_ns;
        } else {
            frame_ts = m_ts_base;
        }
    }

    for (std::size_t ch = 0; ch < M; ++ch) {
        if (!m_output_buffers[ch].has_value()) {
            continue;
        }

        auto buffer = std::move(m_output_buffers[ch].value());
        m_output_buffers[ch].reset();
        m_output_ptrs[ch] = nullptr;

        auto out = composite::immutable_buffer<cf32_t>(std::move(buffer));
        m_data_out.send_data(out, frame_ts);
    }

    m_frame_idx = 0;
    acquire_output_buffers();
}

// =============================================================================
// Tile Processing
// =============================================================================

auto polyphase_channelizer::process_tile() -> void {
    const auto M = static_cast<std::size_t>(m_num_channels);
    const auto N = static_cast<std::size_t>(m_frame_size);

    // Prepare write pointers offset by current frame index
    std::vector<cf32_t*> write_ptrs(M);
    for (std::size_t ch = 0; ch < M; ++ch) {
        write_ptrs[ch] = m_output_ptrs[ch] ? m_output_ptrs[ch] + m_frame_idx : nullptr;
    }

    // Engine handles the processing (fused for M=8, staged otherwise)
    m_engine->process_tile(
        m_ring.read_ptr(),
        m_coeffs->data(),
        write_ptrs.data(),
        m_l1_scratch->data(),
        m_fft.get()
    );

    m_ring.consume(TILE_SIZE * M);
    m_samples_processed += TILE_SIZE * M;
    m_frame_idx += TILE_SIZE;

    if (m_frame_idx >= N) {
        emit_frame();
    }
}

auto polyphase_channelizer::process_partial_tile(std::size_t count) -> void {
    if (count == 0) return;

    const auto M = static_cast<std::size_t>(m_num_channels);
    const auto N = static_cast<std::size_t>(m_frame_size);

    // Prepare write pointers offset by current frame index
    std::vector<cf32_t*> write_ptrs(M);
    for (std::size_t ch = 0; ch < M; ++ch) {
        write_ptrs[ch] = m_output_ptrs[ch] ? m_output_ptrs[ch] + m_frame_idx : nullptr;
    }

    // Engine handles partial processing
    m_engine->process_partial(
        m_ring.read_ptr(),
        m_coeffs->data(),
        write_ptrs.data(),
        m_l1_scratch->data(),
        m_fft.get(),
        count
    );

    m_ring.consume(count * M);
    m_samples_processed += count * M;
    m_frame_idx += count;

    if (m_frame_idx >= N) {
        emit_frame();
    }
}

// =============================================================================
// Main Processing Loop
// =============================================================================

auto polyphase_channelizer::process() -> composite::retval {
    using enum composite::retval;

    if (!m_configured) {
        return NOOP;
    }

    auto [buffer, ts, meta] = m_data_in.get_data();
    if (!buffer) {
        return NOOP;
    }

    // Handle metadata
    if (meta.has_value()) {
        m_metadata = meta.value();
        auto out_metadata = m_metadata;
        if (out_metadata.sample_rate > 0.0) {
            out_metadata.sample_rate /= static_cast<double>(m_num_channels);
        }
        m_data_out.send_metadata(out_metadata);
    }

    // Timestamp tracking
    const composite::timestamp zero_ts{};
    if (ts != zero_ts) {
        m_ts_base = ts;
        m_samples_processed = 0;
        m_ts_initialized = true;
    }

    // Ingest input
    auto span = buffer.as_span();
    m_ring.write(span.data(), span.size());

    // Process tiles
    const auto M = static_cast<std::size_t>(m_num_channels);
    const auto K = static_cast<std::size_t>(m_taps_per_phase);
    const auto N = static_cast<std::size_t>(m_frame_size);

    const std::size_t min_for_tile = (K - 1 + TILE_SIZE) * M;

    while (m_ring.available() >= min_for_tile) {
        const std::size_t remaining_in_frame = N - m_frame_idx;

        if (remaining_in_frame >= TILE_SIZE) {
            process_tile();
        } else {
            // Handle frame boundary - may span multiple frames when TILE_SIZE > N
            std::size_t rows_to_process = TILE_SIZE;

            while (rows_to_process > 0) {
                const std::size_t space_in_frame = N - m_frame_idx;
                const std::size_t batch = std::min(rows_to_process, space_in_frame);

                if (batch == 0 || m_ring.available() < (K - 1 + batch) * M) {
                    break;
                }

                process_partial_tile(batch);
                rows_to_process -= batch;
            }
        }
    }

    // Process remaining partial rows for lower latency
    const std::size_t history_needed = (K - 1) * M;

    while (m_ring.available() > history_needed) {
        const std::size_t avail = m_ring.available();
        const std::size_t rows_available = (avail - history_needed) / M;

        if (rows_available == 0 || rows_available >= TILE_SIZE) {
            break;  // No partial rows or enough for full tile (handled above)
        }

        const std::size_t space_in_frame = N - m_frame_idx;
        const std::size_t rows_to_process = std::min(rows_available, space_in_frame);

        if (rows_to_process == 0) {
            break;
        }

        process_partial_tile(rows_to_process);
    }

    return NORMAL;
}

// =============================================================================
// Component Factory
// =============================================================================

extern "C" {
    auto create(std::string_view id) -> std::shared_ptr<composite::component> {
        return std::make_shared<polyphase_channelizer>(id);
    }
}
