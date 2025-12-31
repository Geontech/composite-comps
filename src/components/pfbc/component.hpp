// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <composite/buffers/aligned_mem.hpp>
#include <composite/buffers/external_buffer.hpp>
#include <composite/buffers/slab_pool.hpp>
#include <composite/composite.hpp>

// Forward declaration
template <typename T>
class fft_plan;

class polyphase_channelizer : public composite::component {
    // Test access
    friend struct PfbcTestFixture;
    friend class pfbc_streaming_tests;

public:
    using cf32_t = std::complex<float>;
    using input_t = composite::immutable_buffer<cf32_t>;
    using input_port_t = composite::input_port<input_t>;
    using output_port_t = composite::output_port<input_t>;

    explicit polyphase_channelizer(std::string_view id);
    ~polyphase_channelizer() override = default;

    auto property_change_handler() -> void override;
    auto process() -> composite::retval override;

private:
    // Configuration and validation
    auto configure() -> bool;
    auto validate_properties() const -> bool;
    auto allocate_buffers() -> void;
    auto reset_state() -> void;
    auto prepare_coefficients() -> void;

    // Processing stages
    auto emit_frames(composite::timestamp base_ts) -> void;

    // Ports
    input_port_t m_data_in{"data_in"};
    output_port_t m_data_out{"data_out"};

    // Properties (configurable)
    uint32_t m_num_channels{64};       // M - must be power of 2
    uint32_t m_taps_per_phase{16};     // K - filter length per polyphase arm
    uint32_t m_frame_size{1024};       // N - samples per channel output frame
    std::vector<float> m_prototype_filter{}; // M*K taps, lowpass prototype
    uint32_t m_num_threads{4};         // OpenMP threads for filter stage

    // Internal state - interleaved coefficients (prepared once at configure)
    // Layout: [tap0: p0,p0,p1,p1,...,pM-1,pM-1][tap1: ...][...]
    // Each coefficient is doubled for complex multiply: h * (re,im) = (h*re, h*im)
    // Size: M * K * 2 floats (64-byte aligned for AVX-512)
    std::unique_ptr<composite::aligned_mem<float>> m_coeffs;

    // History length (K-1 rows needed for FIR filter continuity)
    std::size_t m_history_len{0};  // K - 1

    // Internal state - filter output buffer (direct input to FFT)
    // Layout: [t0: p0,p1,...,pM-1][t1: ...][...]
    // Size: M * max_outputs samples (64-byte aligned)
    std::unique_ptr<composite::aligned_mem<cf32_t>> m_filter_output;
    std::size_t m_filter_output_capacity{0};

    // Internal state - FFT (always FP32 for FFT library compatibility)
    // FFT writes directly to frame buffer, no intermediate output buffer needed
    std::unique_ptr<fft_plan<cf32_t>> m_fft;

    // Internal state - time-major frame accumulator (contiguous FFT output writes)
    // Layout: [t0: ch0,ch1,...,ch(M-1)][t1: ...][...]
    // Size: M * N samples (always FP32 for FFT)
    // Transposed on frame completion to channel-major output buffers.
    std::unique_ptr<composite::aligned_mem<cf32_t>> m_frame_tm;
    std::size_t m_frame_idx{0};  // Current sample index within frame (0..N-1)

    // Output buffer pool - eliminates malloc/free from hot path
    std::shared_ptr<composite::slab_pool<cf32_t>> m_output_pool;
    std::vector<cf32_t*> m_output_ptrs;
    std::vector<std::optional<composite::external_buffer<cf32_t>>> m_output_buffers;

    // History buffer - exactly (K-1)*M samples for FIR filter continuity
    // Persists across calls, updated after each process() with last (K-1) rows
    std::unique_ptr<composite::aligned_mem<cf32_t>> m_history;

    // Boundary buffer - max((2K-2)*M, K*M) samples for boundary outputs + scratch.
    // Used to assemble [history][first (K-1)*M of new data] for outputs that
    // span the history/new-data boundary. Also used as K*M scratch when tail
    // samples require assembling a full row for direct output.
    std::unique_ptr<composite::aligned_mem<cf32_t>> m_boundary_buffer;

    // Tail buffer - holds partial row (0 to M-1 samples) between calls
    // When input doesn't divide evenly by M, leftover samples wait here
    std::vector<cf32_t> m_tail;
    std::size_t m_tail_len{0};

    // Sample counter for timestamp calculation
    // Tracks total input samples processed since the current time base
    uint64_t m_stream_sample_idx{0};
    bool m_ts_initialized{false};
    composite::timestamp m_ts_base{0};  // Base timestamp for stream sample index 0

    // Metadata
    composite::metadata m_metadata{};
    bool m_configured{false};

}; // class polyphase_channelizer
