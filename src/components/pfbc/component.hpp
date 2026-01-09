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

#include "ring_buffer.hpp"

// Forward declarations
template <typename T>
class fft_plan;

namespace pfbc {
class engine_interface;
}

// =============================================================================
// Polyphase Filter Bank Channelizer
// =============================================================================
//
// High-performance PFBC using policy-based processing engine:
//
//   M=8:       Fused in-register pipeline (filter → FFT → transpose in ZMM)
//   M>=64:     L1-tiled processing (8-row tiles stay hot in cache)
//   Other M:   Staged batch processing
//
// Component manages all state; engine provides templated tile processing.
//
class polyphase_channelizer : public composite::component {
    // Test access
    friend struct PfbcTestFixture;
    friend class pfbc_streaming_tests;

public:
    using cf32_t = std::complex<float>;
    using input_t = composite::immutable_buffer<cf32_t>;
    using input_port_t = composite::input_port<input_t>;
    using output_port_t = composite::output_port<input_t>;

    static constexpr std::size_t TILE_SIZE = 8;

    explicit polyphase_channelizer(std::string_view id);
    ~polyphase_channelizer() override;

    auto property_change_handler() -> void override;
    auto process() -> composite::retval override;

private:
    // -------------------------------------------------------------------------
    // Configuration and Lifecycle
    // -------------------------------------------------------------------------
    auto configure() -> bool;
    auto validate_properties() const -> bool;
    auto allocate_buffers() -> void;
    auto reset_state() -> void;
    auto prepare_coefficients() -> void;

    // -------------------------------------------------------------------------
    // Processing Pipeline
    // -------------------------------------------------------------------------
    auto process_tile() -> void;
    auto process_partial_tile(std::size_t count) -> void;

    // -------------------------------------------------------------------------
    // Output Buffer Management
    // -------------------------------------------------------------------------
    auto emit_frame() -> void;
    auto acquire_output_buffers() -> void;

    // -------------------------------------------------------------------------
    // Ports
    // -------------------------------------------------------------------------
    input_port_t m_data_in{"data_in"};
    output_port_t m_data_out{"data_out"};

    // -------------------------------------------------------------------------
    // Properties (configurable)
    // -------------------------------------------------------------------------
    uint32_t m_num_channels{64};
    uint32_t m_taps_per_phase{16};
    uint32_t m_frame_size{1024};
    std::vector<float> m_prototype_filter{};

    // -------------------------------------------------------------------------
    // Processing Engine (templated tile dispatch)
    // -------------------------------------------------------------------------
    std::unique_ptr<pfbc::engine_interface> m_engine;

    // -------------------------------------------------------------------------
    // Ring Buffer
    // -------------------------------------------------------------------------
    pfbc::cf32_ring_buffer m_ring;

    // -------------------------------------------------------------------------
    // Coefficients
    // -------------------------------------------------------------------------
    std::unique_ptr<composite::aligned_mem<float>> m_coeffs;

    // -------------------------------------------------------------------------
    // L1 Scratch Buffer
    // -------------------------------------------------------------------------
    std::unique_ptr<composite::aligned_mem<cf32_t>> m_l1_scratch;

    // -------------------------------------------------------------------------
    // FFT (for partial tile fallback)
    // -------------------------------------------------------------------------
    std::unique_ptr<fft_plan<cf32_t>> m_fft;

    // -------------------------------------------------------------------------
    // Output Buffers
    // -------------------------------------------------------------------------
    std::shared_ptr<composite::slab_pool<cf32_t>> m_output_pool;
    std::vector<cf32_t*> m_output_ptrs;
    std::vector<std::optional<composite::external_buffer<cf32_t>>> m_output_buffers;
    std::size_t m_frame_idx{0};

    // -------------------------------------------------------------------------
    // Timestamp Tracking
    // -------------------------------------------------------------------------
    uint64_t m_samples_processed{0};
    bool m_ts_initialized{false};
    composite::timestamp m_ts_base{0};

    // -------------------------------------------------------------------------
    // Metadata and State
    // -------------------------------------------------------------------------
    composite::metadata m_metadata{};
    bool m_configured{false};

}; // class polyphase_channelizer
