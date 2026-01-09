// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#pragma once

// =============================================================================
// PFBC Engine - Compile-Time Policy Selection
// =============================================================================
// Provides templated tile processing with automatic strategy selection.
// This is a thin dispatch layer - all state management stays in component.cpp.
//
// Policy selection by M (num_channels):
//   M=8:       fused_m8      - In-register vertical FFT-8, fully fused pipeline
//   M=16:      fused_m16     - In-register vertical FFT-16, fully fused pipeline
//   M=32:      hybrid_m32    - In-register FFT-32, L1 scratch transpose (~1.5x faster)
//   M>=64:     staged        - Filter → FFTW → transpose via L1 scratch
//

#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <string_view>

#include "kernels.hpp"
#include "fft_plan.hpp"

namespace pfbc {

using cf32_t = std::complex<float>;

// =============================================================================
// Compile-Time Traits
// =============================================================================

template<std::size_t M, std::size_t K>
struct engine_traits {
    static constexpr std::size_t num_channels = M;
    static constexpr std::size_t taps_per_phase = K;
    static constexpr std::size_t tile_size = 8;
    static constexpr std::size_t history_rows = K - 1;
    static constexpr std::size_t history_samples = history_rows * M;

    // Policy selection: fused for M=8, M=16; hybrid for M=32
    static constexpr bool use_fused_m8 = (M == 8);
    static constexpr bool use_fused_m16 = (M == 16);
    static constexpr bool use_hybrid_m32 = (M == 32);
    static constexpr bool use_fused = use_fused_m8 || use_fused_m16 || use_hybrid_m32;

    static_assert(std::has_single_bit(M), "M must be a power of 2");
    static_assert(K > 0, "K must be positive");
    static_assert(M >= 8, "M must be at least 8");
};

// =============================================================================
// Policy: Fused M=8 (Vertical FFT-8)
// =============================================================================
// For M=8, the entire pipeline (filter → transpose → FFT → store) happens in
// registers using the vertical FFT-8 approach:
//   1. Filter 8 rows → 8 accumulators (row-major)
//   2. 8×8 transpose (row-major → column-major)
//   3. Vertical FFT-8 (no shuffles - each lane is a separate FFT)
//   4. Direct store to channel buffers (already in channel-scatter format)
//
// This provides 2x throughput vs the horizontal FFT approach by eliminating
// the post-FFT transpose entirely.

template<std::size_t K>
struct fused_engine_m8 {
    using traits = engine_traits<8, K>;

    static constexpr std::size_t M = 8;
    static constexpr std::size_t tile_size = 8;

    [[nodiscard]] static constexpr auto name() noexcept -> std::string_view {
        return "fused_m8_vertical";
    }

    // Process a full 8x8 tile directly to channel buffers
    // Uses vertical FFT-8: filter → transpose → FFT → direct store
    static auto process_tile(
        const cf32_t* input,           // Pointer to input samples (K-1+8 rows)
        const float* coeffs,           // Blocked coefficients
        cf32_t* const* channel_ptrs,   // Output pointers (already offset by frame_idx)
        [[maybe_unused]] cf32_t* scratch,  // Unused for fused path
        [[maybe_unused]] fft_plan<cf32_t>* fft  // Unused for fused path
    ) noexcept -> void {
        kernels::filter_fft_transpose_8ch_vertical(
            input,
            coeffs,
            const_cast<cf32_t**>(channel_ptrs),
            tile_size,
            K
        );
    }

    // Partial tile processing - vertical kernel handles any row count
    static auto process_partial(
        const cf32_t* input,
        const float* coeffs,
        cf32_t* const* channel_ptrs,
        [[maybe_unused]] cf32_t* scratch,
        [[maybe_unused]] fft_plan<cf32_t>* fft,
        std::size_t num_rows
    ) noexcept -> void {
        // Vertical kernel handles partial tiles efficiently
        kernels::filter_fft_transpose_8ch_vertical(
            input,
            coeffs,
            const_cast<cf32_t**>(channel_ptrs),
            num_rows,
            K
        );
    }
};

// =============================================================================
// Policy: Fused M=16 (Vertical FFT-16)
// =============================================================================
// For M=16, uses 8-way time unrolling with the vertical FFT-16 kernel.
// Processes 8 outputs at a time using 8 parallel FFT-16s with zero cross-lane
// shuffles during the FFT stage.

template<std::size_t K>
struct fused_engine_m16 {
    using traits = engine_traits<16, K>;

    static constexpr std::size_t M = 16;
    static constexpr std::size_t tile_size = 8;  // Process 8 rows per tile

    [[nodiscard]] static constexpr auto name() noexcept -> std::string_view {
        return "fused_m16_vertical";
    }

    // Process a full tile directly to channel buffers
    // Uses vertical FFT-16: filter → transpose → FFT → direct store
    static auto process_tile(
        const cf32_t* input,           // Pointer to input samples (K-1+8 rows)
        const float* coeffs,           // Blocked coefficients
        cf32_t* const* channel_ptrs,   // Output pointers (already offset by frame_idx)
        [[maybe_unused]] cf32_t* scratch,  // Unused for fused path
        [[maybe_unused]] fft_plan<cf32_t>* fft  // Unused for fused path
    ) noexcept -> void {
        kernels::filter_fft_transpose_16ch_vertical(
            input,
            coeffs,
            const_cast<cf32_t**>(channel_ptrs),
            tile_size,
            K
        );
    }

    // Partial tile processing - vertical kernel handles any row count
    static auto process_partial(
        const cf32_t* input,
        const float* coeffs,
        cf32_t* const* channel_ptrs,
        cf32_t* scratch,
        fft_plan<cf32_t>* fft,
        std::size_t num_rows
    ) noexcept -> void {
        // Vertical kernel handles any count efficiently (main loop + tail)
        kernels::filter_fft_transpose_16ch_vertical(
            input,
            coeffs,
            const_cast<cf32_t**>(channel_ptrs),
            num_rows,
            K
        );
    }
};

// =============================================================================
// Policy: Hybrid M=32 (In-Register FFT-32, L1 Scratch Transpose)
// =============================================================================
// For M=32, uses a hybrid approach:
//   1. Filter 4 rows × 32 channels → 16 ZMM accumulators
//   2. FFT-32 each row in registers → store to L1 scratch
//   3. Transpose 4×32 scratch → 32 channel buffers
//
// This eliminates one memory round-trip vs fully staged approach.

template<std::size_t K>
struct hybrid_engine_m32 {
    using traits = engine_traits<32, K>;

    static constexpr std::size_t M = 32;
    static constexpr std::size_t tile_size = 4;  // Process 4 rows per tile

    [[nodiscard]] static constexpr auto name() noexcept -> std::string_view {
        return "hybrid_m32";
    }

    // Process a full tile directly to channel buffers
    // Uses hybrid approach: filter+FFT in registers, transpose via L1 scratch
    static auto process_tile(
        const cf32_t* input,           // Pointer to input samples (K-1+4 rows)
        const float* coeffs,           // Blocked coefficients
        cf32_t* const* channel_ptrs,   // Output pointers (already offset by frame_idx)
        cf32_t* scratch,               // 4×32 L1 scratch buffer
        [[maybe_unused]] fft_plan<cf32_t>* fft  // Unused for hybrid path
    ) noexcept -> void {
        kernels::filter_fft_transpose_32ch_hybrid(
            input,
            coeffs,
            const_cast<cf32_t**>(channel_ptrs),
            scratch,
            tile_size,
            K
        );
    }

    // Partial tile processing - hybrid kernel handles any row count
    static auto process_partial(
        const cf32_t* input,
        const float* coeffs,
        cf32_t* const* channel_ptrs,
        cf32_t* scratch,
        [[maybe_unused]] fft_plan<cf32_t>* fft,
        std::size_t num_rows
    ) noexcept -> void {
        kernels::filter_fft_transpose_32ch_hybrid(
            input,
            coeffs,
            const_cast<cf32_t**>(channel_ptrs),
            scratch,
            num_rows,
            K
        );
    }
};

// =============================================================================
// Policy: Staged Processing (M >= 64)
// =============================================================================
// For M not in {8, 16, 32}, use the standard staged pipeline:
//   1. Filter → L1 scratch
//   2. In-place FFT via FFTW on each row
//   3. Transpose scratch → channel buffers

template<std::size_t M, std::size_t K>
struct staged_engine {
    using traits = engine_traits<M, K>;

    static constexpr std::size_t tile_size = 8;

    [[nodiscard]] static constexpr auto name() noexcept -> std::string_view {
        return "staged";
    }

    // Process a tile through staged pipeline
    static auto process_tile(
        const cf32_t* input,
        const float* coeffs,
        cf32_t* const* channel_ptrs,
        cf32_t* scratch,
        fft_plan<cf32_t>* fft
    ) noexcept -> void {
        process_partial(input, coeffs, channel_ptrs, scratch, fft, tile_size);
    }

    // Process arbitrary number of rows
    static auto process_partial(
        const cf32_t* input,
        const float* coeffs,
        cf32_t* const* channel_ptrs,
        cf32_t* scratch,
        fft_plan<cf32_t>* fft,
        std::size_t num_rows
    ) noexcept -> void {
        // Stage A: Filter → scratch
        kernels::filter_blocked(input, coeffs, scratch, M, num_rows, K);

        // Stage B: In-place FFT on each row
        for (std::size_t t = 0; t < num_rows; ++t) {
            cf32_t* row = scratch + t * M;
            fft->execute(row, row);
        }

        // Stage C: Transpose to channel buffers
        if (num_rows == tile_size) {
            kernels::transpose_tile_to_channel_buffers(
                scratch,
                const_cast<cf32_t**>(channel_ptrs),
                M,
                0  // channel_ptrs already offset
            );
        } else {
            // Scalar transpose for partial tiles
            for (std::size_t t = 0; t < num_rows; ++t) {
                for (std::size_t ch = 0; ch < M; ++ch) {
                    if (channel_ptrs[ch]) {
                        const_cast<cf32_t*>(channel_ptrs[ch])[t] = scratch[t * M + ch];
                    }
                }
            }
        }
    }
};

// =============================================================================
// Policy Selector
// =============================================================================

template<std::size_t M, std::size_t K>
using selected_engine = std::conditional_t<
    engine_traits<M, K>::use_fused_m8,
    fused_engine_m8<K>,
    std::conditional_t<
        engine_traits<M, K>::use_fused_m16,
        fused_engine_m16<K>,
        std::conditional_t<
            engine_traits<M, K>::use_hybrid_m32,
            hybrid_engine_m32<K>,
            staged_engine<M, K>
        >
    >
>;

// =============================================================================
// Engine Wrapper (Type-Erased Interface for Runtime M/K)
// =============================================================================
// Provides a common interface when M and K are runtime values.

class engine_interface {
public:
    virtual ~engine_interface() = default;

    virtual auto process_tile(
        const cf32_t* input,
        const float* coeffs,
        cf32_t* const* channel_ptrs,
        cf32_t* scratch,
        fft_plan<cf32_t>* fft
    ) noexcept -> void = 0;

    virtual auto process_partial(
        const cf32_t* input,
        const float* coeffs,
        cf32_t* const* channel_ptrs,
        cf32_t* scratch,
        fft_plan<cf32_t>* fft,
        std::size_t num_rows
    ) noexcept -> void = 0;

    [[nodiscard]] virtual auto name() const noexcept -> std::string_view = 0;
    [[nodiscard]] virtual auto tile_size() const noexcept -> std::size_t = 0;
};

template<std::size_t M, std::size_t K>
class engine_impl final : public engine_interface {
    using policy = selected_engine<M, K>;

public:
    auto process_tile(
        const cf32_t* input,
        const float* coeffs,
        cf32_t* const* channel_ptrs,
        cf32_t* scratch,
        fft_plan<cf32_t>* fft
    ) noexcept -> void override {
        policy::process_tile(input, coeffs, channel_ptrs, scratch, fft);
    }

    auto process_partial(
        const cf32_t* input,
        const float* coeffs,
        cf32_t* const* channel_ptrs,
        cf32_t* scratch,
        fft_plan<cf32_t>* fft,
        std::size_t num_rows
    ) noexcept -> void override {
        policy::process_partial(input, coeffs, channel_ptrs, scratch, fft, num_rows);
    }

    [[nodiscard]]
    auto name() const noexcept -> std::string_view override {
        return policy::name();
    }

    [[nodiscard]]
    auto tile_size() const noexcept -> std::size_t override {
        return policy::tile_size;
    }
};

// Factory for runtime M/K selection
template<std::size_t K>
[[nodiscard]]
inline auto make_engine(std::size_t M) -> std::unique_ptr<engine_interface> {
    switch (M) {
        case 8:    return std::make_unique<engine_impl<8, K>>();
        case 16:   return std::make_unique<engine_impl<16, K>>();
        case 32:   return std::make_unique<engine_impl<32, K>>();
        case 64:   return std::make_unique<engine_impl<64, K>>();
        case 128:  return std::make_unique<engine_impl<128, K>>();
        case 256:  return std::make_unique<engine_impl<256, K>>();
        case 512:  return std::make_unique<engine_impl<512, K>>();
        case 1024: return std::make_unique<engine_impl<1024, K>>();
        default:   return nullptr;
    }
}

[[nodiscard]]
inline auto make_engine(std::size_t M, std::size_t K) -> std::unique_ptr<engine_interface> {
    switch (K) {
        case 1:  return make_engine<1>(M);
        case 2:  return make_engine<2>(M);
        case 4:  return make_engine<4>(M);
        case 8:  return make_engine<8>(M);
        case 12: return make_engine<12>(M);
        case 16: return make_engine<16>(M);
        case 24: return make_engine<24>(M);
        case 32: return make_engine<32>(M);
        default: return nullptr;
    }
}

} // namespace pfbc
