// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#pragma once

#include <complex>
#include <cstddef>
#include <cstring>
#include <immintrin.h>

namespace kernels {

static_assert(sizeof(std::complex<float>) == sizeof(double), "transpose_avx512 assumes complex<float> is 64-bit");

// =============================================================================
// Interleaved Polyphase Filter Kernel
// =============================================================================
//
// This kernel processes the polyphase filterbank WITHOUT scatter/gather.
// Input data stays in its natural interleaved format, and all M phases are
// processed together in the inner loop.
//
// Key insight: Instead of M separate memory streams (one per phase lane),
// we have ONE sequential memory stream. This dramatically improves cache
// behavior and prefetcher efficiency.
//
// Memory Layout:
//   Input:  [t0: p0,p1,...,p(M-1)] [t1: p0,p1,...,p(M-1)] ... (time-major)
//   Coeffs: [tap0: h0,h0,h1,h1,...,h(M-1),h(M-1)] [tap1: ...] (doubled for complex)
//   Output: [t0: p0,p1,...,p(M-1)] [t1: ...] (ready for FFT, no gather needed)
//
// For M=64 channels:
//   - Each "row" is 64 complex samples = 512 bytes = 8 cache lines
//   - Sequential access pattern, prefetcher handles it perfectly
//   - All 64 phases computed with shared coefficient loads
//

// -----------------------------------------------------------------------------
// AVX-512 Implementation: 64-Channel Interleaved Filter
// -----------------------------------------------------------------------------
// Processes all 64 phases for one output time in a single pass.
// Uses 8 ZMM registers for accumulators (8 phases each = 64 total).
//
// Inner loop per tap:
//   - Load 64 complex samples (128 floats = 8 ZMM) - contiguous!
//   - Load 64 doubled coefficients (128 floats = 8 ZMM) - contiguous!
//   - 8 FMA instructions
//
// This is memory-bound, not compute-bound, which is what we want.

[[gnu::target("avx512f")]]
inline auto filter_interleaved_64ch_avx512(
    const std::complex<float>* __restrict input,   // [(K + num_outputs - 1) × 64] interleaved
    const float* __restrict coeffs,                // [K × 128] doubled coefficients
    std::complex<float>* __restrict output,        // [num_outputs × 64]
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t M = 64;
    constexpr std::size_t FLOATS_PER_ROW = M * 2;  // 128 floats per time step

    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    for (std::size_t t = 0; t < num_outputs; ++t) {
        // 8 ZMM accumulators for 64 phases (8 complex each)
        __m512 acc0 = _mm512_setzero_ps();  // phases 0-7
        __m512 acc1 = _mm512_setzero_ps();  // phases 8-15
        __m512 acc2 = _mm512_setzero_ps();  // phases 16-23
        __m512 acc3 = _mm512_setzero_ps();  // phases 24-31
        __m512 acc4 = _mm512_setzero_ps();  // phases 32-39
        __m512 acc5 = _mm512_setzero_ps();  // phases 40-47
        __m512 acc6 = _mm512_setzero_ps();  // phases 48-55
        __m512 acc7 = _mm512_setzero_ps();  // phases 56-63

        const float* x_ptr = x_base + t * FLOATS_PER_ROW;
        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            // Load 64 complex samples = 128 floats = 8 ZMM registers
            // These are CONTIGUOUS in memory - perfect for prefetcher
            __m512 x0 = _mm512_loadu_ps(x_ptr + 0);
            __m512 x1 = _mm512_loadu_ps(x_ptr + 16);
            __m512 x2 = _mm512_loadu_ps(x_ptr + 32);
            __m512 x3 = _mm512_loadu_ps(x_ptr + 48);
            __m512 x4 = _mm512_loadu_ps(x_ptr + 64);
            __m512 x5 = _mm512_loadu_ps(x_ptr + 80);
            __m512 x6 = _mm512_loadu_ps(x_ptr + 96);
            __m512 x7 = _mm512_loadu_ps(x_ptr + 112);

            // Load 64 doubled coefficients = 128 floats = 8 ZMM registers
            // Also CONTIGUOUS - shared across all output times for this tap
            __m512 h0 = _mm512_loadu_ps(h_ptr + 0);
            __m512 h1 = _mm512_loadu_ps(h_ptr + 16);
            __m512 h2 = _mm512_loadu_ps(h_ptr + 32);
            __m512 h3 = _mm512_loadu_ps(h_ptr + 48);
            __m512 h4 = _mm512_loadu_ps(h_ptr + 64);
            __m512 h5 = _mm512_loadu_ps(h_ptr + 80);
            __m512 h6 = _mm512_loadu_ps(h_ptr + 96);
            __m512 h7 = _mm512_loadu_ps(h_ptr + 112);

            // FMA: acc += x * h
            acc0 = _mm512_fmadd_ps(x0, h0, acc0);
            acc1 = _mm512_fmadd_ps(x1, h1, acc1);
            acc2 = _mm512_fmadd_ps(x2, h2, acc2);
            acc3 = _mm512_fmadd_ps(x3, h3, acc3);
            acc4 = _mm512_fmadd_ps(x4, h4, acc4);
            acc5 = _mm512_fmadd_ps(x5, h5, acc5);
            acc6 = _mm512_fmadd_ps(x6, h6, acc6);
            acc7 = _mm512_fmadd_ps(x7, h7, acc7);

            x_ptr += FLOATS_PER_ROW;  // Next time row
            h_ptr += FLOATS_PER_ROW;  // Next coefficient row
        }

        // Store 64 complex results - already in FFT order!
        float* out_ptr = out_base + t * FLOATS_PER_ROW;
        _mm512_storeu_ps(out_ptr + 0, acc0);
        _mm512_storeu_ps(out_ptr + 16, acc1);
        _mm512_storeu_ps(out_ptr + 32, acc2);
        _mm512_storeu_ps(out_ptr + 48, acc3);
        _mm512_storeu_ps(out_ptr + 64, acc4);
        _mm512_storeu_ps(out_ptr + 80, acc5);
        _mm512_storeu_ps(out_ptr + 96, acc6);
        _mm512_storeu_ps(out_ptr + 112, acc7);
    }
}

// -----------------------------------------------------------------------------
// AVX-512 Implementation: 64-Channel with 4x Time Unrolling
// -----------------------------------------------------------------------------
// Processes 4 output times simultaneously to improve instruction-level
// parallelism and amortize coefficient loads.
//
// Uses all 32 ZMM registers:
//   - 8 accumulators × 4 time steps = 32 registers
//
// Coefficient loads are shared across all 4 time steps!

[[gnu::target("avx512f")]]
inline auto filter_interleaved_64ch_x4_avx512(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t M = 64;
    constexpr std::size_t FLOATS_PER_ROW = M * 2;

    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

    // Main loop: 4 output times at once
    for (; t + 4 <= num_outputs; t += 4) {
        // 32 ZMM accumulators: 4 times × 8 phase groups
        __m512 acc_t0_p0 = _mm512_setzero_ps();
        __m512 acc_t0_p1 = _mm512_setzero_ps();
        __m512 acc_t0_p2 = _mm512_setzero_ps();
        __m512 acc_t0_p3 = _mm512_setzero_ps();
        __m512 acc_t0_p4 = _mm512_setzero_ps();
        __m512 acc_t0_p5 = _mm512_setzero_ps();
        __m512 acc_t0_p6 = _mm512_setzero_ps();
        __m512 acc_t0_p7 = _mm512_setzero_ps();

        __m512 acc_t1_p0 = _mm512_setzero_ps();
        __m512 acc_t1_p1 = _mm512_setzero_ps();
        __m512 acc_t1_p2 = _mm512_setzero_ps();
        __m512 acc_t1_p3 = _mm512_setzero_ps();
        __m512 acc_t1_p4 = _mm512_setzero_ps();
        __m512 acc_t1_p5 = _mm512_setzero_ps();
        __m512 acc_t1_p6 = _mm512_setzero_ps();
        __m512 acc_t1_p7 = _mm512_setzero_ps();

        __m512 acc_t2_p0 = _mm512_setzero_ps();
        __m512 acc_t2_p1 = _mm512_setzero_ps();
        __m512 acc_t2_p2 = _mm512_setzero_ps();
        __m512 acc_t2_p3 = _mm512_setzero_ps();
        __m512 acc_t2_p4 = _mm512_setzero_ps();
        __m512 acc_t2_p5 = _mm512_setzero_ps();
        __m512 acc_t2_p6 = _mm512_setzero_ps();
        __m512 acc_t2_p7 = _mm512_setzero_ps();

        __m512 acc_t3_p0 = _mm512_setzero_ps();
        __m512 acc_t3_p1 = _mm512_setzero_ps();
        __m512 acc_t3_p2 = _mm512_setzero_ps();
        __m512 acc_t3_p3 = _mm512_setzero_ps();
        __m512 acc_t3_p4 = _mm512_setzero_ps();
        __m512 acc_t3_p5 = _mm512_setzero_ps();
        __m512 acc_t3_p6 = _mm512_setzero_ps();
        __m512 acc_t3_p7 = _mm512_setzero_ps();

        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            // Load coefficients ONCE per tap (shared across all 4 time steps)
            __m512 h0 = _mm512_loadu_ps(h_ptr + 0);
            __m512 h1 = _mm512_loadu_ps(h_ptr + 16);
            __m512 h2 = _mm512_loadu_ps(h_ptr + 32);
            __m512 h3 = _mm512_loadu_ps(h_ptr + 48);
            __m512 h4 = _mm512_loadu_ps(h_ptr + 64);
            __m512 h5 = _mm512_loadu_ps(h_ptr + 80);
            __m512 h6 = _mm512_loadu_ps(h_ptr + 96);
            __m512 h7 = _mm512_loadu_ps(h_ptr + 112);

            // Time 0
            const float* x_t0 = x_base + (t + k) * FLOATS_PER_ROW;
            acc_t0_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 0), h0, acc_t0_p0);
            acc_t0_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 16), h1, acc_t0_p1);
            acc_t0_p2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 32), h2, acc_t0_p2);
            acc_t0_p3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 48), h3, acc_t0_p3);
            acc_t0_p4 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 64), h4, acc_t0_p4);
            acc_t0_p5 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 80), h5, acc_t0_p5);
            acc_t0_p6 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 96), h6, acc_t0_p6);
            acc_t0_p7 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 112), h7, acc_t0_p7);

            // Time 1
            const float* x_t1 = x_base + (t + 1 + k) * FLOATS_PER_ROW;
            acc_t1_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 0), h0, acc_t1_p0);
            acc_t1_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 16), h1, acc_t1_p1);
            acc_t1_p2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 32), h2, acc_t1_p2);
            acc_t1_p3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 48), h3, acc_t1_p3);
            acc_t1_p4 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 64), h4, acc_t1_p4);
            acc_t1_p5 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 80), h5, acc_t1_p5);
            acc_t1_p6 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 96), h6, acc_t1_p6);
            acc_t1_p7 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 112), h7, acc_t1_p7);

            // Time 2
            const float* x_t2 = x_base + (t + 2 + k) * FLOATS_PER_ROW;
            acc_t2_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 0), h0, acc_t2_p0);
            acc_t2_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 16), h1, acc_t2_p1);
            acc_t2_p2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 32), h2, acc_t2_p2);
            acc_t2_p3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 48), h3, acc_t2_p3);
            acc_t2_p4 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 64), h4, acc_t2_p4);
            acc_t2_p5 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 80), h5, acc_t2_p5);
            acc_t2_p6 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 96), h6, acc_t2_p6);
            acc_t2_p7 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 112), h7, acc_t2_p7);

            // Time 3
            const float* x_t3 = x_base + (t + 3 + k) * FLOATS_PER_ROW;
            acc_t3_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 0), h0, acc_t3_p0);
            acc_t3_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 16), h1, acc_t3_p1);
            acc_t3_p2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 32), h2, acc_t3_p2);
            acc_t3_p3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 48), h3, acc_t3_p3);
            acc_t3_p4 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 64), h4, acc_t3_p4);
            acc_t3_p5 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 80), h5, acc_t3_p5);
            acc_t3_p6 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 96), h6, acc_t3_p6);
            acc_t3_p7 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 112), h7, acc_t3_p7);

            h_ptr += FLOATS_PER_ROW;
        }

        // Store all 4 × 64 = 256 complex outputs
        float* out_t0 = out_base + t * FLOATS_PER_ROW;
        _mm512_storeu_ps(out_t0 + 0, acc_t0_p0);
        _mm512_storeu_ps(out_t0 + 16, acc_t0_p1);
        _mm512_storeu_ps(out_t0 + 32, acc_t0_p2);
        _mm512_storeu_ps(out_t0 + 48, acc_t0_p3);
        _mm512_storeu_ps(out_t0 + 64, acc_t0_p4);
        _mm512_storeu_ps(out_t0 + 80, acc_t0_p5);
        _mm512_storeu_ps(out_t0 + 96, acc_t0_p6);
        _mm512_storeu_ps(out_t0 + 112, acc_t0_p7);

        float* out_t1 = out_base + (t + 1) * FLOATS_PER_ROW;
        _mm512_storeu_ps(out_t1 + 0, acc_t1_p0);
        _mm512_storeu_ps(out_t1 + 16, acc_t1_p1);
        _mm512_storeu_ps(out_t1 + 32, acc_t1_p2);
        _mm512_storeu_ps(out_t1 + 48, acc_t1_p3);
        _mm512_storeu_ps(out_t1 + 64, acc_t1_p4);
        _mm512_storeu_ps(out_t1 + 80, acc_t1_p5);
        _mm512_storeu_ps(out_t1 + 96, acc_t1_p6);
        _mm512_storeu_ps(out_t1 + 112, acc_t1_p7);

        float* out_t2 = out_base + (t + 2) * FLOATS_PER_ROW;
        _mm512_storeu_ps(out_t2 + 0, acc_t2_p0);
        _mm512_storeu_ps(out_t2 + 16, acc_t2_p1);
        _mm512_storeu_ps(out_t2 + 32, acc_t2_p2);
        _mm512_storeu_ps(out_t2 + 48, acc_t2_p3);
        _mm512_storeu_ps(out_t2 + 64, acc_t2_p4);
        _mm512_storeu_ps(out_t2 + 80, acc_t2_p5);
        _mm512_storeu_ps(out_t2 + 96, acc_t2_p6);
        _mm512_storeu_ps(out_t2 + 112, acc_t2_p7);

        float* out_t3 = out_base + (t + 3) * FLOATS_PER_ROW;
        _mm512_storeu_ps(out_t3 + 0, acc_t3_p0);
        _mm512_storeu_ps(out_t3 + 16, acc_t3_p1);
        _mm512_storeu_ps(out_t3 + 32, acc_t3_p2);
        _mm512_storeu_ps(out_t3 + 48, acc_t3_p3);
        _mm512_storeu_ps(out_t3 + 64, acc_t3_p4);
        _mm512_storeu_ps(out_t3 + 80, acc_t3_p5);
        _mm512_storeu_ps(out_t3 + 96, acc_t3_p6);
        _mm512_storeu_ps(out_t3 + 112, acc_t3_p7);
    }

    // Tail: remaining 1-3 output times
    for (; t < num_outputs; ++t) {
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps();
        __m512 acc5 = _mm512_setzero_ps();
        __m512 acc6 = _mm512_setzero_ps();
        __m512 acc7 = _mm512_setzero_ps();

        const float* x_ptr = x_base + t * FLOATS_PER_ROW;
        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h0 = _mm512_loadu_ps(h_ptr + 0);
            __m512 h1 = _mm512_loadu_ps(h_ptr + 16);
            __m512 h2 = _mm512_loadu_ps(h_ptr + 32);
            __m512 h3 = _mm512_loadu_ps(h_ptr + 48);
            __m512 h4 = _mm512_loadu_ps(h_ptr + 64);
            __m512 h5 = _mm512_loadu_ps(h_ptr + 80);
            __m512 h6 = _mm512_loadu_ps(h_ptr + 96);
            __m512 h7 = _mm512_loadu_ps(h_ptr + 112);

            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 0), h0, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 16), h1, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 32), h2, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 48), h3, acc3);
            acc4 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 64), h4, acc4);
            acc5 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 80), h5, acc5);
            acc6 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 96), h6, acc6);
            acc7 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 112), h7, acc7);

            x_ptr += FLOATS_PER_ROW;
            h_ptr += FLOATS_PER_ROW;
        }

        float* out_ptr = out_base + t * FLOATS_PER_ROW;
        _mm512_storeu_ps(out_ptr + 0, acc0);
        _mm512_storeu_ps(out_ptr + 16, acc1);
        _mm512_storeu_ps(out_ptr + 32, acc2);
        _mm512_storeu_ps(out_ptr + 48, acc3);
        _mm512_storeu_ps(out_ptr + 64, acc4);
        _mm512_storeu_ps(out_ptr + 80, acc5);
        _mm512_storeu_ps(out_ptr + 96, acc6);
        _mm512_storeu_ps(out_ptr + 112, acc7);
    }
}

// -----------------------------------------------------------------------------
// 8-Channel Specialized Kernel (16x time unroll)
// -----------------------------------------------------------------------------
// M=8: 1 ZMM per output time → can process 16 output times at once

[[gnu::target("avx512f")]]
inline auto filter_interleaved_8ch_x16_avx512(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t M = 8;
    constexpr std::size_t FLOATS_PER_ROW = M * 2;  // 16 floats per time step

    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

    // Main loop: 16 output times at once
    for (; t + 16 <= num_outputs; t += 16) {
        // 16 ZMM accumulators (1 per output time)
        __m512 acc0  = _mm512_setzero_ps();
        __m512 acc1  = _mm512_setzero_ps();
        __m512 acc2  = _mm512_setzero_ps();
        __m512 acc3  = _mm512_setzero_ps();
        __m512 acc4  = _mm512_setzero_ps();
        __m512 acc5  = _mm512_setzero_ps();
        __m512 acc6  = _mm512_setzero_ps();
        __m512 acc7  = _mm512_setzero_ps();
        __m512 acc8  = _mm512_setzero_ps();
        __m512 acc9  = _mm512_setzero_ps();
        __m512 acc10 = _mm512_setzero_ps();
        __m512 acc11 = _mm512_setzero_ps();
        __m512 acc12 = _mm512_setzero_ps();
        __m512 acc13 = _mm512_setzero_ps();
        __m512 acc14 = _mm512_setzero_ps();
        __m512 acc15 = _mm512_setzero_ps();

        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h = _mm512_loadu_ps(h_ptr);

            acc0  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0 + k) * FLOATS_PER_ROW), h, acc0);
            acc1  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1 + k) * FLOATS_PER_ROW), h, acc1);
            acc2  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2 + k) * FLOATS_PER_ROW), h, acc2);
            acc3  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3 + k) * FLOATS_PER_ROW), h, acc3);
            acc4  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 4 + k) * FLOATS_PER_ROW), h, acc4);
            acc5  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 5 + k) * FLOATS_PER_ROW), h, acc5);
            acc6  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 6 + k) * FLOATS_PER_ROW), h, acc6);
            acc7  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 7 + k) * FLOATS_PER_ROW), h, acc7);
            acc8  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 8 + k) * FLOATS_PER_ROW), h, acc8);
            acc9  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 9 + k) * FLOATS_PER_ROW), h, acc9);
            acc10 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 10 + k) * FLOATS_PER_ROW), h, acc10);
            acc11 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 11 + k) * FLOATS_PER_ROW), h, acc11);
            acc12 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 12 + k) * FLOATS_PER_ROW), h, acc12);
            acc13 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 13 + k) * FLOATS_PER_ROW), h, acc13);
            acc14 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 14 + k) * FLOATS_PER_ROW), h, acc14);
            acc15 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 15 + k) * FLOATS_PER_ROW), h, acc15);

            h_ptr += FLOATS_PER_ROW;
        }

        _mm512_storeu_ps(out_base + (t + 0) * FLOATS_PER_ROW, acc0);
        _mm512_storeu_ps(out_base + (t + 1) * FLOATS_PER_ROW, acc1);
        _mm512_storeu_ps(out_base + (t + 2) * FLOATS_PER_ROW, acc2);
        _mm512_storeu_ps(out_base + (t + 3) * FLOATS_PER_ROW, acc3);
        _mm512_storeu_ps(out_base + (t + 4) * FLOATS_PER_ROW, acc4);
        _mm512_storeu_ps(out_base + (t + 5) * FLOATS_PER_ROW, acc5);
        _mm512_storeu_ps(out_base + (t + 6) * FLOATS_PER_ROW, acc6);
        _mm512_storeu_ps(out_base + (t + 7) * FLOATS_PER_ROW, acc7);
        _mm512_storeu_ps(out_base + (t + 8) * FLOATS_PER_ROW, acc8);
        _mm512_storeu_ps(out_base + (t + 9) * FLOATS_PER_ROW, acc9);
        _mm512_storeu_ps(out_base + (t + 10) * FLOATS_PER_ROW, acc10);
        _mm512_storeu_ps(out_base + (t + 11) * FLOATS_PER_ROW, acc11);
        _mm512_storeu_ps(out_base + (t + 12) * FLOATS_PER_ROW, acc12);
        _mm512_storeu_ps(out_base + (t + 13) * FLOATS_PER_ROW, acc13);
        _mm512_storeu_ps(out_base + (t + 14) * FLOATS_PER_ROW, acc14);
        _mm512_storeu_ps(out_base + (t + 15) * FLOATS_PER_ROW, acc15);
    }

    // Tail: remaining output times (scalar-ish, could add 8x/4x/1x tails)
    for (; t < num_outputs; ++t) {
        __m512 acc = _mm512_setzero_ps();
        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h = _mm512_loadu_ps(h_ptr);
            __m512 x = _mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW);
            acc = _mm512_fmadd_ps(x, h, acc);
            h_ptr += FLOATS_PER_ROW;
        }

        _mm512_storeu_ps(out_base + t * FLOATS_PER_ROW, acc);
    }
}

// -----------------------------------------------------------------------------
// 16-Channel Specialized Kernel (8x time unroll)
// -----------------------------------------------------------------------------

[[gnu::target("avx512f")]]
inline auto filter_interleaved_16ch_x8_avx512(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t M = 16;
    constexpr std::size_t FLOATS_PER_ROW = M * 2;  // 32 floats

    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

    // Main loop: 8 output times at once (2 ZMM × 8 = 16 ZMM accumulators)
    for (; t + 8 <= num_outputs; t += 8) {
        // 16 accumulators: 8 times × 2 phase groups
        __m512 acc_t0_p0 = _mm512_setzero_ps(), acc_t0_p1 = _mm512_setzero_ps();
        __m512 acc_t1_p0 = _mm512_setzero_ps(), acc_t1_p1 = _mm512_setzero_ps();
        __m512 acc_t2_p0 = _mm512_setzero_ps(), acc_t2_p1 = _mm512_setzero_ps();
        __m512 acc_t3_p0 = _mm512_setzero_ps(), acc_t3_p1 = _mm512_setzero_ps();
        __m512 acc_t4_p0 = _mm512_setzero_ps(), acc_t4_p1 = _mm512_setzero_ps();
        __m512 acc_t5_p0 = _mm512_setzero_ps(), acc_t5_p1 = _mm512_setzero_ps();
        __m512 acc_t6_p0 = _mm512_setzero_ps(), acc_t6_p1 = _mm512_setzero_ps();
        __m512 acc_t7_p0 = _mm512_setzero_ps(), acc_t7_p1 = _mm512_setzero_ps();

        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h0 = _mm512_loadu_ps(h_ptr + 0);
            __m512 h1 = _mm512_loadu_ps(h_ptr + 16);

            #define PROCESS_TIME(T) \
                acc_t##T##_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + T + k) * FLOATS_PER_ROW + 0), h0, acc_t##T##_p0); \
                acc_t##T##_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + T + k) * FLOATS_PER_ROW + 16), h1, acc_t##T##_p1);

            PROCESS_TIME(0)
            PROCESS_TIME(1)
            PROCESS_TIME(2)
            PROCESS_TIME(3)
            PROCESS_TIME(4)
            PROCESS_TIME(5)
            PROCESS_TIME(6)
            PROCESS_TIME(7)

            #undef PROCESS_TIME

            h_ptr += FLOATS_PER_ROW;
        }

        #define STORE_TIME(T) \
            _mm512_storeu_ps(out_base + (t + T) * FLOATS_PER_ROW + 0, acc_t##T##_p0); \
            _mm512_storeu_ps(out_base + (t + T) * FLOATS_PER_ROW + 16, acc_t##T##_p1);

        STORE_TIME(0)
        STORE_TIME(1)
        STORE_TIME(2)
        STORE_TIME(3)
        STORE_TIME(4)
        STORE_TIME(5)
        STORE_TIME(6)
        STORE_TIME(7)

        #undef STORE_TIME
    }

    // Tail
    for (; t < num_outputs; ++t) {
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h0 = _mm512_loadu_ps(h_ptr + 0);
            __m512 h1 = _mm512_loadu_ps(h_ptr + 16);
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW + 0), h0, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW + 16), h1, acc1);
            h_ptr += FLOATS_PER_ROW;
        }

        _mm512_storeu_ps(out_base + t * FLOATS_PER_ROW + 0, acc0);
        _mm512_storeu_ps(out_base + t * FLOATS_PER_ROW + 16, acc1);
    }
}

// -----------------------------------------------------------------------------
// 32-Channel Specialized Kernel (4x time unroll)
// -----------------------------------------------------------------------------

[[gnu::target("avx512f")]]
inline auto filter_interleaved_32ch_x4_avx512(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t M = 32;
    constexpr std::size_t FLOATS_PER_ROW = M * 2;  // 64 floats

    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

    // Main loop: 4 output times at once (4 ZMM × 4 = 16 ZMM accumulators)
    for (; t + 4 <= num_outputs; t += 4) {
        __m512 acc_t0_p0 = _mm512_setzero_ps(), acc_t0_p1 = _mm512_setzero_ps();
        __m512 acc_t0_p2 = _mm512_setzero_ps(), acc_t0_p3 = _mm512_setzero_ps();
        __m512 acc_t1_p0 = _mm512_setzero_ps(), acc_t1_p1 = _mm512_setzero_ps();
        __m512 acc_t1_p2 = _mm512_setzero_ps(), acc_t1_p3 = _mm512_setzero_ps();
        __m512 acc_t2_p0 = _mm512_setzero_ps(), acc_t2_p1 = _mm512_setzero_ps();
        __m512 acc_t2_p2 = _mm512_setzero_ps(), acc_t2_p3 = _mm512_setzero_ps();
        __m512 acc_t3_p0 = _mm512_setzero_ps(), acc_t3_p1 = _mm512_setzero_ps();
        __m512 acc_t3_p2 = _mm512_setzero_ps(), acc_t3_p3 = _mm512_setzero_ps();

        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h0 = _mm512_loadu_ps(h_ptr + 0);
            __m512 h1 = _mm512_loadu_ps(h_ptr + 16);
            __m512 h2 = _mm512_loadu_ps(h_ptr + 32);
            __m512 h3 = _mm512_loadu_ps(h_ptr + 48);

            #define PROCESS_TIME(T) \
                acc_t##T##_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + T + k) * FLOATS_PER_ROW + 0), h0, acc_t##T##_p0); \
                acc_t##T##_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + T + k) * FLOATS_PER_ROW + 16), h1, acc_t##T##_p1); \
                acc_t##T##_p2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + T + k) * FLOATS_PER_ROW + 32), h2, acc_t##T##_p2); \
                acc_t##T##_p3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + T + k) * FLOATS_PER_ROW + 48), h3, acc_t##T##_p3);

            PROCESS_TIME(0)
            PROCESS_TIME(1)
            PROCESS_TIME(2)
            PROCESS_TIME(3)

            #undef PROCESS_TIME

            h_ptr += FLOATS_PER_ROW;
        }

        #define STORE_TIME(T) \
            _mm512_storeu_ps(out_base + (t + T) * FLOATS_PER_ROW + 0, acc_t##T##_p0); \
            _mm512_storeu_ps(out_base + (t + T) * FLOATS_PER_ROW + 16, acc_t##T##_p1); \
            _mm512_storeu_ps(out_base + (t + T) * FLOATS_PER_ROW + 32, acc_t##T##_p2); \
            _mm512_storeu_ps(out_base + (t + T) * FLOATS_PER_ROW + 48, acc_t##T##_p3);

        STORE_TIME(0)
        STORE_TIME(1)
        STORE_TIME(2)
        STORE_TIME(3)

        #undef STORE_TIME
    }

    // Tail
    for (; t < num_outputs; ++t) {
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h0 = _mm512_loadu_ps(h_ptr + 0);
            __m512 h1 = _mm512_loadu_ps(h_ptr + 16);
            __m512 h2 = _mm512_loadu_ps(h_ptr + 32);
            __m512 h3 = _mm512_loadu_ps(h_ptr + 48);
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW + 0), h0, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW + 16), h1, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW + 32), h2, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW + 48), h3, acc3);
            h_ptr += FLOATS_PER_ROW;
        }

        _mm512_storeu_ps(out_base + t * FLOATS_PER_ROW + 0, acc0);
        _mm512_storeu_ps(out_base + t * FLOATS_PER_ROW + 16, acc1);
        _mm512_storeu_ps(out_base + t * FLOATS_PER_ROW + 32, acc2);
        _mm512_storeu_ps(out_base + t * FLOATS_PER_ROW + 48, acc3);
    }
}

// -----------------------------------------------------------------------------
// 128-Channel Specialized Kernel (2x time unroll)
// -----------------------------------------------------------------------------

[[gnu::target("avx512f")]]
inline auto filter_interleaved_128ch_x2_avx512(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t M = 128;
    constexpr std::size_t FLOATS_PER_ROW = M * 2;  // 256 floats

    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

    // Main loop: 2 output times at once (16 ZMM × 2 = 32 ZMM accumulators)
    for (; t + 2 <= num_outputs; t += 2) {
        // Time 0: 16 accumulators for 128 phases
        __m512 acc_t0_p0  = _mm512_setzero_ps(), acc_t0_p1  = _mm512_setzero_ps();
        __m512 acc_t0_p2  = _mm512_setzero_ps(), acc_t0_p3  = _mm512_setzero_ps();
        __m512 acc_t0_p4  = _mm512_setzero_ps(), acc_t0_p5  = _mm512_setzero_ps();
        __m512 acc_t0_p6  = _mm512_setzero_ps(), acc_t0_p7  = _mm512_setzero_ps();
        __m512 acc_t0_p8  = _mm512_setzero_ps(), acc_t0_p9  = _mm512_setzero_ps();
        __m512 acc_t0_p10 = _mm512_setzero_ps(), acc_t0_p11 = _mm512_setzero_ps();
        __m512 acc_t0_p12 = _mm512_setzero_ps(), acc_t0_p13 = _mm512_setzero_ps();
        __m512 acc_t0_p14 = _mm512_setzero_ps(), acc_t0_p15 = _mm512_setzero_ps();

        // Time 1: 16 accumulators
        __m512 acc_t1_p0  = _mm512_setzero_ps(), acc_t1_p1  = _mm512_setzero_ps();
        __m512 acc_t1_p2  = _mm512_setzero_ps(), acc_t1_p3  = _mm512_setzero_ps();
        __m512 acc_t1_p4  = _mm512_setzero_ps(), acc_t1_p5  = _mm512_setzero_ps();
        __m512 acc_t1_p6  = _mm512_setzero_ps(), acc_t1_p7  = _mm512_setzero_ps();
        __m512 acc_t1_p8  = _mm512_setzero_ps(), acc_t1_p9  = _mm512_setzero_ps();
        __m512 acc_t1_p10 = _mm512_setzero_ps(), acc_t1_p11 = _mm512_setzero_ps();
        __m512 acc_t1_p12 = _mm512_setzero_ps(), acc_t1_p13 = _mm512_setzero_ps();
        __m512 acc_t1_p14 = _mm512_setzero_ps(), acc_t1_p15 = _mm512_setzero_ps();

        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            // Load coefficients (will cause some spilling, but coeffs are reused)
            __m512 h0  = _mm512_loadu_ps(h_ptr + 0);
            __m512 h1  = _mm512_loadu_ps(h_ptr + 16);
            __m512 h2  = _mm512_loadu_ps(h_ptr + 32);
            __m512 h3  = _mm512_loadu_ps(h_ptr + 48);
            __m512 h4  = _mm512_loadu_ps(h_ptr + 64);
            __m512 h5  = _mm512_loadu_ps(h_ptr + 80);
            __m512 h6  = _mm512_loadu_ps(h_ptr + 96);
            __m512 h7  = _mm512_loadu_ps(h_ptr + 112);
            __m512 h8  = _mm512_loadu_ps(h_ptr + 128);
            __m512 h9  = _mm512_loadu_ps(h_ptr + 144);
            __m512 h10 = _mm512_loadu_ps(h_ptr + 160);
            __m512 h11 = _mm512_loadu_ps(h_ptr + 176);
            __m512 h12 = _mm512_loadu_ps(h_ptr + 192);
            __m512 h13 = _mm512_loadu_ps(h_ptr + 208);
            __m512 h14 = _mm512_loadu_ps(h_ptr + 224);
            __m512 h15 = _mm512_loadu_ps(h_ptr + 240);

            // Time 0
            const float* x_t0 = x_base + (t + 0 + k) * FLOATS_PER_ROW;
            acc_t0_p0  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 0), h0, acc_t0_p0);
            acc_t0_p1  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 16), h1, acc_t0_p1);
            acc_t0_p2  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 32), h2, acc_t0_p2);
            acc_t0_p3  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 48), h3, acc_t0_p3);
            acc_t0_p4  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 64), h4, acc_t0_p4);
            acc_t0_p5  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 80), h5, acc_t0_p5);
            acc_t0_p6  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 96), h6, acc_t0_p6);
            acc_t0_p7  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 112), h7, acc_t0_p7);
            acc_t0_p8  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 128), h8, acc_t0_p8);
            acc_t0_p9  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 144), h9, acc_t0_p9);
            acc_t0_p10 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 160), h10, acc_t0_p10);
            acc_t0_p11 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 176), h11, acc_t0_p11);
            acc_t0_p12 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 192), h12, acc_t0_p12);
            acc_t0_p13 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 208), h13, acc_t0_p13);
            acc_t0_p14 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 224), h14, acc_t0_p14);
            acc_t0_p15 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 240), h15, acc_t0_p15);

            // Time 1
            const float* x_t1 = x_base + (t + 1 + k) * FLOATS_PER_ROW;
            acc_t1_p0  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 0), h0, acc_t1_p0);
            acc_t1_p1  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 16), h1, acc_t1_p1);
            acc_t1_p2  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 32), h2, acc_t1_p2);
            acc_t1_p3  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 48), h3, acc_t1_p3);
            acc_t1_p4  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 64), h4, acc_t1_p4);
            acc_t1_p5  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 80), h5, acc_t1_p5);
            acc_t1_p6  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 96), h6, acc_t1_p6);
            acc_t1_p7  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 112), h7, acc_t1_p7);
            acc_t1_p8  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 128), h8, acc_t1_p8);
            acc_t1_p9  = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 144), h9, acc_t1_p9);
            acc_t1_p10 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 160), h10, acc_t1_p10);
            acc_t1_p11 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 176), h11, acc_t1_p11);
            acc_t1_p12 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 192), h12, acc_t1_p12);
            acc_t1_p13 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 208), h13, acc_t1_p13);
            acc_t1_p14 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 224), h14, acc_t1_p14);
            acc_t1_p15 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 240), h15, acc_t1_p15);

            h_ptr += FLOATS_PER_ROW;
        }

        // Store time 0
        float* out_t0 = out_base + (t + 0) * FLOATS_PER_ROW;
        _mm512_storeu_ps(out_t0 + 0, acc_t0_p0);
        _mm512_storeu_ps(out_t0 + 16, acc_t0_p1);
        _mm512_storeu_ps(out_t0 + 32, acc_t0_p2);
        _mm512_storeu_ps(out_t0 + 48, acc_t0_p3);
        _mm512_storeu_ps(out_t0 + 64, acc_t0_p4);
        _mm512_storeu_ps(out_t0 + 80, acc_t0_p5);
        _mm512_storeu_ps(out_t0 + 96, acc_t0_p6);
        _mm512_storeu_ps(out_t0 + 112, acc_t0_p7);
        _mm512_storeu_ps(out_t0 + 128, acc_t0_p8);
        _mm512_storeu_ps(out_t0 + 144, acc_t0_p9);
        _mm512_storeu_ps(out_t0 + 160, acc_t0_p10);
        _mm512_storeu_ps(out_t0 + 176, acc_t0_p11);
        _mm512_storeu_ps(out_t0 + 192, acc_t0_p12);
        _mm512_storeu_ps(out_t0 + 208, acc_t0_p13);
        _mm512_storeu_ps(out_t0 + 224, acc_t0_p14);
        _mm512_storeu_ps(out_t0 + 240, acc_t0_p15);

        // Store time 1
        float* out_t1 = out_base + (t + 1) * FLOATS_PER_ROW;
        _mm512_storeu_ps(out_t1 + 0, acc_t1_p0);
        _mm512_storeu_ps(out_t1 + 16, acc_t1_p1);
        _mm512_storeu_ps(out_t1 + 32, acc_t1_p2);
        _mm512_storeu_ps(out_t1 + 48, acc_t1_p3);
        _mm512_storeu_ps(out_t1 + 64, acc_t1_p4);
        _mm512_storeu_ps(out_t1 + 80, acc_t1_p5);
        _mm512_storeu_ps(out_t1 + 96, acc_t1_p6);
        _mm512_storeu_ps(out_t1 + 112, acc_t1_p7);
        _mm512_storeu_ps(out_t1 + 128, acc_t1_p8);
        _mm512_storeu_ps(out_t1 + 144, acc_t1_p9);
        _mm512_storeu_ps(out_t1 + 160, acc_t1_p10);
        _mm512_storeu_ps(out_t1 + 176, acc_t1_p11);
        _mm512_storeu_ps(out_t1 + 192, acc_t1_p12);
        _mm512_storeu_ps(out_t1 + 208, acc_t1_p13);
        _mm512_storeu_ps(out_t1 + 224, acc_t1_p14);
        _mm512_storeu_ps(out_t1 + 240, acc_t1_p15);
    }

    // Tail: single output time
    for (; t < num_outputs; ++t) {
        __m512 acc0  = _mm512_setzero_ps(), acc1  = _mm512_setzero_ps();
        __m512 acc2  = _mm512_setzero_ps(), acc3  = _mm512_setzero_ps();
        __m512 acc4  = _mm512_setzero_ps(), acc5  = _mm512_setzero_ps();
        __m512 acc6  = _mm512_setzero_ps(), acc7  = _mm512_setzero_ps();
        __m512 acc8  = _mm512_setzero_ps(), acc9  = _mm512_setzero_ps();
        __m512 acc10 = _mm512_setzero_ps(), acc11 = _mm512_setzero_ps();
        __m512 acc12 = _mm512_setzero_ps(), acc13 = _mm512_setzero_ps();
        __m512 acc14 = _mm512_setzero_ps(), acc15 = _mm512_setzero_ps();

        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            const float* x_ptr = x_base + (t + k) * FLOATS_PER_ROW;
            acc0  = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 0), _mm512_loadu_ps(h_ptr + 0), acc0);
            acc1  = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 16), _mm512_loadu_ps(h_ptr + 16), acc1);
            acc2  = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 32), _mm512_loadu_ps(h_ptr + 32), acc2);
            acc3  = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 48), _mm512_loadu_ps(h_ptr + 48), acc3);
            acc4  = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 64), _mm512_loadu_ps(h_ptr + 64), acc4);
            acc5  = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 80), _mm512_loadu_ps(h_ptr + 80), acc5);
            acc6  = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 96), _mm512_loadu_ps(h_ptr + 96), acc6);
            acc7  = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 112), _mm512_loadu_ps(h_ptr + 112), acc7);
            acc8  = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 128), _mm512_loadu_ps(h_ptr + 128), acc8);
            acc9  = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 144), _mm512_loadu_ps(h_ptr + 144), acc9);
            acc10 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 160), _mm512_loadu_ps(h_ptr + 160), acc10);
            acc11 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 176), _mm512_loadu_ps(h_ptr + 176), acc11);
            acc12 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 192), _mm512_loadu_ps(h_ptr + 192), acc12);
            acc13 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 208), _mm512_loadu_ps(h_ptr + 208), acc13);
            acc14 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 224), _mm512_loadu_ps(h_ptr + 224), acc14);
            acc15 = _mm512_fmadd_ps(_mm512_loadu_ps(x_ptr + 240), _mm512_loadu_ps(h_ptr + 240), acc15);
            h_ptr += FLOATS_PER_ROW;
        }

        float* out_ptr = out_base + t * FLOATS_PER_ROW;
        _mm512_storeu_ps(out_ptr + 0, acc0);
        _mm512_storeu_ps(out_ptr + 16, acc1);
        _mm512_storeu_ps(out_ptr + 32, acc2);
        _mm512_storeu_ps(out_ptr + 48, acc3);
        _mm512_storeu_ps(out_ptr + 64, acc4);
        _mm512_storeu_ps(out_ptr + 80, acc5);
        _mm512_storeu_ps(out_ptr + 96, acc6);
        _mm512_storeu_ps(out_ptr + 112, acc7);
        _mm512_storeu_ps(out_ptr + 128, acc8);
        _mm512_storeu_ps(out_ptr + 144, acc9);
        _mm512_storeu_ps(out_ptr + 160, acc10);
        _mm512_storeu_ps(out_ptr + 176, acc11);
        _mm512_storeu_ps(out_ptr + 192, acc12);
        _mm512_storeu_ps(out_ptr + 208, acc13);
        _mm512_storeu_ps(out_ptr + 224, acc14);
        _mm512_storeu_ps(out_ptr + 240, acc15);
    }
}

// -----------------------------------------------------------------------------
// Generic M-Channel AVX-512 Implementation
// -----------------------------------------------------------------------------
// For channel counts other than 8/16/32/64/128, or M=256+
// Uses 8-phase-at-a-time processing with time unrolling

[[gnu::target("avx512f")]]
inline auto filter_interleaved_generic_avx512(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_channels,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    const std::size_t floats_per_row = num_channels * 2;
    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    // Process 8 phases (16 floats) at a time
    const std::size_t num_phase_groups = num_channels / 8;
    const std::size_t remainder_phases = num_channels % 8;

    for (std::size_t t = 0; t < num_outputs; ++t) {
        const float* x_row_base = x_base + t * floats_per_row;
        float* out_row = out_base + t * floats_per_row;

        // Process complete groups of 8 phases
        for (std::size_t pg = 0; pg < num_phase_groups; ++pg) {
            __m512 acc = _mm512_setzero_ps();

            const float* x_ptr = x_row_base + pg * 16;
            const float* h_ptr = coeffs + pg * 16;

            for (std::size_t k = 0; k < num_taps; ++k) {
                __m512 x = _mm512_loadu_ps(x_ptr);
                __m512 h = _mm512_loadu_ps(h_ptr);
                acc = _mm512_fmadd_ps(x, h, acc);

                x_ptr += floats_per_row;
                h_ptr += floats_per_row;
            }

            _mm512_storeu_ps(out_row + pg * 16, acc);
        }

        // Scalar remainder for phases not divisible by 8
        for (std::size_t p = num_phase_groups * 8; p < num_channels; ++p) {
            float acc_re = 0.0f;
            float acc_im = 0.0f;

            for (std::size_t k = 0; k < num_taps; ++k) {
                const float* x_ptr = x_base + (t + k) * floats_per_row + p * 2;
                const float* h_ptr = coeffs + k * floats_per_row + p * 2;

                acc_re += x_ptr[0] * h_ptr[0];
                acc_im += x_ptr[1] * h_ptr[1];
            }

            out_row[p * 2 + 0] = acc_re;
            out_row[p * 2 + 1] = acc_im;
        }
    }
}

// -----------------------------------------------------------------------------
// AVX2 Implementation
// -----------------------------------------------------------------------------

[[gnu::target("avx2,fma")]]
inline auto filter_interleaved_generic_avx2(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_channels,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    const std::size_t floats_per_row = num_channels * 2;
    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    // Process 4 phases (8 floats) at a time with AVX2
    const std::size_t num_phase_groups = num_channels / 4;
    const std::size_t remainder_phases = num_channels % 4;

    for (std::size_t t = 0; t < num_outputs; ++t) {
        const float* x_row_base = x_base + t * floats_per_row;
        float* out_row = out_base + t * floats_per_row;

        for (std::size_t pg = 0; pg < num_phase_groups; ++pg) {
            __m256 acc = _mm256_setzero_ps();

            const float* x_ptr = x_row_base + pg * 8;
            const float* h_ptr = coeffs + pg * 8;

            for (std::size_t k = 0; k < num_taps; ++k) {
                __m256 x = _mm256_loadu_ps(x_ptr);
                __m256 h = _mm256_loadu_ps(h_ptr);
                acc = _mm256_fmadd_ps(x, h, acc);

                x_ptr += floats_per_row;
                h_ptr += floats_per_row;
            }

            _mm256_storeu_ps(out_row + pg * 8, acc);
        }

        // Scalar remainder
        for (std::size_t p = num_phase_groups * 4; p < num_channels; ++p) {
            float acc_re = 0.0f;
            float acc_im = 0.0f;

            for (std::size_t k = 0; k < num_taps; ++k) {
                const float* x_ptr = x_base + (t + k) * floats_per_row + p * 2;
                const float* h_ptr = coeffs + k * floats_per_row + p * 2;

                acc_re += x_ptr[0] * h_ptr[0];
                acc_im += x_ptr[1] * h_ptr[1];
            }

            out_row[p * 2 + 0] = acc_re;
            out_row[p * 2 + 1] = acc_im;
        }
    }
}

// -----------------------------------------------------------------------------
// Scalar Reference Implementation
// -----------------------------------------------------------------------------

[[gnu::target("default")]]
inline auto filter_interleaved_scalar(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_channels,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    const std::size_t floats_per_row = num_channels * 2;
    const float* x_base = reinterpret_cast<const float*>(input);
    const float* h_base = coeffs;
    float* out_base = reinterpret_cast<float*>(output);

    for (std::size_t t = 0; t < num_outputs; ++t) {
        for (std::size_t p = 0; p < num_channels; ++p) {
            float acc_re = 0.0f;
            float acc_im = 0.0f;

            for (std::size_t k = 0; k < num_taps; ++k) {
                const float* x_ptr = x_base + (t + k) * floats_per_row + p * 2;
                const float* h_ptr = h_base + k * floats_per_row + p * 2;

                acc_re += x_ptr[0] * h_ptr[0];
                acc_im += x_ptr[1] * h_ptr[1];
            }

            out_base[t * floats_per_row + p * 2 + 0] = acc_re;
            out_base[t * floats_per_row + p * 2 + 1] = acc_im;
        }
    }
}

// -----------------------------------------------------------------------------
// Dispatcher
// -----------------------------------------------------------------------------
// Selects the optimal kernel based on channel count.
// Each specialized kernel uses the maximum time unrolling that fits in
// 32 ZMM registers while keeping coefficient registers reusable.

inline auto filter_interleaved(
    const std::complex<float>* input,
    const float* coeffs,
    std::complex<float>* output,
    std::size_t num_channels,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    switch (num_channels) {
        case 8:
            filter_interleaved_8ch_x16_avx512(input, coeffs, output, num_outputs, num_taps);
            break;
        case 16:
            filter_interleaved_16ch_x8_avx512(input, coeffs, output, num_outputs, num_taps);
            break;
        case 32:
            filter_interleaved_32ch_x4_avx512(input, coeffs, output, num_outputs, num_taps);
            break;
        case 64:
            filter_interleaved_64ch_x4_avx512(input, coeffs, output, num_outputs, num_taps);
            break;
        case 128:
            filter_interleaved_128ch_x2_avx512(input, coeffs, output, num_outputs, num_taps);
            break;
        default:
            // Generic path for M=256+ or non-power-of-2
            filter_interleaved_generic_avx512(input, coeffs, output, num_channels, num_outputs, num_taps);
            break;
    }
}

// -----------------------------------------------------------------------------
// History Buffer Management
// -----------------------------------------------------------------------------
// Helper to update the interleaved history buffer after processing.
// Copies the last (num_taps - 1) rows to the beginning.

inline auto update_interleaved_history(
    std::complex<float>* buffer,
    std::size_t num_channels,
    std::size_t num_rows_processed,
    std::size_t history_rows  // = num_taps - 1
) -> void
{
    const std::size_t bytes_per_row = num_channels * sizeof(std::complex<float>);
    const std::size_t history_bytes = history_rows * bytes_per_row;

    // Source: end of processed data (last history_rows rows)
    const auto* src = buffer + num_rows_processed * num_channels;

    // Destination: beginning of buffer
    std::memmove(buffer, src, history_bytes);
}

// Transpose an 8x8 block of complex floats using AVX-512.
// Each complex<float> is 8 bytes, so we treat them as doubles for 64-bit operations.
// Input:  8 rows of 8 complex floats, row stride = src_stride complex floats
// Output: 8 rows of 8 complex floats, row stride = dst_stride complex floats
//
// src layout (row-major, time-major):
//   src[0]: t0_ch0, t0_ch1, t0_ch2, t0_ch3, t0_ch4, t0_ch5, t0_ch6, t0_ch7
//   src[1]: t1_ch0, t1_ch1, ...
//   ...
//
// dst layout (column-major, channel-major):
//   dst[0]: t0_ch0, t1_ch0, t2_ch0, t3_ch0, t4_ch0, t5_ch0, t6_ch0, t7_ch0
//   dst[1]: t0_ch1, t1_ch1, ...
//   ...
[[gnu::target("avx512f")]]
inline void transpose_8x8_cf32_avx512(
    const std::complex<float>* src,
    std::complex<float>* dst,
    std::size_t src_stride,  // distance between rows in source (channels per row = M)
    std::size_t dst_stride   // distance between rows in dest (frames per channel = N)
) {
    // Reinterpret as double* since complex<float> is 8 bytes
    const double* s = reinterpret_cast<const double*>(src);
    double* d = reinterpret_cast<double*>(dst);

    // Load 8 rows
    __m512d r0 = _mm512_loadu_pd(s + 0 * src_stride);
    __m512d r1 = _mm512_loadu_pd(s + 1 * src_stride);
    __m512d r2 = _mm512_loadu_pd(s + 2 * src_stride);
    __m512d r3 = _mm512_loadu_pd(s + 3 * src_stride);
    __m512d r4 = _mm512_loadu_pd(s + 4 * src_stride);
    __m512d r5 = _mm512_loadu_pd(s + 5 * src_stride);
    __m512d r6 = _mm512_loadu_pd(s + 6 * src_stride);
    __m512d r7 = _mm512_loadu_pd(s + 7 * src_stride);

    // Stage 1: 2x2 transposes using unpack
    // unpacklo/hi interleave elements from two registers
    __m512d t0 = _mm512_unpacklo_pd(r0, r1);  // r0[0],r1[0], r0[2],r1[2], r0[4],r1[4], r0[6],r1[6]
    __m512d t1 = _mm512_unpackhi_pd(r0, r1);  // r0[1],r1[1], r0[3],r1[3], r0[5],r1[5], r0[7],r1[7]
    __m512d t2 = _mm512_unpacklo_pd(r2, r3);
    __m512d t3 = _mm512_unpackhi_pd(r2, r3);
    __m512d t4 = _mm512_unpacklo_pd(r4, r5);
    __m512d t5 = _mm512_unpackhi_pd(r4, r5);
    __m512d t6 = _mm512_unpacklo_pd(r6, r7);
    __m512d t7 = _mm512_unpackhi_pd(r6, r7);

    // Stage 2: 4x4 transposes using shuffle_f64x2
    // shuffle_f64x2 rearranges 128-bit (2 double) lanes
    // imm8 selects which 128-bit lane from each source goes where
    r0 = _mm512_shuffle_f64x2(t0, t2, 0x88);  // t0[0,1], t2[0,1], t0[4,5], t2[4,5]
    r1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    r2 = _mm512_shuffle_f64x2(t0, t2, 0xDD);  // t0[2,3], t2[2,3], t0[6,7], t2[6,7]
    r3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    r4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
    r5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    r6 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    r7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);

    // Stage 3: 8x8 transpose using shuffle_f64x2 again
    // Now rearrange 256-bit halves
    t0 = _mm512_shuffle_f64x2(r0, r4, 0x88);  // r0[0..3], r4[0..3]
    t1 = _mm512_shuffle_f64x2(r1, r5, 0x88);
    t2 = _mm512_shuffle_f64x2(r2, r6, 0x88);
    t3 = _mm512_shuffle_f64x2(r3, r7, 0x88);
    t4 = _mm512_shuffle_f64x2(r0, r4, 0xDD);  // r0[4..7], r4[4..7]
    t5 = _mm512_shuffle_f64x2(r1, r5, 0xDD);
    t6 = _mm512_shuffle_f64x2(r2, r6, 0xDD);
    t7 = _mm512_shuffle_f64x2(r3, r7, 0xDD);

    // Store 8 transposed rows
    _mm512_storeu_pd(d + 0 * dst_stride, t0);
    _mm512_storeu_pd(d + 1 * dst_stride, t1);
    _mm512_storeu_pd(d + 2 * dst_stride, t2);
    _mm512_storeu_pd(d + 3 * dst_stride, t3);
    _mm512_storeu_pd(d + 4 * dst_stride, t4);
    _mm512_storeu_pd(d + 5 * dst_stride, t5);
    _mm512_storeu_pd(d + 6 * dst_stride, t6);
    _mm512_storeu_pd(d + 7 * dst_stride, t7);
}

// Transpose a full M x N matrix of complex floats from time-major to channel-major.
// Processes in 8x8 tiles for efficiency, with scalar fallback for edges.
//
// Input (time-major):  src[t * M + ch]  where t in [0,N), ch in [0,M)
// Output (channel-major): dst[ch * N + t]
//
// M = number of channels (columns in src, rows in dst)
// N = number of time steps (rows in src, columns in dst)
[[gnu::target("avx512f")]]
inline void transpose_tm_to_cm(
    const std::complex<float>* src,
    std::complex<float>* dst,
    std::size_t M,
    std::size_t N
) {
    const std::size_t M8 = M & ~7ULL;  // M rounded down to multiple of 8
    const std::size_t N8 = N & ~7ULL;  // N rounded down to multiple of 8

    // Process 8x8 tiles
    for (std::size_t ch0 = 0; ch0 < M8; ch0 += 8) {
        for (std::size_t t0 = 0; t0 < N8; t0 += 8) {
            // src[t0][ch0] -> dst[ch0][t0]
            const std::complex<float>* tile_src = src + t0 * M + ch0;
            std::complex<float>* tile_dst = dst + ch0 * N + t0;
            transpose_8x8_cf32_avx512(tile_src, tile_dst, M, N);
        }
    }

    // Handle right edge (channels M8..M-1) with full-height tiles
    for (std::size_t ch = M8; ch < M; ++ch) {
        for (std::size_t t = 0; t < N; ++t) {
            dst[ch * N + t] = src[t * M + ch];
        }
    }

    // Handle bottom edge (times N8..N-1) for the main tile region
    for (std::size_t ch = 0; ch < M8; ++ch) {
        for (std::size_t t = N8; t < N; ++t) {
            dst[ch * N + t] = src[t * M + ch];
        }
    }
}

// Transpose directly to M separate output buffers (channel-major output).
// Each output buffer holds N samples for one channel.
//
// Input (time-major): src[t * M + ch]
// Output: dst_ptrs[ch][t] for each channel
//
// dst_ptrs: array of M pointers, each pointing to buffer of size N
// If dst_ptrs[ch] is nullptr, that channel is skipped.
[[gnu::target("avx512f")]]
inline void transpose_to_channel_buffers(
    const std::complex<float>* src,
    std::complex<float>* const* dst_ptrs,
    std::size_t M,
    std::size_t N
) {
    const std::size_t M8 = M & ~7ULL;
    const std::size_t N8 = N & ~7ULL;

    // For 8x8 tile transpose, we need contiguous output.
    // Since outputs are separate buffers, we transpose 8 time steps at a time
    // and scatter to the appropriate channel buffers.

    // Process in blocks of 8 time steps
    for (std::size_t t0 = 0; t0 < N8; t0 += 8) {
        // For each group of 8 channels
        for (std::size_t ch0 = 0; ch0 < M8; ch0 += 8) {
            // Load 8 rows of 8 elements
            const double* s = reinterpret_cast<const double*>(src + t0 * M + ch0);

            __m512d r0 = _mm512_loadu_pd(s + 0 * M);
            __m512d r1 = _mm512_loadu_pd(s + 1 * M);
            __m512d r2 = _mm512_loadu_pd(s + 2 * M);
            __m512d r3 = _mm512_loadu_pd(s + 3 * M);
            __m512d r4 = _mm512_loadu_pd(s + 4 * M);
            __m512d r5 = _mm512_loadu_pd(s + 5 * M);
            __m512d r6 = _mm512_loadu_pd(s + 6 * M);
            __m512d r7 = _mm512_loadu_pd(s + 7 * M);

            // Transpose
            __m512d t0v = _mm512_unpacklo_pd(r0, r1);
            __m512d t1v = _mm512_unpackhi_pd(r0, r1);
            __m512d t2v = _mm512_unpacklo_pd(r2, r3);
            __m512d t3v = _mm512_unpackhi_pd(r2, r3);
            __m512d t4v = _mm512_unpacklo_pd(r4, r5);
            __m512d t5v = _mm512_unpackhi_pd(r4, r5);
            __m512d t6v = _mm512_unpacklo_pd(r6, r7);
            __m512d t7v = _mm512_unpackhi_pd(r6, r7);

            r0 = _mm512_shuffle_f64x2(t0v, t2v, 0x88);
            r1 = _mm512_shuffle_f64x2(t1v, t3v, 0x88);
            r2 = _mm512_shuffle_f64x2(t0v, t2v, 0xDD);
            r3 = _mm512_shuffle_f64x2(t1v, t3v, 0xDD);
            r4 = _mm512_shuffle_f64x2(t4v, t6v, 0x88);
            r5 = _mm512_shuffle_f64x2(t5v, t7v, 0x88);
            r6 = _mm512_shuffle_f64x2(t4v, t6v, 0xDD);
            r7 = _mm512_shuffle_f64x2(t5v, t7v, 0xDD);

            t0v = _mm512_shuffle_f64x2(r0, r4, 0x88);
            t1v = _mm512_shuffle_f64x2(r1, r5, 0x88);
            t2v = _mm512_shuffle_f64x2(r2, r6, 0x88);
            t3v = _mm512_shuffle_f64x2(r3, r7, 0x88);
            t4v = _mm512_shuffle_f64x2(r0, r4, 0xDD);
            t5v = _mm512_shuffle_f64x2(r1, r5, 0xDD);
            t6v = _mm512_shuffle_f64x2(r2, r6, 0xDD);
            t7v = _mm512_shuffle_f64x2(r3, r7, 0xDD);

            // Store to channel buffers (8 channels, 8 time steps each)
            double* d0 = dst_ptrs[ch0 + 0] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 0] + t0) : nullptr;
            double* d1 = dst_ptrs[ch0 + 1] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 1] + t0) : nullptr;
            double* d2 = dst_ptrs[ch0 + 2] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 2] + t0) : nullptr;
            double* d3 = dst_ptrs[ch0 + 3] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 3] + t0) : nullptr;
            double* d4 = dst_ptrs[ch0 + 4] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 4] + t0) : nullptr;
            double* d5 = dst_ptrs[ch0 + 5] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 5] + t0) : nullptr;
            double* d6 = dst_ptrs[ch0 + 6] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 6] + t0) : nullptr;
            double* d7 = dst_ptrs[ch0 + 7] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 7] + t0) : nullptr;

            if (d0) _mm512_storeu_pd(d0, t0v);
            if (d1) _mm512_storeu_pd(d1, t1v);
            if (d2) _mm512_storeu_pd(d2, t2v);
            if (d3) _mm512_storeu_pd(d3, t3v);
            if (d4) _mm512_storeu_pd(d4, t4v);
            if (d5) _mm512_storeu_pd(d5, t5v);
            if (d6) _mm512_storeu_pd(d6, t6v);
            if (d7) _mm512_storeu_pd(d7, t7v);
        }

        // Handle remaining channels (M8..M-1) for this time block
        for (std::size_t ch = M8; ch < M; ++ch) {
            if (!dst_ptrs[ch]) continue;
            for (std::size_t t = t0; t < t0 + 8; ++t) {
                dst_ptrs[ch][t] = src[t * M + ch];
            }
        }
    }

    // Handle remaining time steps (N8..N-1)
    for (std::size_t t = N8; t < N; ++t) {
        for (std::size_t ch = 0; ch < M; ++ch) {
            if (!dst_ptrs[ch]) continue;
            dst_ptrs[ch][t] = src[t * M + ch];
        }
    }
}

} // namespace kernels
