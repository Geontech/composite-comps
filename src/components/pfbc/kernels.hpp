// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#pragma once

#include <complex>
#include <cstddef>
#include <cstring>
#include <immintrin.h>

namespace kernels {

// =============================================================================
// Interleaved Polyphase Filter Kernel - Template Specializations
// =============================================================================
//
// Template-based dispatch for compile-time optimization based on channel count M.
// Runtime ISA dispatch selects AVX-512/AVX2/scalar implementation.
//
// Memory Layout:
//   Input:  [t0: p0,p1,...,p(M-1)] [t1: p0,p1,...,p(M-1)] ... (time-major)
//   Coeffs: [tap0: h0,h0,h1,h1,...,h(M-1),h(M-1)] [tap1: ...] (doubled for complex)
//   Output: [t0: p0,p1,...,p(M-1)] [t1: ...] (ready for FFT)
//
// Specializations:
//   M=8:  16x time unroll, all phases together (17 ZMM)
//   M=16: 8x time unroll, contiguous all phases (18 ZMM)
//   M=32: 4x time unroll, contiguous all phases (20 ZMM)
//   M>=64: 8x time unroll, sequential phase groups (9 ZMM per group)
//

namespace avx512 {

// =============================================================================
// M=8: 16x Time Unroll - AVX-512
// =============================================================================
template<std::size_t M>
[[gnu::target("avx512f")]]
inline auto filter_interleaved_impl(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void;

template<>
[[gnu::target("avx512f")]]
inline auto filter_interleaved_impl<8>(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t floats_per_row = 8 * 2;
    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

    // Main loop: 16x unroll
    for (; t + 16 <= num_outputs; t += 16) {
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
            __m512 h = _mm512_loadu_ps(h_ptr);

            acc0  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0  + k) * floats_per_row), h, acc0);
            acc1  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1  + k) * floats_per_row), h, acc1);
            acc2  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2  + k) * floats_per_row), h, acc2);
            acc3  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3  + k) * floats_per_row), h, acc3);
            acc4  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 4  + k) * floats_per_row), h, acc4);
            acc5  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 5  + k) * floats_per_row), h, acc5);
            acc6  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 6  + k) * floats_per_row), h, acc6);
            acc7  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 7  + k) * floats_per_row), h, acc7);
            acc8  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 8  + k) * floats_per_row), h, acc8);
            acc9  = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 9  + k) * floats_per_row), h, acc9);
            acc10 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 10 + k) * floats_per_row), h, acc10);
            acc11 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 11 + k) * floats_per_row), h, acc11);
            acc12 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 12 + k) * floats_per_row), h, acc12);
            acc13 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 13 + k) * floats_per_row), h, acc13);
            acc14 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 14 + k) * floats_per_row), h, acc14);
            acc15 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 15 + k) * floats_per_row), h, acc15);

            h_ptr += floats_per_row;
        }

        _mm512_storeu_ps(out_base + (t + 0)  * floats_per_row, acc0);
        _mm512_storeu_ps(out_base + (t + 1)  * floats_per_row, acc1);
        _mm512_storeu_ps(out_base + (t + 2)  * floats_per_row, acc2);
        _mm512_storeu_ps(out_base + (t + 3)  * floats_per_row, acc3);
        _mm512_storeu_ps(out_base + (t + 4)  * floats_per_row, acc4);
        _mm512_storeu_ps(out_base + (t + 5)  * floats_per_row, acc5);
        _mm512_storeu_ps(out_base + (t + 6)  * floats_per_row, acc6);
        _mm512_storeu_ps(out_base + (t + 7)  * floats_per_row, acc7);
        _mm512_storeu_ps(out_base + (t + 8)  * floats_per_row, acc8);
        _mm512_storeu_ps(out_base + (t + 9)  * floats_per_row, acc9);
        _mm512_storeu_ps(out_base + (t + 10) * floats_per_row, acc10);
        _mm512_storeu_ps(out_base + (t + 11) * floats_per_row, acc11);
        _mm512_storeu_ps(out_base + (t + 12) * floats_per_row, acc12);
        _mm512_storeu_ps(out_base + (t + 13) * floats_per_row, acc13);
        _mm512_storeu_ps(out_base + (t + 14) * floats_per_row, acc14);
        _mm512_storeu_ps(out_base + (t + 15) * floats_per_row, acc15);
    }

    // Tail loops
    for (; t + 8 <= num_outputs; t += 8) {
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps();
        __m512 acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();

        const float* h_ptr = coeffs;
        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h = _mm512_loadu_ps(h_ptr);
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0 + k) * floats_per_row), h, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1 + k) * floats_per_row), h, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2 + k) * floats_per_row), h, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3 + k) * floats_per_row), h, acc3);
            acc4 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 4 + k) * floats_per_row), h, acc4);
            acc5 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 5 + k) * floats_per_row), h, acc5);
            acc6 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 6 + k) * floats_per_row), h, acc6);
            acc7 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 7 + k) * floats_per_row), h, acc7);
            h_ptr += floats_per_row;
        }
        _mm512_storeu_ps(out_base + (t + 0) * floats_per_row, acc0);
        _mm512_storeu_ps(out_base + (t + 1) * floats_per_row, acc1);
        _mm512_storeu_ps(out_base + (t + 2) * floats_per_row, acc2);
        _mm512_storeu_ps(out_base + (t + 3) * floats_per_row, acc3);
        _mm512_storeu_ps(out_base + (t + 4) * floats_per_row, acc4);
        _mm512_storeu_ps(out_base + (t + 5) * floats_per_row, acc5);
        _mm512_storeu_ps(out_base + (t + 6) * floats_per_row, acc6);
        _mm512_storeu_ps(out_base + (t + 7) * floats_per_row, acc7);
    }

    for (; t + 4 <= num_outputs; t += 4) {
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
        const float* h_ptr = coeffs;
        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h = _mm512_loadu_ps(h_ptr);
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0 + k) * floats_per_row), h, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1 + k) * floats_per_row), h, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2 + k) * floats_per_row), h, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3 + k) * floats_per_row), h, acc3);
            h_ptr += floats_per_row;
        }
        _mm512_storeu_ps(out_base + (t + 0) * floats_per_row, acc0);
        _mm512_storeu_ps(out_base + (t + 1) * floats_per_row, acc1);
        _mm512_storeu_ps(out_base + (t + 2) * floats_per_row, acc2);
        _mm512_storeu_ps(out_base + (t + 3) * floats_per_row, acc3);
    }

    for (; t < num_outputs; ++t) {
        __m512 acc = _mm512_setzero_ps();
        const float* h_ptr = coeffs;
        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h = _mm512_loadu_ps(h_ptr);
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * floats_per_row), h, acc);
            h_ptr += floats_per_row;
        }
        _mm512_storeu_ps(out_base + t * floats_per_row, acc);
    }
}

// =============================================================================
// M=16: 8x Time Unroll, Contiguous - AVX-512
// =============================================================================
template<>
[[gnu::target("avx512f")]]
inline auto filter_interleaved_impl<16>(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t floats_per_row = 16 * 2;
    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

    for (; t + 8 <= num_outputs; t += 8) {
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

            const float* x_t0 = x_base + (t + 0 + k) * floats_per_row;
            acc_t0_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 0),  h0, acc_t0_p0);
            acc_t0_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 16), h1, acc_t0_p1);

            const float* x_t1 = x_base + (t + 1 + k) * floats_per_row;
            acc_t1_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 0),  h0, acc_t1_p0);
            acc_t1_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 16), h1, acc_t1_p1);

            const float* x_t2 = x_base + (t + 2 + k) * floats_per_row;
            acc_t2_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 0),  h0, acc_t2_p0);
            acc_t2_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 16), h1, acc_t2_p1);

            const float* x_t3 = x_base + (t + 3 + k) * floats_per_row;
            acc_t3_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 0),  h0, acc_t3_p0);
            acc_t3_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 16), h1, acc_t3_p1);

            const float* x_t4 = x_base + (t + 4 + k) * floats_per_row;
            acc_t4_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t4 + 0),  h0, acc_t4_p0);
            acc_t4_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t4 + 16), h1, acc_t4_p1);

            const float* x_t5 = x_base + (t + 5 + k) * floats_per_row;
            acc_t5_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t5 + 0),  h0, acc_t5_p0);
            acc_t5_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t5 + 16), h1, acc_t5_p1);

            const float* x_t6 = x_base + (t + 6 + k) * floats_per_row;
            acc_t6_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t6 + 0),  h0, acc_t6_p0);
            acc_t6_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t6 + 16), h1, acc_t6_p1);

            const float* x_t7 = x_base + (t + 7 + k) * floats_per_row;
            acc_t7_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t7 + 0),  h0, acc_t7_p0);
            acc_t7_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t7 + 16), h1, acc_t7_p1);

            h_ptr += floats_per_row;
        }

        _mm512_storeu_ps(out_base + (t + 0) * floats_per_row + 0,  acc_t0_p0);
        _mm512_storeu_ps(out_base + (t + 0) * floats_per_row + 16, acc_t0_p1);
        _mm512_storeu_ps(out_base + (t + 1) * floats_per_row + 0,  acc_t1_p0);
        _mm512_storeu_ps(out_base + (t + 1) * floats_per_row + 16, acc_t1_p1);
        _mm512_storeu_ps(out_base + (t + 2) * floats_per_row + 0,  acc_t2_p0);
        _mm512_storeu_ps(out_base + (t + 2) * floats_per_row + 16, acc_t2_p1);
        _mm512_storeu_ps(out_base + (t + 3) * floats_per_row + 0,  acc_t3_p0);
        _mm512_storeu_ps(out_base + (t + 3) * floats_per_row + 16, acc_t3_p1);
        _mm512_storeu_ps(out_base + (t + 4) * floats_per_row + 0,  acc_t4_p0);
        _mm512_storeu_ps(out_base + (t + 4) * floats_per_row + 16, acc_t4_p1);
        _mm512_storeu_ps(out_base + (t + 5) * floats_per_row + 0,  acc_t5_p0);
        _mm512_storeu_ps(out_base + (t + 5) * floats_per_row + 16, acc_t5_p1);
        _mm512_storeu_ps(out_base + (t + 6) * floats_per_row + 0,  acc_t6_p0);
        _mm512_storeu_ps(out_base + (t + 6) * floats_per_row + 16, acc_t6_p1);
        _mm512_storeu_ps(out_base + (t + 7) * floats_per_row + 0,  acc_t7_p0);
        _mm512_storeu_ps(out_base + (t + 7) * floats_per_row + 16, acc_t7_p1);
    }

    for (; t < num_outputs; ++t) {
        __m512 acc_p0 = _mm512_setzero_ps(), acc_p1 = _mm512_setzero_ps();
        const float* h_ptr = coeffs;
        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h0 = _mm512_loadu_ps(h_ptr + 0);
            __m512 h1 = _mm512_loadu_ps(h_ptr + 16);
            const float* x = x_base + (t + k) * floats_per_row;
            acc_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + 0),  h0, acc_p0);
            acc_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + 16), h1, acc_p1);
            h_ptr += floats_per_row;
        }
        _mm512_storeu_ps(out_base + t * floats_per_row + 0,  acc_p0);
        _mm512_storeu_ps(out_base + t * floats_per_row + 16, acc_p1);
    }
}

// =============================================================================
// M=32: 4x Time Unroll, Contiguous - AVX-512
// =============================================================================
template<>
[[gnu::target("avx512f")]]
inline auto filter_interleaved_impl<32>(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t floats_per_row = 32 * 2;
    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

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

            const float* x_t0 = x_base + (t + 0 + k) * floats_per_row;
            acc_t0_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 0),  h0, acc_t0_p0);
            acc_t0_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 16), h1, acc_t0_p1);
            acc_t0_p2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 32), h2, acc_t0_p2);
            acc_t0_p3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t0 + 48), h3, acc_t0_p3);

            const float* x_t1 = x_base + (t + 1 + k) * floats_per_row;
            acc_t1_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 0),  h0, acc_t1_p0);
            acc_t1_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 16), h1, acc_t1_p1);
            acc_t1_p2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 32), h2, acc_t1_p2);
            acc_t1_p3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t1 + 48), h3, acc_t1_p3);

            const float* x_t2 = x_base + (t + 2 + k) * floats_per_row;
            acc_t2_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 0),  h0, acc_t2_p0);
            acc_t2_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 16), h1, acc_t2_p1);
            acc_t2_p2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 32), h2, acc_t2_p2);
            acc_t2_p3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t2 + 48), h3, acc_t2_p3);

            const float* x_t3 = x_base + (t + 3 + k) * floats_per_row;
            acc_t3_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 0),  h0, acc_t3_p0);
            acc_t3_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 16), h1, acc_t3_p1);
            acc_t3_p2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 32), h2, acc_t3_p2);
            acc_t3_p3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_t3 + 48), h3, acc_t3_p3);

            h_ptr += floats_per_row;
        }

        _mm512_storeu_ps(out_base + (t + 0) * floats_per_row + 0,  acc_t0_p0);
        _mm512_storeu_ps(out_base + (t + 0) * floats_per_row + 16, acc_t0_p1);
        _mm512_storeu_ps(out_base + (t + 0) * floats_per_row + 32, acc_t0_p2);
        _mm512_storeu_ps(out_base + (t + 0) * floats_per_row + 48, acc_t0_p3);
        _mm512_storeu_ps(out_base + (t + 1) * floats_per_row + 0,  acc_t1_p0);
        _mm512_storeu_ps(out_base + (t + 1) * floats_per_row + 16, acc_t1_p1);
        _mm512_storeu_ps(out_base + (t + 1) * floats_per_row + 32, acc_t1_p2);
        _mm512_storeu_ps(out_base + (t + 1) * floats_per_row + 48, acc_t1_p3);
        _mm512_storeu_ps(out_base + (t + 2) * floats_per_row + 0,  acc_t2_p0);
        _mm512_storeu_ps(out_base + (t + 2) * floats_per_row + 16, acc_t2_p1);
        _mm512_storeu_ps(out_base + (t + 2) * floats_per_row + 32, acc_t2_p2);
        _mm512_storeu_ps(out_base + (t + 2) * floats_per_row + 48, acc_t2_p3);
        _mm512_storeu_ps(out_base + (t + 3) * floats_per_row + 0,  acc_t3_p0);
        _mm512_storeu_ps(out_base + (t + 3) * floats_per_row + 16, acc_t3_p1);
        _mm512_storeu_ps(out_base + (t + 3) * floats_per_row + 32, acc_t3_p2);
        _mm512_storeu_ps(out_base + (t + 3) * floats_per_row + 48, acc_t3_p3);
    }

    for (; t < num_outputs; ++t) {
        __m512 acc_p0 = _mm512_setzero_ps(), acc_p1 = _mm512_setzero_ps();
        __m512 acc_p2 = _mm512_setzero_ps(), acc_p3 = _mm512_setzero_ps();
        const float* h_ptr = coeffs;
        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h0 = _mm512_loadu_ps(h_ptr + 0);
            __m512 h1 = _mm512_loadu_ps(h_ptr + 16);
            __m512 h2 = _mm512_loadu_ps(h_ptr + 32);
            __m512 h3 = _mm512_loadu_ps(h_ptr + 48);
            const float* x = x_base + (t + k) * floats_per_row;
            acc_p0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + 0),  h0, acc_p0);
            acc_p1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + 16), h1, acc_p1);
            acc_p2 = _mm512_fmadd_ps(_mm512_loadu_ps(x + 32), h2, acc_p2);
            acc_p3 = _mm512_fmadd_ps(_mm512_loadu_ps(x + 48), h3, acc_p3);
            h_ptr += floats_per_row;
        }
        _mm512_storeu_ps(out_base + t * floats_per_row + 0,  acc_p0);
        _mm512_storeu_ps(out_base + t * floats_per_row + 16, acc_p1);
        _mm512_storeu_ps(out_base + t * floats_per_row + 32, acc_p2);
        _mm512_storeu_ps(out_base + t * floats_per_row + 48, acc_p3);
    }
}

// =============================================================================
// Generic M >= 64: Sequential Phase Groups - AVX-512
// =============================================================================
template<std::size_t M>
[[gnu::target("avx512f")]]
inline auto filter_interleaved_impl(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t floats_per_row = M * 2;
    constexpr std::size_t num_phase_groups = M / 8;

    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

    for (; t + 8 <= num_outputs; t += 8) {
        for (std::size_t pg = 0; pg < num_phase_groups; ++pg) {
            __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps(), acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps(), acc7 = _mm512_setzero_ps();

            const float* h_ptr = coeffs + pg * 16;

            for (std::size_t k = 0; k < num_taps; ++k) {
                __m512 h = _mm512_loadu_ps(h_ptr);
                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0 + k) * floats_per_row + pg * 16), h, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1 + k) * floats_per_row + pg * 16), h, acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2 + k) * floats_per_row + pg * 16), h, acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3 + k) * floats_per_row + pg * 16), h, acc3);
                acc4 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 4 + k) * floats_per_row + pg * 16), h, acc4);
                acc5 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 5 + k) * floats_per_row + pg * 16), h, acc5);
                acc6 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 6 + k) * floats_per_row + pg * 16), h, acc6);
                acc7 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 7 + k) * floats_per_row + pg * 16), h, acc7);
                h_ptr += floats_per_row;
            }

            _mm512_storeu_ps(out_base + (t + 0) * floats_per_row + pg * 16, acc0);
            _mm512_storeu_ps(out_base + (t + 1) * floats_per_row + pg * 16, acc1);
            _mm512_storeu_ps(out_base + (t + 2) * floats_per_row + pg * 16, acc2);
            _mm512_storeu_ps(out_base + (t + 3) * floats_per_row + pg * 16, acc3);
            _mm512_storeu_ps(out_base + (t + 4) * floats_per_row + pg * 16, acc4);
            _mm512_storeu_ps(out_base + (t + 5) * floats_per_row + pg * 16, acc5);
            _mm512_storeu_ps(out_base + (t + 6) * floats_per_row + pg * 16, acc6);
            _mm512_storeu_ps(out_base + (t + 7) * floats_per_row + pg * 16, acc7);
        }
    }

    for (; t < num_outputs; ++t) {
        for (std::size_t pg = 0; pg < num_phase_groups; ++pg) {
            __m512 acc = _mm512_setzero_ps();
            const float* h_ptr = coeffs + pg * 16;
            for (std::size_t k = 0; k < num_taps; ++k) {
                __m512 h = _mm512_loadu_ps(h_ptr);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * floats_per_row + pg * 16), h, acc);
                h_ptr += floats_per_row;
            }
            _mm512_storeu_ps(out_base + t * floats_per_row + pg * 16, acc);
        }
    }
}

} // namespace avx512

namespace avx2 {

// =============================================================================
// Generic AVX2 Implementation
// =============================================================================
template<std::size_t M>
[[gnu::target("avx2,fma")]]
inline auto filter_interleaved_impl(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t floats_per_row = M * 2;
    constexpr std::size_t num_phase_groups = M / 4;

    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

    for (; t + 4 <= num_outputs; t += 4) {
        for (std::size_t pg = 0; pg < num_phase_groups; ++pg) {
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();

            const float* h_ptr = coeffs + pg * 8;

            for (std::size_t k = 0; k < num_taps; ++k) {
                __m256 h = _mm256_loadu_ps(h_ptr);
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 0 + k) * floats_per_row + pg * 8), h, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 1 + k) * floats_per_row + pg * 8), h, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 2 + k) * floats_per_row + pg * 8), h, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 3 + k) * floats_per_row + pg * 8), h, acc3);
                h_ptr += floats_per_row;
            }

            _mm256_storeu_ps(out_base + (t + 0) * floats_per_row + pg * 8, acc0);
            _mm256_storeu_ps(out_base + (t + 1) * floats_per_row + pg * 8, acc1);
            _mm256_storeu_ps(out_base + (t + 2) * floats_per_row + pg * 8, acc2);
            _mm256_storeu_ps(out_base + (t + 3) * floats_per_row + pg * 8, acc3);
        }
    }

    for (; t < num_outputs; ++t) {
        for (std::size_t pg = 0; pg < num_phase_groups; ++pg) {
            __m256 acc = _mm256_setzero_ps();
            const float* h_ptr = coeffs + pg * 8;
            for (std::size_t k = 0; k < num_taps; ++k) {
                __m256 h = _mm256_loadu_ps(h_ptr);
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + k) * floats_per_row + pg * 8), h, acc);
                h_ptr += floats_per_row;
            }
            _mm256_storeu_ps(out_base + t * floats_per_row + pg * 8, acc);
        }
    }
}

} // namespace avx2

namespace scalar {

// =============================================================================
// Scalar Implementation
// =============================================================================
template<std::size_t M>
inline auto filter_interleaved_impl(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    constexpr std::size_t floats_per_row = M * 2;
    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    for (std::size_t t = 0; t < num_outputs; ++t) {
        for (std::size_t p = 0; p < M; ++p) {
            float acc_re = 0.0f, acc_im = 0.0f;
            for (std::size_t k = 0; k < num_taps; ++k) {
                acc_re += x_base[(t + k) * floats_per_row + p * 2] * coeffs[k * floats_per_row + p * 2];
                acc_im += x_base[(t + k) * floats_per_row + p * 2 + 1] * coeffs[k * floats_per_row + p * 2 + 1];
            }
            out_base[t * floats_per_row + p * 2] = acc_re;
            out_base[t * floats_per_row + p * 2 + 1] = acc_im;
        }
    }
}

} // namespace scalar

// =============================================================================
// Public API - Runtime ISA Dispatch
// =============================================================================
template<std::size_t M>
inline auto filter_interleaved(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    if (__builtin_cpu_supports("avx512f")) {
        avx512::filter_interleaved_impl<M>(input, coeffs, output, num_outputs, num_taps);
    } else if (__builtin_cpu_supports("avx2")) {
        avx2::filter_interleaved_impl<M>(input, coeffs, output, num_outputs, num_taps);
    } else {
        scalar::filter_interleaved_impl<M>(input, coeffs, output, num_outputs, num_taps);
    }
}

// =============================================================================
// Runtime Dispatch Wrapper
// =============================================================================
// For use when M is determined at runtime (e.g., from configuration).
// Dispatches to the appropriate compile-time template instantiation.

inline auto filter_interleaved_dispatch(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t M,
    std::size_t num_outputs,
    std::size_t num_taps
) -> void
{
    switch (M) {
        case 8:   filter_interleaved<8>(input, coeffs, output, num_outputs, num_taps); break;
        case 16:  filter_interleaved<16>(input, coeffs, output, num_outputs, num_taps); break;
        case 32:  filter_interleaved<32>(input, coeffs, output, num_outputs, num_taps); break;
        case 64:  filter_interleaved<64>(input, coeffs, output, num_outputs, num_taps); break;
        case 128: filter_interleaved<128>(input, coeffs, output, num_outputs, num_taps); break;
        case 256: filter_interleaved<256>(input, coeffs, output, num_outputs, num_taps); break;
        default:
            // Fallback: use scalar for unsupported M values
            scalar::filter_interleaved_impl<8>(input, coeffs, output, num_outputs, num_taps);
            break;
    }
}

// -----------------------------------------------------------------------------
// History Buffer Management
// -----------------------------------------------------------------------------
inline auto update_interleaved_history(
    std::complex<float>* buffer,
    std::size_t num_channels,
    std::size_t num_rows_processed,
    std::size_t history_rows
) -> void
{
    const std::size_t bytes_per_row = num_channels * sizeof(std::complex<float>);
    const std::size_t history_bytes = history_rows * bytes_per_row;
    const auto* src = buffer + num_rows_processed * num_channels;
    std::memmove(buffer, src, history_bytes);
}

// =============================================================================
// Transpose Functions
// =============================================================================

static_assert(sizeof(std::complex<float>) == sizeof(double),
              "transpose functions assume complex<float> is 64-bit");

[[gnu::target("avx512f")]]
inline void transpose_8x8_cf32_avx512(
    const std::complex<float>* src,
    std::complex<float>* dst,
    std::size_t src_stride,
    std::size_t dst_stride
) {
    const double* s = reinterpret_cast<const double*>(src);
    double* d = reinterpret_cast<double*>(dst);

    __m512d r0 = _mm512_loadu_pd(s + 0 * src_stride);
    __m512d r1 = _mm512_loadu_pd(s + 1 * src_stride);
    __m512d r2 = _mm512_loadu_pd(s + 2 * src_stride);
    __m512d r3 = _mm512_loadu_pd(s + 3 * src_stride);
    __m512d r4 = _mm512_loadu_pd(s + 4 * src_stride);
    __m512d r5 = _mm512_loadu_pd(s + 5 * src_stride);
    __m512d r6 = _mm512_loadu_pd(s + 6 * src_stride);
    __m512d r7 = _mm512_loadu_pd(s + 7 * src_stride);

    __m512d t0 = _mm512_unpacklo_pd(r0, r1);
    __m512d t1 = _mm512_unpackhi_pd(r0, r1);
    __m512d t2 = _mm512_unpacklo_pd(r2, r3);
    __m512d t3 = _mm512_unpackhi_pd(r2, r3);
    __m512d t4 = _mm512_unpacklo_pd(r4, r5);
    __m512d t5 = _mm512_unpackhi_pd(r4, r5);
    __m512d t6 = _mm512_unpacklo_pd(r6, r7);
    __m512d t7 = _mm512_unpackhi_pd(r6, r7);

    r0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    r1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    r2 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    r3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    r4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
    r5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    r6 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    r7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);

    t0 = _mm512_shuffle_f64x2(r0, r4, 0x88);
    t1 = _mm512_shuffle_f64x2(r1, r5, 0x88);
    t2 = _mm512_shuffle_f64x2(r2, r6, 0x88);
    t3 = _mm512_shuffle_f64x2(r3, r7, 0x88);
    t4 = _mm512_shuffle_f64x2(r0, r4, 0xDD);
    t5 = _mm512_shuffle_f64x2(r1, r5, 0xDD);
    t6 = _mm512_shuffle_f64x2(r2, r6, 0xDD);
    t7 = _mm512_shuffle_f64x2(r3, r7, 0xDD);

    _mm512_storeu_pd(d + 0 * dst_stride, t0);
    _mm512_storeu_pd(d + 1 * dst_stride, t1);
    _mm512_storeu_pd(d + 2 * dst_stride, t2);
    _mm512_storeu_pd(d + 3 * dst_stride, t3);
    _mm512_storeu_pd(d + 4 * dst_stride, t4);
    _mm512_storeu_pd(d + 5 * dst_stride, t5);
    _mm512_storeu_pd(d + 6 * dst_stride, t6);
    _mm512_storeu_pd(d + 7 * dst_stride, t7);
}

[[gnu::target("avx512f")]]
inline void transpose_tm_to_cm(
    const std::complex<float>* src,
    std::complex<float>* dst,
    std::size_t M,
    std::size_t N
) {
    const std::size_t M8 = M & ~7ULL;
    const std::size_t N8 = N & ~7ULL;

    for (std::size_t ch0 = 0; ch0 < M8; ch0 += 8) {
        for (std::size_t t0 = 0; t0 < N8; t0 += 8) {
            const std::complex<float>* tile_src = src + t0 * M + ch0;
            std::complex<float>* tile_dst = dst + ch0 * N + t0;
            transpose_8x8_cf32_avx512(tile_src, tile_dst, M, N);
        }
    }

    for (std::size_t ch = M8; ch < M; ++ch) {
        for (std::size_t t = 0; t < N; ++t) {
            dst[ch * N + t] = src[t * M + ch];
        }
    }

    for (std::size_t ch = 0; ch < M8; ++ch) {
        for (std::size_t t = N8; t < N; ++t) {
            dst[ch * N + t] = src[t * M + ch];
        }
    }
}

[[gnu::target("avx512f")]]
inline void transpose_to_channel_buffers(
    const std::complex<float>* src,
    std::complex<float>* const* dst_ptrs,
    std::size_t M,
    std::size_t N
) {
    const std::size_t M8 = M & ~7ULL;
    const std::size_t N8 = N & ~7ULL;

    for (std::size_t t0 = 0; t0 < N8; t0 += 8) {
        for (std::size_t ch0 = 0; ch0 < M8; ch0 += 8) {
            const double* s = reinterpret_cast<const double*>(src + t0 * M + ch0);

            __m512d r0 = _mm512_loadu_pd(s + 0 * M);
            __m512d r1 = _mm512_loadu_pd(s + 1 * M);
            __m512d r2 = _mm512_loadu_pd(s + 2 * M);
            __m512d r3 = _mm512_loadu_pd(s + 3 * M);
            __m512d r4 = _mm512_loadu_pd(s + 4 * M);
            __m512d r5 = _mm512_loadu_pd(s + 5 * M);
            __m512d r6 = _mm512_loadu_pd(s + 6 * M);
            __m512d r7 = _mm512_loadu_pd(s + 7 * M);

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

        for (std::size_t ch = M8; ch < M; ++ch) {
            if (!dst_ptrs[ch]) continue;
            for (std::size_t t = t0; t < t0 + 8; ++t) {
                dst_ptrs[ch][t] = src[t * M + ch];
            }
        }
    }

    for (std::size_t t = N8; t < N; ++t) {
        for (std::size_t ch = 0; ch < M; ++ch) {
            if (!dst_ptrs[ch]) continue;
            dst_ptrs[ch][t] = src[t * M + ch];
        }
    }
}

// =============================================================================
// Tile Transpose: 8 rows to channel buffers at offset
// =============================================================================
// Specialized transpose for exactly 8 rows (one processing tile).
// Used in fused filter→FFT→transpose pipeline to write directly to channel
// buffers while data is L1-hot.
//
// Input:  tile[8][M] - 8 time steps, M channels (row-major, L1-resident)
// Output: 8 samples written to each channel buffer at the given offset
//

[[gnu::target("avx512f")]]
inline void transpose_tile_to_channel_buffers(
    const std::complex<float>* tile,   // [8 × M] tile, row-major
    std::complex<float>* const* dst_ptrs,  // M channel buffer pointers
    std::size_t M,
    std::size_t offset                 // Write position in each channel buffer
) {
    const std::size_t M8 = M & ~7ULL;

    // Process 8×8 blocks
    for (std::size_t ch0 = 0; ch0 < M8; ch0 += 8) {
        const double* s = reinterpret_cast<const double*>(tile + ch0);

        // Load 8 rows of 8 channels each
        __m512d r0 = _mm512_loadu_pd(s + 0 * M);
        __m512d r1 = _mm512_loadu_pd(s + 1 * M);
        __m512d r2 = _mm512_loadu_pd(s + 2 * M);
        __m512d r3 = _mm512_loadu_pd(s + 3 * M);
        __m512d r4 = _mm512_loadu_pd(s + 4 * M);
        __m512d r5 = _mm512_loadu_pd(s + 5 * M);
        __m512d r6 = _mm512_loadu_pd(s + 6 * M);
        __m512d r7 = _mm512_loadu_pd(s + 7 * M);

        // 8×8 transpose using AVX-512 shuffles
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

        // Store 8 samples to each of 8 channel buffers
        double* d0 = dst_ptrs[ch0 + 0] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 0] + offset) : nullptr;
        double* d1 = dst_ptrs[ch0 + 1] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 1] + offset) : nullptr;
        double* d2 = dst_ptrs[ch0 + 2] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 2] + offset) : nullptr;
        double* d3 = dst_ptrs[ch0 + 3] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 3] + offset) : nullptr;
        double* d4 = dst_ptrs[ch0 + 4] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 4] + offset) : nullptr;
        double* d5 = dst_ptrs[ch0 + 5] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 5] + offset) : nullptr;
        double* d6 = dst_ptrs[ch0 + 6] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 6] + offset) : nullptr;
        double* d7 = dst_ptrs[ch0 + 7] ? reinterpret_cast<double*>(dst_ptrs[ch0 + 7] + offset) : nullptr;

        if (d0) _mm512_storeu_pd(d0, t0v);
        if (d1) _mm512_storeu_pd(d1, t1v);
        if (d2) _mm512_storeu_pd(d2, t2v);
        if (d3) _mm512_storeu_pd(d3, t3v);
        if (d4) _mm512_storeu_pd(d4, t4v);
        if (d5) _mm512_storeu_pd(d5, t5v);
        if (d6) _mm512_storeu_pd(d6, t6v);
        if (d7) _mm512_storeu_pd(d7, t7v);
    }

    // Handle remaining channels (M not divisible by 8)
    for (std::size_t ch = M8; ch < M; ++ch) {
        if (!dst_ptrs[ch]) continue;
        for (std::size_t t = 0; t < 8; ++t) {
            dst_ptrs[ch][offset + t] = tile[t * M + ch];
        }
    }
}

// Variant for partial tiles (fewer than 8 rows) - used for tail handling
inline void transpose_partial_tile_to_channel_buffers(
    const std::complex<float>* tile,
    std::complex<float>* const* dst_ptrs,
    std::size_t M,
    std::size_t num_rows,              // 1-7 rows
    std::size_t offset
) {
    for (std::size_t ch = 0; ch < M; ++ch) {
        if (!dst_ptrs[ch]) continue;
        for (std::size_t t = 0; t < num_rows; ++t) {
            dst_ptrs[ch][offset + t] = tile[t * M + ch];
        }
    }
}

} // namespace kernels
