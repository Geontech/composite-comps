// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#pragma once

#include <complex>
#include <immintrin.h>

namespace kernels {

// =============================================================================
// KERNEL 1: DE-INTERLEAVER - Multi-Function Versioning
// =============================================================================
// Splits interleaved complex data into two contiguous linear buffers.
// Runtime CPU dispatch: AVX-512 → AVX2 → Scalar
// Alignment: Inputs can be unaligned, Outputs should be 64B aligned.

// -----------------------------------------------------------------------------
// AVX-512 Version
// -----------------------------------------------------------------------------
[[gnu::target("avx512f")]]
inline auto deinterleave_block(
  const std::complex<float>* __restrict__ src,
  std::complex<float>* __restrict__ dst_even,
  std::complex<float>* __restrict__ dst_odd,
  std::size_t num_pairs
) -> void {
    // Cast to float pointers once to avoid repeated complex pointer arithmetic
    const float* src_f = reinterpret_cast<const float*>(src);
    float* dst_even_f = reinterpret_cast<float*>(dst_even);
    float* dst_odd_f = reinterpret_cast<float*>(dst_odd);

    // Indices to permute 32-bit floats across two 512-bit registers.
    // Indices 0-15 from first vec, 16-31 from second vec
    // idx_even: Selects real/imag of even-indexed complex samples
    const auto idx_even = _mm512_set_epi32(
        29, 28, 25, 24, 21, 20, 17, 16, // High lane
        13, 12,  9,  8,  5,  4,  1,  0  // Low lane
    );
    // idx_odd: Selects real/imag of odd-indexed complex samples
    const auto idx_odd = _mm512_set_epi32(
        31, 30, 27, 26, 23, 22, 19, 18,
        15, 14, 11, 10,  7,  6,  3,  2
    );

    // Unroll 2x: Process 16 pairs per iteration
    // 16 pairs = 32 complex inputs = 64 floats = 4 ZMM registers
    auto i = std::size_t{};
    for (; i + 16 <= num_pairs; i += 16) {
        // 1. Load 4 blocks of interleaved data (64 floats total)
        // src_f index: 4 floats per pair (RealEven, ImagEven, RealOdd, ImagOdd)
        // 4 * i is the float offset for pair i
        auto t0 = _mm512_loadu_ps(src_f + 4*i);
        auto t1 = _mm512_loadu_ps(src_f + 4*i + 16);
        auto t2 = _mm512_loadu_ps(src_f + 4*i + 32);
        auto t3 = _mm512_loadu_ps(src_f + 4*i + 48);

        // 2. Permute Group 1 (t0 + t1) -> Yields 8 pairs
        auto ev0 = _mm512_permutex2var_ps(t0, idx_even, t1);
        auto od0 = _mm512_permutex2var_ps(t0, idx_odd, t1);

        // 3. Permute Group 2 (t2 + t3) -> Yields 8 pairs
        auto ev1 = _mm512_permutex2var_ps(t2, idx_even, t3);
        auto od1 = _mm512_permutex2var_ps(t2, idx_odd, t3);

        // 4. Store Group 1 (16 floats = 8 complex)
        _mm512_storeu_ps(dst_even_f + 2*i, ev0);
        _mm512_storeu_ps(dst_odd_f + 2*i, od0);

        // 5. Store Group 2 (16 floats = 8 complex, starting at i+8)
        _mm512_storeu_ps(dst_even_f + 2*i + 16, ev1);
        _mm512_storeu_ps(dst_odd_f + 2*i + 16, od1);
    }

    // 2. Tail: Process 8 pairs (16 complex inputs = 32 floats = 2 ZMMs)
    if (i + 8 <= num_pairs) {
        auto t0 = _mm512_loadu_ps(src_f + 4*i);
        auto t1 = _mm512_loadu_ps(src_f + 4*i + 16);

        auto ev = _mm512_permutex2var_ps(t0, idx_even, t1);
        auto od = _mm512_permutex2var_ps(t0, idx_odd, t1);

        _mm512_storeu_ps(dst_even_f + 2*i, ev);
        _mm512_storeu_ps(dst_odd_f + 2*i, od);
        i += 8;
    }

    // 3. Scalar tail handling
    for (; i < num_pairs; ++i) {
        dst_even[i] = src[2*i];
        dst_odd[i]  = src[2*i + 1];
    }
}

// -----------------------------------------------------------------------------
// AVX2 Version
// -----------------------------------------------------------------------------
[[gnu::target("avx2,fma")]]
inline auto deinterleave_block(
  const std::complex<float>* __restrict__ src,
  std::complex<float>* __restrict__ dst_even,
  std::complex<float>* __restrict__ dst_odd,
  std::size_t num_pairs
) -> void {
    // Process 4 output pairs per loop (8 input complex samples)
    // AVX2: YMM registers hold 4 complex<float> = 8 floats = 4 doubles
    std::size_t i = 0;
    for (; i + 4 <= num_pairs; i += 4) {
        // Load 8 interleaved complex samples
        auto a = _mm256_loadu_pd(reinterpret_cast<const double*>(src + 2*i));     // [0,1,2,3]
        auto b = _mm256_loadu_pd(reinterpret_cast<const double*>(src + 2*i + 4)); // [4,5,6,7]

        // Shuffle to separate even and odd
        // unpacklo: [0,2] from a and b → [0,2,4,6]
        // unpackhi: [1,3] from a and b → [1,3,5,7]
        auto evens_lo = _mm256_unpacklo_pd(a, b);  // [0,4,2,6]
        auto odds_lo  = _mm256_unpackhi_pd(a, b);  // [1,5,3,7]

        // Permute to correct order: [0,4,2,6] → [0,2,4,6]
        auto evens = _mm256_permute4x64_pd(evens_lo, 0xD8); // 0xD8 = [3,1,2,0] → [0,2,1,3]
        auto odds  = _mm256_permute4x64_pd(odds_lo, 0xD8);

        _mm256_storeu_pd(reinterpret_cast<double*>(dst_even + i), evens);
        _mm256_storeu_pd(reinterpret_cast<double*>(dst_odd + i), odds);
    }

    // Scalar tail
    for (; i < num_pairs; ++i) {
        dst_even[i] = src[2 * i];
        dst_odd[i]  = src[2 * i + 1];
    }
}

// -----------------------------------------------------------------------------
// Scalar Version
// -----------------------------------------------------------------------------
[[gnu::target("default")]]
inline auto deinterleave_block(
  const std::complex<float>* __restrict__ src,
  std::complex<float>* __restrict__ dst_even,
  std::complex<float>* __restrict__ dst_odd,
  std::size_t num_pairs
) -> void {
    for (std::size_t i = 0; i < num_pairs; ++i) {
        dst_even[i] = src[2*i];
        dst_odd[i]  = src[2*i + 1];
    }
}

// =============================================================================
// KERNEL 2: VERTICAL HALF-BAND FILTER - Multi-Function Versioning
// =============================================================================
// Computes FIR (Even) + Delay (Odd) simultaneously
// Runtime CPU dispatch: AVX-512 → AVX2 → Scalar
// Strategy: Accumulator Splitting (A/B) to hide 4-cycle FMA latency.
//           Unrolling inner loop by 2 to support splitting.
//           Masked store for tail handling.

// -----------------------------------------------------------------------------
// AVX-512 Version
// -----------------------------------------------------------------------------
[[gnu::target("avx512f")]]
inline auto halfband_filter_vertical(
  const std::complex<float>* __restrict__ even_hist,
  const std::complex<float>* __restrict__ odd_hist,
  const float* __restrict__ coeffs,
  std::size_t num_taps,
  float center_tap,
  std::size_t delay_offset,
  std::complex<float>* __restrict__ output,
  std::size_t num_outputs
) -> void {
    const auto* odd_hist_f = reinterpret_cast<const float*>(odd_hist);
    const auto* even_hist_f = reinterpret_cast<const float*>(even_hist);
    auto* output_f = reinterpret_cast<float*>(output);
    auto center_reg = _mm512_set1_ps(center_tap);
    auto i = std::size_t{};

    // Primary loop: 32 outputs (4 ZMM registers) per iteration
    for (; i + 32 <= num_outputs; i += 32) {
        // --- 1. Initialize Accumulators (Split A/B) ---
        // 'A' accumulators start with the Center Tap contribution
        auto odd_0 = _mm512_loadu_ps(odd_hist_f + 2*(i + delay_offset));
        auto odd_1 = _mm512_loadu_ps(odd_hist_f + 2*(i + delay_offset) + 16);
        auto odd_2 = _mm512_loadu_ps(odd_hist_f + 2*(i + delay_offset) + 32);
        auto odd_3 = _mm512_loadu_ps(odd_hist_f + 2*(i + delay_offset) + 48);

        auto acc_0a = _mm512_mul_ps(odd_0, center_reg);
        auto acc_1a = _mm512_mul_ps(odd_1, center_reg);
        auto acc_2a = _mm512_mul_ps(odd_2, center_reg);
        auto acc_3a = _mm512_mul_ps(odd_3, center_reg);

        // 'B' accumulators start at zero
        auto acc_0b = _mm512_setzero_ps();
        auto acc_1b = _mm512_setzero_ps();
        auto acc_2b = _mm512_setzero_ps();
        auto acc_3b = _mm512_setzero_ps();

        // --- 2. Inner FIR Loop (Unrolled x2 for dependency breaking) ---
        for (std::size_t k = 0; k < num_taps; k += 2) {
            // Tap K -> Set A
            auto h_even = _mm512_set1_ps(coeffs[k]);

            auto ev_0a = _mm512_loadu_ps(even_hist_f + 2*(i + k));
            auto ev_1a = _mm512_loadu_ps(even_hist_f + 2*(i + k) + 16);
            auto ev_2a = _mm512_loadu_ps(even_hist_f + 2*(i + k) + 32);
            auto ev_3a = _mm512_loadu_ps(even_hist_f + 2*(i + k) + 48);

            acc_0a = _mm512_fmadd_ps(ev_0a, h_even, acc_0a);
            acc_1a = _mm512_fmadd_ps(ev_1a, h_even, acc_1a);
            acc_2a = _mm512_fmadd_ps(ev_2a, h_even, acc_2a);
            acc_3a = _mm512_fmadd_ps(ev_3a, h_even, acc_3a);

            // Tap K+1 -> Set B
            // These FMAs are independent of the 'A' set above.
            // This hides the latency of the previous instruction.
            if (k + 1 < num_taps) {
                auto h_odd = _mm512_set1_ps(coeffs[k+1]);

                auto ev_0b = _mm512_loadu_ps(even_hist_f + 2*(i + k + 1));
                auto ev_1b = _mm512_loadu_ps(even_hist_f + 2*(i + k + 1) + 16);
                auto ev_2b = _mm512_loadu_ps(even_hist_f + 2*(i + k + 1) + 32);
                auto ev_3b = _mm512_loadu_ps(even_hist_f + 2*(i + k + 1) + 48);

                acc_0b = _mm512_fmadd_ps(ev_0b, h_odd, acc_0b);
                acc_1b = _mm512_fmadd_ps(ev_1b, h_odd, acc_1b);
                acc_2b = _mm512_fmadd_ps(ev_2b, h_odd, acc_2b);
                acc_3b = _mm512_fmadd_ps(ev_3b, h_odd, acc_3b);
            }
        }

        // --- 3. Reduction ---
        _mm512_storeu_ps(output_f + 2*i,      _mm512_add_ps(acc_0a, acc_0b));
        _mm512_storeu_ps(output_f + 2*i + 16, _mm512_add_ps(acc_1a, acc_1b));
        _mm512_storeu_ps(output_f + 2*i + 32, _mm512_add_ps(acc_2a, acc_2b));
        _mm512_storeu_ps(output_f + 2*i + 48, _mm512_add_ps(acc_3a, acc_3b));
    }

    // 2. Tail: 16 outputs (2 ZMMs)
    if (i + 16 <= num_outputs) {
        auto odd_0 = _mm512_loadu_ps(odd_hist_f + 2*(i + delay_offset));
        auto odd_1 = _mm512_loadu_ps(odd_hist_f + 2*(i + delay_offset) + 16);

        auto acc_0a = _mm512_mul_ps(odd_0, center_reg);
        auto acc_1a = _mm512_mul_ps(odd_1, center_reg);
        auto acc_0b = _mm512_setzero_ps();
        auto acc_1b = _mm512_setzero_ps();

        for (std::size_t k = 0; k < num_taps; k += 2) {
            auto h_even = _mm512_set1_ps(coeffs[k]);
            auto ev_0a = _mm512_loadu_ps(even_hist_f + 2*(i + k));
            auto ev_1a = _mm512_loadu_ps(even_hist_f + 2*(i + k) + 16);
            acc_0a = _mm512_fmadd_ps(ev_0a, h_even, acc_0a);
            acc_1a = _mm512_fmadd_ps(ev_1a, h_even, acc_1a);

            if (k + 1 < num_taps) {
                auto h_odd = _mm512_set1_ps(coeffs[k+1]);
                auto ev_0b = _mm512_loadu_ps(even_hist_f + 2*(i + k + 1));
                auto ev_1b = _mm512_loadu_ps(even_hist_f + 2*(i + k + 1) + 16);
                acc_0b = _mm512_fmadd_ps(ev_0b, h_odd, acc_0b);
                acc_1b = _mm512_fmadd_ps(ev_1b, h_odd, acc_1b);
            }
        }
        _mm512_storeu_ps(output_f + 2*i,      _mm512_add_ps(acc_0a, acc_0b));
        _mm512_storeu_ps(output_f + 2*i + 16, _mm512_add_ps(acc_1a, acc_1b));
        i += 16;
    }

    // 3. Tail: 8 outputs (1 ZMM)
    if (i + 8 <= num_outputs) {
        auto odd_0 = _mm512_loadu_ps(odd_hist_f + 2*(i + delay_offset));
        auto acc_0a = _mm512_mul_ps(odd_0, center_reg);
        auto acc_0b = _mm512_setzero_ps();

        for (std::size_t k = 0; k < num_taps; k += 2) {
            auto h_even = _mm512_set1_ps(coeffs[k]);
            auto ev_0a = _mm512_loadu_ps(even_hist_f + 2*(i + k));
            acc_0a = _mm512_fmadd_ps(ev_0a, h_even, acc_0a);

            if (k + 1 < num_taps) {
                auto h_odd = _mm512_set1_ps(coeffs[k+1]);
                auto ev_0b = _mm512_loadu_ps(even_hist_f + 2*(i + k + 1));
                acc_0b = _mm512_fmadd_ps(ev_0b, h_odd, acc_0b);
            }
        }
        _mm512_storeu_ps(output_f + 2*i, _mm512_add_ps(acc_0a, acc_0b));
        i += 8;
    }

    // 4. Scalar tail
    for (; i < num_outputs; ++i) {
        auto sum = odd_hist[i + delay_offset] * center_tap;
        for (std::size_t k = 0; k < num_taps; ++k) {
            sum += even_hist[i + k] * coeffs[k];
        }
        output[i] = sum;
    }
}

// -----------------------------------------------------------------------------
// AVX2 Version
// -----------------------------------------------------------------------------
[[gnu::target("avx2,fma")]]
inline
auto halfband_filter_vertical(
  const std::complex<float>* __restrict__ even_hist,
  const std::complex<float>* __restrict__ odd_hist,
  const float* __restrict__ coeffs,
  std::size_t num_taps,
  float center_tap,
  std::size_t delay_offset,
  std::complex<float>* __restrict__ output,
  std::size_t num_outputs
) -> void {
    auto center_reg = _mm256_set1_ps(center_tap);
    auto i = std::size_t{};

    // Primary loop: 16 outputs (4 YMM registers) per iteration
    for (; i + 16 <= num_outputs; i += 16) {
        // Init accum A
        auto odd_0 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset));
        auto odd_1 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 4));
        auto odd_2 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 8));
        auto odd_3 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 12));

        auto acc_0a = _mm256_mul_ps(odd_0, center_reg);
        auto acc_1a = _mm256_mul_ps(odd_1, center_reg);
        auto acc_2a = _mm256_mul_ps(odd_2, center_reg);
        auto acc_3a = _mm256_mul_ps(odd_3, center_reg);

        // Init accum B
        auto acc_0b = _mm256_setzero_ps();
        auto acc_1b = _mm256_setzero_ps();
        auto acc_2b = _mm256_setzero_ps();
        auto acc_3b = _mm256_setzero_ps();

        for (std::size_t k = 0; k < num_taps; k += 2) {
            // Tap K -> A
            auto h_even = _mm256_set1_ps(coeffs[k]);
            auto ev_0a = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k));
            auto ev_1a = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 4));
            auto ev_2a = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 8));
            auto ev_3a = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 12));

            acc_0a = _mm256_fmadd_ps(ev_0a, h_even, acc_0a);
            acc_1a = _mm256_fmadd_ps(ev_1a, h_even, acc_1a);
            acc_2a = _mm256_fmadd_ps(ev_2a, h_even, acc_2a);
            acc_3a = _mm256_fmadd_ps(ev_3a, h_even, acc_3a);

            // Tap K+1 -> B
            if (k + 1 < num_taps) {
                auto h_odd = _mm256_set1_ps(coeffs[k+1]);
                auto ev_0b = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 1));
                auto ev_1b = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 5));
                auto ev_2b = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 9));
                auto ev_3b = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 13));

                acc_0b = _mm256_fmadd_ps(ev_0b, h_odd, acc_0b);
                acc_1b = _mm256_fmadd_ps(ev_1b, h_odd, acc_1b);
                acc_2b = _mm256_fmadd_ps(ev_2b, h_odd, acc_2b);
                acc_3b = _mm256_fmadd_ps(ev_3b, h_odd, acc_3b);
            }
        }

        // Sum A + B
        _mm256_storeu_ps(reinterpret_cast<float*>(output + i),      _mm256_add_ps(acc_0a, acc_0b));
        _mm256_storeu_ps(reinterpret_cast<float*>(output + i + 4),  _mm256_add_ps(acc_1a, acc_1b));
        _mm256_storeu_ps(reinterpret_cast<float*>(output + i + 8),  _mm256_add_ps(acc_2a, acc_2b));
        _mm256_storeu_ps(reinterpret_cast<float*>(output + i + 12), _mm256_add_ps(acc_3a, acc_3b));
    }

    // Tail loop: 8 outputs
    if (i + 8 <= num_outputs) {
        auto odd_0 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset));
        auto odd_1 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 4));

        auto acc_0a = _mm256_mul_ps(odd_0, center_reg);
        auto acc_1a = _mm256_mul_ps(odd_1, center_reg);
        auto acc_0b = _mm256_setzero_ps();
        auto acc_1b = _mm256_setzero_ps();

        for (std::size_t k = 0; k < num_taps; k += 2) {
            auto h_reg = _mm256_set1_ps(coeffs[k]);
            auto ev_0 = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k));
            auto ev_1 = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 4));
            acc_0a = _mm256_fmadd_ps(ev_0, h_reg, acc_0a);
            acc_1a = _mm256_fmadd_ps(ev_1, h_reg, acc_1a);

            if (k + 1 < num_taps) {
                auto h_odd = _mm256_set1_ps(coeffs[k+1]);
                auto ev_0b = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 1));
                auto ev_1b = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 5));
                acc_0b = _mm256_fmadd_ps(ev_0b, h_odd, acc_0b);
                acc_1b = _mm256_fmadd_ps(ev_1b, h_odd, acc_1b);
            }
        }

        _mm256_storeu_ps(reinterpret_cast<float*>(output + i),     _mm256_add_ps(acc_0a, acc_0b));
        _mm256_storeu_ps(reinterpret_cast<float*>(output + i + 4), _mm256_add_ps(acc_1a, acc_1b));
        i += 8;
    }

    // 4. Scalar tail
    for (; i < num_outputs; ++i) {
        auto sum = odd_hist[i + delay_offset] * center_tap;
        for (std::size_t k = 0; k < num_taps; ++k) {
            sum += even_hist[i + k] * coeffs[k];
        }
        output[i] = sum;
    }
}

// -----------------------------------------------------------------------------
// Scalar Version
// -----------------------------------------------------------------------------
[[gnu::target("default")]]
inline
auto halfband_filter_vertical(
  const std::complex<float>* __restrict__ even_hist,
  const std::complex<float>* __restrict__ odd_hist,
  const float* __restrict__ coeffs,
  std::size_t num_taps,
  float center_tap,
  std::size_t delay_offset,
  std::complex<float>* __restrict__ output,
  std::size_t num_outputs
) -> void {
    for (std::size_t i = 0; i < num_outputs; ++i) {
        auto sum = odd_hist[i + delay_offset] * center_tap;
        for (std::size_t k = 0; k < num_taps; ++k) {
            sum += even_hist[i + k] * coeffs[k];
        }
        output[i] = sum;
    }
}

} // namespace kernels
