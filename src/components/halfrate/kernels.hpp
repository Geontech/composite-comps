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

// -----------------------------------------------------------------------------
// AVX-512 Version
// -----------------------------------------------------------------------------
[[gnu::target("avx512f")]]
inline
auto deinterleave_block(
  const std::complex<float>* src,
  std::complex<float>* dst_even,
  std::complex<float>* dst_odd,
  std::size_t num_pairs
) -> void {
    // Indices for shuffling 64-bit blocks (treating complex<float> as double)
    const auto idx_even = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    const auto idx_odd  = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

    // Process 8 output pairs per loop (16 input complex samples)
    // For pair i: dst_even[i] = src[2*i], dst_odd[i] = src[2*i+1]
    std::size_t i = 0;
    for (; i + 8 <= num_pairs; i += 8) {
        // Load 16 interleaved complex samples (8 pairs)
        auto a = _mm512_loadu_pd(reinterpret_cast<const double*>(src + 2*i));
        auto b = _mm512_loadu_pd(reinterpret_cast<const double*>(src + 2*i + 8));

        // Permute to separate even and odd
        auto evens = _mm512_permutex2var_pd(a, idx_even, b);
        auto odds  = _mm512_permutex2var_pd(a, idx_odd, b);

        _mm512_storeu_pd(reinterpret_cast<double*>(dst_even + i), evens);
        _mm512_storeu_pd(reinterpret_cast<double*>(dst_odd + i), odds);
    }

    // Scalar tail handling
    for (; i < num_pairs; ++i) {
        dst_even[i] = src[2 * i];
        dst_odd[i]  = src[2 * i + 1];
    }
}

// -----------------------------------------------------------------------------
// AVX2 Version
// -----------------------------------------------------------------------------
[[gnu::target("avx2,fma")]]
inline
auto deinterleave_block(
  const std::complex<float>* src,
  std::complex<float>* dst_even,
  std::complex<float>* dst_odd,
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
inline
auto deinterleave_block(
  const std::complex<float>* src,
  std::complex<float>* dst_even,
  std::complex<float>* dst_odd,
  std::size_t num_pairs
) -> void {
    for (std::size_t i = 0; i < num_pairs; ++i) {
        dst_even[i] = src[2 * i];
        dst_odd[i]  = src[2 * i + 1];
    }
}

// =============================================================================
// KERNEL 2: VERTICAL HALF-BAND FILTER - Multi-Function Versioning
// =============================================================================
// Computes FIR (Even) + Delay (Odd) simultaneously
// Runtime CPU dispatch: AVX-512 → AVX2 → Scalar

// -----------------------------------------------------------------------------
// AVX-512 Version
// -----------------------------------------------------------------------------
[[gnu::target("avx512f")]]
inline
auto halfband_filter_vertical(
  const std::complex<float>* even_hist,
  const std::complex<float>* odd_hist,
  const float* coeffs,
  std::size_t num_taps,
  float center_tap,
  std::size_t delay_offset,
  std::complex<float>* output,
  std::size_t num_outputs
) -> void {
    auto center_reg = _mm512_set1_ps(center_tap);

    // Primary loop: 32 outputs (4 ZMM registers)
    std::size_t i = 0;
    for (; i + 32 <= num_outputs; i += 32) {
        // Load odd lane + initialize with center tap
        auto odd_0 = _mm512_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset));
        auto odd_1 = _mm512_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 8));
        auto odd_2 = _mm512_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 16));
        auto odd_3 = _mm512_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 24));

        auto acc_0 = _mm512_mul_ps(odd_0, center_reg);
        auto acc_1 = _mm512_mul_ps(odd_1, center_reg);
        auto acc_2 = _mm512_mul_ps(odd_2, center_reg);
        auto acc_3 = _mm512_mul_ps(odd_3, center_reg);

        // FIR over even lane
        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h_reg = _mm512_set1_ps(coeffs[k]);
            auto ev_0 = _mm512_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k));
            auto ev_1 = _mm512_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 8));
            auto ev_2 = _mm512_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 16));
            auto ev_3 = _mm512_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 24));

            acc_0 = _mm512_fmadd_ps(ev_0, h_reg, acc_0);
            acc_1 = _mm512_fmadd_ps(ev_1, h_reg, acc_1);
            acc_2 = _mm512_fmadd_ps(ev_2, h_reg, acc_2);
            acc_3 = _mm512_fmadd_ps(ev_3, h_reg, acc_3);
        }

        _mm512_storeu_ps(reinterpret_cast<float*>(output + i), acc_0);
        _mm512_storeu_ps(reinterpret_cast<float*>(output + i + 8), acc_1);
        _mm512_storeu_ps(reinterpret_cast<float*>(output + i + 16), acc_2);
        _mm512_storeu_ps(reinterpret_cast<float*>(output + i + 24), acc_3);
    }

    // Tail loop: 16 outputs
    if (i + 16 <= num_outputs) {
        auto odd_0 = _mm512_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset));
        auto odd_1 = _mm512_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 8));

        auto acc_0 = _mm512_mul_ps(odd_0, center_reg);
        auto acc_1 = _mm512_mul_ps(odd_1, center_reg);

        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h_reg = _mm512_set1_ps(coeffs[k]);
            auto ev_0 = _mm512_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k));
            auto ev_1 = _mm512_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 8));
            acc_0 = _mm512_fmadd_ps(ev_0, h_reg, acc_0);
            acc_1 = _mm512_fmadd_ps(ev_1, h_reg, acc_1);
        }

        _mm512_storeu_ps(reinterpret_cast<float*>(output + i), acc_0);
        _mm512_storeu_ps(reinterpret_cast<float*>(output + i + 8), acc_1);
        i += 16;
    }

    // Scalar tail
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
  const std::complex<float>* even_hist,
  const std::complex<float>* odd_hist,
  const float* coeffs,
  std::size_t num_taps,
  float center_tap,
  std::size_t delay_offset,
  std::complex<float>* output,
  std::size_t num_outputs
) -> void {
    auto center_reg = _mm256_set1_ps(center_tap);

    // Primary loop: 16 outputs (2 YMM registers)
    std::size_t i = 0;
    for (; i + 16 <= num_outputs; i += 16) {
        // Load odd lane + initialize with center tap
        auto odd_0 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset));
        auto odd_1 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 4));
        auto odd_2 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 8));
        auto odd_3 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 12));

        auto acc_0 = _mm256_mul_ps(odd_0, center_reg);
        auto acc_1 = _mm256_mul_ps(odd_1, center_reg);
        auto acc_2 = _mm256_mul_ps(odd_2, center_reg);
        auto acc_3 = _mm256_mul_ps(odd_3, center_reg);

        // FIR over even lane
        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h_reg = _mm256_set1_ps(coeffs[k]);
            auto ev_0 = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k));
            auto ev_1 = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 4));
            auto ev_2 = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 8));
            auto ev_3 = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 12));

            acc_0 = _mm256_fmadd_ps(ev_0, h_reg, acc_0);
            acc_1 = _mm256_fmadd_ps(ev_1, h_reg, acc_1);
            acc_2 = _mm256_fmadd_ps(ev_2, h_reg, acc_2);
            acc_3 = _mm256_fmadd_ps(ev_3, h_reg, acc_3);
        }

        _mm256_storeu_ps(reinterpret_cast<float*>(output + i), acc_0);
        _mm256_storeu_ps(reinterpret_cast<float*>(output + i + 4), acc_1);
        _mm256_storeu_ps(reinterpret_cast<float*>(output + i + 8), acc_2);
        _mm256_storeu_ps(reinterpret_cast<float*>(output + i + 12), acc_3);
    }

    // Tail loop: 8 outputs
    if (i + 8 <= num_outputs) {
        auto odd_0 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset));
        auto odd_1 = _mm256_loadu_ps(reinterpret_cast<const float*>(odd_hist + i + delay_offset + 4));

        auto acc_0 = _mm256_mul_ps(odd_0, center_reg);
        auto acc_1 = _mm256_mul_ps(odd_1, center_reg);

        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h_reg = _mm256_set1_ps(coeffs[k]);
            auto ev_0 = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k));
            auto ev_1 = _mm256_loadu_ps(reinterpret_cast<const float*>(even_hist + i + k + 4));
            acc_0 = _mm256_fmadd_ps(ev_0, h_reg, acc_0);
            acc_1 = _mm256_fmadd_ps(ev_1, h_reg, acc_1);
        }

        _mm256_storeu_ps(reinterpret_cast<float*>(output + i), acc_0);
        _mm256_storeu_ps(reinterpret_cast<float*>(output + i + 4), acc_1);
        i += 8;
    }

    // Scalar tail
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
  const std::complex<float>* even_hist,
  const std::complex<float>* odd_hist,
  const float* coeffs,
  std::size_t num_taps,
  float center_tap,
  std::size_t delay_offset,
  std::complex<float>* output,
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
