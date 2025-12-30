// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#pragma once

#include <cassert>
#include <complex>
#include <cstring>
#include <immintrin.h>

namespace kernels {

// =============================================================================
// FUSED DEINTERLEAVE + FILTER - Tiled for Cache Locality
// =============================================================================
// Combines deinterleave and filter operations to improve cache utilization.
// Processes data in tiles that fit in L1 cache, filtering immediately after
// deinterleaving while data is still hot.
//
// Memory savings: Only writes tail samples to history buffers (not full block).
// Cache benefit: Deinterleaved data is consumed before eviction from L1.
//
// Tile buffer layout:
//   tile_even: [history_len samples][tile_outputs samples]
//   tile_odd:  [history_len samples][tile_outputs samples]
//
// Filter indexing (same as non-fused kernel):
//   - Center tap: tile_even[i + delay_offset] for output i
//   - FIR taps:   tile_odd[i + k] for output i, tap k

// Tile size chosen to fit working set in L1 cache (32KB)
inline constexpr std::size_t FUSED_TILE_SIZE = 512;

// Maximum supported history length (limits filter_semi_length to 128)
// This ensures: 1) tile buffers don't overflow, 2) in-place output is safe
inline constexpr std::size_t MAX_HISTORY_LEN = 256;
static_assert(
    MAX_HISTORY_LEN <= (FUSED_TILE_SIZE / 2),
    "MAX_HISTORY_LEN must be <= FUSED_TILE_SIZE/2 for in-place safety"
);

// -----------------------------------------------------------------------------
// AVX-512 Version
// -----------------------------------------------------------------------------
[[gnu::target("avx512f")]]
inline auto halfband_filter_fused(
    const std::complex<float>* __restrict__ input,       // Interleaved input pairs
    std::complex<float>* __restrict__ even_hist,         // History buffer for even lane (read/write)
    std::complex<float>* __restrict__ odd_hist,          // History buffer for odd lane (read/write)
    const float* __restrict__ coeffs,
    std::size_t num_taps,
    float center_tap,
    std::size_t delay_offset,
    std::complex<float>* __restrict__ output,
    std::size_t num_outputs,
    std::size_t history_len
) -> void {
    // Validate parameters to prevent buffer overflow and in-place corruption
    assert(history_len <= MAX_HISTORY_LEN && "history_len exceeds MAX_HISTORY_LEN (256)");

    // Stack-allocated tile buffers (L1 resident)
    // Layout: [history_len][tile_outputs]
    alignas(64) std::complex<float> tile_even[FUSED_TILE_SIZE + MAX_HISTORY_LEN];
    alignas(64) std::complex<float> tile_odd[FUSED_TILE_SIZE + MAX_HISTORY_LEN];

    const auto* input_f = reinterpret_cast<const float*>(input);
    auto* output_f = reinterpret_cast<float*>(output);

    auto center_reg = _mm512_set1_ps(center_tap);

    // Permute indices for deinterleave
    const auto idx_even = _mm512_set_epi32(
        29, 28, 25, 24, 21, 20, 17, 16,
        13, 12,  9,  8,  5,  4,  1,  0
    );
    const auto idx_odd = _mm512_set_epi32(
        31, 30, 27, 26, 23, 22, 19, 18,
        15, 14, 11, 10,  7,  6,  3,  2
    );

    std::size_t output_pos = 0;

    while (output_pos < num_outputs) {
        // Determine tile size
        std::size_t tile_outputs = std::min(FUSED_TILE_SIZE, num_outputs - output_pos);

        // --- PHASE 1: Set up tile buffers with history + new data ---

        // Copy history to the beginning of tile buffers
        if (output_pos == 0) {
            // First tile: copy from persistent history buffers
            std::memcpy(tile_even, even_hist, history_len * sizeof(std::complex<float>));
            std::memcpy(tile_odd, odd_hist, history_len * sizeof(std::complex<float>));
        } else {
            // Subsequent tiles: re-deinterleave overlap from input
            // We need history_len samples that precede this tile's output_pos
            for (std::size_t j = 0; j < history_len; ++j) {
                std::size_t input_idx = output_pos - history_len + j;
                tile_even[j] = input[2 * input_idx];
                tile_odd[j] = input[2 * input_idx + 1];
            }
        }

        // Deinterleave new samples into tile buffer AFTER history
        auto* tile_even_new = tile_even + history_len;
        auto* tile_odd_new = tile_odd + history_len;
        auto* tile_even_f = reinterpret_cast<float*>(tile_even_new);
        auto* tile_odd_f = reinterpret_cast<float*>(tile_odd_new);
        const auto* tile_input_f = input_f + 4 * output_pos;

        // Deinterleave using AVX-512
        std::size_t di = 0;
        for (; di + 16 <= tile_outputs; di += 16) {
            auto t0 = _mm512_loadu_ps(tile_input_f + 4*di);
            auto t1 = _mm512_loadu_ps(tile_input_f + 4*di + 16);
            auto t2 = _mm512_loadu_ps(tile_input_f + 4*di + 32);
            auto t3 = _mm512_loadu_ps(tile_input_f + 4*di + 48);

            auto ev0 = _mm512_permutex2var_ps(t0, idx_even, t1);
            auto od0 = _mm512_permutex2var_ps(t0, idx_odd, t1);
            auto ev1 = _mm512_permutex2var_ps(t2, idx_even, t3);
            auto od1 = _mm512_permutex2var_ps(t2, idx_odd, t3);

            _mm512_storeu_ps(tile_even_f + 2*di, ev0);
            _mm512_storeu_ps(tile_odd_f + 2*di, od0);
            _mm512_storeu_ps(tile_even_f + 2*di + 16, ev1);
            _mm512_storeu_ps(tile_odd_f + 2*di + 16, od1);
        }

        // Scalar tail for deinterleave
        for (; di < tile_outputs; ++di) {
            tile_even_new[di] = input[2 * (output_pos + di)];
            tile_odd_new[di] = input[2 * (output_pos + di) + 1];
        }

        // --- PHASE 2: Filter (data is hot in L1!) ---
        // tile_even/tile_odd now have [history_len][tile_outputs] layout
        // Filter reads from indices starting at 0 (history region)

        const auto* filter_even = reinterpret_cast<const float*>(tile_even);
        const auto* filter_odd = reinterpret_cast<const float*>(tile_odd);

        // Filter this tile
        std::size_t fi = 0;
        for (; fi + 32 <= tile_outputs; fi += 32) {
            // Load even samples for center tap (with delay)
            // Access pattern: tile_even[fi + delay_offset] where fi is relative to history_len
            // Since tile starts with history, actual index is fi + delay_offset
            auto even_0 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset));
            auto even_1 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 16);
            auto even_2 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 32);
            auto even_3 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 48);

            auto acc_0a = _mm512_mul_ps(even_0, center_reg);
            auto acc_1a = _mm512_mul_ps(even_1, center_reg);
            auto acc_2a = _mm512_mul_ps(even_2, center_reg);
            auto acc_3a = _mm512_mul_ps(even_3, center_reg);

            auto acc_0b = _mm512_setzero_ps();
            auto acc_1b = _mm512_setzero_ps();
            auto acc_2b = _mm512_setzero_ps();
            auto acc_3b = _mm512_setzero_ps();

            for (std::size_t k = 0; k < num_taps; k += 2) {
                auto h_even = _mm512_set1_ps(coeffs[k]);

                auto od_0a = _mm512_loadu_ps(filter_odd + 2*(fi + k));
                auto od_1a = _mm512_loadu_ps(filter_odd + 2*(fi + k) + 16);
                auto od_2a = _mm512_loadu_ps(filter_odd + 2*(fi + k) + 32);
                auto od_3a = _mm512_loadu_ps(filter_odd + 2*(fi + k) + 48);

                acc_0a = _mm512_fmadd_ps(od_0a, h_even, acc_0a);
                acc_1a = _mm512_fmadd_ps(od_1a, h_even, acc_1a);
                acc_2a = _mm512_fmadd_ps(od_2a, h_even, acc_2a);
                acc_3a = _mm512_fmadd_ps(od_3a, h_even, acc_3a);

                if (k + 1 < num_taps) {
                    auto h_odd = _mm512_set1_ps(coeffs[k+1]);

                    auto od_0b = _mm512_loadu_ps(filter_odd + 2*(fi + k + 1));
                    auto od_1b = _mm512_loadu_ps(filter_odd + 2*(fi + k + 1) + 16);
                    auto od_2b = _mm512_loadu_ps(filter_odd + 2*(fi + k + 1) + 32);
                    auto od_3b = _mm512_loadu_ps(filter_odd + 2*(fi + k + 1) + 48);

                    acc_0b = _mm512_fmadd_ps(od_0b, h_odd, acc_0b);
                    acc_1b = _mm512_fmadd_ps(od_1b, h_odd, acc_1b);
                    acc_2b = _mm512_fmadd_ps(od_2b, h_odd, acc_2b);
                    acc_3b = _mm512_fmadd_ps(od_3b, h_odd, acc_3b);
                }
            }

            _mm512_storeu_ps(output_f + 2*(output_pos + fi),      _mm512_add_ps(acc_0a, acc_0b));
            _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 16, _mm512_add_ps(acc_1a, acc_1b));
            _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 32, _mm512_add_ps(acc_2a, acc_2b));
            _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 48, _mm512_add_ps(acc_3a, acc_3b));
        }

        // Scalar tail for filter
        for (; fi < tile_outputs; ++fi) {
            auto sum = tile_even[fi + delay_offset] * center_tap;
            for (std::size_t k = 0; k < num_taps; ++k) {
                sum += tile_odd[fi + k] * coeffs[k];
            }
            output[output_pos + fi] = sum;
        }

        output_pos += tile_outputs;
    }

    // --- PHASE 3: Update history with ONLY the tail samples ---
    if (num_outputs >= history_len) {
        // Normal case: copy last history_len samples from input
        for (std::size_t i = 0; i < history_len; ++i) {
            std::size_t src_idx = num_outputs - history_len + i;
            even_hist[i] = input[2 * src_idx];
            odd_hist[i] = input[2 * src_idx + 1];
        }
    } else {
        // Small input: shift history left, append new samples
        // This preserves filter continuity for tiny blocks
        std::size_t keep = history_len - num_outputs;
        std::memmove(even_hist, even_hist + num_outputs, keep * sizeof(std::complex<float>));
        std::memmove(odd_hist, odd_hist + num_outputs, keep * sizeof(std::complex<float>));
        for (std::size_t i = 0; i < num_outputs; ++i) {
            even_hist[keep + i] = input[2 * i];
            odd_hist[keep + i] = input[2 * i + 1];
        }
    }
}

// -----------------------------------------------------------------------------
// Scalar Version (fallback)
// -----------------------------------------------------------------------------
[[gnu::target("default")]]
inline auto halfband_filter_fused(
    const std::complex<float>* __restrict__ input,
    std::complex<float>* __restrict__ even_hist,
    std::complex<float>* __restrict__ odd_hist,
    const float* __restrict__ coeffs,
    std::size_t num_taps,
    float center_tap,
    std::size_t delay_offset,
    std::complex<float>* __restrict__ output,
    std::size_t num_outputs,
    std::size_t history_len
) -> void {
    // Validate parameters to prevent buffer overflow and in-place corruption
    assert(history_len <= MAX_HISTORY_LEN && "history_len exceeds MAX_HISTORY_LEN (256)");

    // Simple scalar implementation matching AVX version logic
    // Tile buffer layout: [history_len samples][tile_outputs samples]

    alignas(64) std::complex<float> tile_even[FUSED_TILE_SIZE + MAX_HISTORY_LEN];
    alignas(64) std::complex<float> tile_odd[FUSED_TILE_SIZE + MAX_HISTORY_LEN];

    std::size_t output_pos = 0;

    while (output_pos < num_outputs) {
        std::size_t tile_outputs = std::min(FUSED_TILE_SIZE, num_outputs - output_pos);

        // Copy history to the beginning of tile buffers
        if (output_pos == 0) {
            // First tile: copy from persistent history buffers
            std::memcpy(tile_even, even_hist, history_len * sizeof(std::complex<float>));
            std::memcpy(tile_odd, odd_hist, history_len * sizeof(std::complex<float>));
        } else {
            // Subsequent tiles: re-deinterleave overlap from input
            for (std::size_t j = 0; j < history_len; ++j) {
                std::size_t input_idx = output_pos - history_len + j;
                tile_even[j] = input[2 * input_idx];
                tile_odd[j] = input[2 * input_idx + 1];
            }
        }

        // Deinterleave new samples AFTER history
        for (std::size_t i = 0; i < tile_outputs; ++i) {
            tile_even[history_len + i] = input[2 * (output_pos + i)];
            tile_odd[history_len + i] = input[2 * (output_pos + i) + 1];
        }

        // Filter - same indexing as non-fused kernel
        for (std::size_t i = 0; i < tile_outputs; ++i) {
            auto sum = tile_even[i + delay_offset] * center_tap;
            for (std::size_t k = 0; k < num_taps; ++k) {
                sum += tile_odd[i + k] * coeffs[k];
            }
            output[output_pos + i] = sum;
        }

        output_pos += tile_outputs;
    }

    // Update history with tail samples
    if (num_outputs >= history_len) {
        // Normal case: copy last history_len samples from input
        for (std::size_t i = 0; i < history_len; ++i) {
            std::size_t src_idx = num_outputs - history_len + i;
            even_hist[i] = input[2 * src_idx];
            odd_hist[i] = input[2 * src_idx + 1];
        }
    } else {
        // Small input: shift history left, append new samples
        std::size_t keep = history_len - num_outputs;
        std::memmove(even_hist, even_hist + num_outputs, keep * sizeof(std::complex<float>));
        std::memmove(odd_hist, odd_hist + num_outputs, keep * sizeof(std::complex<float>));
        for (std::size_t i = 0; i < num_outputs; ++i) {
            even_hist[keep + i] = input[2 * i];
            odd_hist[keep + i] = input[2 * i + 1];
        }
    }
}

} // namespace kernels
