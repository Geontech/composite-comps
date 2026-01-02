// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#pragma once

#include <array>
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
// AVX-512 Helper: Process tile (deinterleave new samples + filter)
// Must be always_inline with matching target attributes for loop peeling
// -----------------------------------------------------------------------------
[[gnu::target("avx512f,fma"), gnu::always_inline]]
static inline auto process_tile_avx512(
  const std::complex<float>* input,
  std::complex<float>* __restrict__ tile_even,
  std::complex<float>* __restrict__ tile_odd,
  std::complex<float>* output,
  const float* __restrict__ coeffs,
  std::size_t num_taps,
  float center_tap,
  std::size_t delay_offset,
  std::size_t output_pos,
  std::size_t tile_outputs,
  std::size_t history_len,
  __m512& center_reg,
  const __m512i& idx_even,
  const __m512i& idx_odd
) -> void {
    const auto* input_f = reinterpret_cast<const float*>(input);
    auto* output_f = reinterpret_cast<float*>(output);

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

    // --- Filter (data is hot in L1!) ---
    const auto* filter_even = reinterpret_cast<const float*>(tile_even);
    const auto* filter_odd = reinterpret_cast<const float*>(tile_odd);

    // Filter this tile using cascaded loops: 64 -> 32 -> 16 -> 8 -> scalar
    std::size_t fi = 0;

    // --- 64 outputs (8 ZMM accumulators) - main workhorse ---
    for (; fi + 64 <= tile_outputs; fi += 64) {
        auto ev_0 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset));
        auto ev_1 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 16);
        auto ev_2 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 32);
        auto ev_3 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 48);
        auto ev_4 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 64);
        auto ev_5 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 80);
        auto ev_6 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 96);
        auto ev_7 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 112);

        auto acc_0 = _mm512_mul_ps(ev_0, center_reg);
        auto acc_1 = _mm512_mul_ps(ev_1, center_reg);
        auto acc_2 = _mm512_mul_ps(ev_2, center_reg);
        auto acc_3 = _mm512_mul_ps(ev_3, center_reg);
        auto acc_4 = _mm512_mul_ps(ev_4, center_reg);
        auto acc_5 = _mm512_mul_ps(ev_5, center_reg);
        auto acc_6 = _mm512_mul_ps(ev_6, center_reg);
        auto acc_7 = _mm512_mul_ps(ev_7, center_reg);

        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h = _mm512_set1_ps(coeffs[k]);
            acc_0 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k)),       h, acc_0);
            acc_1 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k) + 16),  h, acc_1);
            acc_2 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k) + 32),  h, acc_2);
            acc_3 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k) + 48),  h, acc_3);
            acc_4 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k) + 64),  h, acc_4);
            acc_5 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k) + 80),  h, acc_5);
            acc_6 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k) + 96),  h, acc_6);
            acc_7 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k) + 112), h, acc_7);
        }

        _mm512_storeu_ps(output_f + 2*(output_pos + fi),       acc_0);
        _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 16,  acc_1);
        _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 32,  acc_2);
        _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 48,  acc_3);
        _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 64,  acc_4);
        _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 80,  acc_5);
        _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 96,  acc_6);
        _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 112, acc_7);
    }

    // --- 32 outputs (4 ZMM accumulators) ---
    for (; fi + 32 <= tile_outputs; fi += 32) {
        auto ev_0 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset));
        auto ev_1 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 16);
        auto ev_2 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 32);
        auto ev_3 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 48);

        auto acc_0 = _mm512_mul_ps(ev_0, center_reg);
        auto acc_1 = _mm512_mul_ps(ev_1, center_reg);
        auto acc_2 = _mm512_mul_ps(ev_2, center_reg);
        auto acc_3 = _mm512_mul_ps(ev_3, center_reg);

        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h = _mm512_set1_ps(coeffs[k]);
            acc_0 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k)),      h, acc_0);
            acc_1 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k) + 16), h, acc_1);
            acc_2 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k) + 32), h, acc_2);
            acc_3 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k) + 48), h, acc_3);
        }

        _mm512_storeu_ps(output_f + 2*(output_pos + fi),      acc_0);
        _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 16, acc_1);
        _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 32, acc_2);
        _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 48, acc_3);
    }

    // --- 16 outputs (2 ZMM accumulators) ---
    for (; fi + 16 <= tile_outputs; fi += 16) {
        auto ev_0 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset));
        auto ev_1 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset) + 16);

        auto acc_0 = _mm512_mul_ps(ev_0, center_reg);
        auto acc_1 = _mm512_mul_ps(ev_1, center_reg);

        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h = _mm512_set1_ps(coeffs[k]);
            acc_0 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k)),      h, acc_0);
            acc_1 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k) + 16), h, acc_1);
        }

        _mm512_storeu_ps(output_f + 2*(output_pos + fi),      acc_0);
        _mm512_storeu_ps(output_f + 2*(output_pos + fi) + 16, acc_1);
    }

    // --- 8 outputs (1 ZMM accumulator) ---
    for (; fi + 8 <= tile_outputs; fi += 8) {
        auto ev_0 = _mm512_loadu_ps(filter_even + 2*(fi + delay_offset));
        auto acc_0 = _mm512_mul_ps(ev_0, center_reg);

        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h = _mm512_set1_ps(coeffs[k]);
            acc_0 = _mm512_fmadd_ps(_mm512_loadu_ps(filter_odd + 2*(fi + k)), h, acc_0);
        }

        _mm512_storeu_ps(output_f + 2*(output_pos + fi), acc_0);
    }

    // --- Scalar tail (0-7 outputs) ---
    for (; fi < tile_outputs; ++fi) {
        auto sum = tile_even[fi + delay_offset] * center_tap;
        for (std::size_t k = 0; k < num_taps; ++k) {
            sum += tile_odd[fi + k] * coeffs[k];
        }
        output[output_pos + fi] = sum;
    }
}

// -----------------------------------------------------------------------------
// AVX-512 Version
// -----------------------------------------------------------------------------
[[gnu::target("avx512f,fma")]]
inline auto halfband_filter_fused(
  const std::complex<float>* input,                 // Interleaved input pairs
  std::complex<float>* __restrict__ even_hist,      // History buffer for even lane (read/write)
  std::complex<float>* __restrict__ odd_hist,       // History buffer for odd lane (read/write)
  const float* __restrict__ coeffs,
  std::size_t num_taps,
  float center_tap,
  std::size_t delay_offset,
  std::complex<float>* output,
  std::size_t num_outputs,
  std::size_t history_len
) -> void {
    // Validate parameters to prevent buffer overflow and in-place corruption
    assert(history_len <= MAX_HISTORY_LEN && "history_len exceeds MAX_HISTORY_LEN (256)");

    if (num_outputs == 0) return;

    // Stack-allocated tile buffers (L1 resident)
    // Layout: [history_len][tile_outputs]
    alignas(64) std::array<std::complex<float>, FUSED_TILE_SIZE + MAX_HISTORY_LEN> tile_even{};
    alignas(64) std::array<std::complex<float>, FUSED_TILE_SIZE + MAX_HISTORY_LEN> tile_odd{};

    const auto* input_f = reinterpret_cast<const float*>(input);

    auto center_reg = _mm512_set1_ps(center_tap);

    // Permute indices for deinterleave
    const auto idx_even = _mm512_set_epi32(29,28,25,24,21,20,17,16,13,12,9,8,5,4,1,0);
    const auto idx_odd = _mm512_set_epi32(31,30,27,26,23,22,19,18,15,14,11,10,7,6,3,2);

    // -------------------------------------------------------------------------
    // PEELED: First tile (uses persistent history buffers)
    // -------------------------------------------------------------------------
    std::size_t output_pos = 0;
    std::size_t tile_outputs = std::min(FUSED_TILE_SIZE, num_outputs);

    // Copy from persistent history buffers
    std::memcpy(tile_even.data(), even_hist, history_len * sizeof(std::complex<float>));
    std::memcpy(tile_odd.data(), odd_hist, history_len * sizeof(std::complex<float>));

    // Process first tile
    process_tile_avx512(
        input, tile_even.data(), tile_odd.data(), output,
        coeffs, num_taps, center_tap, delay_offset,
        output_pos, tile_outputs, history_len,
        center_reg, idx_even, idx_odd
    );
    output_pos += tile_outputs;

    // -------------------------------------------------------------------------
    // MAIN LOOP: Subsequent tiles (no if-check needed, always re-deinterleave)
    // -------------------------------------------------------------------------
    while (output_pos < num_outputs) {
        tile_outputs = std::min(FUSED_TILE_SIZE, num_outputs - output_pos);

        // Re-deinterleave overlap from input (VECTORIZED)
        std::size_t reload_base = output_pos - history_len;
        const auto* reload_f = input_f + 4 * reload_base;
        auto* hist_even_f = reinterpret_cast<float*>(tile_even.data());
        auto* hist_odd_f = reinterpret_cast<float*>(tile_odd.data());

        std::size_t hi = 0;
        for (; hi + 16 <= history_len; hi += 16) {
            auto t0 = _mm512_loadu_ps(reload_f + 4*hi);
            auto t1 = _mm512_loadu_ps(reload_f + 4*hi + 16);
            auto t2 = _mm512_loadu_ps(reload_f + 4*hi + 32);
            auto t3 = _mm512_loadu_ps(reload_f + 4*hi + 48);

            // Aligned stores: tile buffers are alignas(64), hi advances by 16 (128 bytes)
            _mm512_store_ps(hist_even_f + 2*hi,      _mm512_permutex2var_ps(t0, idx_even, t1));
            _mm512_store_ps(hist_odd_f + 2*hi,       _mm512_permutex2var_ps(t0, idx_odd, t1));
            _mm512_store_ps(hist_even_f + 2*hi + 16, _mm512_permutex2var_ps(t2, idx_even, t3));
            _mm512_store_ps(hist_odd_f + 2*hi + 16,  _mm512_permutex2var_ps(t2, idx_odd, t3));
        }
        // Scalar tail for history reload
        for (; hi < history_len; ++hi) {
            tile_even[hi] = input[2 * (reload_base + hi)];
            tile_odd[hi] = input[2 * (reload_base + hi) + 1];
        }

        // Process tile
        process_tile_avx512(
            input, tile_even.data(), tile_odd.data(), output,
            coeffs, num_taps, center_tap, delay_offset,
            output_pos, tile_outputs, history_len,
            center_reg, idx_even, idx_odd
        );
        output_pos += tile_outputs;
    }

    // -------------------------------------------------------------------------
    // Update history with tail samples
    // -------------------------------------------------------------------------
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
// AVX2 Helper: Process tile (deinterleave new samples + filter)
// Must be always_inline with matching target attributes for loop peeling
// -----------------------------------------------------------------------------
[[gnu::target("avx2,fma"), gnu::always_inline]]
static inline auto process_tile_avx2(
  const std::complex<float>* input,
  std::complex<float>* __restrict__ tile_even,
  std::complex<float>* __restrict__ tile_odd,
  std::complex<float>* output,
  const float* __restrict__ coeffs,
  std::size_t num_taps,
  float center_tap,
  std::size_t delay_offset,
  std::size_t output_pos,
  std::size_t tile_outputs,
  std::size_t history_len,
  __m256& center_reg
) -> void {
    auto* output_f = reinterpret_cast<float*>(output);

    // Deinterleave new samples into tile buffer AFTER history
    auto* tile_even_new = tile_even + history_len;
    auto* tile_odd_new = tile_odd + history_len;

    // AVX2 deinterleave: process 4 output pairs per iteration
    std::size_t di = 0;
    for (; di + 4 <= tile_outputs; di += 4) {
        auto a = _mm256_loadu_pd(reinterpret_cast<const double*>(input + 2*(output_pos + di)));
        auto b = _mm256_loadu_pd(reinterpret_cast<const double*>(input + 2*(output_pos + di) + 4));

        auto evens_lo = _mm256_unpacklo_pd(a, b);
        auto odds_lo  = _mm256_unpackhi_pd(a, b);

        auto evens = _mm256_permute4x64_pd(evens_lo, 0xD8);
        auto odds  = _mm256_permute4x64_pd(odds_lo, 0xD8);

        _mm256_storeu_pd(reinterpret_cast<double*>(tile_even_new + di), evens);
        _mm256_storeu_pd(reinterpret_cast<double*>(tile_odd_new + di), odds);
    }

    // Scalar tail for deinterleave
    for (; di < tile_outputs; ++di) {
        tile_even_new[di] = input[2 * (output_pos + di)];
        tile_odd_new[di] = input[2 * (output_pos + di) + 1];
    }

    // --- Filter (data is hot in L1!) ---
    const auto* filter_even = reinterpret_cast<const float*>(tile_even);
    const auto* filter_odd = reinterpret_cast<const float*>(tile_odd);

    // Filter this tile using cascaded loops: 32 -> 16 -> 8 -> 4 -> scalar
    std::size_t fi = 0;

    // --- 32 outputs (8 YMM accumulators) - main workhorse ---
    for (; fi + 32 <= tile_outputs; fi += 32) {
        auto ev_0 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset));
        auto ev_1 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset) + 8);
        auto ev_2 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset) + 16);
        auto ev_3 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset) + 24);
        auto ev_4 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset) + 32);
        auto ev_5 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset) + 40);
        auto ev_6 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset) + 48);
        auto ev_7 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset) + 56);

        auto acc_0 = _mm256_mul_ps(ev_0, center_reg);
        auto acc_1 = _mm256_mul_ps(ev_1, center_reg);
        auto acc_2 = _mm256_mul_ps(ev_2, center_reg);
        auto acc_3 = _mm256_mul_ps(ev_3, center_reg);
        auto acc_4 = _mm256_mul_ps(ev_4, center_reg);
        auto acc_5 = _mm256_mul_ps(ev_5, center_reg);
        auto acc_6 = _mm256_mul_ps(ev_6, center_reg);
        auto acc_7 = _mm256_mul_ps(ev_7, center_reg);

        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h = _mm256_set1_ps(coeffs[k]);
            acc_0 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k)),      h, acc_0);
            acc_1 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k) + 8),  h, acc_1);
            acc_2 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k) + 16), h, acc_2);
            acc_3 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k) + 24), h, acc_3);
            acc_4 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k) + 32), h, acc_4);
            acc_5 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k) + 40), h, acc_5);
            acc_6 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k) + 48), h, acc_6);
            acc_7 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k) + 56), h, acc_7);
        }

        _mm256_storeu_ps(output_f + 2*(output_pos + fi),      acc_0);
        _mm256_storeu_ps(output_f + 2*(output_pos + fi) + 8,  acc_1);
        _mm256_storeu_ps(output_f + 2*(output_pos + fi) + 16, acc_2);
        _mm256_storeu_ps(output_f + 2*(output_pos + fi) + 24, acc_3);
        _mm256_storeu_ps(output_f + 2*(output_pos + fi) + 32, acc_4);
        _mm256_storeu_ps(output_f + 2*(output_pos + fi) + 40, acc_5);
        _mm256_storeu_ps(output_f + 2*(output_pos + fi) + 48, acc_6);
        _mm256_storeu_ps(output_f + 2*(output_pos + fi) + 56, acc_7);
    }

    // --- 16 outputs (4 YMM accumulators) ---
    for (; fi + 16 <= tile_outputs; fi += 16) {
        auto ev_0 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset));
        auto ev_1 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset) + 8);
        auto ev_2 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset) + 16);
        auto ev_3 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset) + 24);

        auto acc_0 = _mm256_mul_ps(ev_0, center_reg);
        auto acc_1 = _mm256_mul_ps(ev_1, center_reg);
        auto acc_2 = _mm256_mul_ps(ev_2, center_reg);
        auto acc_3 = _mm256_mul_ps(ev_3, center_reg);

        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h = _mm256_set1_ps(coeffs[k]);
            acc_0 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k)),      h, acc_0);
            acc_1 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k) + 8),  h, acc_1);
            acc_2 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k) + 16), h, acc_2);
            acc_3 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k) + 24), h, acc_3);
        }

        _mm256_storeu_ps(output_f + 2*(output_pos + fi),      acc_0);
        _mm256_storeu_ps(output_f + 2*(output_pos + fi) + 8,  acc_1);
        _mm256_storeu_ps(output_f + 2*(output_pos + fi) + 16, acc_2);
        _mm256_storeu_ps(output_f + 2*(output_pos + fi) + 24, acc_3);
    }

    // --- 8 outputs (2 YMM accumulators) ---
    for (; fi + 8 <= tile_outputs; fi += 8) {
        auto ev_0 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset));
        auto ev_1 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset) + 8);

        auto acc_0 = _mm256_mul_ps(ev_0, center_reg);
        auto acc_1 = _mm256_mul_ps(ev_1, center_reg);

        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h = _mm256_set1_ps(coeffs[k]);
            acc_0 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k)),     h, acc_0);
            acc_1 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k) + 8), h, acc_1);
        }

        _mm256_storeu_ps(output_f + 2*(output_pos + fi),     acc_0);
        _mm256_storeu_ps(output_f + 2*(output_pos + fi) + 8, acc_1);
    }

    // --- 4 outputs (1 YMM accumulator) ---
    for (; fi + 4 <= tile_outputs; fi += 4) {
        auto ev_0 = _mm256_loadu_ps(filter_even + 2*(fi + delay_offset));
        auto acc_0 = _mm256_mul_ps(ev_0, center_reg);

        for (std::size_t k = 0; k < num_taps; ++k) {
            auto h = _mm256_set1_ps(coeffs[k]);
            acc_0 = _mm256_fmadd_ps(_mm256_loadu_ps(filter_odd + 2*(fi + k)), h, acc_0);
        }

        _mm256_storeu_ps(output_f + 2*(output_pos + fi), acc_0);
    }

    // --- Scalar tail (0-3 outputs) ---
    for (; fi < tile_outputs; ++fi) {
        auto sum = tile_even[fi + delay_offset] * center_tap;
        for (std::size_t k = 0; k < num_taps; ++k) {
            sum += tile_odd[fi + k] * coeffs[k];
        }
        output[output_pos + fi] = sum;
    }
}

// -----------------------------------------------------------------------------
// AVX2 Version
// -----------------------------------------------------------------------------
[[gnu::target("avx2,fma")]]
inline auto halfband_filter_fused(
  const std::complex<float>* input,
  std::complex<float>* __restrict__ even_hist,
  std::complex<float>* __restrict__ odd_hist,
  const float* __restrict__ coeffs,
  std::size_t num_taps,
  float center_tap,
  std::size_t delay_offset,
  std::complex<float>* output,
  std::size_t num_outputs,
  std::size_t history_len
) -> void {
    assert(history_len <= MAX_HISTORY_LEN && "history_len exceeds MAX_HISTORY_LEN (256)");
    if (num_outputs == 0) { return; }

    alignas(64) std::array<std::complex<float>, FUSED_TILE_SIZE + MAX_HISTORY_LEN> tile_even{};
    alignas(64) std::array<std::complex<float>, FUSED_TILE_SIZE + MAX_HISTORY_LEN> tile_odd{};

    auto center_reg = _mm256_set1_ps(center_tap);

    // -------------------------------------------------------------------------
    // PEELED: First tile (uses persistent history buffers)
    // -------------------------------------------------------------------------
    std::size_t output_pos = 0;
    std::size_t tile_outputs = std::min(FUSED_TILE_SIZE, num_outputs);

    // Copy from persistent history buffers
    std::memcpy(tile_even.data(), even_hist, history_len * sizeof(std::complex<float>));
    std::memcpy(tile_odd.data(), odd_hist, history_len * sizeof(std::complex<float>));

    // Process first tile
    process_tile_avx2(
        input, tile_even.data(), tile_odd.data(), output,
        coeffs, num_taps, center_tap, delay_offset,
        output_pos, tile_outputs, history_len,
        center_reg
    );
    output_pos += tile_outputs;

    // -------------------------------------------------------------------------
    // MAIN LOOP: Subsequent tiles (no if-check needed, always re-deinterleave)
    // -------------------------------------------------------------------------
    while (output_pos < num_outputs) {
        tile_outputs = std::min(FUSED_TILE_SIZE, num_outputs - output_pos);

        // Re-deinterleave overlap from input (VECTORIZED)
        std::size_t reload_base = output_pos - history_len;

        std::size_t hi = 0;
        for (; hi + 4 <= history_len; hi += 4) {
            auto a = _mm256_loadu_pd(reinterpret_cast<const double*>(input + 2*(reload_base + hi)));
            auto b = _mm256_loadu_pd(reinterpret_cast<const double*>(input + 2*(reload_base + hi) + 4));

            auto evens_lo = _mm256_unpacklo_pd(a, b);
            auto odds_lo  = _mm256_unpackhi_pd(a, b);

            // Aligned stores: tile buffers are alignas(64), hi advances by 4 (32 bytes)
            _mm256_store_pd(reinterpret_cast<double*>(&tile_even[hi]), _mm256_permute4x64_pd(evens_lo, 0xD8));
            _mm256_store_pd(reinterpret_cast<double*>(&tile_odd[hi]),  _mm256_permute4x64_pd(odds_lo, 0xD8));
        }
        // Scalar tail for history reload
        for (; hi < history_len; ++hi) {
            tile_even[hi] = input[2 * (reload_base + hi)];
            tile_odd[hi] = input[2 * (reload_base + hi) + 1];
        }

        // Process tile
        process_tile_avx2(
            input, tile_even.data(), tile_odd.data(), output,
            coeffs, num_taps, center_tap, delay_offset,
            output_pos, tile_outputs, history_len,
            center_reg
        );
        output_pos += tile_outputs;
    }

    // -------------------------------------------------------------------------
    // Update history with tail samples
    // -------------------------------------------------------------------------
    if (num_outputs >= history_len) {
        for (std::size_t i = 0; i < history_len; ++i) {
            std::size_t src_idx = num_outputs - history_len + i;
            even_hist[i] = input[2 * src_idx];
            odd_hist[i] = input[2 * src_idx + 1];
        }
    } else {
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

    alignas(64) std::array<std::complex<float>, FUSED_TILE_SIZE + MAX_HISTORY_LEN> tile_even{};
    alignas(64) std::array<std::complex<float>, FUSED_TILE_SIZE + MAX_HISTORY_LEN> tile_odd{};

    std::size_t output_pos = 0;

    while (output_pos < num_outputs) {
        std::size_t tile_outputs = std::min(FUSED_TILE_SIZE, num_outputs - output_pos);

        // Copy history to the beginning of tile buffers
        if (output_pos == 0) {
            // First tile: copy from persistent history buffers
            std::memcpy(tile_even.data(), even_hist, history_len * sizeof(std::complex<float>));
            std::memcpy(tile_odd.data(), odd_hist, history_len * sizeof(std::complex<float>));
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
