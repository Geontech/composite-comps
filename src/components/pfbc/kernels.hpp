// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace kernels {

// =============================================================================
// Streaming Store Helpers
// =============================================================================
// Streaming stores bypass the cache, writing directly to memory. This is
// beneficial for write-only output buffers that won't be read soon by the
// same core, as it avoids cache pollution.
//
// Requirements:
// - 64-byte alignment for _mm512_stream_ps (AVX-512)
// - 32-byte alignment for _mm256_stream_ps (AVX2)
// - Call _mm_sfence() after streaming stores if consumer needs immediate visibility
//

// Set to 1 to enable streaming (non-temporal) stores for output buffers
// Streaming stores bypass the cache, beneficial when output won't be read soon
#define PFBC_USE_STREAMING_STORES 0

[[gnu::target("avx512f")]]
inline void store_512(float* ptr, __m512 val) {
#if PFBC_USE_STREAMING_STORES
    if ((reinterpret_cast<std::uintptr_t>(ptr) & 63) == 0) {
        _mm512_stream_ps(ptr, val);
    } else {
        _mm512_storeu_ps(ptr, val);
    }
#else
    _mm512_storeu_ps(ptr, val);
#endif
}

[[gnu::target("avx512f")]]
inline void store_512d(double* ptr, __m512d val) {
#if PFBC_USE_STREAMING_STORES
    if ((reinterpret_cast<std::uintptr_t>(ptr) & 63) == 0) {
        _mm512_stream_pd(ptr, val);
    } else {
        _mm512_storeu_pd(ptr, val);
    }
#else
    _mm512_storeu_pd(ptr, val);
#endif
}

[[gnu::target("avx2")]]
inline void store_256(float* ptr, __m256 val) {
#if PFBC_USE_STREAMING_STORES
    if ((reinterpret_cast<std::uintptr_t>(ptr) & 31) == 0) {
        _mm256_stream_ps(ptr, val);
    } else {
        _mm256_storeu_ps(ptr, val);
    }
#else
    _mm256_storeu_ps(ptr, val);
#endif
}

// =============================================================================
// Transpose Functions
// =============================================================================

static_assert(
    sizeof(std::complex<float>) == sizeof(double),
    "transpose functions assume complex<float> is 64-bit"
);

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

        if (d0) { store_512d(d0, t0v); }
        if (d1) { store_512d(d1, t1v); }
        if (d2) { store_512d(d2, t2v); }
        if (d3) { store_512d(d3, t3v); }
        if (d4) { store_512d(d4, t4v); }
        if (d5) { store_512d(d5, t5v); }
        if (d6) { store_512d(d6, t6v); }
        if (d7) { store_512d(d7, t7v); }
    }

    // Handle remaining channels (M not divisible by 8)
    for (std::size_t ch = M8; ch < M; ++ch) {
        if (!dst_ptrs[ch]) { continue; }
        for (std::size_t t = 0; t < 8; ++t) {
            dst_ptrs[ch][offset + t] = tile[t * M + ch];
        }
    }
}

// =============================================================================
// M=8 Fused Filter + FFT + Transpose (Register-Resident)
// =============================================================================
// For M=8, the entire pipeline stays in registers:
//   Filter → Deinterleave → FFT-8 → Reorder → Interleave → Transpose → Store
//
// No intermediate memory writes between stages.

namespace fft8 {

namespace detail {
    inline constexpr float SQRT2_2 = 0.7071067811865475244f;

    // Matrix DFT twiddle factors for interleaved format
    // Each row n contains W^(n*k) for k=0..7, duplicated for interleaved complex
    // Format: [W^(n*0).re, W^(n*0).re, W^(n*1).re, W^(n*1).re, ...]
    inline constexpr float S = 0.70710678118f;

    alignas(64) inline constexpr float DFT8_TW_RR[8][16] = {
        {1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1, 1,1},
        {1,1, S,S, 0,0, -S,-S, -1,-1, -S,-S, 0,0, S,S},
        {1,1, 0,0, -1,-1, 0,0, 1,1, 0,0, -1,-1, 0,0},
        {1,1, -S,-S, 0,0, S,S, -1,-1, S,S, 0,0, -S,-S},
        {1,1, -1,-1, 1,1, -1,-1, 1,1, -1,-1, 1,1, -1,-1},
        {1,1, -S,-S, 0,0, S,S, -1,-1, S,S, 0,0, -S,-S},
        {1,1, 0,0, -1,-1, 0,0, 1,1, 0,0, -1,-1, 0,0},
        {1,1, S,S, 0,0, -S,-S, -1,-1, -S,-S, 0,0, S,S},
    };

    alignas(64) inline constexpr float DFT8_TW_II[8][16] = {
        {0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
        {0,0, -S,-S, -1,-1, -S,-S, 0,0, S,S, 1,1, S,S},
        {0,0, -1,-1, 0,0, 1,1, 0,0, -1,-1, 0,0, 1,1},
        {0,0, -S,-S, 1,1, -S,-S, 0,0, S,S, -1,-1, S,S},
        {0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0},
        {0,0, S,S, -1,-1, S,S, 0,0, -S,-S, 1,1, -S,-S},
        {0,0, 1,1, 0,0, -1,-1, 0,0, 1,1, 0,0, -1,-1},
        {0,0, S,S, 1,1, S,S, 0,0, -S,-S, -1,-1, -S,-S},
    };
}

// =============================================================================
// Matrix DFT FFT-8 for Interleaved Format
// =============================================================================
//
// Pure matrix-vector multiply: X[k] = sum_n { x[n] * W^(n*k) }
// Works directly on interleaved [re0,im0,re1,im1,...] format - no format conversion!
// Uses 4-way accumulator chains for instruction-level parallelism.
//
// Validated against FFTW with 1000+ random test vectors.

// Complex multiply-accumulate for interleaved format: acc += z * w
[[gnu::always_inline, gnu::target("avx512f,avx512dq")]]
inline __m512 cmul_acc(__m512 acc, __m512 z, __m512 w_rr, __m512 w_ii) noexcept {
    __m512 z_swap = _mm512_permute_ps(z, 0xB1);
    __m512 t = _mm512_fmadd_ps(z, w_rr, acc);
    __m512 temp = _mm512_fmaddsub_ps(z_swap, w_ii, t);
    const __m512 neg_even = _mm512_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f,
                                           -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);
    return _mm512_xor_ps(temp, neg_even);
}

// Complex multiply (no accumulate) for interleaved format: returns z * w
[[gnu::always_inline, gnu::target("avx512f,avx512dq")]]
inline __m512 cmul(__m512 z, __m512 w_rr, __m512 w_ii) noexcept {
    __m512 z_swap = _mm512_permute_ps(z, 0xB1);
    __m512 prod_rr = _mm512_mul_ps(z, w_rr);
    __m512 temp = _mm512_fmaddsub_ps(z_swap, w_ii, prod_rr);
    const __m512 neg_even = _mm512_setr_ps(-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f,
                                           -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f);
    return _mm512_xor_ps(temp, neg_even);
}

// Register-to-register FFT-8 matrix DFT for fused kernel (no memory traffic)
// Input:  ZMM register with 8 interleaved complex samples
// Output: ZMM register with 8 interleaved FFT bins
[[gnu::always_inline, gnu::target("avx512f,avx512dq")]]
inline __m512 fft8_interleaved_reg(__m512 input) noexcept {
    using namespace detail;

    // Extract and broadcast each complex sample using permute
    __m512 z0 = _mm512_permutexvar_ps(_mm512_set1_epi32(0), input);
    __m512 z1 = _mm512_permutexvar_ps(_mm512_set1_epi32(2), input);
    __m512 z2 = _mm512_permutexvar_ps(_mm512_set1_epi32(4), input);
    __m512 z3 = _mm512_permutexvar_ps(_mm512_set1_epi32(6), input);
    __m512 z4 = _mm512_permutexvar_ps(_mm512_set1_epi32(8), input);
    __m512 z5 = _mm512_permutexvar_ps(_mm512_set1_epi32(10), input);
    __m512 z6 = _mm512_permutexvar_ps(_mm512_set1_epi32(12), input);
    __m512 z7 = _mm512_permutexvar_ps(_mm512_set1_epi32(14), input);
    z0 = _mm512_mask_permutexvar_ps(z0, 0xAAAA, _mm512_set1_epi32(1), input);
    z1 = _mm512_mask_permutexvar_ps(z1, 0xAAAA, _mm512_set1_epi32(3), input);
    z2 = _mm512_mask_permutexvar_ps(z2, 0xAAAA, _mm512_set1_epi32(5), input);
    z3 = _mm512_mask_permutexvar_ps(z3, 0xAAAA, _mm512_set1_epi32(7), input);
    z4 = _mm512_mask_permutexvar_ps(z4, 0xAAAA, _mm512_set1_epi32(9), input);
    z5 = _mm512_mask_permutexvar_ps(z5, 0xAAAA, _mm512_set1_epi32(11), input);
    z6 = _mm512_mask_permutexvar_ps(z6, 0xAAAA, _mm512_set1_epi32(13), input);
    z7 = _mm512_mask_permutexvar_ps(z7, 0xAAAA, _mm512_set1_epi32(15), input);

    // 4 independent accumulator chains for ILP
    __m512 acc0 = _mm512_fmadd_ps(z4, _mm512_load_ps(DFT8_TW_RR[4]), z0);
    __m512 acc1 = cmul(z1, _mm512_load_ps(DFT8_TW_RR[1]), _mm512_load_ps(DFT8_TW_II[1]));
    __m512 acc2 = cmul(z2, _mm512_load_ps(DFT8_TW_RR[2]), _mm512_load_ps(DFT8_TW_II[2]));
    __m512 acc3 = cmul(z3, _mm512_load_ps(DFT8_TW_RR[3]), _mm512_load_ps(DFT8_TW_II[3]));
    acc1 = cmul_acc(acc1, z5, _mm512_load_ps(DFT8_TW_RR[5]), _mm512_load_ps(DFT8_TW_II[5]));
    acc2 = cmul_acc(acc2, z6, _mm512_load_ps(DFT8_TW_RR[6]), _mm512_load_ps(DFT8_TW_II[6]));
    acc3 = cmul_acc(acc3, z7, _mm512_load_ps(DFT8_TW_RR[7]), _mm512_load_ps(DFT8_TW_II[7]));

    // Final reduction
    __m512 sum01 = _mm512_add_ps(acc0, acc1);
    __m512 sum23 = _mm512_add_ps(acc2, acc3);
    return _mm512_add_ps(sum01, sum23);
}

// 8x8 complex float transpose entirely in registers
// Input:  r0..r7 each hold 8 complex floats (time-major: row = time step)
// Output: r0..r7 transposed (channel-major: row = channel)
[[gnu::always_inline, gnu::target("avx512f")]]
inline void transpose_8x8_cf32_inplace(
    __m512& r0, __m512& r1, __m512& r2, __m512& r3,
    __m512& r4, __m512& r5, __m512& r6, __m512& r7
) noexcept
{
    // Treat complex floats as doubles (both 8 bytes) for 64-bit shuffles
    __m512d d0 = _mm512_castps_pd(r0);
    __m512d d1 = _mm512_castps_pd(r1);
    __m512d d2 = _mm512_castps_pd(r2);
    __m512d d3 = _mm512_castps_pd(r3);
    __m512d d4 = _mm512_castps_pd(r4);
    __m512d d5 = _mm512_castps_pd(r5);
    __m512d d6 = _mm512_castps_pd(r6);
    __m512d d7 = _mm512_castps_pd(r7);

    // Stage 1: 2x2 transposes using unpack
    __m512d t0 = _mm512_unpacklo_pd(d0, d1);
    __m512d t1 = _mm512_unpackhi_pd(d0, d1);
    __m512d t2 = _mm512_unpacklo_pd(d2, d3);
    __m512d t3 = _mm512_unpackhi_pd(d2, d3);
    __m512d t4 = _mm512_unpacklo_pd(d4, d5);
    __m512d t5 = _mm512_unpackhi_pd(d4, d5);
    __m512d t6 = _mm512_unpacklo_pd(d6, d7);
    __m512d t7 = _mm512_unpackhi_pd(d6, d7);

    // Stage 2: 4x4 transposes using 128-bit lane shuffles
    d0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    d1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    d2 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    d3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    d4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
    d5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    d6 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    d7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);

    // Stage 3: 8x8 transpose using 256-bit lane shuffles
    t0 = _mm512_shuffle_f64x2(d0, d4, 0x88);
    t1 = _mm512_shuffle_f64x2(d1, d5, 0x88);
    t2 = _mm512_shuffle_f64x2(d2, d6, 0x88);
    t3 = _mm512_shuffle_f64x2(d3, d7, 0x88);
    t4 = _mm512_shuffle_f64x2(d0, d4, 0xDD);
    t5 = _mm512_shuffle_f64x2(d1, d5, 0xDD);
    t6 = _mm512_shuffle_f64x2(d2, d6, 0xDD);
    t7 = _mm512_shuffle_f64x2(d3, d7, 0xDD);

    // Cast back to float
    r0 = _mm512_castpd_ps(t0);
    r1 = _mm512_castpd_ps(t1);
    r2 = _mm512_castpd_ps(t2);
    r3 = _mm512_castpd_ps(t3);
    r4 = _mm512_castpd_ps(t4);
    r5 = _mm512_castpd_ps(t5);
    r6 = _mm512_castpd_ps(t6);
    r7 = _mm512_castpd_ps(t7);
}

} // namespace fft8

// =============================================================================
// Vertical FFT-8: 8 Parallel FFTs for Maximum Throughput
// =============================================================================
// "Vertical" FFT processes 8 independent FFT-8s in parallel.
// Each SIMD lane handles a different FFT instance.
//
// Data layout:
//   r[i] = [fft0_x[i], fft1_x[i], ..., fft7_x[i]]  (8 complex values)
//
// Benefits:
//   - Zero shuffle operations (all data movement is between registers)
//   - Maximum arithmetic throughput
//   - Output naturally arranged for channel scatter
//
// Algorithm: Radix-2 DIT with bit-reversed input ordering

namespace fft8_vertical {

// √2/2 constant for W8 twiddles
inline constexpr float SQRT2_2 = 0.70710678118654752f;

// Multiply by -j: (a+bi)(-j) = b - ai
// For interleaved [re,im,re,im,...]: swap pairs and negate new im
[[gnu::always_inline, gnu::target("avx512f")]]
inline __m512 mul_neg_j(__m512 x) noexcept
{
    // Swap re<->im within each complex
    __m512 swapped = _mm512_shuffle_ps(x, x, 0xB1);
    // Negate the new imaginary parts (positions 1,3,5,7,9,11,13,15)
    __m512 neg_mask = _mm512_setr_ps(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1);
    return _mm512_mul_ps(swapped, neg_mask);
}

// Multiply by W8^1 = (√2/2)(1-j)
// (a+bi)(1-j) = (a+b) + (b-a)i
// Result: √2/2 * [(a+b), (b-a)]
[[gnu::always_inline, gnu::target("avx512f")]]
inline __m512 mul_w8_1(__m512 x) noexcept
{
    // x = [a0, b0, a1, b1, ...]
    __m512 swapped = _mm512_shuffle_ps(x, x, 0xB1);  // [b0, a0, b1, a1, ...]
    // sum = a+b in even positions, b+a in odd (same value)
    __m512 sum = _mm512_add_ps(x, swapped);
    // diff = a-b in even positions, b-a in odd positions
    __m512 diff = _mm512_sub_ps(x, swapped);
    // We want [a+b, b-a] = [sum[even], diff[odd]]
    // Blend: take even positions from sum, odd positions from diff
    __m512 result = _mm512_mask_blend_ps(0xAAAA, sum, diff);
    return _mm512_mul_ps(result, _mm512_set1_ps(SQRT2_2));
}

// Multiply by W8^3 = (√2/2)(-1-j)
// (a+bi)(-1-j) = (-a+b) + (-b-a)i = (b-a) - (a+b)i
// Result: √2/2 * [(b-a), -(a+b)]
[[gnu::always_inline, gnu::target("avx512f,avx512dq")]]
inline __m512 mul_w8_3(__m512 x) noexcept
{
    // x = [a0, b0, a1, b1, ...]
    __m512 swapped = _mm512_shuffle_ps(x, x, 0xB1);  // [b0, a0, b1, a1, ...]
    // diff_ba = b-a in even positions, a-b in odd positions
    __m512 diff_ba = _mm512_sub_ps(swapped, x);
    // sum = a+b
    __m512 sum = _mm512_add_ps(x, swapped);
    // We want [b-a, -(a+b)] = [diff_ba[even], -sum[odd]]
    __m512 neg_sum = _mm512_xor_ps(sum, _mm512_set1_ps(-0.0f));
    __m512 result = _mm512_mask_blend_ps(0xAAAA, diff_ba, neg_sum);
    return _mm512_mul_ps(result, _mm512_set1_ps(SQRT2_2));
}

// Butterfly: compute (a + tw*b) and (a - tw*b) where tw=1
// Simple add/sub butterfly
[[gnu::always_inline, gnu::target("avx512f")]]
inline void butterfly(__m512& a, __m512& b) noexcept
{
    __m512 t = b;
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

// Butterfly with -j twiddle: compute (a + (-j)*b) and (a - (-j)*b)
[[gnu::always_inline, gnu::target("avx512f")]]
inline void butterfly_neg_j(__m512& a, __m512& b) noexcept
{
    __m512 t = mul_neg_j(b);
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

// Butterfly with W8^1 twiddle
[[gnu::always_inline, gnu::target("avx512f")]]
inline void butterfly_w8_1(__m512& a, __m512& b) noexcept
{
    __m512 t = mul_w8_1(b);
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

// Butterfly with W8^3 twiddle
[[gnu::always_inline, gnu::target("avx512f,avx512dq")]]
inline void butterfly_w8_3(__m512& a, __m512& b) noexcept
{
    __m512 t = mul_w8_3(b);
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

// Vertical FFT-8: process 8 parallel FFT-8s
// Input:  r[0..7] where r[i] holds x[i] from all 8 FFTs (interleaved complex)
// Output: r[0..7] where r[i] holds X[i] from all 8 FFTs
[[gnu::target("avx512f,avx512dq")]]
inline void fft8_vertical(
    __m512& r0, __m512& r1, __m512& r2, __m512& r3,
    __m512& r4, __m512& r5, __m512& r6, __m512& r7
) noexcept
{
    // Bit-reverse input order: [0,1,2,3,4,5,6,7] -> [0,4,2,6,1,5,3,7]
    // This is done by register renaming (no actual instructions)
    // b0=r0, b1=r4, b2=r2, b3=r6, b4=r1, b5=r5, b6=r3, b7=r7
    __m512& b0 = r0;
    __m512& b1 = r4;
    __m512& b2 = r2;
    __m512& b3 = r6;
    __m512& b4 = r1;
    __m512& b5 = r5;
    __m512& b6 = r3;
    __m512& b7 = r7;

    // Stage 1: stride-1 butterflies, all twiddles = 1
    butterfly(b0, b1);
    butterfly(b2, b3);
    butterfly(b4, b5);
    butterfly(b6, b7);

    // Stage 2: stride-2 butterflies, twiddles = [1, -j, 1, -j]
    butterfly(b0, b2);       // W4^0 = 1
    butterfly_neg_j(b1, b3); // W4^1 = -j
    butterfly(b4, b6);       // W4^0 = 1
    butterfly_neg_j(b5, b7); // W4^1 = -j

    // Stage 3: stride-4 butterflies, twiddles = [1, W8^1, W8^2=-j, W8^3]
    butterfly(b0, b4);       // W8^0 = 1
    butterfly_w8_1(b1, b5);  // W8^1 = (√2/2)(1-j)
    butterfly_neg_j(b2, b6); // W8^2 = -j
    butterfly_w8_3(b3, b7);  // W8^3 = (√2/2)(-1-j)

    // Output is now in natural order in b0..b7
    // Since b[i] are references to r[bit_reverse(i)], we need to
    // physically move the values back to r0..r7 in order

    // Current mapping after bit-reverse aliasing:
    // b0=r0 has X[0], b1=r4 has X[1], b2=r2 has X[2], b3=r6 has X[3]
    // b4=r1 has X[4], b5=r5 has X[5], b6=r3 has X[6], b7=r7 has X[7]

    // We need: r0=X[0], r1=X[1], r2=X[2], r3=X[3], r4=X[4], r5=X[5], r6=X[6], r7=X[7]
    // Current: r0=X[0]✓, r4=X[1], r2=X[2]✓, r6=X[3], r1=X[4], r5=X[5]✓, r3=X[6], r7=X[7]✓

    // Swaps needed: (r1,r4), (r3,r6)
    __m512 tmp = r1;
    r1 = r4;
    r4 = tmp;

    tmp = r3;
    r3 = r6;
    r6 = tmp;
}

// Convenience wrapper for memory-based input/output
// Input:  data[8][8] where data[i] holds x[i] from 8 FFTs (column-major)
// Output: data[8][8] where data[i] holds X[i] from 8 FFTs
[[gnu::target("avx512f,avx512dq")]]
inline void fft8_vertical_inplace(float* data) noexcept
{
    __m512 r0 = _mm512_loadu_ps(data + 0 * 16);
    __m512 r1 = _mm512_loadu_ps(data + 1 * 16);
    __m512 r2 = _mm512_loadu_ps(data + 2 * 16);
    __m512 r3 = _mm512_loadu_ps(data + 3 * 16);
    __m512 r4 = _mm512_loadu_ps(data + 4 * 16);
    __m512 r5 = _mm512_loadu_ps(data + 5 * 16);
    __m512 r6 = _mm512_loadu_ps(data + 6 * 16);
    __m512 r7 = _mm512_loadu_ps(data + 7 * 16);

    fft8_vertical(r0, r1, r2, r3, r4, r5, r6, r7);

    _mm512_storeu_ps(data + 0 * 16, r0);
    _mm512_storeu_ps(data + 1 * 16, r1);
    _mm512_storeu_ps(data + 2 * 16, r2);
    _mm512_storeu_ps(data + 3 * 16, r3);
    _mm512_storeu_ps(data + 4 * 16, r4);
    _mm512_storeu_ps(data + 5 * 16, r5);
    _mm512_storeu_ps(data + 6 * 16, r6);
    _mm512_storeu_ps(data + 7 * 16, r7);
}

// =============================================================================
// AVX2 Vertical FFT-8 (4 Parallel FFTs) - for tail handling
// =============================================================================

// Multiply by -j using AVX2
[[gnu::always_inline, gnu::target("avx2")]]
inline __m256 mul_neg_j_avx2(__m256 x) noexcept
{
    __m256 swapped = _mm256_shuffle_ps(x, x, 0xB1);
    __m256 neg_mask = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);
    return _mm256_mul_ps(swapped, neg_mask);
}

// Multiply by W8^1 using AVX2
[[gnu::always_inline, gnu::target("avx2")]]
inline __m256 mul_w8_1_avx2(__m256 x) noexcept
{
    __m256 swapped = _mm256_shuffle_ps(x, x, 0xB1);
    __m256 sum = _mm256_add_ps(x, swapped);
    __m256 diff = _mm256_sub_ps(x, swapped);
    __m256 result = _mm256_blend_ps(sum, diff, 0xAA);
    return _mm256_mul_ps(result, _mm256_set1_ps(SQRT2_2));
}

// Multiply by W8^3 using AVX2
[[gnu::always_inline, gnu::target("avx2")]]
inline __m256 mul_w8_3_avx2(__m256 x) noexcept
{
    __m256 swapped = _mm256_shuffle_ps(x, x, 0xB1);
    __m256 diff_ba = _mm256_sub_ps(swapped, x);
    __m256 sum = _mm256_add_ps(x, swapped);
    __m256 neg_sum = _mm256_xor_ps(sum, _mm256_set1_ps(-0.0f));
    __m256 result = _mm256_blend_ps(diff_ba, neg_sum, 0xAA);
    return _mm256_mul_ps(result, _mm256_set1_ps(SQRT2_2));
}

// AVX2 butterflies
[[gnu::always_inline, gnu::target("avx2")]]
inline void butterfly_avx2(__m256& a, __m256& b) noexcept
{
    __m256 t = b;
    b = _mm256_sub_ps(a, t);
    a = _mm256_add_ps(a, t);
}

[[gnu::always_inline, gnu::target("avx2")]]
inline void butterfly_neg_j_avx2(__m256& a, __m256& b) noexcept
{
    __m256 t = mul_neg_j_avx2(b);
    b = _mm256_sub_ps(a, t);
    a = _mm256_add_ps(a, t);
}

[[gnu::always_inline, gnu::target("avx2")]]
inline void butterfly_w8_1_avx2(__m256& a, __m256& b) noexcept
{
    __m256 t = mul_w8_1_avx2(b);
    b = _mm256_sub_ps(a, t);
    a = _mm256_add_ps(a, t);
}

[[gnu::always_inline, gnu::target("avx2")]]
inline void butterfly_w8_3_avx2(__m256& a, __m256& b) noexcept
{
    __m256 t = mul_w8_3_avx2(b);
    b = _mm256_sub_ps(a, t);
    a = _mm256_add_ps(a, t);
}

// Vertical FFT-8: process 4 parallel FFT-8s using AVX2
[[gnu::target("avx2")]]
inline void fft8_vertical_avx2(
    __m256& r0, __m256& r1, __m256& r2, __m256& r3,
    __m256& r4, __m256& r5, __m256& r6, __m256& r7
) noexcept
{
    // Bit-reverse via register aliasing
    __m256& b0 = r0;
    __m256& b1 = r4;
    __m256& b2 = r2;
    __m256& b3 = r6;
    __m256& b4 = r1;
    __m256& b5 = r5;
    __m256& b6 = r3;
    __m256& b7 = r7;

    // Stage 1
    butterfly_avx2(b0, b1);
    butterfly_avx2(b2, b3);
    butterfly_avx2(b4, b5);
    butterfly_avx2(b6, b7);

    // Stage 2
    butterfly_avx2(b0, b2);
    butterfly_neg_j_avx2(b1, b3);
    butterfly_avx2(b4, b6);
    butterfly_neg_j_avx2(b5, b7);

    // Stage 3
    butterfly_avx2(b0, b4);
    butterfly_w8_1_avx2(b1, b5);
    butterfly_neg_j_avx2(b2, b6);
    butterfly_w8_3_avx2(b3, b7);

    // Fix output order
    __m256 tmp = r1; r1 = r4; r4 = tmp;
    tmp = r3; r3 = r6; r6 = tmp;
}

// 4x4 complex transpose using AVX2
// Input:  4 YMM registers, each holding 4 complex floats (row-major)
//         r0 = [t0c0, t0c1, t0c2, t0c3], r1 = [t1c0, ...], etc.
// Output: 4 YMM registers transposed (column-major)
//         r0 = [t0c0, t1c0, t2c0, t3c0], r1 = [t0c1, ...], etc.
[[gnu::target("avx2")]]
inline void transpose_4x4_cf32_avx2(
    __m256& r0, __m256& r1, __m256& r2, __m256& r3
) noexcept
{
    // Treat as 64-bit (complex) elements using double intrinsics
    __m256d d0 = _mm256_castps_pd(r0);  // [t0c0, t0c1 | t0c2, t0c3]
    __m256d d1 = _mm256_castps_pd(r1);  // [t1c0, t1c1 | t1c2, t1c3]
    __m256d d2 = _mm256_castps_pd(r2);  // [t2c0, t2c1 | t2c2, t2c3]
    __m256d d3 = _mm256_castps_pd(r3);  // [t3c0, t3c1 | t3c2, t3c3]

    // Step 1: Interleave pairs within 128-bit lanes
    // unpacklo_pd: takes element 0 from each lane
    // unpackhi_pd: takes element 1 from each lane
    __m256d t0 = _mm256_unpacklo_pd(d0, d1);  // [t0c0, t1c0 | t0c2, t1c2]
    __m256d t1 = _mm256_unpackhi_pd(d0, d1);  // [t0c1, t1c1 | t0c3, t1c3]
    __m256d t2 = _mm256_unpacklo_pd(d2, d3);  // [t2c0, t3c0 | t2c2, t3c2]
    __m256d t3 = _mm256_unpackhi_pd(d2, d3);  // [t2c1, t3c1 | t2c3, t3c3]

    // Step 2: Swap 128-bit lanes to complete transpose
    r0 = _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t2, 0x20));  // [t0c0, t1c0, t2c0, t3c0]
    r2 = _mm256_castpd_ps(_mm256_permute2f128_pd(t0, t2, 0x31));  // [t0c2, t1c2, t2c2, t3c2]
    r1 = _mm256_castpd_ps(_mm256_permute2f128_pd(t1, t3, 0x20));  // [t0c1, t1c1, t2c1, t3c1]
    r3 = _mm256_castpd_ps(_mm256_permute2f128_pd(t1, t3, 0x31));  // [t0c3, t1c3, t2c3, t3c3]
}

} // namespace fft8_vertical

// =============================================================================
// Vertical FFT-16 (8 Parallel FFT-16s)
// =============================================================================
// Each ZMM register holds one frequency bin across 8 parallel FFTs.
// Uses 16 registers for data, zero cross-lane shuffles during butterflies.
// Radix-2 DIT with 4 stages.

namespace fft16_vertical {

// W16 twiddle factor constants
inline constexpr float COS_PI_8  = 0.92387953251128676f;  // cos(π/8)
inline constexpr float SIN_PI_8  = 0.38268343236508977f;  // sin(π/8)
inline constexpr float SQRT2_2   = 0.70710678118654752f;  // cos(π/4) = sin(π/4)

// Multiply by -j (same as FFT-8)
[[gnu::always_inline, gnu::target("avx512f")]]
inline __m512 mul_neg_j(__m512 x) noexcept
{
    __m512 swapped = _mm512_shuffle_ps(x, x, 0xB1);
    __m512 neg_mask = _mm512_setr_ps(1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1);
    return _mm512_mul_ps(swapped, neg_mask);
}

// Multiply by W8^1 = √2/2(1-j) (same as FFT-8)
[[gnu::always_inline, gnu::target("avx512f")]]
inline __m512 mul_w8_1(__m512 x) noexcept
{
    __m512 swapped = _mm512_shuffle_ps(x, x, 0xB1);
    __m512 sum = _mm512_add_ps(x, swapped);
    __m512 diff = _mm512_sub_ps(x, swapped);
    __m512 result = _mm512_mask_blend_ps(0xAAAA, sum, diff);
    return _mm512_mul_ps(result, _mm512_set1_ps(SQRT2_2));
}

// Multiply by W8^3 = √2/2(-1-j) (same as FFT-8)
[[gnu::always_inline, gnu::target("avx512f,avx512dq")]]
inline __m512 mul_w8_3(__m512 x) noexcept
{
    __m512 swapped = _mm512_shuffle_ps(x, x, 0xB1);
    __m512 diff_ba = _mm512_sub_ps(swapped, x);
    __m512 sum = _mm512_add_ps(x, swapped);
    __m512 neg_sum = _mm512_xor_ps(sum, _mm512_set1_ps(-0.0f));
    __m512 result = _mm512_mask_blend_ps(0xAAAA, diff_ba, neg_sum);
    return _mm512_mul_ps(result, _mm512_set1_ps(SQRT2_2));
}

// Multiply by W16^1 = cos(π/8) - j*sin(π/8)
// (a+bi)(c-si) = (ac+bs) + (bc-as)i where c=cos(π/8), s=sin(π/8)
[[gnu::always_inline, gnu::target("avx512f,fma")]]
inline __m512 mul_w16_1(__m512 x) noexcept
{
    __m512 swapped = _mm512_shuffle_ps(x, x, 0xB1);  // [b, a, ...]
    __m512 cos_val = _mm512_set1_ps(COS_PI_8);
    __m512 sin_val = _mm512_set1_ps(SIN_PI_8);

    // For real: a*cos + b*sin
    // For imag: b*cos - a*sin
    __m512 term1 = _mm512_mul_ps(x, cos_val);        // [a*c, b*c, ...]
    __m512 term2 = _mm512_mul_ps(swapped, sin_val);  // [b*s, a*s, ...]

    // Blend: even positions get add (a*c + b*s), odd positions get sub (b*c - a*s)
    __m512 result = _mm512_mask_blend_ps(0xAAAA,
        _mm512_add_ps(term1, term2),   // a*c + b*s for real
        _mm512_sub_ps(term1, term2));  // b*c - a*s for imag
    return result;
}

// Multiply by W16^3 = sin(π/8) - j*cos(π/8) = 0.3827 - 0.9239j
// (a+bi)(s-ci) = (as+bc) + (bs-ac)i
[[gnu::always_inline, gnu::target("avx512f,fma")]]
inline __m512 mul_w16_3(__m512 x) noexcept
{
    __m512 swapped = _mm512_shuffle_ps(x, x, 0xB1);
    __m512 cos_val = _mm512_set1_ps(COS_PI_8);
    __m512 sin_val = _mm512_set1_ps(SIN_PI_8);

    __m512 term1 = _mm512_mul_ps(x, sin_val);        // [a*s, b*s, ...]
    __m512 term2 = _mm512_mul_ps(swapped, cos_val);  // [b*c, a*c, ...]

    __m512 result = _mm512_mask_blend_ps(0xAAAA,
        _mm512_add_ps(term1, term2),   // a*s + b*c for real
        _mm512_sub_ps(term1, term2));  // b*s - a*c for imag
    return result;
}

// Multiply by W16^5 = -sin(π/8) - j*cos(π/8) = -0.3827 - 0.9239j
// (a+bi)(-s-jc) = (bc - as) + (-(ac + bs))i
[[gnu::always_inline, gnu::target("avx512f,fma,avx512dq")]]
inline __m512 mul_w16_5(__m512 x) noexcept
{
    __m512 swapped = _mm512_shuffle_ps(x, x, 0xB1);  // [b, a, ...]
    __m512 cos_val = _mm512_set1_ps(COS_PI_8);
    __m512 sin_val = _mm512_set1_ps(SIN_PI_8);

    // diff = swapped*cos - x*sin = [bc-as, ac-bs, ...] - need even positions
    __m512 diff = _mm512_fmsub_ps(swapped, cos_val, _mm512_mul_ps(x, sin_val));

    // sum = x*cos + swapped*sin = [ac+bs, bc+as, ...] - need even positions negated
    __m512 sum = _mm512_fmadd_ps(x, cos_val, _mm512_mul_ps(swapped, sin_val));
    __m512 neg_sum = _mm512_xor_ps(sum, _mm512_set1_ps(-0.0f));

    // Broadcast even elements (correct values) to both positions within each pair
    __m512 diff_bcast = _mm512_shuffle_ps(diff, diff, 0xA0);          // [bc-as, bc-as, ...]
    __m512 neg_sum_bcast = _mm512_shuffle_ps(neg_sum, neg_sum, 0xA0); // [-(ac+bs), -(ac+bs), ...]

    // Blend: even from diff_bcast (real), odd from neg_sum_bcast (imag)
    return _mm512_mask_blend_ps(0xAAAA, diff_bcast, neg_sum_bcast);
}

// Multiply by W16^7 = -cos(π/8) - j*sin(π/8) = -0.9239 - 0.3827j
// (a+bi)(-c-js) = (bs - ac) + (-(as + bc))i
[[gnu::always_inline, gnu::target("avx512f,fma,avx512dq")]]
inline __m512 mul_w16_7(__m512 x) noexcept
{
    __m512 swapped = _mm512_shuffle_ps(x, x, 0xB1);  // [b, a, ...]
    __m512 cos_val = _mm512_set1_ps(COS_PI_8);
    __m512 sin_val = _mm512_set1_ps(SIN_PI_8);

    // diff = swapped*sin - x*cos = [bs-ac, as-bc, ...] - need even positions
    __m512 diff = _mm512_fmsub_ps(swapped, sin_val, _mm512_mul_ps(x, cos_val));

    // sum = x*sin + swapped*cos = [as+bc, bs+ac, ...] - need even positions negated
    __m512 sum = _mm512_fmadd_ps(x, sin_val, _mm512_mul_ps(swapped, cos_val));
    __m512 neg_sum = _mm512_xor_ps(sum, _mm512_set1_ps(-0.0f));

    // Broadcast even elements (correct values) to both positions within each pair
    __m512 diff_bcast = _mm512_shuffle_ps(diff, diff, 0xA0);          // [bs-ac, bs-ac, ...]
    __m512 neg_sum_bcast = _mm512_shuffle_ps(neg_sum, neg_sum, 0xA0); // [-(as+bc), -(as+bc), ...]

    // Blend: even from diff_bcast (real), odd from neg_sum_bcast (imag)
    return _mm512_mask_blend_ps(0xAAAA, diff_bcast, neg_sum_bcast);
}

// Simple butterfly
[[gnu::always_inline, gnu::target("avx512f")]]
inline void butterfly(__m512& a, __m512& b) noexcept
{
    __m512 t = b;
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

// Butterfly with twiddle functions
[[gnu::always_inline, gnu::target("avx512f")]]
inline void butterfly_neg_j(__m512& a, __m512& b) noexcept
{
    __m512 t = mul_neg_j(b);
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

[[gnu::always_inline, gnu::target("avx512f")]]
inline void butterfly_w8_1(__m512& a, __m512& b) noexcept
{
    __m512 t = mul_w8_1(b);
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

[[gnu::always_inline, gnu::target("avx512f,avx512dq")]]
inline void butterfly_w8_3(__m512& a, __m512& b) noexcept
{
    __m512 t = mul_w8_3(b);
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

[[gnu::always_inline, gnu::target("avx512f,fma")]]
inline void butterfly_w16_1(__m512& a, __m512& b) noexcept
{
    __m512 t = mul_w16_1(b);
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

[[gnu::always_inline, gnu::target("avx512f,fma")]]
inline void butterfly_w16_3(__m512& a, __m512& b) noexcept
{
    __m512 t = mul_w16_3(b);
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

[[gnu::always_inline, gnu::target("avx512f,fma,avx512dq")]]
inline void butterfly_w16_5(__m512& a, __m512& b) noexcept
{
    __m512 t = mul_w16_5(b);
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

[[gnu::always_inline, gnu::target("avx512f,fma,avx512dq")]]
inline void butterfly_w16_7(__m512& a, __m512& b) noexcept
{
    __m512 t = mul_w16_7(b);
    b = _mm512_sub_ps(a, t);
    a = _mm512_add_ps(a, t);
}

// Vertical FFT-16: process 8 parallel FFT-16s
// Input:  r[0..15] where r[i] holds x[i] from all 8 FFTs
// Output: r[0..15] where r[i] holds X[i] from all 8 FFTs
[[gnu::target("avx512f,avx512dq,fma")]]
inline void fft16_vertical(
    __m512& r0,  __m512& r1,  __m512& r2,  __m512& r3,
    __m512& r4,  __m512& r5,  __m512& r6,  __m512& r7,
    __m512& r8,  __m512& r9,  __m512& r10, __m512& r11,
    __m512& r12, __m512& r13, __m512& r14, __m512& r15
) noexcept
{
    // Bit-reverse for N=16: [0,1,2,...,15] -> [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
    __m512& b0  = r0;   // 0  -> 0
    __m512& b1  = r8;   // 1  -> 8
    __m512& b2  = r4;   // 2  -> 4
    __m512& b3  = r12;  // 3  -> 12
    __m512& b4  = r2;   // 4  -> 2
    __m512& b5  = r10;  // 5  -> 10
    __m512& b6  = r6;   // 6  -> 6
    __m512& b7  = r14;  // 7  -> 14
    __m512& b8  = r1;   // 8  -> 1
    __m512& b9  = r9;   // 9  -> 9
    __m512& b10 = r5;   // 10 -> 5
    __m512& b11 = r13;  // 11 -> 13
    __m512& b12 = r3;   // 12 -> 3
    __m512& b13 = r11;  // 13 -> 11
    __m512& b14 = r7;   // 14 -> 7
    __m512& b15 = r15;  // 15 -> 15

    // Stage 1: stride-1 butterflies, all twiddles = 1
    butterfly(b0, b1);   butterfly(b2, b3);   butterfly(b4, b5);   butterfly(b6, b7);
    butterfly(b8, b9);   butterfly(b10, b11); butterfly(b12, b13); butterfly(b14, b15);

    // Stage 2: stride-2 butterflies, twiddles = [1, W4^1=-j]
    butterfly(b0, b2);       butterfly_neg_j(b1, b3);
    butterfly(b4, b6);       butterfly_neg_j(b5, b7);
    butterfly(b8, b10);      butterfly_neg_j(b9, b11);
    butterfly(b12, b14);     butterfly_neg_j(b13, b15);

    // Stage 3: stride-4 butterflies, twiddles = [1, W8^1, W8^2=-j, W8^3]
    butterfly(b0, b4);       butterfly_w8_1(b1, b5);
    butterfly_neg_j(b2, b6); butterfly_w8_3(b3, b7);
    butterfly(b8, b12);      butterfly_w8_1(b9, b13);
    butterfly_neg_j(b10, b14); butterfly_w8_3(b11, b15);

    // Stage 4: stride-8 butterflies, twiddles = [1, W16^1, W16^2, W16^3, W16^4=-j, W16^5, W16^6, W16^7]
    butterfly(b0, b8);           butterfly_w16_1(b1, b9);
    butterfly_w8_1(b2, b10);     butterfly_w16_3(b3, b11);
    butterfly_neg_j(b4, b12);    butterfly_w16_5(b5, b13);
    butterfly_w8_3(b6, b14);     butterfly_w16_7(b7, b15);

    // Fix output order: need to swap registers to get natural order
    // Current: b[i] = r[bit_rev(i)]
    // After bit-reverse aliasing, output is in b0..b15 in natural order
    // but physically stored in r[bit_rev(0..15)]
    //
    // We need: r[i] = X[i]
    // Current mapping:
    //   b0=r0=X[0]✓, b1=r8=X[1], b2=r4=X[2], b3=r12=X[3]
    //   b4=r2=X[4], b5=r10=X[5], b6=r6=X[6]✓, b7=r14=X[7]
    //   b8=r1=X[8], b9=r9=X[9]✓, b10=r5=X[10], b11=r13=X[11]
    //   b12=r3=X[12], b13=r11=X[13], b14=r7=X[14], b15=r15=X[15]✓
    //
    // Swaps needed to fix: (r1,r8), (r2,r4), (r3,r12), (r5,r10), (r7,r14), (r11,r13)

    __m512 tmp;
    tmp = r1;  r1 = r8;   r8 = tmp;    // X[1] <-> X[8]
    tmp = r2;  r2 = r4;   r4 = tmp;    // X[2] <-> X[4]
    tmp = r3;  r3 = r12;  r12 = tmp;   // X[3] <-> X[12]
    tmp = r5;  r5 = r10;  r10 = tmp;   // X[5] <-> X[10]
    tmp = r7;  r7 = r14;  r14 = tmp;   // X[7] <-> X[14]
    tmp = r11; r11 = r13; r13 = tmp;   // X[11] <-> X[13]
}

} // namespace fft16_vertical

// =============================================================================
// FFT-16 Kernels (Radix-4 Four-Step Algorithm)
// =============================================================================
// Uses split re/im format internally for efficient radix-4 computation.
// Public API accepts interleaved format (2 ZMM registers = 16 complex samples).

namespace fft16 {

// Transpose index for 4x4 blocks - reorders data for radix-4 stages
alignas(64) inline constexpr int idx_transpose_16[] = {
    0, 4, 8, 12,  1, 5, 9, 13,  2, 6, 10, 14,  3, 7, 11, 15
};

// FFT-16 twiddle factors W_16^k for k=0..15 (arranged for 4x4 structure)
alignas(64) inline constexpr float tw16_re[] = {
    1.0f,  1.0f,        1.0f,        1.0f,
    1.0f,  0.92387953f, 0.70710678f, 0.38268343f,
    1.0f,  0.70710678f, 0.0f,       -0.70710678f,
    1.0f,  0.38268343f,-0.70710678f,-0.92387953f
};

alignas(64) inline constexpr float tw16_im[] = {
    0.0f,  0.0f,        0.0f,        0.0f,
    0.0f, -0.38268343f,-0.70710678f,-0.92387953f,
    0.0f, -0.70710678f,-1.0f,       -0.70710678f,
    0.0f, -0.92387953f,-0.70710678f, 0.38268343f
};

// Fused deinterleave + first transpose indices
// Input: [r0,i0,r1,i1,...,r7,i7] || [r8,i8,...,r15,i15] (two ZMM)
// Output: re=[r0,r4,r8,r12,...], im=[i0,i4,i8,i12,...] (transposed for radix-4)
alignas(64) inline constexpr int idx_fused_deint_tr_re[] = {
    0, 8, 16, 24,  2, 10, 18, 26,  4, 12, 20, 28,  6, 14, 22, 30
};
alignas(64) inline constexpr int idx_fused_deint_tr_im[] = {
    1, 9, 17, 25,  3, 11, 19, 27,  5, 13, 21, 29,  7, 15, 23, 31
};

// Fused last transpose + interleave indices
// Input: re/im in column-major form after second FFT-4
// Output: interleaved [r0,i0,r1,i1,...] in natural order
alignas(64) inline constexpr int idx_fused_tr_int_lo[] = {
    0, 16, 4, 20, 8, 24, 12, 28, 1, 17, 5, 21, 9, 25, 13, 29
};
alignas(64) inline constexpr int idx_fused_tr_int_hi[] = {
    2, 18, 6, 22, 10, 26, 14, 30, 3, 19, 7, 23, 11, 27, 15, 31
};

// Parallel FFT-4 on 4 independent groups (for split re/im format)
// Each lane group of 4 floats is an independent FFT-4
[[gnu::always_inline, gnu::target("avx512f")]]
inline void fft4_parallel_x4(__m512& re, __m512& im) noexcept
{
    // Get elements 0,1 and 2,3 of each group
    __m512 re_01 = _mm512_permute_ps(re, 0x44);  // [0,1,0,1, 0,1,0,1, ...]
    __m512 im_01 = _mm512_permute_ps(im, 0x44);
    __m512 re_23 = _mm512_permute_ps(re, 0xEE);  // [2,3,2,3, 2,3,2,3, ...]
    __m512 im_23 = _mm512_permute_ps(im, 0xEE);

    // Butterfly: sum = 01 + 23, diff = 01 - 23
    __m512 s_re = _mm512_add_ps(re_01, re_23);
    __m512 s_im = _mm512_add_ps(im_01, im_23);
    __m512 d_re = _mm512_sub_ps(re_01, re_23);
    __m512 d_im = _mm512_sub_ps(im_01, im_23);

    // Extract A=s[0], B=s[1], C=d[0], D=d[1]
    __m512 A_re = _mm512_permute_ps(s_re, 0x00);
    __m512 A_im = _mm512_permute_ps(s_im, 0x00);
    __m512 B_re = _mm512_permute_ps(s_re, 0x55);
    __m512 B_im = _mm512_permute_ps(s_im, 0x55);

    // out[0] = A + B, out[2] = A - B
    __m512 out0_re = _mm512_add_ps(A_re, B_re);
    __m512 out0_im = _mm512_add_ps(A_im, B_im);
    __m512 out2_re = _mm512_sub_ps(A_re, B_re);
    __m512 out2_im = _mm512_sub_ps(A_im, B_im);

    __m512 C_re = _mm512_permute_ps(d_re, 0x00);
    __m512 C_im = _mm512_permute_ps(d_im, 0x00);
    __m512 D_re = _mm512_permute_ps(d_re, 0x55);
    __m512 D_im = _mm512_permute_ps(d_im, 0x55);

    // D * (-j) = (D_im, -D_re)
    __m512 Dnj_re = D_im;
    __m512 Dnj_im = _mm512_sub_ps(_mm512_setzero_ps(), D_re);

    // out[1] = C + D*(-j), out[3] = C - D*(-j)
    __m512 out1_re = _mm512_add_ps(C_re, Dnj_re);
    __m512 out1_im = _mm512_add_ps(C_im, Dnj_im);
    __m512 out3_re = _mm512_sub_ps(C_re, Dnj_re);
    __m512 out3_im = _mm512_sub_ps(C_im, Dnj_im);

    // Reassemble: blend outputs into correct positions
    __m512 tmpA_re = _mm512_mask_blend_ps(0xAAAA, out0_re, out1_re);
    __m512 tmpA_im = _mm512_mask_blend_ps(0xAAAA, out0_im, out1_im);
    __m512 tmpB_re = _mm512_mask_blend_ps(0xAAAA, out2_re, out3_re);
    __m512 tmpB_im = _mm512_mask_blend_ps(0xAAAA, out2_im, out3_im);

    re = _mm512_mask_blend_ps(0xCCCC, tmpA_re, tmpB_re);
    im = _mm512_mask_blend_ps(0xCCCC, tmpA_im, tmpB_im);
}

// FFT-16 core using Radix-4 Four-Step algorithm (split re/im format)
// Expects input already in transposed form; produces output in column-major form
[[gnu::always_inline, gnu::target("avx512f,avx512vl,fma")]]
inline void fft16_split_opt(__m512& re, __m512& im) noexcept
{
    // Step 1: Four parallel FFT-4s (input already transposed)
    fft4_parallel_x4(re, im);

    // Step 2: Apply twiddle factors using FMA
    __m512 tw_re = _mm512_load_ps(tw16_re);
    __m512 tw_im = _mm512_load_ps(tw16_im);
    __m512 re_tw_re = _mm512_mul_ps(re, tw_re);
    __m512 re_tw_im = _mm512_mul_ps(re, tw_im);
    __m512 tmp_re = _mm512_fnmadd_ps(im, tw_im, re_tw_re);  // re*tw_re - im*tw_im
    __m512 tmp_im = _mm512_fmadd_ps(im, tw_re, re_tw_im);   // re*tw_im + im*tw_re
    re = tmp_re;
    im = tmp_im;

    // Step 3: Transpose
    __m512i idx_tr = _mm512_load_si512(idx_transpose_16);
    re = _mm512_permutexvar_ps(idx_tr, re);
    im = _mm512_permutexvar_ps(idx_tr, im);

    // Step 4: Four parallel FFT-4s
    fft4_parallel_x4(re, im);

    // Output is in column-major form; caller applies fused transpose+interleave
}

// FFT-16 on interleaved data (2 ZMM registers in, 2 ZMM registers out)
// Input:  lo=[r0,i0,r1,i1,...,r7,i7], hi=[r8,i8,...,r15,i15]
// Output: lo=[r0,i0,r1,i1,...,r7,i7], hi=[r8,i8,...,r15,i15] (FFT result)
[[gnu::always_inline, gnu::target("avx512f,avx512vl,fma")]]
inline void fft16_interleaved_reg(__m512& lo, __m512& hi) noexcept
{
    // Fused deinterleave + first transpose (single permutation each)
    __m512i idx_re = _mm512_load_si512(idx_fused_deint_tr_re);
    __m512i idx_im = _mm512_load_si512(idx_fused_deint_tr_im);
    __m512 re = _mm512_permutex2var_ps(lo, idx_re, hi);
    __m512 im = _mm512_permutex2var_ps(lo, idx_im, hi);

    // Compute FFT-16 core (expects transposed input, produces column-major output)
    fft16_split_opt(re, im);

    // Fused last transpose + interleave
    __m512i idx_lo = _mm512_load_si512(idx_fused_tr_int_lo);
    __m512i idx_hi = _mm512_load_si512(idx_fused_tr_int_hi);
    lo = _mm512_permutex2var_ps(re, idx_lo, im);
    hi = _mm512_permutex2var_ps(re, idx_hi, im);
}

// FFT-16 in-place on memory (for benchmarking/testing)
[[gnu::target("avx512f,avx512vl,fma")]]
inline void fft16_interleaved_inplace(float* data) noexcept
{
    __m512 lo = _mm512_load_ps(data);
    __m512 hi = _mm512_load_ps(data + 16);
    fft16_interleaved_reg(lo, hi);
    _mm512_store_ps(data, lo);
    _mm512_store_ps(data + 16, hi);
}

} // namespace fft16

// =============================================================================
// FFT-32 via Cooley-Tukey DIT (Decimation-in-Time)
// =============================================================================
// Uses FFT-16 as building block:
//   1. Gather evens (0,2,4,...,30) and odds (1,3,5,...,31)
//   2. FFT-16 on evens, FFT-16 on odds
//   3. Apply W32 twiddles to odds
//   4. Butterfly: X[k] = E[k] + W*O[k], X[k+16] = E[k] - W*O[k]
//
// Input: 4 ZMM registers (32 interleaved complex = 64 floats)
// Register pressure: ~20 ZMM (feasible with 32 available)

namespace fft32 {

// Twiddle factors for FFT-32: W32^k = exp(-2πik/32) for k=0..15
// These are applied to the odd-indexed FFT-16 output before butterfly
alignas(64) inline constexpr float tw32_re[16] = {
    1.0f,                    // k=0
    0.98078528040323043f,    // k=1
    0.92387953251128674f,    // k=2
    0.83146961230254524f,    // k=3
    0.70710678118654752f,    // k=4
    0.55557023301960218f,    // k=5
    0.38268343236508978f,    // k=6
    0.19509032201612828f,    // k=7
    0.0f,                    // k=8
    -0.19509032201612828f,   // k=9
    -0.38268343236508978f,   // k=10
    -0.55557023301960218f,   // k=11
    -0.70710678118654752f,   // k=12
    -0.83146961230254524f,   // k=13
    -0.92387953251128674f,   // k=14
    -0.98078528040323043f    // k=15
};

alignas(64) inline constexpr float tw32_im[16] = {
    0.0f,                    // k=0
    -0.19509032201612828f,   // k=1
    -0.38268343236508978f,   // k=2
    -0.55557023301960218f,   // k=3
    -0.70710678118654752f,   // k=4
    -0.83146961230254524f,   // k=5
    -0.92387953251128674f,   // k=6
    -0.98078528040323043f,   // k=7
    -1.0f,                   // k=8
    -0.98078528040323043f,   // k=9
    -0.92387953251128674f,   // k=10
    -0.83146961230254524f,   // k=11
    -0.70710678118654752f,   // k=12
    -0.55557023301960218f,   // k=13
    -0.38268343236508978f,   // k=14
    -0.19509032201612828f    // k=15
};

// Gather indices for extracting even samples from concatenated register pair
// Input: r0=[c0,c1,c2,c3,c4,c5,c6,c7], r1=[c8,c9,c10,c11,c12,c13,c14,c15]
// Output: [c0,c2,c4,c6,c8,c10,c12,c14] (evens)
// Complex ci at float positions 2i,2i+1; we want i=0,2,4,6 from r0, i=8,10,12,14 from r1
alignas(64) inline constexpr int idx_gather_even[16] = {
    0, 1, 4, 5, 8, 9, 12, 13,        // From r0: samples 0,2,4,6
    16, 17, 20, 21, 24, 25, 28, 29   // From r1: samples 8,10,12,14 (offset +16)
};

// Gather indices for extracting odd samples
// Output: [c1,c3,c5,c7,c9,c11,c13,c15] (odds)
alignas(64) inline constexpr int idx_gather_odd[16] = {
    2, 3, 6, 7, 10, 11, 14, 15,      // From r0: samples 1,3,5,7
    18, 19, 22, 23, 26, 27, 30, 31   // From r1: samples 9,11,13,15 (offset +16)
};

// Scatter indices for writing butterfly results back to natural order
// After butterfly, we have:
//   X[0..7] in first half of res_lo, X[8..15] in second half (from e_lo ± twiddle*o_lo)
//   X[16..23] in first half of res_hi, X[24..31] in second half (from e_hi ± twiddle*o_hi)
// But the output is split as X[k] and X[k+16], so we need to interleave them
// back into 4 contiguous ZMM registers for natural order output.

// Interleave twiddle factors for vectorized complex multiply
// Format: [re0,re0,re1,re1,...,re7,re7] for use with interleaved complex data
alignas(64) inline constexpr float tw32_re_interleaved_lo[16] = {
    1.0f, 1.0f,                                     // k=0
    0.98078528040323043f, 0.98078528040323043f,     // k=1
    0.92387953251128674f, 0.92387953251128674f,     // k=2
    0.83146961230254524f, 0.83146961230254524f,     // k=3
    0.70710678118654752f, 0.70710678118654752f,     // k=4
    0.55557023301960218f, 0.55557023301960218f,     // k=5
    0.38268343236508978f, 0.38268343236508978f,     // k=6
    0.19509032201612828f, 0.19509032201612828f      // k=7
};

alignas(64) inline constexpr float tw32_im_interleaved_lo[16] = {
    0.0f, -0.0f,                                    // k=0: sin(0)=0
    0.19509032201612828f, -0.19509032201612828f,    // k=1: [sin(θ), -sin(θ)]
    0.38268343236508978f, -0.38268343236508978f,    // k=2
    0.55557023301960218f, -0.55557023301960218f,    // k=3
    0.70710678118654752f, -0.70710678118654752f,    // k=4
    0.83146961230254524f, -0.83146961230254524f,    // k=5
    0.92387953251128674f, -0.92387953251128674f,    // k=6
    0.98078528040323043f, -0.98078528040323043f     // k=7
};

alignas(64) inline constexpr float tw32_re_interleaved_hi[16] = {
    0.0f, 0.0f,                                     // k=8
    -0.19509032201612828f, -0.19509032201612828f,   // k=9
    -0.38268343236508978f, -0.38268343236508978f,   // k=10
    -0.55557023301960218f, -0.55557023301960218f,   // k=11
    -0.70710678118654752f, -0.70710678118654752f,   // k=12
    -0.83146961230254524f, -0.83146961230254524f,   // k=13
    -0.92387953251128674f, -0.92387953251128674f,   // k=14
    -0.98078528040323043f, -0.98078528040323043f    // k=15
};

alignas(64) inline constexpr float tw32_im_interleaved_hi[16] = {
    1.0f, -1.0f,                                    // k=8: sin(π/2)=1
    0.98078528040323043f, -0.98078528040323043f,    // k=9
    0.92387953251128674f, -0.92387953251128674f,    // k=10
    0.83146961230254524f, -0.83146961230254524f,    // k=11
    0.70710678118654752f, -0.70710678118654752f,    // k=12
    0.55557023301960218f, -0.55557023301960218f,    // k=13
    0.38268343236508978f, -0.38268343236508978f,    // k=14
    0.19509032201612828f, -0.19509032201612828f     // k=15
};

// Vectorized complex multiply for interleaved data: (a+bi) * (c+di)
// Input:  val = [re0,im0,re1,im1,...], tw_re = [c,c,c,c,...], tw_im = [-d,d,-d,d,...]
// Output: [re0*c-im0*d, re0*d+im0*c, re1*c-im1*d, re1*d+im1*c, ...]
[[gnu::always_inline, gnu::target("avx512f,fma")]]
inline __m512 complex_mul_interleaved(__m512 val, __m512 tw_re, __m512 tw_im) noexcept
{
    // tw_re has [c,c,c,c,...] - same value for re and im positions
    // tw_im has [-d,d,-d,d,...] - negated for re, positive for im
    // Result: val * tw_re + val_swapped * tw_im
    //   re_out = re*c + im*(-d) = re*c - im*d
    //   im_out = im*c + re*d
    __m512 val_swap = _mm512_shuffle_ps(val, val, 0xB1);  // Swap re<->im pairs
    return _mm512_fmadd_ps(val, tw_re, _mm512_mul_ps(val_swap, tw_im));
}

// FFT-32 on 4 interleaved ZMM registers
// Input:  r0=[c0..c7], r1=[c8..c15], r2=[c16..c23], r3=[c24..c31]
// Output: r0=[X0..X7], r1=[X8..X15], r2=[X16..X23], r3=[X24..X31]
[[gnu::always_inline, gnu::target("avx512f,avx512vl,avx512dq,fma")]]
inline void fft32_interleaved_reg(__m512& r0, __m512& r1, __m512& r2, __m512& r3) noexcept
{
    // Load gather indices
    __m512i idx_even = _mm512_load_epi32(idx_gather_even);
    __m512i idx_odd = _mm512_load_epi32(idx_gather_odd);

    // Step 1: Gather evens and odds
    // e_lo = evens from samples 0-15: [c0,c2,c4,c6,c8,c10,c12,c14]
    // o_lo = odds from samples 0-15:  [c1,c3,c5,c7,c9,c11,c13,c15]
    __m512 e_lo = _mm512_permutex2var_ps(r0, idx_even, r1);
    __m512 o_lo = _mm512_permutex2var_ps(r0, idx_odd, r1);

    // e_hi = evens from samples 16-31: [c16,c18,c20,c22,c24,c26,c28,c30]
    // o_hi = odds from samples 16-31:  [c17,c19,c21,c23,c25,c27,c29,c31]
    __m512 e_hi = _mm512_permutex2var_ps(r2, idx_even, r3);
    __m512 o_hi = _mm512_permutex2var_ps(r2, idx_odd, r3);

    // Step 2: FFT-16 on evens -> E[0..15]
    fft16::fft16_interleaved_reg(e_lo, e_hi);

    // Step 3: FFT-16 on odds -> O[0..15]
    fft16::fft16_interleaved_reg(o_lo, o_hi);

    // Step 4: Apply W32 twiddles to odds
    // o_lo has O[0..7], needs W32^0..W32^7
    // o_hi has O[8..15], needs W32^8..W32^15
    __m512 tw_re_lo = _mm512_load_ps(tw32_re_interleaved_lo);
    __m512 tw_im_lo = _mm512_load_ps(tw32_im_interleaved_lo);
    __m512 tw_re_hi = _mm512_load_ps(tw32_re_interleaved_hi);
    __m512 tw_im_hi = _mm512_load_ps(tw32_im_interleaved_hi);

    __m512 wo_lo = complex_mul_interleaved(o_lo, tw_re_lo, tw_im_lo);  // W*O[0..7]
    __m512 wo_hi = complex_mul_interleaved(o_hi, tw_re_hi, tw_im_hi);  // W*O[8..15]

    // Step 5: Butterfly combine
    // X[k] = E[k] + W*O[k]     for k=0..15
    // X[k+16] = E[k] - W*O[k]  for k=0..15
    __m512 x_0_7   = _mm512_add_ps(e_lo, wo_lo);   // X[0..7]
    __m512 x_8_15  = _mm512_add_ps(e_hi, wo_hi);   // X[8..15]
    __m512 x_16_23 = _mm512_sub_ps(e_lo, wo_lo);   // X[16..23]
    __m512 x_24_31 = _mm512_sub_ps(e_hi, wo_hi);   // X[24..31]

    // Output in natural order
    r0 = x_0_7;
    r1 = x_8_15;
    r2 = x_16_23;
    r3 = x_24_31;
}

// FFT-32 in-place on memory (for testing/benchmarking)
[[gnu::target("avx512f,avx512vl,avx512dq,fma")]]
inline void fft32_interleaved_inplace(float* data) noexcept
{
    __m512 r0 = _mm512_loadu_ps(data);
    __m512 r1 = _mm512_loadu_ps(data + 16);
    __m512 r2 = _mm512_loadu_ps(data + 32);
    __m512 r3 = _mm512_loadu_ps(data + 48);

    fft32_interleaved_reg(r0, r1, r2, r3);

    _mm512_storeu_ps(data, r0);
    _mm512_storeu_ps(data + 16, r1);
    _mm512_storeu_ps(data + 32, r2);
    _mm512_storeu_ps(data + 48, r3);
}

} // namespace fft32

// =============================================================================
// FFT-32 via 2D approach: 4 rows × 8 columns
// =============================================================================
// View 32 samples as a 4×8 matrix (already the natural layout of 4 ZMM registers):
//   r0 = [x0..x7], r1 = [x8..x15], r2 = [x16..x23], r3 = [x24..x31]
//
// Algorithm:
//   1. FFT-8 on each row (4 independent FFT-8s) - uses our fast kernel
//   2. Multiply by twiddle factors W32^(row*col)
//   3. 4×8 transpose to get columns as contiguous groups
//   4. FFT-4 on each group (8 independent FFT-4s)
//   5. Output is already in natural order (no final transpose needed!)
//
// This avoids the expensive cross-lane gathers of the DIT decomposition.

namespace fft32_2d {

// Twiddles for 2D FFT-32: W32^(row*col)
// Row 0: all 1s (no twiddle needed)
// Row 1: W32^k for k=0..7
// Row 2: W32^(2k) for k=0..7
// Row 3: W32^(3k) for k=0..7
// Format: interleaved [re, re, re, re, ...] for broadcast multiply

alignas(64) inline constexpr float tw32_2d_row1_re[16] = {
    1.0f, 1.0f,                                     // k=0: W32^0
    0.98078528040323043f, 0.98078528040323043f,     // k=1: W32^1
    0.92387953251128674f, 0.92387953251128674f,     // k=2: W32^2
    0.83146961230254524f, 0.83146961230254524f,     // k=3: W32^3
    0.70710678118654752f, 0.70710678118654752f,     // k=4: W32^4
    0.55557023301960218f, 0.55557023301960218f,     // k=5: W32^5
    0.38268343236508978f, 0.38268343236508978f,     // k=6: W32^6
    0.19509032201612828f, 0.19509032201612828f      // k=7: W32^7
};

alignas(64) inline constexpr float tw32_2d_row1_im[16] = {
    0.0f, -0.0f,
    0.19509032201612828f, -0.19509032201612828f,
    0.38268343236508978f, -0.38268343236508978f,
    0.55557023301960218f, -0.55557023301960218f,
    0.70710678118654752f, -0.70710678118654752f,
    0.83146961230254524f, -0.83146961230254524f,
    0.92387953251128674f, -0.92387953251128674f,
    0.98078528040323043f, -0.98078528040323043f
};

alignas(64) inline constexpr float tw32_2d_row2_re[16] = {
    1.0f, 1.0f,                                     // k=0: W32^0
    0.92387953251128674f, 0.92387953251128674f,     // k=1: W32^2
    0.70710678118654752f, 0.70710678118654752f,     // k=2: W32^4
    0.38268343236508978f, 0.38268343236508978f,     // k=3: W32^6
    0.0f, 0.0f,                                     // k=4: W32^8
    -0.38268343236508978f, -0.38268343236508978f,   // k=5: W32^10
    -0.70710678118654752f, -0.70710678118654752f,   // k=6: W32^12
    -0.92387953251128674f, -0.92387953251128674f    // k=7: W32^14
};

alignas(64) inline constexpr float tw32_2d_row2_im[16] = {
    0.0f, -0.0f,
    0.38268343236508978f, -0.38268343236508978f,
    0.70710678118654752f, -0.70710678118654752f,
    0.92387953251128674f, -0.92387953251128674f,
    1.0f, -1.0f,
    0.92387953251128674f, -0.92387953251128674f,
    0.70710678118654752f, -0.70710678118654752f,
    0.38268343236508978f, -0.38268343236508978f
};

alignas(64) inline constexpr float tw32_2d_row3_re[16] = {
    1.0f, 1.0f,                                     // k=0: W32^0
    0.83146961230254524f, 0.83146961230254524f,     // k=1: W32^3
    0.38268343236508978f, 0.38268343236508978f,     // k=2: W32^6
    -0.19509032201612828f, -0.19509032201612828f,   // k=3: W32^9
    -0.70710678118654752f, -0.70710678118654752f,   // k=4: W32^12
    -0.98078528040323043f, -0.98078528040323043f,   // k=5: W32^15
    -0.92387953251128674f, -0.92387953251128674f,   // k=6: W32^18
    -0.55557023301960218f, -0.55557023301960218f    // k=7: W32^21
};

alignas(64) inline constexpr float tw32_2d_row3_im[16] = {
    0.0f, -0.0f,
    0.55557023301960218f, -0.55557023301960218f,
    0.92387953251128674f, -0.92387953251128674f,
    0.98078528040323043f, -0.98078528040323043f,
    0.70710678118654752f, -0.70710678118654752f,
    0.19509032201612828f, -0.19509032201612828f,
    -0.38268343236508978f, 0.38268343236508978f,
    -0.83146961230254524f, 0.83146961230254524f
};

// Transpose indices for 4×8 -> columns as groups
// Input (as doubles): d0=[Y00..Y07], d1=[Y10..Y17], d2=[Y20..Y27], d3=[Y30..Y37]
// Output: cols01=[Y00,Y10,Y20,Y30, Y01,Y11,Y21,Y31], etc.
alignas(64) inline constexpr long long idx_interleave_lo[8] = {0, 8, 1, 9, 2, 10, 3, 11};
alignas(64) inline constexpr long long idx_interleave_hi[8] = {4, 12, 5, 13, 6, 14, 7, 15};
alignas(64) inline constexpr long long idx_combine_01[8] = {0, 1, 8, 9, 2, 3, 10, 11};
alignas(64) inline constexpr long long idx_combine_23[8] = {4, 5, 12, 13, 6, 7, 14, 15};

// Complex multiply using same formula as fft32::complex_mul_interleaved
[[gnu::always_inline, gnu::target("avx512f,fma")]]
inline __m512 complex_mul(__m512 val, const float* tw_re, const float* tw_im) noexcept
{
    __m512 tre = _mm512_load_ps(tw_re);
    __m512 tim = _mm512_load_ps(tw_im);
    __m512 val_swap = _mm512_shuffle_ps(val, val, 0xB1);
    return _mm512_fmadd_ps(val, tre, _mm512_mul_ps(val_swap, tim));
}

// FFT-4 on a single 256-bit register containing 4 interleaved complex values
// Input:  [x0.r, x0.i, x1.r, x1.i, x2.r, x2.i, x3.r, x3.i]
// Output: [X0.r, X0.i, X1.r, X1.i, X2.r, X2.i, X3.r, X3.i]
[[gnu::always_inline, gnu::target("avx,avx2,fma")]]
inline __m256 fft4_avx(__m256 x) noexcept
{
    // Butterfly: t0 = x0+x2, t1 = x0-x2, t2 = x1+x3, t3 = x1-x3
    // Then: X0 = t0+t2, X2 = t0-t2, X1 = t1-j*t3, X3 = t1+j*t3

    // Create x0+x2 and x1+x3 pairs using permute
    // x = [x0.r, x0.i, x1.r, x1.i, x2.r, x2.i, x3.r, x3.i]
    __m256 x_lo = _mm256_permutevar8x32_ps(x, _mm256_setr_epi32(0, 1, 2, 3, 0, 1, 2, 3));
    __m256 x_hi = _mm256_permutevar8x32_ps(x, _mm256_setr_epi32(4, 5, 6, 7, 4, 5, 6, 7));
    // x_lo = [x0, x1, x0, x1], x_hi = [x2, x3, x2, x3]

    __m256 sum = _mm256_add_ps(x_lo, x_hi);   // [t0, t2, t0, t2]
    __m256 diff = _mm256_sub_ps(x_lo, x_hi);  // [t1, t3, t1, t3]

    // For X0, X2: simple add/sub of t0 and t2
    __m256 t0 = _mm256_permutevar8x32_ps(sum, _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1));
    __m256 t2 = _mm256_permutevar8x32_ps(sum, _mm256_setr_epi32(2, 3, 2, 3, 2, 3, 2, 3));
    __m256 X0 = _mm256_add_ps(t0, t2);  // [X0, X0, X0, X0]
    __m256 X2 = _mm256_sub_ps(t0, t2);  // [X2, X2, X2, X2]

    // For X1, X3: t1 ± j*t3
    // j*t3 = [-t3.im, t3.re]
    __m256 t1 = _mm256_permutevar8x32_ps(diff, _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1));
    __m256 t3 = _mm256_permutevar8x32_ps(diff, _mm256_setr_epi32(2, 3, 2, 3, 2, 3, 2, 3));
    __m256 t3_swap = _mm256_shuffle_ps(t3, t3, 0xB1);  // [t3.i, t3.r, ...]
    __m256 j_t3 = _mm256_mul_ps(t3_swap, _mm256_setr_ps(-1, 1, -1, 1, -1, 1, -1, 1));
    __m256 X1 = _mm256_sub_ps(t1, j_t3);  // t1 - j*t3
    __m256 X3 = _mm256_add_ps(t1, j_t3);  // t1 + j*t3

    // Combine: output = [X0, X1, X2, X3]
    __m256 r01 = _mm256_shuffle_ps(X0, X1, 0b01000100);  // [X0, X1] in each 128-bit lane
    __m256 r23 = _mm256_shuffle_ps(X2, X3, 0b01000100);  // [X2, X3] in each 128-bit lane
    return _mm256_permute2f128_ps(r01, r23, 0x20);  // [X0, X1, X2, X3]
}

// FFT-4 on 2 groups within a 512-bit register
[[gnu::always_inline, gnu::target("avx512f,avx512dq,avx,avx2,fma")]]
inline void fft4_x2(__m512& reg) noexcept
{
    __m256 lo = _mm512_castps512_ps256(reg);
    __m256 hi = _mm512_extractf32x8_ps(reg, 1);
    lo = fft4_avx(lo);
    hi = fft4_avx(hi);
    reg = _mm512_insertf32x8(_mm512_castps256_ps512(lo), hi, 1);
}

// 4×8 transpose: rearrange 4 rows of 8 complex values to 8 columns of 4 complex values
// After this, each register holds 2 columns ready for FFT-4
[[gnu::always_inline, gnu::target("avx512f")]]
inline void transpose_4x8(__m512& r0, __m512& r1, __m512& r2, __m512& r3) noexcept
{
    // Treat complex values as doubles for 8-element transpose
    __m512d d0 = _mm512_castps_pd(r0);
    __m512d d1 = _mm512_castps_pd(r1);
    __m512d d2 = _mm512_castps_pd(r2);
    __m512d d3 = _mm512_castps_pd(r3);

    __m512i idx_lo = _mm512_load_epi64(idx_interleave_lo);
    __m512i idx_hi = _mm512_load_epi64(idx_interleave_hi);
    __m512i idx_01 = _mm512_load_epi64(idx_combine_01);
    __m512i idx_23 = _mm512_load_epi64(idx_combine_23);

    // Interleave rows 0,1 and rows 2,3
    __m512d d01_lo = _mm512_permutex2var_pd(d0, idx_lo, d1);  // [Y00,Y10,Y01,Y11,Y02,Y12,Y03,Y13]
    __m512d d01_hi = _mm512_permutex2var_pd(d0, idx_hi, d1);  // [Y04,Y14,Y05,Y15,Y06,Y16,Y07,Y17]
    __m512d d23_lo = _mm512_permutex2var_pd(d2, idx_lo, d3);  // [Y20,Y30,Y21,Y31,Y22,Y32,Y23,Y33]
    __m512d d23_hi = _mm512_permutex2var_pd(d2, idx_hi, d3);  // [Y24,Y34,Y25,Y35,Y26,Y36,Y27,Y37]

    // Combine to get column pairs
    __m512d cols01 = _mm512_permutex2var_pd(d01_lo, idx_01, d23_lo);  // [Y00,Y10,Y20,Y30, Y01,Y11,Y21,Y31]
    __m512d cols23 = _mm512_permutex2var_pd(d01_lo, idx_23, d23_lo);  // [Y02,Y12,Y22,Y32, Y03,Y13,Y23,Y33]
    __m512d cols45 = _mm512_permutex2var_pd(d01_hi, idx_01, d23_hi);  // [Y04,Y14,Y24,Y34, Y05,Y15,Y25,Y35]
    __m512d cols67 = _mm512_permutex2var_pd(d01_hi, idx_23, d23_hi);  // [Y06,Y16,Y26,Y36, Y07,Y17,Y27,Y37]

    r0 = _mm512_castpd_ps(cols01);
    r1 = _mm512_castpd_ps(cols23);
    r2 = _mm512_castpd_ps(cols45);
    r3 = _mm512_castpd_ps(cols67);
}

// Indices for final 8×4 transpose (column-major to row-major)
// Input: r0=[X00,X10,X20,X30,X01,X11,X21,X31], r1=[X02,...], r2=[X04,...], r3=[X06,...]
// Output: r0=[X00,X01,X02,X03,X04,X05,X06,X07] (row 0), etc.
alignas(64) inline constexpr long long idx_row0[8] = {0, 4, 8, 12, 16, 20, 24, 28};  // k1=0 from all cols
alignas(64) inline constexpr long long idx_row1[8] = {1, 5, 9, 13, 17, 21, 25, 29};  // k1=1 from all cols
alignas(64) inline constexpr long long idx_row2[8] = {2, 6, 10, 14, 18, 22, 26, 30}; // k1=2 from all cols
alignas(64) inline constexpr long long idx_row3[8] = {3, 7, 11, 15, 19, 23, 27, 31}; // k1=3 from all cols

// 8×4 transpose: convert column-major output back to row-major (natural order)
// Input: 4 regs each with 2 columns [col_i, col_i+1] where each column has 4 elements
// Output: 4 regs each with one row of 8 elements
[[gnu::always_inline, gnu::target("avx512f")]]
inline void transpose_8x4(__m512& r0, __m512& r1, __m512& r2, __m512& r3) noexcept
{
    __m512d d0 = _mm512_castps_pd(r0);  // cols 0,1
    __m512d d1 = _mm512_castps_pd(r1);  // cols 2,3
    __m512d d2 = _mm512_castps_pd(r2);  // cols 4,5
    __m512d d3 = _mm512_castps_pd(r3);  // cols 6,7

    // First, combine d0,d1 and d2,d3 to get all 32 elements accessible via permutex2var
    // d0||d1 gives 16 doubles, but we need all 32. Use two stages.

    // Stage 1: gather from pairs
    // For row 0, need: d0[0], d0[4], d1[0], d1[4], d2[0], d2[4], d3[0], d3[4]
    // Indices within d0||d1: 0, 4, 8, 12 and within d2||d3: 0, 4, 8, 12

    __m512i idx_04 = _mm512_setr_epi64(0, 4, 8, 12, 0, 4, 8, 12);  // Same pattern for both halves

    __m512d t01_04 = _mm512_permutex2var_pd(d0, idx_04, d1);  // [d0[0],d0[4],d1[0],d1[4], d0[0],d0[4],d1[0],d1[4]]
    __m512d t23_04 = _mm512_permutex2var_pd(d2, idx_04, d3);  // [d2[0],d2[4],d3[0],d3[4], ...]

    // Combine first halves: row0 = [t01_04[0:3], t23_04[0:3]]
    __m512d row0 = _mm512_shuffle_f64x2(t01_04, t23_04, 0x44);  // [t01_04.lo, t23_04.lo]

    __m512i idx_15 = _mm512_setr_epi64(1, 5, 9, 13, 1, 5, 9, 13);
    __m512d t01_15 = _mm512_permutex2var_pd(d0, idx_15, d1);
    __m512d t23_15 = _mm512_permutex2var_pd(d2, idx_15, d3);
    __m512d row1 = _mm512_shuffle_f64x2(t01_15, t23_15, 0x44);

    __m512i idx_26 = _mm512_setr_epi64(2, 6, 10, 14, 2, 6, 10, 14);
    __m512d t01_26 = _mm512_permutex2var_pd(d0, idx_26, d1);
    __m512d t23_26 = _mm512_permutex2var_pd(d2, idx_26, d3);
    __m512d row2 = _mm512_shuffle_f64x2(t01_26, t23_26, 0x44);

    __m512i idx_37 = _mm512_setr_epi64(3, 7, 11, 15, 3, 7, 11, 15);
    __m512d t01_37 = _mm512_permutex2var_pd(d0, idx_37, d1);
    __m512d t23_37 = _mm512_permutex2var_pd(d2, idx_37, d3);
    __m512d row3 = _mm512_shuffle_f64x2(t01_37, t23_37, 0x44);

    r0 = _mm512_castpd_ps(row0);
    r1 = _mm512_castpd_ps(row1);
    r2 = _mm512_castpd_ps(row2);
    r3 = _mm512_castpd_ps(row3);
}

// Main FFT-32 function using 2D decomposition
[[gnu::target("avx512f,avx512dq,avx,avx2,fma")]]
inline void fft32_2d_reg(__m512& r0, __m512& r1, __m512& r2, __m512& r3) noexcept
{
    // Step 1: FFT-8 on each row
    r0 = fft8::fft8_interleaved_reg(r0);
    r1 = fft8::fft8_interleaved_reg(r1);
    r2 = fft8::fft8_interleaved_reg(r2);
    r3 = fft8::fft8_interleaved_reg(r3);

    // Step 2: Apply twiddle factors W32^(row*col)
    // Row 0: no twiddle (W32^0 = 1)
    r1 = complex_mul(r1, tw32_2d_row1_re, tw32_2d_row1_im);
    r2 = complex_mul(r2, tw32_2d_row2_re, tw32_2d_row2_im);
    r3 = complex_mul(r3, tw32_2d_row3_re, tw32_2d_row3_im);

    // Step 3: 4×8 transpose
    transpose_4x8(r0, r1, r2, r3);

    // Step 4: FFT-4 on each column (2 columns per register)
    fft4_x2(r0);  // Columns 0,1
    fft4_x2(r1);  // Columns 2,3
    fft4_x2(r2);  // Columns 4,5
    fft4_x2(r3);  // Columns 6,7

    // Step 5: Transpose 8×4 to get natural output order
    // After FFT-4, each register has column-major layout: [col0_k0..k3, col1_k0..k3]
    // Transpose gathers all k=0 outputs from columns 0-7 into r0, etc.
    transpose_8x4(r0, r1, r2, r3);

    // Output is now in natural order:
    // r0 = [X0..X7], r1 = [X8..X15], r2 = [X16..X23], r3 = [X24..X31]
}

// In-place wrapper for testing/benchmarking
[[gnu::target("avx512f,avx512dq,avx,avx2,fma")]]
inline void fft32_2d_inplace(float* data) noexcept
{
    __m512 r0 = _mm512_loadu_ps(data);
    __m512 r1 = _mm512_loadu_ps(data + 16);
    __m512 r2 = _mm512_loadu_ps(data + 32);
    __m512 r3 = _mm512_loadu_ps(data + 48);

    fft32_2d_reg(r0, r1, r2, r3);

    _mm512_storeu_ps(data, r0);
    _mm512_storeu_ps(data + 16, r1);
    _mm512_storeu_ps(data + 32, r2);
    _mm512_storeu_ps(data + 48, r3);
}

} // namespace fft32_2d

// =============================================================================
// M=16 Fused Filter + FFT + Transpose (Register-Resident)
// =============================================================================
// For M=16, uses 4-way time unrolling (vs 8-way for M=8) due to register pressure.
// Each time step requires 2 ZMM registers (lo/hi for 16 complex samples).
//
// Pipeline stays largely in registers:
//   Filter → FFT-16 → Transpose → Store

// 4x16 transpose helper: transposes 4 rows of 16 complex values to 16 channels × 4 samples
// Uses AVX-512 shuffles to minimize memory operations.
// Handles null channel pointers safely.
[[gnu::always_inline, gnu::target("avx512f,avx512dq")]]
inline void transpose_4x16_to_channels(
    __m512 r0_lo, __m512 r0_hi,  // Row 0: channels 0-7, 8-15
    __m512 r1_lo, __m512 r1_hi,  // Row 1
    __m512 r2_lo, __m512 r2_hi,  // Row 2
    __m512 r3_lo, __m512 r3_hi,  // Row 3
    std::complex<float>* const* channel_ptrs,
    std::size_t offset
) noexcept
{
    // Treat complex floats as doubles (8 bytes) for efficient 4x4 transposes
    // Process in two groups: channels 0-7 (lo) and channels 8-15 (hi)

    // Group 1: Channels 0-7 from lo registers
    {
        __m512d d0 = _mm512_castps_pd(r0_lo);
        __m512d d1 = _mm512_castps_pd(r1_lo);
        __m512d d2 = _mm512_castps_pd(r2_lo);
        __m512d d3 = _mm512_castps_pd(r3_lo);

        // 4x8 -> partial transpose: we have 4 rows of 8 complex values
        // Output 8 channels each getting 4 values

        // Interleave pairs: [r0c0,r1c0,r0c1,r1c1,...] etc
        __m512d t0 = _mm512_unpacklo_pd(d0, d1);  // [r0c0,r1c0, r0c2,r1c2, r0c4,r1c4, r0c6,r1c6]
        __m512d t1 = _mm512_unpackhi_pd(d0, d1);  // [r0c1,r1c1, r0c3,r1c3, r0c5,r1c5, r0c7,r1c7]
        __m512d t2 = _mm512_unpacklo_pd(d2, d3);  // [r2c0,r3c0, r2c2,r3c2, r2c4,r3c4, r2c6,r3c6]
        __m512d t3 = _mm512_unpackhi_pd(d2, d3);  // [r2c1,r3c1, r2c3,r3c3, r2c5,r3c5, r2c7,r3c7]

        // Combine 128-bit lanes to get 4 consecutive time samples per channel
        // Channel 0: [r0c0,r1c0,r2c0,r3c0] from t0[0:1] and t2[0:1]
        if (channel_ptrs[0]) {
            __m256d ch0 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_castpd512_pd128(t0)),
                _mm512_castpd512_pd128(t2), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[0] + offset), ch0);
        }

        // Channel 1: from t1[0:1] and t3[0:1]
        if (channel_ptrs[1]) {
            __m256d ch1 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_castpd512_pd128(t1)),
                _mm512_castpd512_pd128(t3), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[1] + offset), ch1);
        }

        // Channels 2-3: from lane 1 (indices 2,3)
        if (channel_ptrs[2]) {
            __m256d ch2 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t0, 1)),
                _mm512_extractf64x2_pd(t2, 1), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[2] + offset), ch2);
        }

        if (channel_ptrs[3]) {
            __m256d ch3 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t1, 1)),
                _mm512_extractf64x2_pd(t3, 1), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[3] + offset), ch3);
        }

        // Channels 4-5: from lane 2 (indices 4,5)
        if (channel_ptrs[4]) {
            __m256d ch4 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t0, 2)),
                _mm512_extractf64x2_pd(t2, 2), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[4] + offset), ch4);
        }

        if (channel_ptrs[5]) {
            __m256d ch5 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t1, 2)),
                _mm512_extractf64x2_pd(t3, 2), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[5] + offset), ch5);
        }

        // Channels 6-7: from lane 3 (indices 6,7)
        if (channel_ptrs[6]) {
            __m256d ch6 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t0, 3)),
                _mm512_extractf64x2_pd(t2, 3), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[6] + offset), ch6);
        }

        if (channel_ptrs[7]) {
            __m256d ch7 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t1, 3)),
                _mm512_extractf64x2_pd(t3, 3), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[7] + offset), ch7);
        }
    }

    // Group 2: Channels 8-15 from hi registers (same pattern)
    {
        __m512d d0 = _mm512_castps_pd(r0_hi);
        __m512d d1 = _mm512_castps_pd(r1_hi);
        __m512d d2 = _mm512_castps_pd(r2_hi);
        __m512d d3 = _mm512_castps_pd(r3_hi);

        __m512d t0 = _mm512_unpacklo_pd(d0, d1);
        __m512d t1 = _mm512_unpackhi_pd(d0, d1);
        __m512d t2 = _mm512_unpacklo_pd(d2, d3);
        __m512d t3 = _mm512_unpackhi_pd(d2, d3);

        if (channel_ptrs[8]) {
            __m256d ch8 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_castpd512_pd128(t0)),
                _mm512_castpd512_pd128(t2), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[8] + offset), ch8);
        }

        if (channel_ptrs[9]) {
            __m256d ch9 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_castpd512_pd128(t1)),
                _mm512_castpd512_pd128(t3), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[9] + offset), ch9);
        }

        if (channel_ptrs[10]) {
            __m256d ch10 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t0, 1)),
                _mm512_extractf64x2_pd(t2, 1), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[10] + offset), ch10);
        }

        if (channel_ptrs[11]) {
            __m256d ch11 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t1, 1)),
                _mm512_extractf64x2_pd(t3, 1), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[11] + offset), ch11);
        }

        if (channel_ptrs[12]) {
            __m256d ch12 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t0, 2)),
                _mm512_extractf64x2_pd(t2, 2), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[12] + offset), ch12);
        }

        if (channel_ptrs[13]) {
            __m256d ch13 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t1, 2)),
                _mm512_extractf64x2_pd(t3, 2), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[13] + offset), ch13);
        }

        if (channel_ptrs[14]) {
            __m256d ch14 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t0, 3)),
                _mm512_extractf64x2_pd(t2, 3), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[14] + offset), ch14);
        }

        if (channel_ptrs[15]) {
            __m256d ch15 = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm512_extractf64x2_pd(t1, 3)),
                _mm512_extractf64x2_pd(t3, 3), 1);
            _mm256_storeu_pd(reinterpret_cast<double*>(channel_ptrs[15] + offset), ch15);
        }
    }
}

// =============================================================================
// Fused Filter + Vertical FFT-8 + Direct Scatter (Throughput-Optimized)
// =============================================================================
// Uses vertical FFT for maximum throughput:
//   1. Filter 8 time samples into 8 accumulators (row-major)
//   2. 8x8 register transpose (row-major -> column-major)
//   3. Vertical FFT-8 (8 parallel FFTs, NO shuffles)
//   4. Direct store to channel buffers (already in scatter format)
//
// Key difference from horizontal version:
//   - Transpose happens BEFORE FFT (not after)
//   - FFT has zero internal shuffles
//   - Output needs no transpose
[[gnu::target("avx512f,avx512dq")]]
inline void filter_fft_transpose_8ch_vertical(
    const std::complex<float>* input,
    const float* coeffs,
    std::complex<float>* const* channel_ptrs,  // 8 output pointers
    std::size_t num_outputs,
    std::size_t num_taps
) noexcept
{
    constexpr std::size_t M = 8;
    constexpr std::size_t FLOATS_PER_ROW = M * 2;

    const float* x_base = reinterpret_cast<const float*>(input);

    std::size_t t = 0;

    // Main loop: 8 outputs at a time
    for (; t + 8 <= num_outputs; t += 8) {
        // =================================================================
        // Stage 1: Polyphase filter - 8 accumulators, row-major (same as horizontal)
        // =================================================================
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        __m512 acc4 = _mm512_setzero_ps();
        __m512 acc5 = _mm512_setzero_ps();
        __m512 acc6 = _mm512_setzero_ps();
        __m512 acc7 = _mm512_setzero_ps();

        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h = _mm512_loadu_ps(h_ptr);

            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0 + k) * FLOATS_PER_ROW), h, acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1 + k) * FLOATS_PER_ROW), h, acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2 + k) * FLOATS_PER_ROW), h, acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3 + k) * FLOATS_PER_ROW), h, acc3);
            acc4 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 4 + k) * FLOATS_PER_ROW), h, acc4);
            acc5 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 5 + k) * FLOATS_PER_ROW), h, acc5);
            acc6 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 6 + k) * FLOATS_PER_ROW), h, acc6);
            acc7 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 7 + k) * FLOATS_PER_ROW), h, acc7);

            h_ptr += FLOATS_PER_ROW;
        }

        // =================================================================
        // Stage 2: 8x8 transpose BEFORE FFT (row-major -> column-major)
        // After this: acc0 = [t0_ch0, t1_ch0, ..., t7_ch0] (channel 0 across all times)
        // =================================================================
        fft8::transpose_8x8_cf32_inplace(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

        // =================================================================
        // Stage 3: Vertical FFT-8 (8 parallel FFTs, NO internal shuffles)
        // Input: acc[i] = channel i across all 8 time samples
        // Output: acc[i] = bin i across all 8 FFTs (already in scatter format!)
        // =================================================================
        fft8_vertical::fft8_vertical(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);

        // =================================================================
        // Stage 4: Direct store to channel buffers (no transpose needed!)
        // acc0 = X[0] for all 8 time samples -> channel 0 buffer
        // acc1 = X[1] for all 8 time samples -> channel 1 buffer
        // =================================================================
        if (channel_ptrs[0]) { store_512(reinterpret_cast<float*>(channel_ptrs[0] + t), acc0); }
        if (channel_ptrs[1]) { store_512(reinterpret_cast<float*>(channel_ptrs[1] + t), acc1); }
        if (channel_ptrs[2]) { store_512(reinterpret_cast<float*>(channel_ptrs[2] + t), acc2); }
        if (channel_ptrs[3]) { store_512(reinterpret_cast<float*>(channel_ptrs[3] + t), acc3); }
        if (channel_ptrs[4]) { store_512(reinterpret_cast<float*>(channel_ptrs[4] + t), acc4); }
        if (channel_ptrs[5]) { store_512(reinterpret_cast<float*>(channel_ptrs[5] + t), acc5); }
        if (channel_ptrs[6]) { store_512(reinterpret_cast<float*>(channel_ptrs[6] + t), acc6); }
        if (channel_ptrs[7]) { store_512(reinterpret_cast<float*>(channel_ptrs[7] + t), acc7); }
    }

    // =====================================================================
    // Tail: 4-7 outputs - use AVX2 vertical FFT (4 parallel FFT-8s)
    // =====================================================================
    if (t + 4 <= num_outputs) {
        // Process 4 outputs using AVX2 vertical approach
        // Filter: 4 rows × 8 channels → 8 YMM accumulators
        // Layout: acc[0,2,4,6] = first 4 channels, acc[1,3,5,7] = last 4 channels
        __m256 acc0 = _mm256_setzero_ps();  // t0 ch0-3
        __m256 acc1 = _mm256_setzero_ps();  // t0 ch4-7
        __m256 acc2 = _mm256_setzero_ps();  // t1 ch0-3
        __m256 acc3 = _mm256_setzero_ps();  // t1 ch4-7
        __m256 acc4 = _mm256_setzero_ps();  // t2 ch0-3
        __m256 acc5 = _mm256_setzero_ps();  // t2 ch4-7
        __m256 acc6 = _mm256_setzero_ps();  // t3 ch0-3
        __m256 acc7 = _mm256_setzero_ps();  // t3 ch4-7

        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m256 h_lo = _mm256_loadu_ps(h_ptr);      // ch0-3 coeffs
            __m256 h_hi = _mm256_loadu_ps(h_ptr + 8);  // ch4-7 coeffs

            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 0 + k) * FLOATS_PER_ROW), h_lo, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 0 + k) * FLOATS_PER_ROW + 8), h_hi, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 1 + k) * FLOATS_PER_ROW), h_lo, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 1 + k) * FLOATS_PER_ROW + 8), h_hi, acc3);
            acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 2 + k) * FLOATS_PER_ROW), h_lo, acc4);
            acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 2 + k) * FLOATS_PER_ROW + 8), h_hi, acc5);
            acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 3 + k) * FLOATS_PER_ROW), h_lo, acc6);
            acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(x_base + (t + 3 + k) * FLOATS_PER_ROW + 8), h_hi, acc7);

            h_ptr += FLOATS_PER_ROW;
        }

        // Transpose 4×8 → 8×4 using two 4×4 transposes
        // First half: channels 0-3 across times 0-3
        fft8_vertical::transpose_4x4_cf32_avx2(acc0, acc2, acc4, acc6);
        // Second half: channels 4-7 across times 0-3
        fft8_vertical::transpose_4x4_cf32_avx2(acc1, acc3, acc5, acc7);

        // Rearrange for vertical FFT: r[i] = channel i across all 4 times
        // After transpose: acc0=ch0, acc2=ch1, acc4=ch2, acc6=ch3
        //                  acc1=ch4, acc3=ch5, acc5=ch6, acc7=ch7
        __m256 r0 = acc0, r1 = acc2, r2 = acc4, r3 = acc6;
        __m256 r4 = acc1, r5 = acc3, r6 = acc5, r7 = acc7;

        // Vertical FFT-8 using AVX2 (4 parallel FFTs)
        fft8_vertical::fft8_vertical_avx2(r0, r1, r2, r3, r4, r5, r6, r7);

        // Direct store to channel buffers (4 complex values each)
        if (channel_ptrs[0]) { store_256(reinterpret_cast<float*>(channel_ptrs[0] + t), r0); }
        if (channel_ptrs[1]) { store_256(reinterpret_cast<float*>(channel_ptrs[1] + t), r1); }
        if (channel_ptrs[2]) { store_256(reinterpret_cast<float*>(channel_ptrs[2] + t), r2); }
        if (channel_ptrs[3]) { store_256(reinterpret_cast<float*>(channel_ptrs[3] + t), r3); }
        if (channel_ptrs[4]) { store_256(reinterpret_cast<float*>(channel_ptrs[4] + t), r4); }
        if (channel_ptrs[5]) { store_256(reinterpret_cast<float*>(channel_ptrs[5] + t), r5); }
        if (channel_ptrs[6]) { store_256(reinterpret_cast<float*>(channel_ptrs[6] + t), r6); }
        if (channel_ptrs[7]) { store_256(reinterpret_cast<float*>(channel_ptrs[7] + t), r7); }

        t += 4;
    }

    // =====================================================================
    // Tail: remaining 1-3 outputs - scalar fallback
    // =====================================================================
    for (; t < num_outputs; ++t) {
        __m512 acc = _mm512_setzero_ps();
        const float* h_ptr = coeffs;

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h = _mm512_loadu_ps(h_ptr);
            __m512 x = _mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW);
            acc = _mm512_fmadd_ps(x, h, acc);
            h_ptr += FLOATS_PER_ROW;
        }

        // Use horizontal FFT for single row
        acc = fft8::fft8_interleaved_reg(acc);

        // Scalar scatter to channel buffers
        alignas(64) float out_arr[16];
        _mm512_store_ps(out_arr, acc);

        for (std::size_t ch = 0; ch < 8; ++ch) {
            if (channel_ptrs[ch]) {
                channel_ptrs[ch][t] = std::complex<float>(out_arr[ch * 2], out_arr[ch * 2 + 1]);
            }
        }
    }
}

// =============================================================================
// Fused Filter+FFT+Transpose for M=16 (Vertical FFT-16 Approach)
// =============================================================================
// For M=16 channelizer, processes 8 time outputs at once using vertical FFT:
//   1. Filter: 8 rows × 16 complex → 16 ZMM accumulators (2 per row)
//   2. Transpose: 8×16 → 16×8 (two independent 8×8 transposes)
//   3. Vertical FFT-16: 8 parallel FFT-16s with zero cross-lane shuffles
//   4. Direct store to 16 channel buffers
//
// This is faster than the horizontal version because:
//   - FFT has zero cross-lane shuffles (each SIMD lane is independent FFT)
//   - Output is already in scatter format after FFT (no post-FFT transpose)
//
// Register budget per stage (sequential, reuses registers):
//   Filter:    16 accumulators + 2-4 temp = ~20 ZMM
//   Transpose: 16 data + 4 temp = ~20 ZMM
//   FFT-16:    16 data + 10 temp = ~26 ZMM (peak)
//   Store:     16 data = 16 ZMM
//
[[gnu::target("avx512f,avx512dq,avx512vl,fma")]]
inline void filter_fft_transpose_16ch_vertical(
    const std::complex<float>* input,
    const float* coeffs,
    std::complex<float>* const* channel_ptrs,  // 16 output pointers
    std::size_t num_outputs,
    std::size_t num_taps
) noexcept
{
    constexpr std::size_t M = 16;
    constexpr std::size_t FLOATS_PER_ROW = M * 2;  // 32 floats per row

    const float* x_base = reinterpret_cast<const float*>(input);

    std::size_t t = 0;

    // Main loop: 8 outputs at a time
    for (; t + 8 <= num_outputs; t += 8) {
        // =================================================================
        // Stage 1: Polyphase filter - 8 rows × 16 channels = 16 ZMM accumulators
        // Row t uses (acc_t_lo, acc_t_hi) for channels 0-7 and 8-15
        // =================================================================
        __m512 acc0_lo = _mm512_setzero_ps(), acc0_hi = _mm512_setzero_ps();
        __m512 acc1_lo = _mm512_setzero_ps(), acc1_hi = _mm512_setzero_ps();
        __m512 acc2_lo = _mm512_setzero_ps(), acc2_hi = _mm512_setzero_ps();
        __m512 acc3_lo = _mm512_setzero_ps(), acc3_hi = _mm512_setzero_ps();
        __m512 acc4_lo = _mm512_setzero_ps(), acc4_hi = _mm512_setzero_ps();
        __m512 acc5_lo = _mm512_setzero_ps(), acc5_hi = _mm512_setzero_ps();
        __m512 acc6_lo = _mm512_setzero_ps(), acc6_hi = _mm512_setzero_ps();
        __m512 acc7_lo = _mm512_setzero_ps(), acc7_hi = _mm512_setzero_ps();

        // Blocked coefficient layout: block0 = channels 0-7, block1 = channels 8-15
        // Block 0 at offset 0, Block 1 at offset K*16
        constexpr std::size_t FLOATS_PER_REG = 16;
        const std::size_t block1_offset = num_taps * FLOATS_PER_REG;

        for (std::size_t k = 0; k < num_taps; ++k) {
            // Load from blocked layout: tap k of each block
            __m512 h_lo = _mm512_load_ps(coeffs + k * FLOATS_PER_REG);
            __m512 h_hi = _mm512_load_ps(coeffs + block1_offset + k * FLOATS_PER_REG);

            acc0_lo = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0 + k) * FLOATS_PER_ROW), h_lo, acc0_lo);
            acc0_hi = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0 + k) * FLOATS_PER_ROW + 16), h_hi, acc0_hi);
            acc1_lo = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1 + k) * FLOATS_PER_ROW), h_lo, acc1_lo);
            acc1_hi = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1 + k) * FLOATS_PER_ROW + 16), h_hi, acc1_hi);
            acc2_lo = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2 + k) * FLOATS_PER_ROW), h_lo, acc2_lo);
            acc2_hi = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2 + k) * FLOATS_PER_ROW + 16), h_hi, acc2_hi);
            acc3_lo = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3 + k) * FLOATS_PER_ROW), h_lo, acc3_lo);
            acc3_hi = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3 + k) * FLOATS_PER_ROW + 16), h_hi, acc3_hi);
            acc4_lo = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 4 + k) * FLOATS_PER_ROW), h_lo, acc4_lo);
            acc4_hi = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 4 + k) * FLOATS_PER_ROW + 16), h_hi, acc4_hi);
            acc5_lo = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 5 + k) * FLOATS_PER_ROW), h_lo, acc5_lo);
            acc5_hi = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 5 + k) * FLOATS_PER_ROW + 16), h_hi, acc5_hi);
            acc6_lo = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 6 + k) * FLOATS_PER_ROW), h_lo, acc6_lo);
            acc6_hi = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 6 + k) * FLOATS_PER_ROW + 16), h_hi, acc6_hi);
            acc7_lo = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 7 + k) * FLOATS_PER_ROW), h_lo, acc7_lo);
            acc7_hi = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 7 + k) * FLOATS_PER_ROW + 16), h_hi, acc7_hi);
        }

        // =================================================================
        // Stage 2: Transpose 8×16 → 16×8 using two independent 8×8 transposes
        // Before: acc[row]_lo = channels 0-7 for time row
        //         acc[row]_hi = channels 8-15 for time row
        // After:  acc[row]_lo = bin row across times 0-7
        //         acc[row]_hi = bin (row+8) across times 0-7
        // =================================================================

        // First 8×8 transpose: channels 0-7 across all 8 time outputs
        fft8::transpose_8x8_cf32_inplace(acc0_lo, acc1_lo, acc2_lo, acc3_lo,
                                          acc4_lo, acc5_lo, acc6_lo, acc7_lo);

        // Second 8×8 transpose: channels 8-15 across all 8 time outputs
        fft8::transpose_8x8_cf32_inplace(acc0_hi, acc1_hi, acc2_hi, acc3_hi,
                                          acc4_hi, acc5_hi, acc6_hi, acc7_hi);

        // =================================================================
        // Stage 3: Vertical FFT-16 (8 parallel FFT-16s)
        // Map registers to FFT inputs:
        //   r0..r7   = bins 0-7 (from _lo registers after transpose)
        //   r8..r15  = bins 8-15 (from _hi registers after transpose)
        // =================================================================
        fft16_vertical::fft16_vertical(
            acc0_lo, acc1_lo, acc2_lo, acc3_lo, acc4_lo, acc5_lo, acc6_lo, acc7_lo,
            acc0_hi, acc1_hi, acc2_hi, acc3_hi, acc4_hi, acc5_hi, acc6_hi, acc7_hi
        );

        // =================================================================
        // Stage 4: Direct store to channel buffers
        // After FFT: acc[i]_lo = X[i] for times 0-7
        //            acc[i]_hi = X[i+8] for times 0-7
        // =================================================================
        if (channel_ptrs[0])  { store_512(reinterpret_cast<float*>(channel_ptrs[0] + t), acc0_lo); }
        if (channel_ptrs[1])  { store_512(reinterpret_cast<float*>(channel_ptrs[1] + t), acc1_lo); }
        if (channel_ptrs[2])  { store_512(reinterpret_cast<float*>(channel_ptrs[2] + t), acc2_lo); }
        if (channel_ptrs[3])  { store_512(reinterpret_cast<float*>(channel_ptrs[3] + t), acc3_lo); }
        if (channel_ptrs[4])  { store_512(reinterpret_cast<float*>(channel_ptrs[4] + t), acc4_lo); }
        if (channel_ptrs[5])  { store_512(reinterpret_cast<float*>(channel_ptrs[5] + t), acc5_lo); }
        if (channel_ptrs[6])  { store_512(reinterpret_cast<float*>(channel_ptrs[6] + t), acc6_lo); }
        if (channel_ptrs[7])  { store_512(reinterpret_cast<float*>(channel_ptrs[7] + t), acc7_lo); }
        if (channel_ptrs[8])  { store_512(reinterpret_cast<float*>(channel_ptrs[8] + t), acc0_hi); }
        if (channel_ptrs[9])  { store_512(reinterpret_cast<float*>(channel_ptrs[9] + t), acc1_hi); }
        if (channel_ptrs[10]) { store_512(reinterpret_cast<float*>(channel_ptrs[10] + t), acc2_hi); }
        if (channel_ptrs[11]) { store_512(reinterpret_cast<float*>(channel_ptrs[11] + t), acc3_hi); }
        if (channel_ptrs[12]) { store_512(reinterpret_cast<float*>(channel_ptrs[12] + t), acc4_hi); }
        if (channel_ptrs[13]) { store_512(reinterpret_cast<float*>(channel_ptrs[13] + t), acc5_hi); }
        if (channel_ptrs[14]) { store_512(reinterpret_cast<float*>(channel_ptrs[14] + t), acc6_hi); }
        if (channel_ptrs[15]) { store_512(reinterpret_cast<float*>(channel_ptrs[15] + t), acc7_hi); }
    }

    // =====================================================================
    // Tail: remaining 1-7 outputs - use staged approach (filter → FFT → scatter)
    // =====================================================================
    constexpr std::size_t FLOATS_PER_REG_TAIL = 16;
    const std::size_t block1_offset_tail = num_taps * FLOATS_PER_REG_TAIL;

    for (; t < num_outputs; ++t) {
        // Filter single row using blocked coefficient layout
        __m512 acc_lo = _mm512_setzero_ps();
        __m512 acc_hi = _mm512_setzero_ps();

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h_lo = _mm512_load_ps(coeffs + k * FLOATS_PER_REG_TAIL);
            __m512 h_hi = _mm512_load_ps(coeffs + block1_offset_tail + k * FLOATS_PER_REG_TAIL);
            acc_lo = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW), h_lo, acc_lo);
            acc_hi = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW + 16), h_hi, acc_hi);
        }

        // Horizontal FFT-16 on single row
        fft16::fft16_interleaved_reg(acc_lo, acc_hi);

        // Scatter to channel buffers
        alignas(64) float out_arr[32];
        _mm512_store_ps(out_arr, acc_lo);
        _mm512_store_ps(out_arr + 16, acc_hi);

        for (std::size_t ch = 0; ch < 16; ++ch) {
            if (channel_ptrs[ch]) {
                channel_ptrs[ch][t] = std::complex<float>(out_arr[ch * 2], out_arr[ch * 2 + 1]);
            }
        }
    }
}

// =============================================================================
// M=32 Hybrid: Filter+FFT in Registers, Transpose via L1 Scratch
// =============================================================================
// For M=32, we can't keep 8 time rows in registers (would need 32 ZMMs), so we
// use a hybrid approach:
//   1. Filter 4 rows × 32 channels → 16 ZMM accumulators
//   2. FFT-32 each row (reuse 4 ZMMs) → store to L1 scratch
//   3. Transpose 4×32 scratch → 32 channel buffers (two 4×16 transposes)
//
// This eliminates one memory round-trip vs fully staged:
//   - Staged: input → filter → scratch → FFT → scratch → transpose → channels
//   - Hybrid: input → filter/FFT (in regs) → scratch → transpose → channels
//
// Register budget:
//   Filter: 16 accumulators (4 rows × 4 ZMMs) + 4 coeff loads = 20 ZMMs
//   FFT-32: 4 data + ~8 temp = 12 ZMMs (per row, sequential)
//   Transpose: done from L1 scratch, minimal registers

// Helper: transpose 4 rows × 32 channels from scratch to channel buffers
// Uses two 4×16 transpose passes (channels 0-15, then 16-31)
[[gnu::always_inline, gnu::target("avx512f,avx512dq")]]
inline void transpose_4x32_scratch_to_channels(
    const std::complex<float>* scratch,  // 4×32 row-major layout
    std::complex<float>* const* channel_ptrs,
    std::size_t offset
) noexcept
{
    // Scratch layout: [row0: ch0..ch31][row1: ch0..ch31][row2: ch0..ch31][row3: ch0..ch31]
    const float* s = reinterpret_cast<const float*>(scratch);
    constexpr std::size_t FLOATS_PER_ROW = 32 * 2;  // 32 complex = 64 floats

    // Load all 4 rows (8 ZMMs total - 2 per row for 32 channels)
    // Row N: channels 0-7 in rN_0, channels 8-15 in rN_1, channels 16-23 in rN_2, channels 24-31 in rN_3
    __m512 r0_0 = _mm512_loadu_ps(s + 0 * FLOATS_PER_ROW + 0);
    __m512 r0_1 = _mm512_loadu_ps(s + 0 * FLOATS_PER_ROW + 16);
    __m512 r0_2 = _mm512_loadu_ps(s + 0 * FLOATS_PER_ROW + 32);
    __m512 r0_3 = _mm512_loadu_ps(s + 0 * FLOATS_PER_ROW + 48);
    __m512 r1_0 = _mm512_loadu_ps(s + 1 * FLOATS_PER_ROW + 0);
    __m512 r1_1 = _mm512_loadu_ps(s + 1 * FLOATS_PER_ROW + 16);
    __m512 r1_2 = _mm512_loadu_ps(s + 1 * FLOATS_PER_ROW + 32);
    __m512 r1_3 = _mm512_loadu_ps(s + 1 * FLOATS_PER_ROW + 48);
    __m512 r2_0 = _mm512_loadu_ps(s + 2 * FLOATS_PER_ROW + 0);
    __m512 r2_1 = _mm512_loadu_ps(s + 2 * FLOATS_PER_ROW + 16);
    __m512 r2_2 = _mm512_loadu_ps(s + 2 * FLOATS_PER_ROW + 32);
    __m512 r2_3 = _mm512_loadu_ps(s + 2 * FLOATS_PER_ROW + 48);
    __m512 r3_0 = _mm512_loadu_ps(s + 3 * FLOATS_PER_ROW + 0);
    __m512 r3_1 = _mm512_loadu_ps(s + 3 * FLOATS_PER_ROW + 16);
    __m512 r3_2 = _mm512_loadu_ps(s + 3 * FLOATS_PER_ROW + 32);
    __m512 r3_3 = _mm512_loadu_ps(s + 3 * FLOATS_PER_ROW + 48);

    // Transpose channels 0-15 (using existing 4x16 transpose helper)
    transpose_4x16_to_channels(r0_0, r0_1, r1_0, r1_1, r2_0, r2_1, r3_0, r3_1,
                                channel_ptrs, offset);

    // Transpose channels 16-31
    transpose_4x16_to_channels(r0_2, r0_3, r1_2, r1_3, r2_2, r2_3, r3_2, r3_3,
                                channel_ptrs + 16, offset);
}

// Main M=32 hybrid kernel
[[gnu::target("avx512f,avx512vl,avx512dq,fma")]]
inline void filter_fft_transpose_32ch_hybrid(
    const std::complex<float>* input,
    const float* coeffs,             // Blocked layout (4 blocks × K × 16 floats)
    std::complex<float>* const* channel_ptrs,  // 32 output pointers
    std::complex<float>* scratch,    // 4×32 = 128 complex L1 scratch buffer
    std::size_t num_outputs,
    std::size_t num_taps
) noexcept
{
    constexpr std::size_t M = 32;
    constexpr std::size_t FLOATS_PER_ROW = M * 2;  // 64 floats per row
    constexpr std::size_t FLOATS_PER_REG = 16;
    constexpr std::size_t TILE_ROWS = 4;  // Process 4 time outputs per tile

    const float* x_base = reinterpret_cast<const float*>(input);
    float* scratch_f = reinterpret_cast<float*>(scratch);

    // Block offsets for blocked coefficient layout (4 blocks of 8 channels each)
    const std::size_t block_stride = num_taps * FLOATS_PER_REG;

    std::size_t t = 0;

    // Main loop: 4 outputs at a time
    for (; t + TILE_ROWS <= num_outputs; t += TILE_ROWS) {
        // =================================================================
        // Stage 1: Filter 4 rows × 32 channels → 16 ZMM accumulators
        // Row i: (acc_i_0, acc_i_1, acc_i_2, acc_i_3) for channels 0-7, 8-15, 16-23, 24-31
        // =================================================================
        __m512 acc0_0 = _mm512_setzero_ps(), acc0_1 = _mm512_setzero_ps();
        __m512 acc0_2 = _mm512_setzero_ps(), acc0_3 = _mm512_setzero_ps();
        __m512 acc1_0 = _mm512_setzero_ps(), acc1_1 = _mm512_setzero_ps();
        __m512 acc1_2 = _mm512_setzero_ps(), acc1_3 = _mm512_setzero_ps();
        __m512 acc2_0 = _mm512_setzero_ps(), acc2_1 = _mm512_setzero_ps();
        __m512 acc2_2 = _mm512_setzero_ps(), acc2_3 = _mm512_setzero_ps();
        __m512 acc3_0 = _mm512_setzero_ps(), acc3_1 = _mm512_setzero_ps();
        __m512 acc3_2 = _mm512_setzero_ps(), acc3_3 = _mm512_setzero_ps();

        for (std::size_t k = 0; k < num_taps; ++k) {
            // Load coefficients from blocked layout (4 blocks)
            __m512 h0 = _mm512_load_ps(coeffs + 0 * block_stride + k * FLOATS_PER_REG);
            __m512 h1 = _mm512_load_ps(coeffs + 1 * block_stride + k * FLOATS_PER_REG);
            __m512 h2 = _mm512_load_ps(coeffs + 2 * block_stride + k * FLOATS_PER_REG);
            __m512 h3 = _mm512_load_ps(coeffs + 3 * block_stride + k * FLOATS_PER_REG);

            // Row 0
            acc0_0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0 + k) * FLOATS_PER_ROW + 0), h0, acc0_0);
            acc0_1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0 + k) * FLOATS_PER_ROW + 16), h1, acc0_1);
            acc0_2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0 + k) * FLOATS_PER_ROW + 32), h2, acc0_2);
            acc0_3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 0 + k) * FLOATS_PER_ROW + 48), h3, acc0_3);

            // Row 1
            acc1_0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1 + k) * FLOATS_PER_ROW + 0), h0, acc1_0);
            acc1_1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1 + k) * FLOATS_PER_ROW + 16), h1, acc1_1);
            acc1_2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1 + k) * FLOATS_PER_ROW + 32), h2, acc1_2);
            acc1_3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 1 + k) * FLOATS_PER_ROW + 48), h3, acc1_3);

            // Row 2
            acc2_0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2 + k) * FLOATS_PER_ROW + 0), h0, acc2_0);
            acc2_1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2 + k) * FLOATS_PER_ROW + 16), h1, acc2_1);
            acc2_2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2 + k) * FLOATS_PER_ROW + 32), h2, acc2_2);
            acc2_3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 2 + k) * FLOATS_PER_ROW + 48), h3, acc2_3);

            // Row 3
            acc3_0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3 + k) * FLOATS_PER_ROW + 0), h0, acc3_0);
            acc3_1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3 + k) * FLOATS_PER_ROW + 16), h1, acc3_1);
            acc3_2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3 + k) * FLOATS_PER_ROW + 32), h2, acc3_2);
            acc3_3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + 3 + k) * FLOATS_PER_ROW + 48), h3, acc3_3);
        }

        // =================================================================
        // Stage 2: FFT-32 on each row, store to scratch
        // Process one row at a time to limit register pressure
        // =================================================================

        // Row 0: FFT-32 → scratch row 0
        fft32::fft32_interleaved_reg(acc0_0, acc0_1, acc0_2, acc0_3);
        _mm512_storeu_ps(scratch_f + 0 * FLOATS_PER_ROW + 0, acc0_0);
        _mm512_storeu_ps(scratch_f + 0 * FLOATS_PER_ROW + 16, acc0_1);
        _mm512_storeu_ps(scratch_f + 0 * FLOATS_PER_ROW + 32, acc0_2);
        _mm512_storeu_ps(scratch_f + 0 * FLOATS_PER_ROW + 48, acc0_3);

        // Row 1: FFT-32 → scratch row 1
        fft32::fft32_interleaved_reg(acc1_0, acc1_1, acc1_2, acc1_3);
        _mm512_storeu_ps(scratch_f + 1 * FLOATS_PER_ROW + 0, acc1_0);
        _mm512_storeu_ps(scratch_f + 1 * FLOATS_PER_ROW + 16, acc1_1);
        _mm512_storeu_ps(scratch_f + 1 * FLOATS_PER_ROW + 32, acc1_2);
        _mm512_storeu_ps(scratch_f + 1 * FLOATS_PER_ROW + 48, acc1_3);

        // Row 2: FFT-32 → scratch row 2
        fft32::fft32_interleaved_reg(acc2_0, acc2_1, acc2_2, acc2_3);
        _mm512_storeu_ps(scratch_f + 2 * FLOATS_PER_ROW + 0, acc2_0);
        _mm512_storeu_ps(scratch_f + 2 * FLOATS_PER_ROW + 16, acc2_1);
        _mm512_storeu_ps(scratch_f + 2 * FLOATS_PER_ROW + 32, acc2_2);
        _mm512_storeu_ps(scratch_f + 2 * FLOATS_PER_ROW + 48, acc2_3);

        // Row 3: FFT-32 → scratch row 3
        fft32::fft32_interleaved_reg(acc3_0, acc3_1, acc3_2, acc3_3);
        _mm512_storeu_ps(scratch_f + 3 * FLOATS_PER_ROW + 0, acc3_0);
        _mm512_storeu_ps(scratch_f + 3 * FLOATS_PER_ROW + 16, acc3_1);
        _mm512_storeu_ps(scratch_f + 3 * FLOATS_PER_ROW + 32, acc3_2);
        _mm512_storeu_ps(scratch_f + 3 * FLOATS_PER_ROW + 48, acc3_3);

        // =================================================================
        // Stage 3: Transpose 4×32 scratch → 32 channel buffers
        // =================================================================
        transpose_4x32_scratch_to_channels(scratch, channel_ptrs, t);
    }

    // =====================================================================
    // Tail: remaining 1-3 outputs - scalar fallback
    // =====================================================================
    for (; t < num_outputs; ++t) {
        // Filter single row
        __m512 acc_0 = _mm512_setzero_ps();
        __m512 acc_1 = _mm512_setzero_ps();
        __m512 acc_2 = _mm512_setzero_ps();
        __m512 acc_3 = _mm512_setzero_ps();

        for (std::size_t k = 0; k < num_taps; ++k) {
            __m512 h0 = _mm512_load_ps(coeffs + 0 * block_stride + k * FLOATS_PER_REG);
            __m512 h1 = _mm512_load_ps(coeffs + 1 * block_stride + k * FLOATS_PER_REG);
            __m512 h2 = _mm512_load_ps(coeffs + 2 * block_stride + k * FLOATS_PER_REG);
            __m512 h3 = _mm512_load_ps(coeffs + 3 * block_stride + k * FLOATS_PER_REG);

            acc_0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW + 0), h0, acc_0);
            acc_1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW + 16), h1, acc_1);
            acc_2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW + 32), h2, acc_2);
            acc_3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_base + (t + k) * FLOATS_PER_ROW + 48), h3, acc_3);
        }

        // FFT-32
        fft32::fft32_interleaved_reg(acc_0, acc_1, acc_2, acc_3);

        // Scalar scatter to channels
        alignas(64) float out_arr[64];
        _mm512_store_ps(out_arr + 0, acc_0);
        _mm512_store_ps(out_arr + 16, acc_1);
        _mm512_store_ps(out_arr + 32, acc_2);
        _mm512_store_ps(out_arr + 48, acc_3);

        for (std::size_t ch = 0; ch < 32; ++ch) {
            if (channel_ptrs[ch]) {
                channel_ptrs[ch][t] = std::complex<float>(out_arr[ch * 2], out_arr[ch * 2 + 1]);
            }
        }
    }
}

// =============================================================================
// Blocked Coefficient Filter Kernels
// =============================================================================
//
// Uses blocked coefficient layout where taps are contiguous within each block:
//   coeffs[block * K * 16 + tap * 16]
//
// Memory access pattern:
//   - Coefficient stride between taps: 16 floats (LINEAR! prefetcher optimal)
//   - Data stride between taps: M samples (unavoidable due to polyphase structure)
//
// Main API:
//   filter_blocked() - Adaptive dispatcher, picks optimal strategy based on M
//
// Implementation details:
//   filter_blocked_impl() - Core AVX-512 kernel for all channels
//   filter_blocked_channel_tiled() - Channel-tiled variant for large M (>= 256)
//

namespace blocked {

constexpr std::size_t BLOCK_WIDTH = 8;
constexpr std::size_t FLOATS_PER_REG = 16;
constexpr std::size_t CHANNEL_TILE_THRESHOLD = 256;  // M >= this triggers channel tiling
constexpr std::size_t CHANNEL_TILE_SIZE = 128;       // Process 128 channels per slab

// Core AVX-512 implementation for blocked coefficient filter
[[gnu::target("avx512f")]]
inline void filter_blocked_impl(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,          // Blocked layout
    std::complex<float>* __restrict output,
    std::size_t M,
    std::size_t num_outputs,
    std::size_t num_taps
) {
    const std::size_t floats_per_row = M * 2;
    const std::size_t num_blocks = M / BLOCK_WIDTH;
    const std::size_t block_coeff_stride = num_taps * FLOATS_PER_REG;

    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

    // Main loop: 8x time unroll
    for (; t + 8 <= num_outputs; t += 8) {
        // Process each block of 8 channels
        for (std::size_t block = 0; block < num_blocks; ++block) {
            // Coefficient pointer for this block (taps are contiguous!)
            const float* h_ptr = coeffs + block * block_coeff_stride;

            // Phase group offset in data
            const std::size_t pg_offset = block * FLOATS_PER_REG;

            // Initialize 8 accumulators (one per time step)
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps();
            __m512 acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps();
            __m512 acc7 = _mm512_setzero_ps();

            // Inner loop over taps - LINEAR coefficient access!
            for (std::size_t k = 0; k < num_taps; ++k) {
                // Load coefficients - stride 16, LINEAR!
                __m512 h = _mm512_load_ps(h_ptr);
                h_ptr += FLOATS_PER_REG;  // Next tap is 16 floats away

                // Load data for each time step (strided by M*2 per tap)
                const float* x_tap = x_base + (t + k) * floats_per_row + pg_offset;

                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 0 * floats_per_row), h, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 1 * floats_per_row), h, acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 2 * floats_per_row), h, acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 3 * floats_per_row), h, acc3);
                acc4 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 4 * floats_per_row), h, acc4);
                acc5 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 5 * floats_per_row), h, acc5);
                acc6 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 6 * floats_per_row), h, acc6);
                acc7 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 7 * floats_per_row), h, acc7);
            }

            // Store results
            _mm512_storeu_ps(out_base + (t + 0) * floats_per_row + pg_offset, acc0);
            _mm512_storeu_ps(out_base + (t + 1) * floats_per_row + pg_offset, acc1);
            _mm512_storeu_ps(out_base + (t + 2) * floats_per_row + pg_offset, acc2);
            _mm512_storeu_ps(out_base + (t + 3) * floats_per_row + pg_offset, acc3);
            _mm512_storeu_ps(out_base + (t + 4) * floats_per_row + pg_offset, acc4);
            _mm512_storeu_ps(out_base + (t + 5) * floats_per_row + pg_offset, acc5);
            _mm512_storeu_ps(out_base + (t + 6) * floats_per_row + pg_offset, acc6);
            _mm512_storeu_ps(out_base + (t + 7) * floats_per_row + pg_offset, acc7);
        }
    }

    // Tail: remaining 1-7 outputs
    for (; t < num_outputs; ++t) {
        for (std::size_t block = 0; block < num_blocks; ++block) {
            const float* h_ptr = coeffs + block * block_coeff_stride;
            const std::size_t pg_offset = block * FLOATS_PER_REG;

            __m512 acc = _mm512_setzero_ps();

            for (std::size_t k = 0; k < num_taps; ++k) {
                __m512 h = _mm512_load_ps(h_ptr);
                h_ptr += FLOATS_PER_REG;

                __m512 x = _mm512_loadu_ps(x_base + (t + k) * floats_per_row + pg_offset);
                acc = _mm512_fmadd_ps(x, h, acc);
            }

            _mm512_storeu_ps(out_base + t * floats_per_row + pg_offset, acc);
        }
    }
}

// Channel-tiled filter for large M where history exceeds L1
// Processes a SUBSET of channels while respecting the full row stride.
//
// Use case: M=512, K=16 → History = 64KB (2x L1)
//   Split into 4 slabs of 128 channels each → History = 16KB per slab (fits L1)
[[gnu::target("avx512f")]]
inline void filter_blocked_channel_tiled(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,          // Coefficients for THIS slab only
    std::complex<float>* __restrict output,
    std::size_t M_process,   // Channels to process (e.g., 128)
    std::size_t M_stride,    // Full row stride (e.g., 512)
    std::size_t num_outputs,
    std::size_t num_taps
) {
    const std::size_t floats_per_row = M_stride * 2;  // Stride based on FULL M
    const std::size_t num_blocks = M_process / BLOCK_WIDTH;
    const std::size_t block_coeff_stride = num_taps * FLOATS_PER_REG;

    const float* x_base = reinterpret_cast<const float*>(input);
    float* out_base = reinterpret_cast<float*>(output);

    std::size_t t = 0;

    // Main loop: 8x time unroll
    for (; t + 8 <= num_outputs; t += 8) {
        for (std::size_t block = 0; block < num_blocks; ++block) {
            const float* h_ptr = coeffs + block * block_coeff_stride;
            const std::size_t pg_offset = block * FLOATS_PER_REG;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps();
            __m512 acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps();
            __m512 acc7 = _mm512_setzero_ps();

            for (std::size_t k = 0; k < num_taps; ++k) {
                __m512 h = _mm512_loadu_ps(h_ptr);
                h_ptr += FLOATS_PER_REG;

                const float* x_tap = x_base + (t + k) * floats_per_row + pg_offset;

                acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 0 * floats_per_row), h, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 1 * floats_per_row), h, acc1);
                acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 2 * floats_per_row), h, acc2);
                acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 3 * floats_per_row), h, acc3);
                acc4 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 4 * floats_per_row), h, acc4);
                acc5 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 5 * floats_per_row), h, acc5);
                acc6 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 6 * floats_per_row), h, acc6);
                acc7 = _mm512_fmadd_ps(_mm512_loadu_ps(x_tap + 7 * floats_per_row), h, acc7);
            }

            _mm512_storeu_ps(out_base + (t + 0) * floats_per_row + pg_offset, acc0);
            _mm512_storeu_ps(out_base + (t + 1) * floats_per_row + pg_offset, acc1);
            _mm512_storeu_ps(out_base + (t + 2) * floats_per_row + pg_offset, acc2);
            _mm512_storeu_ps(out_base + (t + 3) * floats_per_row + pg_offset, acc3);
            _mm512_storeu_ps(out_base + (t + 4) * floats_per_row + pg_offset, acc4);
            _mm512_storeu_ps(out_base + (t + 5) * floats_per_row + pg_offset, acc5);
            _mm512_storeu_ps(out_base + (t + 6) * floats_per_row + pg_offset, acc6);
            _mm512_storeu_ps(out_base + (t + 7) * floats_per_row + pg_offset, acc7);
        }
    }

    // Tail: remaining 1-7 outputs
    for (; t < num_outputs; ++t) {
        for (std::size_t block = 0; block < num_blocks; ++block) {
            const float* h_ptr = coeffs + block * block_coeff_stride;
            const std::size_t pg_offset = block * FLOATS_PER_REG;

            __m512 acc = _mm512_setzero_ps();

            for (std::size_t k = 0; k < num_taps; ++k) {
                __m512 h = _mm512_loadu_ps(h_ptr);
                h_ptr += FLOATS_PER_REG;

                __m512 x = _mm512_loadu_ps(x_base + (t + k) * floats_per_row + pg_offset);
                acc = _mm512_fmadd_ps(x, h, acc);
            }

            _mm512_storeu_ps(out_base + t * floats_per_row + pg_offset, acc);
        }
    }
}

} // namespace blocked

// =============================================================================
// Main Blocked Filter API
// =============================================================================
// Adaptive dispatcher that picks optimal strategy based on M and hardware.
// This is the primary entry point for blocked coefficient filtering.

inline void filter_blocked(
    const std::complex<float>* __restrict input,
    const float* __restrict coeffs,
    std::complex<float>* __restrict output,
    std::size_t M,
    std::size_t num_outputs,
    std::size_t num_taps
) {
    // AVX-512 is required for this kernel
    // Component validates AVX-512 support at configure time

    if (M < blocked::CHANNEL_TILE_THRESHOLD) {
        // Small/medium M: process all channels at once (fits L1)
        blocked::filter_blocked_impl(input, coeffs, output, M, num_outputs, num_taps);
    } else {
        // Large M: tile by channels to keep history working set in L1
        const std::size_t floats_per_block_coeffs = num_taps * blocked::FLOATS_PER_REG;

        for (std::size_t ch_start = 0; ch_start < M; ch_start += blocked::CHANNEL_TILE_SIZE) {
            const std::size_t ch_count = std::min(blocked::CHANNEL_TILE_SIZE, M - ch_start);

            const auto* slab_input = input + ch_start;
            auto* slab_output = output + ch_start;
            const float* slab_coeffs = coeffs + (ch_start / blocked::BLOCK_WIDTH) * floats_per_block_coeffs;

            blocked::filter_blocked_channel_tiled(
                slab_input,
                slab_coeffs,
                slab_output,
                ch_count,
                M,
                num_outputs,
                num_taps
            );
        }
    }
}

} // namespace kernels
