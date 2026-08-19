/*
 * Copyright (C) 2024-2025 Geon Technologies, LLC
 *
 * This file is part of composite-comps.
 *
 * composite-comps is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * composite-comps is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include <composite/buffers/buffer.hpp>
#include <composite/buffers/aligned_mem.hpp>

#include <atomic>
#include <complex>
#include <immintrin.h>
#include <limits>
#include <numeric>
#include "simd_fmv.hpp"

/*
 * =====================================================================================
 * PSD Calculation Algorithm
 * =====================================================================================
 * This class calculates the Power Spectral Density (PSD) of a complex signal and
 * returns the result in decibels (dB). The calculation follows these main steps:
 *
 * 1.  **Power Calculation**:
 * For each complex input sample `z = r + i*j`, the instantaneous power is
 * calculated as the squared magnitude: `Power = r² + i²`.
 *
 * 2.  **Normalization**:
 * The calculated power is multiplied by a user-provided normalization constant:
 * `Normalized_Power = Power * norm_const`.
 *
 * 3.  **Logarithmic Scaling (dB)**:
 * The final PSD value in dB is calculated using the formula:
 * `PSD_dB = 10 * log10(Normalized_Power)`.
 * It is implemented as:
 * `PSD_dB = (10 / log2(10)) * log2(Normalized_Power)
 *
 * -------------------------------------------------------------------------------------
 * SIMD Implementation Details (AVX2 & AVX-512)
 * -------------------------------------------------------------------------------------
 * The SIMD versions are heavily optimized to perform these calculations in parallel.
 *
 * A. **Data Handling (De-interleaving)**:
 * The input data is interleaved complex values `[r0, i0, r1, i1, ...]`. To
 * process this efficiently, a `shuffle -> compute -> permute` pattern is used:
 * - **Load & Shuffle**: Two contiguous blocks of data are loaded. Fast in-lane
 * `shuffle` instructions are used to separate the real and imaginary components
 * into two separate vectors. The elements in these vectors are now in a
 * "scrambled" (non-sequential) order.
 * - **Compute**: The power mathematical operationsa re performed on
 * these scrambled vectors. Since the operations are element-wise, the
 * scrambled order does not affect the correctness of the intermediate results.
 * - **Permute**: After all calculations are complete, a final cross-lane
 * `permute` instruction unscrambles the final result vector back into
 * sequential order just before it is stored in the output array.
 *
 * B. **Fast `log2` Approximation**:
 * To avoid the high latency of `log10` or `log2` instructions, a fast polynomial
 * approximation is used to calculate `log2(x)`. This result is then scaled to
 * get `10 * log10(x)`. The approximation involves:
 * - **Decomposition**: The input `x` is bit-manipulated and decomposed into its
 * floating-point components `x = m * 2^e`, where `m` is the mantissa in the
 * range `[1.0, 2.0)` and `e` is the integer exponent. This uses the identity:
 * `log2(x) = log2(m) + e`.
 * - **Range Reduction**: To improve polynomial accuracy, the mantissa `m` is
 * mapped to a variable `z` in a smaller input range `[-1/3, 1/3]` using the
 * formula `z = (m - 1) / (m + 1)`.
 * - **Polynomial Evaluation**: A polynomial `P(w)` where `w = z*z` is evaluated
 * to approximate `log2(m) / z`. Estrin's method is used to structure the
 * polynomial evaluation for maximum instruction-level parallelism.
 * - **Recomposition**: The final `log2(x)` result is constructed by combining
 * the polynomial result with the exponent `e`.
 * - **Special Values**: Inputs of `0`, `inf`, `NaN`, and negative numbers are
 * explicitly handled using masks to produce the correct corresponding outputs
 * (`-inf`, `inf`, `NaN`).
 */

template <typename T>
class work {};

template <>
class work<float> {
    using cplx_data_type = composite::immutable_buffer<std::complex<float>>;
    using real_data_type = composite::mutable_buffer<float>;
    static constexpr std::size_t ALIGNMENT = 64;
    static constexpr auto STRIDE_256 = std::size_t{256u / 8u / sizeof(float)};
    static constexpr auto STRIDE_512 = std::size_t{512u / 8u / sizeof(float)};
    static constexpr auto log_const = 3.010299956639812f; // 10.f / std::log2f(10.f);
    // Pre-calculated coefficients for the polynomial P(w)
    // C_k = 2.0f / (log(2.0f) * (2*k + 1))
    static constexpr auto C0 = 2.88539008178f;
    static constexpr auto C1 = 0.96179669392f;
    static constexpr auto C2 = 0.57707801635f;
    static constexpr auto C3 = 0.41219858311f;
    static constexpr auto C4 = 0.32059889797f;
    static constexpr auto C5 = 0.26230818925f;
    static constexpr auto C6 = 0.22195308321f;
public:
    work() = default;
    explicit work(float normalization_const) : m_norm_const(normalization_const) {}

    auto norm_const() const noexcept -> float {
        return m_norm_const;
    }

    auto norm_const(float val) -> void {
        m_norm_const = val;
    }

    COMPS_FMV_DEFAULT
    auto process(const cplx_data_type& data) -> real_data_type {
        // Make output data
        auto psd = composite::make_aligned_buffer_uninitialized<float>(ALIGNMENT, data.size());
        // Process data
        for (auto i = 0u; i < data.size(); ++i) {
            const auto& val = data[i];
            const auto power = val.real() * val.real() + val.imag() * val.imag();
            psd[i] = log_const * std::log2f(m_norm_const * power);
        }
        return psd;
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx512f")]]
    auto process(const cplx_data_type& data) -> real_data_type {
        // Constant registers
        static const auto unscramble_idx = _mm512_set_epi32(15,14,11,10,7,6,3,2,13,12,9,8,5,4,1,0);
        static const auto v_log_const = _mm512_set1_ps(log_const);
        static const auto v_c0 = _mm512_set1_ps(C0);
        static const auto v_c1 = _mm512_set1_ps(C1);
        static const auto v_c2 = _mm512_set1_ps(C2);
        static const auto v_c3 = _mm512_set1_ps(C3);
        static const auto v_c4 = _mm512_set1_ps(C4);
        static const auto v_c5 = _mm512_set1_ps(C5);
        static const auto v_c6 = _mm512_set1_ps(C6);
        static const auto v_zero = _mm512_setzero_ps();
        static const auto v_one = _mm512_set1_ps(1.0f);
        static const auto v_pos_inf = _mm512_set1_ps(std::numeric_limits<float>::infinity());
        static const auto v_neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        static const auto v_nan = _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN());
        const auto v_norm_const = _mm512_set1_ps(m_norm_const);
        const auto norm_const_scalar = m_norm_const;

        // Make output data
        auto psd = composite::make_aligned_buffer_uninitialized<float>(ALIGNMENT, data.size());

        // Process data in SIMD chunks
        const auto simd_end = data.size() - (data.size() % STRIDE_512);
        std::size_t i = 0;
        for (; i < simd_end; i += STRIDE_512) {
            // Load 16 complex numbers = 32 floats from data, starting at index i
            auto a = _mm512_loadu_ps(reinterpret_cast<const float*>(data.data() + i));                   // [r0, i0, r1, i1, r2, i2, r3, i3, r4, i4, r5, i5, r6, i6, r7, i7]
            auto b = _mm512_loadu_ps(reinterpret_cast<const float*>(data.data() + i + STRIDE_512 / 2u)); // [r8, i8, r9, i9, r10, i10, r11, i11, r12, i12, r13, i13, r14, i14, r15, i15]
            // Shuffle to split real/imag
            auto real = _mm512_shuffle_ps(a, b, 0x88); // real: [r0, r1, r8, r9, r2, r3, r10, r11, r4, r5, r12, r13, r6, r7, r14, r15]
            auto imag = _mm512_shuffle_ps(a, b, 0xDD); // imag: [i0, i1, i8, i9, i2, i3, i10, i11, i4, i5, i12, i13, i6, i7, i14, i15]

            // --- Calculate power ---
            // Square reals
            auto vdata = _mm512_mul_ps(real, real);
            // Square imags and add to squared reals to get power
            vdata = _mm512_fmadd_ps(imag, imag, vdata);
            // Multiply by normalization contstant
            vdata = _mm512_mul_ps(vdata, v_norm_const);
            // Reorder the data
            vdata = _mm512_permutexvar_ps(unscramble_idx, vdata);

            // --- log2 implementation ---
            // 1) Decompose values into m and e
            // Reinterpret the float vector `vdata` as a vector of 32-bit integers (`vdatai`).
            auto vdatai = _mm512_castps_si512(vdata);
            // Extract the integer exponent `e`
            auto ve_int = _mm512_sub_epi32(_mm512_srli_epi32(vdatai, 23), _mm512_set1_epi32(127));
            // Convert the integer exponent vector `ve_int` to a floating-point vector `ve`
            auto ve = _mm512_cvtepi32_ps(ve_int);
            // Isolate the 23 bits of the mantissa
            auto mant_bits = _mm512_and_si512(vdatai, _mm512_set1_epi32(0x007FFFFF));
            // Construct the mantissa `m` in the range `[1.0, 2.0)`
            auto vm = _mm512_castsi512_ps(_mm512_or_si512(mant_bits, _mm512_set1_epi32(0x3F800000)));
            // Range reduction: m -> z
            auto vz = _mm512_div_ps(_mm512_sub_ps(vm, v_one), _mm512_add_ps(vm, v_one));
            // 2) Evaluate Polynomial P(w) using Estrin's Method
            auto vw = _mm512_mul_ps(vz, vz); // w = z*z
            // Estrin Step 1: Calculate independent pairs in parallel
            auto p01 = _mm512_fmadd_ps(v_c1, vw, v_c0); // C0 + C1*w
            auto p23 = _mm512_fmadd_ps(v_c3, vw, v_c2); // C2 + C3*w
            auto p45 = _mm512_fmadd_ps(v_c5, vw, v_c4); // C4 + C5*w
            // Estrin Step 2: Combine pairs
            auto vw2 = _mm512_mul_ps(vw, vw); // w^2
            auto p03 = _mm512_fmadd_ps(p23, vw2, p01); // p01 + p23*w^2
            auto p46 = _mm512_fmadd_ps(v_c6, vw2, p45); // p45 + C6*w^2
            // Estrin Step 3: Final combination
            auto vw4 = _mm512_mul_ps(vw2, vw2); // w^4
            auto poly = _mm512_fmadd_ps(p46, vw4, p03);
            // Final combination & blending
            auto v_log2_m = _mm512_mul_ps(vz, poly);
            auto v_poly_res = _mm512_add_ps(v_log2_m, ve);
            // 3) Handle special values
            const auto zero_mask = _mm512_cmpeq_ps_mask(vdata, v_zero);
            const auto neg_mask = _mm512_cmplt_ps_mask(vdata, v_zero);
            const auto nan_mask = _mm512_cmp_ps_mask(vdata, vdata, _CMP_UNORD_Q);
            const auto inf_mask = _mm512_cmpeq_ps_mask(vdata, v_pos_inf);
            const auto domain_error_mask = _mm512_kor(neg_mask, nan_mask);
            auto v_res = v_poly_res;
            v_res = _mm512_mask_blend_ps(inf_mask, v_res, v_pos_inf);
            v_res = _mm512_mask_blend_ps(zero_mask, v_res, v_neg_inf);
            v_res = _mm512_mask_blend_ps(domain_error_mask, v_res, v_nan);

            // --- Apply constant for log2 -> 10*log10 ---
            v_res = _mm512_mul_ps(v_res, v_log_const);

            // Store result into psd
            _mm512_storeu_ps(psd.data() + i, v_res);
        }

        // Handle remainder with scalar code
        for (; i < data.size(); ++i) {
            const auto& val = data[i];
            const auto power = val.real() * val.real() + val.imag() * val.imag();
            psd[i] = log_const * std::log2f(norm_const_scalar * power);
        }

        return psd;
    }
#endif

#if COMPS_FMV_ENABLED
    [[gnu::target("avx2,fma")]]
    auto process(const cplx_data_type& data) -> real_data_type {
        // Constant registers
        static const auto unscramble_idx_256 = _mm256_set_epi32(7,6,3,2,5,4,1,0);
        static const auto v_log_const_256 = _mm256_set1_ps(log_const);
        static const auto v_c0_256 = _mm256_set1_ps(C0);
        static const auto v_c1_256 = _mm256_set1_ps(C1);
        static const auto v_c2_256 = _mm256_set1_ps(C2);
        static const auto v_c3_256 = _mm256_set1_ps(C3);
        static const auto v_c4_256 = _mm256_set1_ps(C4);
        static const auto v_c5_256 = _mm256_set1_ps(C5);
        static const auto v_c6_256 = _mm256_set1_ps(C6);
        static const auto v_zero_256 = _mm256_setzero_ps();
        static const auto v_one_256 = _mm256_set1_ps(1.0f);
        static const auto v_pos_inf_256 = _mm256_set1_ps(std::numeric_limits<float>::infinity());
        static const auto v_neg_inf_256 = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        static const auto v_nan_256 = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
        const auto v_norm_const_256 = _mm256_set1_ps(m_norm_const);
        const auto norm_const_scalar = m_norm_const;

        // Make output data
        auto psd = composite::make_aligned_buffer_uninitialized<float>(ALIGNMENT, data.size());

        // Process data in SIMD chunks
        const auto simd_end = data.size() - (data.size() % STRIDE_256);
        std::size_t i = 0;
        for (; i < simd_end; i += STRIDE_256) {
            // Load 8 complex numbers = 16 floats from data, starting at index i
            auto a = _mm256_loadu_ps(reinterpret_cast<const float*>(data.data() + i));                   // [r0, i0, r1, i1, r2, i2, r3, i3]
            auto b = _mm256_loadu_ps(reinterpret_cast<const float*>(data.data() + i + STRIDE_256 / 2u)); // [r4, i4, r5, i5, r6, i6, r7, i7]
            // Shuffle to split real/imag
            auto real = _mm256_shuffle_ps(a, b, 0x88); // real: [r0, r1, r4, r5, r2, r3, r6, r7]
            auto imag = _mm256_shuffle_ps(a, b, 0xDD); // imag: [i0, i1, i4, i5, i2, i3, i6, i7]

            // --- Calculate power (see AVX-512 version above for more detailed comments) ---
            auto vdata = _mm256_mul_ps(real, real);
            vdata = _mm256_fmadd_ps(imag, imag, vdata);
            vdata = _mm256_mul_ps(vdata, v_norm_const_256);
            vdata = _mm256_permutexvar_ps(unscramble_idx_256, vdata);

            // --- log2 implementation (see AVX-512 version above for steps) ---
            auto vdatai = _mm256_castps_si256(vdata);
            auto ve_int = _mm256_sub_epi32(_mm256_srli_epi32(vdatai, 23), _mm256_set1_epi32(127));
            auto ve = _mm256_cvtepi32_ps(ve_int);
            auto mant_bits = _mm256_and_si256(vdatai, _mm256_set1_epi32(0x007FFFFF));
            auto vm = _mm256_castsi256_ps(_mm256_or_si256(mant_bits, _mm256_set1_epi32(0x3F800000)));
            auto vz = _mm256_div_ps(_mm256_sub_ps(vm, v_one_256), _mm256_add_ps(vm, v_one_256));
            auto vw = _mm256_mul_ps(vz, vz);
            auto p01 = _mm256_fmadd_ps(v_c1_256, vw, v_c0_256);
            auto p23 = _mm256_fmadd_ps(v_c3_256, vw, v_c2_256);
            auto p45 = _mm256_fmadd_ps(v_c5_256, vw, v_c4_256);
            auto vw2 = _mm256_mul_ps(vw, vw);
            auto p03 = _mm256_fmadd_ps(p23, vw2, p01);
            auto p46 = _mm256_fmadd_ps(v_c6_256, vw2, p45);
            auto vw4 = _mm256_mul_ps(vw2, vw2);
            auto poly = _mm256_fmadd_ps(p46, vw4, p03);
            auto v_log2_m = _mm256_mul_ps(vz, poly);
            auto v_poly_res = _mm256_add_ps(v_log2_m, ve);
            // Handle special values using blendv (mask-based blend is AVX-512 only)
            auto zero_mask = _mm256_cmp_ps(vdata, v_zero_256, _CMP_EQ_OQ);
            auto neg_mask = _mm256_cmp_ps(vdata, v_zero_256, _CMP_LT_OQ);
            auto nan_mask = _mm256_cmp_ps(vdata, vdata, _CMP_UNORD_Q);
            auto inf_mask = _mm256_cmp_ps(vdata, v_pos_inf_256, _CMP_EQ_OQ);
            auto domain_error_mask = _mm256_or_ps(neg_mask, nan_mask);
            auto v_res = v_poly_res;
            v_res = _mm256_blendv_ps(v_res, v_pos_inf_256, inf_mask);
            v_res = _mm256_blendv_ps(v_res, v_neg_inf_256, zero_mask);
            v_res = _mm256_blendv_ps(v_res, v_nan_256, domain_error_mask);

            // --- Apply constant for log2 -> 10*log10 ---
            v_res = _mm256_mul_ps(v_res, v_log_const_256);

            // Store result into psd
            _mm256_storeu_ps(psd.data() + i, v_res);
        }

        // Handle remainder with scalar code
        for (; i < data.size(); ++i) {
            const auto& val = data[i];
            const auto power = val.real() * val.real() + val.imag() * val.imag();
            psd[i] = log_const * std::log2f(norm_const_scalar * power);
        }

        return psd;
    }
#endif

private:
    float m_norm_const{1};

}; // class work<float>

template <>
class work<double> {
    using cplx_data_type = composite::immutable_buffer<std::complex<double>>;
    using real_data_type = composite::mutable_buffer<double>;
    static constexpr std::size_t ALIGNMENT = 64;
    static constexpr auto STRIDE_256 = std::size_t{256u / 8u / sizeof(double)};
    static constexpr auto STRIDE_512 = std::size_t{512u / 8u / sizeof(double)};
    static constexpr auto log_const = 3.010299956639812; // 10. / std::log2(10.);
    // Pre-calculated coefficients for the polynomial P(w)
    // C_k = 2.0 / (log(2.0) * (2*k + 1))
    static constexpr auto C0 = 2.88539008178;
    static constexpr auto C1 = 0.96179669392;
    static constexpr auto C2 = 0.57707801635;
    static constexpr auto C3 = 0.41219858311;
    static constexpr auto C4 = 0.32059889797;
    static constexpr auto C5 = 0.26230818925;
    static constexpr auto C6 = 0.22195308321;
public:
    work() = default;
    explicit work(double normalization_const) : m_norm_const(normalization_const) {}

    auto norm_const() const noexcept -> double {
        return m_norm_const;
    }

    auto norm_const(double val) -> void {
        m_norm_const = val;
    }

    COMPS_FMV_DEFAULT
    auto process(const cplx_data_type& data) -> real_data_type {
        // Make output data
        auto psd = composite::make_aligned_buffer_uninitialized<double>(ALIGNMENT, data.size());
        // Process data
        for (auto i = 0u; i < data.size(); ++i) {
            const auto& val = data[i];
            const auto power = val.real() * val.real() + val.imag() * val.imag();
            psd[i] = log_const * std::log2(m_norm_const * power);
        }
        return psd;
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx512f,avx512dq")]]
    auto process(const cplx_data_type& data) -> real_data_type {
        // Constant registers
        static const auto unscramble_idx = _mm512_set_epi64(7,5,3,1,6,4,2,0);
        static const auto v_log_const = _mm512_set1_pd(log_const);
        static const auto v_c0 = _mm512_set1_pd(C0);
        static const auto v_c1 = _mm512_set1_pd(C1);
        static const auto v_c2 = _mm512_set1_pd(C2);
        static const auto v_c3 = _mm512_set1_pd(C3);
        static const auto v_c4 = _mm512_set1_pd(C4);
        static const auto v_c5 = _mm512_set1_pd(C5);
        static const auto v_c6 = _mm512_set1_pd(C6);
        static const auto v_zero = _mm512_setzero_pd();
        static const auto v_one = _mm512_set1_pd(1.0);
        static const auto v_pos_inf = _mm512_set1_pd(std::numeric_limits<double>::infinity());
        static const auto v_neg_inf = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
        static const auto v_nan = _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN());
        const auto v_norm_const = _mm512_set1_pd(m_norm_const);
        const auto norm_const_scalar = m_norm_const;

        // Make output data
        auto psd = composite::make_aligned_buffer_uninitialized<double>(ALIGNMENT, data.size());

        // Process data in SIMD chunks
        const auto simd_end = data.size() - (data.size() % STRIDE_512);
        std::size_t i = 0;
        for (; i < simd_end; i += STRIDE_512) {
            // Load 8 complex numbers = 16 doubles from data, starting at index i
            auto a = _mm512_loadu_pd(reinterpret_cast<const double*>(data.data() + i));                   // [r0, i0, r1, i1, r2, i2, r3, i3]
            auto b = _mm512_loadu_pd(reinterpret_cast<const double*>(data.data() + i + STRIDE_512 / 2u)); // [r4, i4, r5, i5, r6, i6, r7, i7]
            // Shuffle to split real/imag
            auto real = _mm512_shuffle_pd(a, b, 0x00); // real: [r0, r4, r1, r5, r2, r6, r3, r7]
            auto imag = _mm512_shuffle_pd(a, b, 0xFF); // imag: [i0, i4, i1, i5, i2, i6, i3, i7]

            // --- Calculate power ---
            auto vdata = _mm512_mul_pd(real, real);
            vdata = _mm512_fmadd_pd(imag, imag, vdata);
            vdata = _mm512_mul_pd(vdata, v_norm_const);
            vdata = _mm512_permutexvar_pd(unscramble_idx, vdata);

            // --- log2 implementation ---
            auto vdatai = _mm512_castpd_si512(vdata);
            auto ve_int = _mm512_sub_epi64(_mm512_srli_epi64(vdatai, 52), _mm512_set1_epi64(1023));
            auto ve = _mm512_cvtepi64_pd(ve_int);
            auto mant_bits = _mm512_and_si512(vdatai, _mm512_set1_epi64(0x000FFFFFFFFFFFFFLL));
            auto vm = _mm512_castsi512_pd(_mm512_or_si512(mant_bits, _mm512_set1_epi64(0x3FF0000000000000LL)));
            auto vz = _mm512_div_pd(_mm512_sub_pd(vm, v_one), _mm512_add_pd(vm, v_one));
            auto vw = _mm512_mul_pd(vz, vz);
            auto p01 = _mm512_fmadd_pd(v_c1, vw, v_c0);
            auto p23 = _mm512_fmadd_pd(v_c3, vw, v_c2);
            auto p45 = _mm512_fmadd_pd(v_c5, vw, v_c4);
            auto vw2 = _mm512_mul_pd(vw, vw);
            auto p03 = _mm512_fmadd_pd(p23, vw2, p01);
            auto p46 = _mm512_fmadd_pd(v_c6, vw2, p45);
            auto vw4 = _mm512_mul_pd(vw2, vw2);
            auto poly = _mm512_fmadd_pd(p46, vw4, p03);
            auto v_log2_m = _mm512_mul_pd(vz, poly);
            auto v_poly_res = _mm512_add_pd(v_log2_m, ve);
            const auto zero_mask = _mm512_cmpeq_pd_mask(vdata, v_zero);
            const auto neg_mask = _mm512_cmplt_pd_mask(vdata, v_zero);
            const auto nan_mask = _mm512_cmp_pd_mask(vdata, vdata, _CMP_UNORD_Q);
            const auto inf_mask = _mm512_cmpeq_pd_mask(vdata, v_pos_inf);
            const auto domain_error_mask = _mm512_kor(neg_mask, nan_mask);
            auto v_res = v_poly_res;
            v_res = _mm512_mask_blend_pd(inf_mask, v_res, v_pos_inf);
            v_res = _mm512_mask_blend_pd(zero_mask, v_res, v_neg_inf);
            v_res = _mm512_mask_blend_pd(domain_error_mask, v_res, v_nan);

            // --- Apply constant for log2 -> 10*log10 ---
            v_res = _mm512_mul_pd(v_res, v_log_const);

            // Store result into psd
            _mm512_storeu_pd(psd.data() + i, v_res);
        }

        // Handle remainder with scalar code
        for (; i < data.size(); ++i) {
            const auto& val = data[i];
            const auto power = val.real() * val.real() + val.imag() * val.imag();
            psd[i] = log_const * std::log2(norm_const_scalar * power);
        }

        return psd;
    }
#endif

private:
    double m_norm_const{1};

}; // class work<double>
