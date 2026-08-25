/*
 * Copyright (C) 2024-2025 Geon Technologies, LLC
 *
 * This file is part of composite-comps.
 *
 * composite-comps is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your option) any later version.
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

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <stdexcept>
#include "simd_fmv.hpp"

template <typename T>
class work {};

template <>
class work<float> {
    static constexpr auto ALIGNMENT_AVX2 = std::size_t{32};
    static constexpr auto ALIGNMENT_AVX512 = std::size_t{64};
    static constexpr auto STRIDE_256 = std::size_t{8};   // 256 bits / 32 bits per float
    static constexpr auto STRIDE_512 = std::size_t{16};  // 512 bits / 32 bits per float
public:
    explicit work(float alpha) : m_alpha(alpha) {}

    auto process(composite::mutable_buffer<float>& curr_psd, composite::mutable_buffer<float>& prev_psd) const -> void {
        if (curr_psd.size() != prev_psd.size()) {
            throw std::runtime_error("buffer size mismatch in exp_smooth");
        }
        process_impl(curr_psd.data(), prev_psd.data(), curr_psd.size(), m_alpha);
    }

private:
    // Helper: scalar processing for any range. Sanitization must guard BOTH operands:
    // a non-finite prev (e.g. a -inf PSD bin adopted as the baseline frame) would otherwise
    // latch that bin non-finite FOREVER — alpha*x + (1-alpha)*(-inf) is -inf no matter how
    // the input recovers. prev bad + curr good re-seeds the bin from curr; both bad leaves
    // it unchanged (no finite information exists yet). NOTE: relies on std::isfinite — this
    // component must never opt into FAST_MATH (see CMakeLists.txt).
    static auto process_scalar_range(float* curr, float* prev, std::size_t start, std::size_t end, float alpha) -> void {
        const auto one_minus_alpha = 1.0f - alpha;
        for (auto i = start; i < end; ++i) {
            const auto seed = !std::isfinite(curr[i]) ? prev[i] : curr[i];
            curr[i] = std::isfinite(prev[i]) ? (alpha * seed) + (one_minus_alpha * prev[i]) : seed;
        }
    }

    // Helper: AVX2 vectorized loop (no alignment checking, caller ensures it)
    [[gnu::target("avx2,fma")]]
    static auto process_avx2_aligned(float* curr, float* prev, std::size_t size, float alpha) -> void {
        const auto alpha_vec = _mm256_set1_ps(alpha);
        const auto one_minus_alpha_vec = _mm256_set1_ps(1.0f - alpha);
        const auto inf_vec = _mm256_set1_ps(std::numeric_limits<float>::infinity());
        const auto abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        auto i = std::size_t{};
        for (; i + STRIDE_256 <= size; i += STRIDE_256) {
            auto curr_data = _mm256_load_ps(curr + i);
            auto prev_data = _mm256_load_ps(prev + i);

            auto nan_mask = _mm256_cmp_ps(curr_data, curr_data, _CMP_UNORD_Q);
            auto abs_curr = _mm256_and_ps(curr_data, abs_mask);
            auto inf_mask = _mm256_cmp_ps(abs_curr, inf_vec, _CMP_EQ_OQ);
            auto bad_curr = _mm256_or_ps(nan_mask, inf_mask);
            // prev must be guarded too: a non-finite baseline bin otherwise latches forever.
            auto nan_prev = _mm256_cmp_ps(prev_data, prev_data, _CMP_UNORD_Q);
            auto abs_prev = _mm256_and_ps(prev_data, abs_mask);
            auto inf_prev = _mm256_cmp_ps(abs_prev, inf_vec, _CMP_EQ_OQ);
            auto bad_prev = _mm256_or_ps(nan_prev, inf_prev);

            auto seed = _mm256_blendv_ps(curr_data, prev_data, bad_curr);
            auto result = _mm256_fmadd_ps(prev_data, one_minus_alpha_vec, _mm256_mul_ps(seed, alpha_vec));
            result = _mm256_blendv_ps(result, seed, bad_prev);  // prev bad: re-seed from curr
            _mm256_store_ps(curr + i, result);
        }

        // Remainder
        process_scalar_range(curr, prev, i, size, alpha);
    }

    // Helper: AVX-512 vectorized loop (no alignment checking, caller ensures it)
    [[gnu::target("avx512f,avx512dq")]]
    static auto process_avx512_aligned(float* curr, float* prev, std::size_t size, float alpha) -> void {
        const auto alpha_vec = _mm512_set1_ps(alpha);
        const auto one_minus_alpha_vec = _mm512_set1_ps(1.0f - alpha);
        const auto inf_vec = _mm512_set1_ps(std::numeric_limits<float>::infinity());
        const auto abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));

        auto i = std::size_t{};
        for (; i + STRIDE_512 <= size; i += STRIDE_512) {
            auto curr_data = _mm512_load_ps(curr + i);
            auto prev_data = _mm512_load_ps(prev + i);

            auto nan_mask = _mm512_cmp_ps_mask(curr_data, curr_data, _CMP_UNORD_Q);
            auto abs_curr = _mm512_and_ps(curr_data, abs_mask);
            auto inf_mask = _mm512_cmp_ps_mask(abs_curr, inf_vec, _CMP_EQ_OQ);
            auto bad_curr = nan_mask | inf_mask;
            // prev must be guarded too: a non-finite baseline bin otherwise latches forever.
            auto nan_prev = _mm512_cmp_ps_mask(prev_data, prev_data, _CMP_UNORD_Q);
            auto abs_prev = _mm512_and_ps(prev_data, abs_mask);
            auto inf_prev = _mm512_cmp_ps_mask(abs_prev, inf_vec, _CMP_EQ_OQ);
            auto bad_prev = nan_prev | inf_prev;

            auto seed = _mm512_mask_blend_ps(bad_curr, curr_data, prev_data);
            auto result = _mm512_fmadd_ps(prev_data, one_minus_alpha_vec, _mm512_mul_ps(seed, alpha_vec));
            result = _mm512_mask_blend_ps(bad_prev, result, seed);  // prev bad: re-seed from curr
            _mm512_store_ps(curr + i, result);
        }

        // Remainder
        process_scalar_range(curr, prev, i, size, alpha);
    }

    // MFV target: scalar baseline
    COMPS_FMV_DEFAULT
    static auto process_impl(float* curr, float* prev, std::size_t size, float alpha) -> void {
        process_scalar_range(curr, prev, 0, size, alpha);
    }

    // MFV target: AVX2 with alignment check
#if COMPS_FMV_ENABLED
    [[gnu::target("avx2,fma")]]
    static auto process_impl(float* curr, float* prev, std::size_t size, float alpha) -> void {
        auto curr_addr = reinterpret_cast<std::uintptr_t>(curr);
        auto prev_addr = reinterpret_cast<std::uintptr_t>(prev);
        bool aligned = (curr_addr % ALIGNMENT_AVX2 == 0) && (prev_addr % ALIGNMENT_AVX2 == 0);

        if (aligned && size >= STRIDE_256) {
            process_avx2_aligned(curr, prev, size, alpha);
        } else {
            process_scalar_range(curr, prev, 0, size, alpha);
        }
    }
#endif

    // MFV target: AVX-512 with alignment check and AVX2 fallback
#if COMPS_FMV_ENABLED
    [[gnu::target("avx512f,avx512dq")]]
    static auto process_impl(float* curr, float* prev, std::size_t size, float alpha) -> void {
        auto curr_addr = reinterpret_cast<std::uintptr_t>(curr);
        auto prev_addr = reinterpret_cast<std::uintptr_t>(prev);
        bool aligned_512 = (curr_addr % ALIGNMENT_AVX512 == 0) && (prev_addr % ALIGNMENT_AVX512 == 0);
        bool aligned_256 = (curr_addr % ALIGNMENT_AVX2 == 0) && (prev_addr % ALIGNMENT_AVX2 == 0);

        if (aligned_512 && size >= STRIDE_512) {
            process_avx512_aligned(curr, prev, size, alpha);
        } else if (aligned_256 && size >= STRIDE_256) {
            process_avx2_aligned(curr, prev, size, alpha);
        } else {
            process_scalar_range(curr, prev, 0, size, alpha);
        }
    }
#endif

    float m_alpha{1.0f};
};

template <>
class work<double> {
    static constexpr auto ALIGNMENT_AVX2 = std::size_t{32};
    static constexpr auto ALIGNMENT_AVX512 = std::size_t{64};
    static constexpr auto STRIDE_256 = std::size_t{4};
    static constexpr auto STRIDE_512 = std::size_t{8};
public:
    explicit work(double alpha) : m_alpha(alpha) {}

    auto process(composite::mutable_buffer<double>& curr_psd, composite::mutable_buffer<double>& prev_psd) const -> void {
        if (curr_psd.size() != prev_psd.size()) {
            throw std::runtime_error("buffer size mismatch in exp_smooth");
        }
        process_impl(curr_psd.data(), prev_psd.data(), curr_psd.size(), m_alpha);
    }

private:
    // Helper: scalar processing for any range (see work<float> for the two-sided
    // sanitization rationale: a non-finite prev must re-seed, not latch forever).
    static auto process_scalar_range(double* curr, double* prev, std::size_t start, std::size_t end, double alpha) -> void {
        const auto one_minus_alpha = 1.0 - alpha;
        for (auto i = start; i < end; ++i) {
            const auto seed = !std::isfinite(curr[i]) ? prev[i] : curr[i];
            curr[i] = std::isfinite(prev[i]) ? (alpha * seed) + (one_minus_alpha * prev[i]) : seed;
        }
    }

    // Helper: AVX2 vectorized loop
    [[gnu::target("avx2,fma")]]
    static auto process_avx2_aligned(double* curr, double* prev, std::size_t size, double alpha) -> void {
        const auto alpha_vec = _mm256_set1_pd(alpha);
        const auto one_minus_alpha_vec = _mm256_set1_pd(1.0 - alpha);
        const auto inf_vec = _mm256_set1_pd(std::numeric_limits<double>::infinity());
        const auto abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));

        auto i = std::size_t{};
        for (; i + STRIDE_256 <= size; i += STRIDE_256) {
            auto curr_data = _mm256_load_pd(curr + i);
            auto prev_data = _mm256_load_pd(prev + i);

            auto nan_mask = _mm256_cmp_pd(curr_data, curr_data, _CMP_UNORD_Q);
            auto abs_curr = _mm256_and_pd(curr_data, abs_mask);
            auto inf_mask = _mm256_cmp_pd(abs_curr, inf_vec, _CMP_EQ_OQ);
            auto bad_curr = _mm256_or_pd(nan_mask, inf_mask);
            // prev must be guarded too: a non-finite baseline bin otherwise latches forever.
            auto nan_prev = _mm256_cmp_pd(prev_data, prev_data, _CMP_UNORD_Q);
            auto abs_prev = _mm256_and_pd(prev_data, abs_mask);
            auto inf_prev = _mm256_cmp_pd(abs_prev, inf_vec, _CMP_EQ_OQ);
            auto bad_prev = _mm256_or_pd(nan_prev, inf_prev);

            auto seed = _mm256_blendv_pd(curr_data, prev_data, bad_curr);
            auto result = _mm256_fmadd_pd(prev_data, one_minus_alpha_vec, _mm256_mul_pd(seed, alpha_vec));
            result = _mm256_blendv_pd(result, seed, bad_prev);  // prev bad: re-seed from curr
            _mm256_store_pd(curr + i, result);
        }

        process_scalar_range(curr, prev, i, size, alpha);
    }

    // Helper: AVX-512 vectorized loop
    [[gnu::target("avx512f,avx512dq")]]
    static auto process_avx512_aligned(double* curr, double* prev, std::size_t size, double alpha) -> void {
        const auto alpha_vec = _mm512_set1_pd(alpha);
        const auto one_minus_alpha_vec = _mm512_set1_pd(1.0 - alpha);
        const auto inf_vec = _mm512_set1_pd(std::numeric_limits<double>::infinity());
        const auto abs_mask = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL));

        auto i = std::size_t{};
        for (; i + STRIDE_512 <= size; i += STRIDE_512) {
            auto curr_data = _mm512_load_pd(curr + i);
            auto prev_data = _mm512_load_pd(prev + i);

            auto nan_mask = _mm512_cmp_pd_mask(curr_data, curr_data, _CMP_UNORD_Q);
            auto abs_curr = _mm512_and_pd(curr_data, abs_mask);
            auto inf_mask = _mm512_cmp_pd_mask(abs_curr, inf_vec, _CMP_EQ_OQ);
            auto bad_curr = nan_mask | inf_mask;
            // prev must be guarded too: a non-finite baseline bin otherwise latches forever.
            auto nan_prev = _mm512_cmp_pd_mask(prev_data, prev_data, _CMP_UNORD_Q);
            auto abs_prev = _mm512_and_pd(prev_data, abs_mask);
            auto inf_prev = _mm512_cmp_pd_mask(abs_prev, inf_vec, _CMP_EQ_OQ);
            auto bad_prev = nan_prev | inf_prev;

            auto seed = _mm512_mask_blend_pd(bad_curr, curr_data, prev_data);
            auto result = _mm512_fmadd_pd(prev_data, one_minus_alpha_vec, _mm512_mul_pd(seed, alpha_vec));
            result = _mm512_mask_blend_pd(bad_prev, result, seed);  // prev bad: re-seed from curr
            _mm512_store_pd(curr + i, result);
        }

        process_scalar_range(curr, prev, i, size, alpha);
    }

    // MFV target: scalar baseline
    COMPS_FMV_DEFAULT
    static auto process_impl(double* curr, double* prev, std::size_t size, double alpha) -> void {
        process_scalar_range(curr, prev, 0, size, alpha);
    }

    // MFV target: AVX2 with alignment check
#if COMPS_FMV_ENABLED
    [[gnu::target("avx2,fma")]]
    static auto process_impl(double* curr, double* prev, std::size_t size, double alpha) -> void {
        auto curr_addr = reinterpret_cast<std::uintptr_t>(curr);
        auto prev_addr = reinterpret_cast<std::uintptr_t>(prev);
        bool aligned = (curr_addr % ALIGNMENT_AVX2 == 0) && (prev_addr % ALIGNMENT_AVX2 == 0);

        if (aligned && size >= STRIDE_256) {
            process_avx2_aligned(curr, prev, size, alpha);
        } else {
            process_scalar_range(curr, prev, 0, size, alpha);
        }
    }
#endif

    // MFV target: AVX-512 with alignment check and AVX2 fallback
#if COMPS_FMV_ENABLED
    [[gnu::target("avx512f,avx512dq")]]
    static auto process_impl(double* curr, double* prev, std::size_t size, double alpha) -> void {
        auto curr_addr = reinterpret_cast<std::uintptr_t>(curr);
        auto prev_addr = reinterpret_cast<std::uintptr_t>(prev);
        bool aligned_512 = (curr_addr % ALIGNMENT_AVX512 == 0) && (prev_addr % ALIGNMENT_AVX512 == 0);
        bool aligned_256 = (curr_addr % ALIGNMENT_AVX2 == 0) && (prev_addr % ALIGNMENT_AVX2 == 0);

        if (aligned_512 && size >= STRIDE_512) {
            process_avx512_aligned(curr, prev, size, alpha);
        } else if (aligned_256 && size >= STRIDE_256) {
            process_avx2_aligned(curr, prev, size, alpha);
        } else {
            process_scalar_range(curr, prev, 0, size, alpha);
        }
    }
#endif

    double m_alpha{1.0};
};
