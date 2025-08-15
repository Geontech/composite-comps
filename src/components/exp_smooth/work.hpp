/*
 * Copyright (C) 2024 Geon Technologies, LLC
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

#include <aligned_mem.hpp>

#include <cmath>
#include <immintrin.h>
#include <limits>

template <typename T>
class work {};

template <>
class work<float> {
    using data_type = aligned::aligned_mem<float>;
    static constexpr auto STRIDE_256 = std::size_t{256u / 8u / sizeof(float)};
    static constexpr auto STRIDE_512 = std::size_t{512u / 8u / sizeof(float)};
public:
    explicit work(float alpha) : m_alpha(alpha) {}

    [[gnu::target("default")]]
    auto process(data_type* curr_psd, data_type* prev_psd) const -> void {
        const auto one_minus_alpha = 1.f - m_alpha;
        for (auto i=0u; i < curr_psd->size(); ++i) {
            auto curr_val = curr_psd->data()[i];
            auto prev_val = prev_psd->data()[i];

            // If value is not finite, replace it with the previous value
            auto sanitized_curr_val = !std::isfinite(curr_val) ? prev_val : curr_val;
            
            auto result = (m_alpha * sanitized_curr_val) + (one_minus_alpha * prev_val);
            curr_psd->data()[i] = result;
        }
    }

    [[gnu::target("avx2,fma")]]
    auto process(data_type* curr_psd, data_type* prev_psd) const -> void {
        const auto alpha_vec_256 = _mm256_set1_ps(m_alpha);
        const auto one_minus_alpha_vec_256 = _mm256_set1_ps(1.f - m_alpha);
        static const auto inf_vec_256 = _mm256_set1_ps(std::numeric_limits<float>::infinity());
        static const auto abs_mask_256 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        for (auto i=0u; i < curr_psd->size(); i += STRIDE_256) {
            // Load data
            auto curr_data = _mm256_load_ps(curr_psd->data() + i);
            auto prev_data = _mm256_load_ps(prev_psd->data() + i);
            // Create masks for specific 'bad' values (NaN, +/-INF)
            auto nan_mask = _mm256_cmp_ps(curr_data, curr_data, _CMP_UNORD_Q);
            auto abs_curr_data = _mm256_and_ps(curr_data, abs_mask_256);
            auto inf_mask = _mm256_cmp_ps(abs_curr_data, inf_vec_256, _CMP_EQ_OQ);
            auto non_finite_mask = _mm256_or_ps(nan_mask, inf_mask);
            // Replace non-finite values with previous values
            auto sanitized_curr_data = _mm256_blendv_ps(curr_data, prev_data, non_finite_mask);
            // Multiply current data by alpha
            sanitized_curr_data = _mm256_mul_ps(sanitized_curr_data, alpha_vec_256);
            // Multiply prev by (1-alpha) and add to current
            auto result = _mm256_fmadd_ps(prev_data, one_minus_alpha_vec_256, sanitized_curr_data);
            // Store result into psd
            _mm256_store_ps(curr_psd->data() + i, result);
        }
    }

    [[gnu::target("avx512f,avx512dq")]]
    auto process(data_type* curr_psd, data_type* prev_psd) const -> void {
        const auto alpha_vec_512 = _mm512_set1_ps(m_alpha);
        const auto one_minus_alpha_vec_512 = _mm512_set1_ps(1.f - m_alpha);
        static const auto inf_vec_512 = _mm512_set1_ps(std::numeric_limits<float>::infinity());
        static const auto abs_mask_512 = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
        for (auto i=0u; i < curr_psd->size(); i += STRIDE_512) {
            // Load data
            auto curr_data = _mm512_load_ps(curr_psd->data() + i);
            auto prev_data = _mm512_load_ps(prev_psd->data() + i);
            // Create masks for specific 'bad' values (NaN, +/-INF)
            auto nan_mask = _mm512_cmp_ps_mask(curr_data, curr_data, _MM_CMPINT_NE);
            auto abs_curr_data = _mm512_and_ps(curr_data, abs_mask_512);
            auto inf_mask = _mm512_cmp_ps_mask(abs_curr_data, inf_vec_512, _MM_CMPINT_EQ);
            auto non_finite_mask = nan_mask | inf_mask;
            // Replace non-finite values with previous values
            auto sanitized_curr_data = _mm512_mask_blend_ps(non_finite_mask, curr_data, prev_data);
            // Multiply current data by alpha
            sanitized_curr_data = _mm512_mul_ps(sanitized_curr_data, alpha_vec_512);
            // Multiply prev by (1-alpha) and add to current
            auto result = _mm512_fmadd_ps(prev_data, one_minus_alpha_vec_512, sanitized_curr_data);
            // Store result into psd
            _mm512_store_ps(curr_psd->data() + i, result);
        }
    }

private:
    float m_alpha{1};

}; // class work<float>

template <>
class work<double> {
    using data_type = aligned::aligned_mem<double>;
    static constexpr auto STRIDE_256 = std::size_t{256u / 8u / sizeof(double)};
    static constexpr auto STRIDE_512 = std::size_t{512u / 8u / sizeof(double)};
public:
    explicit work(double alpha) : m_alpha(alpha) {}

    [[gnu::target("default")]]
    auto process(data_type* curr_psd, data_type* prev_psd) const -> void {
        const auto one_minus_alpha = 1.0 - m_alpha;
        for (auto i=0u; i < curr_psd->size(); ++i) {
            auto curr_val = curr_psd->data()[i];
            auto prev_val = prev_psd->data()[i];

            // If value is not finite, replace it with the previous value
            auto sanitized_curr_val = !std::isfinite(curr_val) ? prev_val : curr_val;
            
            auto result = (m_alpha * sanitized_curr_val) + (one_minus_alpha * prev_val);
            curr_psd->data()[i] = result;
        }
    }

    [[gnu::target("avx2,fma")]]
    auto process(data_type* curr_psd, data_type* prev_psd) const -> void {
        const auto alpha_vec_256 = _mm256_set1_pd(m_alpha);
        const auto one_minus_alpha_vec_256 = _mm256_set1_pd(1.0 - m_alpha);
        static const auto inf_vec_256 = _mm256_set1_pd(std::numeric_limits<double>::infinity());
        static const auto abs_mask_256 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
        for (auto i=0u; i < curr_psd->size(); i += STRIDE_256) {
            // Load data
            auto curr_data = _mm256_load_pd(curr_psd->data() + i);
            auto prev_data = _mm256_load_pd(prev_psd->data() + i);
            // Create masks for specific 'bad' values (NaN, +/-INF)
            auto nan_mask = _mm256_cmp_pd(curr_data, curr_data, _CMP_UNORD_Q);
            auto abs_curr_data = _mm256_and_pd(curr_data, abs_mask_256);
            auto inf_mask = _mm256_cmp_pd(abs_curr_data, inf_vec_256, _CMP_EQ_OQ);
            auto non_finite_mask = _mm256_or_pd(nan_mask, inf_mask);
            // Replace non-finite values with previous values
            auto sanitized_curr_data = _mm256_blendv_pd(curr_data, prev_data, non_finite_mask);
            // Multiply current data by alpha
            sanitized_curr_data = _mm256_mul_pd(sanitized_curr_data, alpha_vec_256);
            // Multiply prev by (1-alpha) and add to current
            auto result = _mm256_fmadd_pd(prev_data, one_minus_alpha_vec_256, sanitized_curr_data);
            // Store result into psd
            _mm256_store_pd(curr_psd->data() + i, result);
        }
    }

    [[gnu::target("avx512f,avx512dq")]]
    auto process(data_type* curr_psd, data_type* prev_psd) const -> void {
        const auto alpha_vec_512 = _mm512_set1_pd(m_alpha);
        const auto one_minus_alpha_vec_512 = _mm512_set1_pd(1.0 - m_alpha);
        static const auto inf_vec_512 = _mm512_set1_pd(std::numeric_limits<double>::infinity());
        static const auto abs_mask_512 = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL));
        for (auto i=0u; i < curr_psd->size(); i += STRIDE_512) {
            // Load data
            auto curr_data = _mm512_load_pd(curr_psd->data() + i);
            auto prev_data = _mm512_load_pd(prev_psd->data() + i);
            // Create masks for specific 'bad' values (NaN, +/-INF)
            auto nan_mask = _mm512_cmp_pd_mask(curr_data, curr_data, _MM_CMPINT_NE);
            auto abs_curr_data = _mm512_and_pd(curr_data, abs_mask_512);
            auto inf_mask = _mm512_cmp_pd_mask(abs_curr_data, inf_vec_512, _MM_CMPINT_EQ);
            auto non_finite_mask = nan_mask | inf_mask;
            // Replace non-finite values with previous values
            auto sanitized_curr_data = _mm512_mask_blend_pd(non_finite_mask, curr_data, prev_data);
            // Multiply current data by alpha
            sanitized_curr_data = _mm512_mul_pd(sanitized_curr_data, alpha_vec_512);
            // Multiply prev by (1-alpha) and add to current
            auto result = _mm512_fmadd_pd(prev_data, one_minus_alpha_vec_512, sanitized_curr_data);
            // Store result into psd
            _mm512_store_pd(curr_psd->data() + i, result);
        }
    }

private:
    double m_alpha{1};

}; // class work<double>
