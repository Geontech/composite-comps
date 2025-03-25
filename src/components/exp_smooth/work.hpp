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
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include <aligned_mem.hpp>

#include <immintrin.h>

template <typename T>
class work {};

template <>
class work<float> {
    using data_type = aligned::aligned_mem<float>;
    static constexpr auto STRIDE_256 = std::size_t{256u / 8u / sizeof(float)};
    static constexpr auto STRIDE_512 = std::size_t{512u / 8u / sizeof(float)};
public:
    explicit work(float alpha) {
        // Initialize the function pointer based on CPU features
        if (__builtin_cpu_supports("avx512f")) {
            init_avx512(alpha);
        } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            init_avx2(alpha);
        }
    }

    auto process(data_type* curr_psd, data_type* prev_psd) const -> void {
        (this->*process_func)(curr_psd, prev_psd);
    }

private:
    [[gnu::target("avx2,fma")]]
    auto init_avx2(double alpha) -> void {
        process_func = &work::process_avx2;
        m_alpha_vec_256 = _mm256_set1_ps(alpha);
        m_one_minus_alpha_vec_256 = _mm256_set1_ps(1 - alpha);
    }

    [[gnu::target("avx512f")]]
    auto init_avx512(double alpha) -> void {
        process_func = &work::process_avx512;
        m_alpha_vec_512 = _mm512_set1_ps(alpha);
        m_one_minus_alpha_vec_512 = _mm512_set1_ps(1 - alpha);
    }

    [[gnu::target("avx2,fma")]]
    auto process_avx2(data_type* curr_psd, data_type* prev_psd) const -> void {
        for (auto i=0u; i < curr_psd->size(); i += STRIDE_256) {
            // Load data
            auto curr_data = _mm256_load_ps(curr_psd->data() + i);
            auto prev_data = _mm256_load_ps(prev_psd->data() + i);
            // Multiply current data by alpha
            curr_data = _mm256_mul_ps(curr_data, m_alpha_vec_256);
            // Multiply prev by (1-alpha) and add to current
            curr_data = _mm256_fmadd_ps(prev_data, m_one_minus_alpha_vec_256, curr_data);
            // Store result into psd
            _mm256_store_ps(curr_psd->data() + i, curr_data);
        }
    }

    [[gnu::target("avx512f")]]
    auto process_avx512(data_type* curr_psd, data_type* prev_psd) const -> void {
        for (auto i=0u; i < curr_psd->size(); i += STRIDE_512) {
            // Load data
            auto curr_data = _mm512_load_ps(curr_psd->data() + i);
            auto prev_data = _mm512_load_ps(prev_psd->data() + i);
            // Multiply current data by alpha
            curr_data = _mm512_mul_ps(curr_data, m_alpha_vec_512);
            // Multiply prev by (1-alpha) and add to current
            curr_data = _mm512_fmadd_ps(prev_data, m_one_minus_alpha_vec_512, curr_data);
            // Store result into psd
            _mm512_store_ps(curr_psd->data() + i, curr_data);
        }
    }

    auto (work::*process_func)(data_type*, data_type*) const -> void;
    __m256 m_alpha_vec_256;
    __m512 m_alpha_vec_512;
    __m256 m_one_minus_alpha_vec_256;
    __m512 m_one_minus_alpha_vec_512;

}; // class work<float>

template <>
class work<double> {
    using data_type = aligned::aligned_mem<double>;
    static constexpr auto STRIDE_256 = std::size_t{256u / 8u / sizeof(double)};
    static constexpr auto STRIDE_512 = std::size_t{512u / 8u / sizeof(double)};
public:
    explicit work(double alpha) {
        // Initialize the function pointer based on CPU features
        if (__builtin_cpu_supports("avx512f")) {
            init_avx512(alpha);
        } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            init_avx2(alpha);
        }
    }

    auto process(data_type* curr_psd, data_type* prev_psd) const -> void {
        (this->*process_func)(curr_psd, prev_psd);
    }

private:
    [[gnu::target("avx2,fma")]]
    auto init_avx2(double alpha) -> void {
        process_func = &work::process_avx2;
        m_alpha_vec_256 = _mm256_set1_pd(alpha);
        m_one_minus_alpha_vec_256 = _mm256_set1_pd(1 - alpha);
    }

    [[gnu::target("avx512f")]]
    auto init_avx512(double alpha) -> void {
        process_func = &work::process_avx512;
        m_alpha_vec_512 = _mm512_set1_pd(alpha);
        m_one_minus_alpha_vec_512 = _mm512_set1_pd(1 - alpha);
    }

    [[gnu::target("avx2,fma")]]
    auto process_avx2(data_type* curr_psd, data_type* prev_psd) const -> void {
        for (auto i=0u; i < curr_psd->size(); i += STRIDE_256) {
            // Load data
            auto curr_data = _mm256_load_pd(curr_psd->data() + i);
            auto prev_data = _mm256_load_pd(prev_psd->data() + i);
            // Multiply current data by alpha
            curr_data = _mm256_mul_pd(curr_data, m_alpha_vec_256);
            // Multiply prev by (1-alpha) and add to current
            curr_data = _mm256_fmadd_pd(prev_data, m_one_minus_alpha_vec_256, curr_data);
            // Store result into psd
            _mm256_store_pd(curr_psd->data() + i, curr_data);
        }
    }

    [[gnu::target("avx512f")]]
    auto process_avx512(data_type* curr_psd, data_type* prev_psd) const -> void {
        for (auto i=0u; i < curr_psd->size(); i += STRIDE_512) {
            // Load data
            auto curr_data = _mm512_load_pd(curr_psd->data() + i);
            auto prev_data = _mm512_load_pd(prev_psd->data() + i);
            // Multiply current data by alpha
            curr_data = _mm512_mul_pd(curr_data, m_alpha_vec_512);
            // Multiply prev by (1-alpha) and add to current
            curr_data = _mm512_fmadd_pd(prev_data, m_one_minus_alpha_vec_512, curr_data);
            // Store result into psd
            _mm512_store_pd(curr_psd->data() + i, curr_data);
        }
    }

    auto (work::*process_func)(data_type*, data_type*) const -> void;
    __m256d m_alpha_vec_256;
    __m512d m_alpha_vec_512;
    __m256d m_one_minus_alpha_vec_256;
    __m512d m_one_minus_alpha_vec_512;

}; // class work<double>
