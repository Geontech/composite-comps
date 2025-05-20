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

#include <algorithm>
#include <complex>
#include <numeric>
#include <immintrin.h>

template <typename T>
class work {};

template <>
class work<float> {
    using cplx_data_type = aligned::aligned_mem<std::complex<float>>;
    using real_data_type = aligned::aligned_mem<float>;
    static constexpr auto STRIDE_256 = std::size_t{256u / 8u / sizeof(float)};
    static constexpr auto STRIDE_512 = std::size_t{512u / 8u / sizeof(float)};
public:
    explicit work(float normalization_const) : m_norm_const(normalization_const) {
        // Initialize the function pointer based on CPU features
        if (__builtin_cpu_supports("avx512f")) {
            init_avx512();
        } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            init_avx2();
        }
    }

    auto norm_const() const -> float {
        return m_norm_const;
    }

    auto process(cplx_data_type* data) -> std::unique_ptr<real_data_type> {
        return (this->*process_func)(data);
    }

    auto apply_multiplier(real_data_type* data) const -> void {
        (this->*apply_multiplier_func)(data);
    }

private:
    [[gnu::target("avx512f")]]
    auto init_avx512() -> void {
        process_func = &work::process_avx512;
        apply_multiplier_func = &work::apply_multiplier_avx512;
        m_norm_const_512 = _mm512_set1_ps(m_norm_const);
        m_log_const_512 = _mm512_set1_ps(float{10} / std::log2f(10));
        m_real_idx_512i = _mm512_set_epi32(30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0);
        m_imag_idx_512i = _mm512_set_epi32(31,29,27,25,23,21,19,17,15,13,11,9,7,5,3,1);
    }

    [[gnu::target("avx2,fma")]]
    auto init_avx2() -> void {
        process_func = &work::process_avx2;
        apply_multiplier_func = &work::apply_multiplier_avx2;
        m_norm_const_256 = _mm256_set1_ps(m_norm_const);
        m_log_const_256 = _mm256_set1_ps(float{10} / std::log2f(10));
        m_real_idx_256i = _mm256_set_epi32(14,12,10,8,6,4,2,0);
        m_imag_idx_256i = _mm256_set_epi32(15,13,11,9,7,5,3,1);
    }

    [[gnu::target("avx512f")]]
    auto process_avx512(cplx_data_type* data) -> std::unique_ptr<real_data_type> {
        auto psd = aligned::make_aligned<float>(data->alignment(), data->size());
        for (auto i=0u; i < data->size(); i += STRIDE_512) {
            // Load real and imag parts separately
            auto real_m512 = _mm512_i32gather_ps(m_real_idx_512i, data->data() + i, 4);
            auto imag_m512 = _mm512_i32gather_ps(m_imag_idx_512i, data->data() + i, 4);
            // Calculate power          
            // Square reals
            real_m512 = _mm512_mul_ps(real_m512, real_m512);
            // Square imags and add to squared reals to get power
            real_m512 = _mm512_fmadd_ps(imag_m512, imag_m512, real_m512);
            // Multiply by normalization contstant
            real_m512 = _mm512_mul_ps(real_m512, m_norm_const_512);
            // Store result into psd
            _mm512_store_ps(psd->data() + i, real_m512);
        }
        return psd;
    }

    [[gnu::target("avx2,fma")]]
    auto process_avx2(cplx_data_type* data) -> std::unique_ptr<real_data_type> {
        auto psd = aligned::make_aligned<float>(data->alignment(), data->size());
        for (auto i=0u; i < data->size(); i += STRIDE_256) {
            // Load real and imag parts separately
            auto real = _mm256_i32gather_ps(reinterpret_cast<float*>(data->data() + i), m_real_idx_256i, 4);
            auto imag = _mm256_i32gather_ps(reinterpret_cast<float*>(data->data() + i), m_imag_idx_256i, 4);
            // Calculate power          
            // Square reals
            real = _mm256_mul_ps(real, real);
            // Square imags and add to squared reals to get power
            real = _mm256_fmadd_ps(imag, imag, real);
            // Multiply by normalization constant
            real = _mm256_mul_ps(real, m_norm_const_256);
            // Store result into psd
            _mm256_store_ps(psd->data() + i, real);
        }
        return psd;
    }

    [[gnu::target("avx512f")]]
    auto apply_multiplier_avx512(real_data_type* data) const -> void {
        for (auto i=0u; i < data->size(); i += STRIDE_512) {
            // Load data
            auto ps_data = _mm512_load_ps(data->data() + i);  
            // Apply constant
            ps_data = _mm512_mul_ps(ps_data, m_log_const_512);
            // Store result
            _mm512_store_ps(data->data() + i, ps_data);
        }
    }

    [[gnu::target("avx2")]]
    auto apply_multiplier_avx2(real_data_type* data) const -> void {
        for (auto i=0u; i < data->size(); i += STRIDE_256) {
            // Load data
            auto ps_data = _mm256_load_ps(data->data() + i);  
            // Apply constant
            ps_data = _mm256_mul_ps(ps_data, m_log_const_256);
            // Store result
            _mm256_store_ps(data->data() + i, ps_data);
        }
    }

    auto (work::*process_func)(cplx_data_type*) -> std::unique_ptr<real_data_type>;
    auto (work::*apply_multiplier_func)(real_data_type*) const -> void;
    float m_norm_const{1};
    __m256 m_norm_const_256;
    __m512 m_norm_const_512;
    __m256 m_log_const_256;
    __m512 m_log_const_512;
    __m256i m_real_idx_256i;
    __m512i m_real_idx_512i;
    __m256i m_imag_idx_256i;
    __m512i m_imag_idx_512i;

}; // class psd_work<float>

template <>
class work<double> {
    using cplx_data_type = aligned::aligned_mem<std::complex<double>>;
    using real_data_type = aligned::aligned_mem<double>;
    static constexpr auto STRIDE_256 = std::size_t{256u / 8u / sizeof(double)};
    static constexpr auto STRIDE_512 = std::size_t{512u / 8u / sizeof(double)};
public:
    explicit work(double normalization_const) : m_norm_const(normalization_const) {
        // Initialize the function pointer based on CPU features
        if (__builtin_cpu_supports("avx512f")) {
            init_avx512();
        } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
           init_avx2();
        }
    }

    auto norm_const() const -> double {
        return m_norm_const;
    }

    auto process(cplx_data_type* data) -> std::unique_ptr<real_data_type> {
        return (this->*process_func)(data);
    }

    auto apply_multiplier(real_data_type* data) const -> void {
        (this->*apply_multiplier_func)(data);
    }

private:
    [[gnu::target("avx512f")]]
    auto init_avx512() -> void {
        process_func = &work::process_avx512;
        apply_multiplier_func = &work::apply_multiplier_avx512;
        m_norm_const_512 = _mm512_set1_pd(m_norm_const);
        m_log_const_512 = _mm512_set1_pd(double{10} / std::log2(10));
        m_real_idx_512i = _mm512_set_epi64(14,12,10,8,6,4,2,0);
        m_imag_idx_512i = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    }

    [[gnu::target("avx2,fma")]]
    auto init_avx2() -> void {
        process_func = &work::process_avx2;
        apply_multiplier_func = &work::apply_multiplier_avx2;
        m_norm_const_256 = _mm256_set1_pd(m_norm_const);
        m_log_const_256 = _mm256_set1_pd(double{10} / std::log2(10));
        m_real_idx_256i = _mm256_set_epi64x(6,4,2,0);
        m_imag_idx_256i = _mm256_set_epi64x(7,5,3,1);
    }

    [[gnu::target("avx512f")]]
    auto process_avx512(cplx_data_type* data) -> std::unique_ptr<real_data_type> {
        auto psd = aligned::make_aligned<double>(data->alignment(), data->size());
        for (auto i=0u; i < data->size(); i += STRIDE_512) {
            // Load real and imag parts separately
            auto real = _mm512_i64gather_pd(m_real_idx_512i, data->data() + i, 8);
            auto imag = _mm512_i64gather_pd(m_imag_idx_512i, data->data() + i, 8);
            // Calculate power          
            // Square reals
            real = _mm512_mul_pd(real, real);
            // Square imags and add to squared reals to get power
            real = _mm512_fmadd_pd(imag, imag, real);
            // Multiply by normalization constant
            real = _mm512_mul_pd(real, m_norm_const_512);
            // Store result into psd
            _mm512_store_pd(psd->data() + i, real);
        }
        return psd;
    }

    [[gnu::target("avx2,fma")]]
    auto process_avx2(cplx_data_type* data) -> std::unique_ptr<real_data_type> {
        auto psd = aligned::make_aligned<double>(data->alignment(), data->size());
        for (auto i=0u; i < data->size(); i += STRIDE_256) {
            // Load real and imag parts separately
            auto real = _mm256_i64gather_pd(reinterpret_cast<double*>(data->data() + i), m_real_idx_256i, 8);
            auto imag = _mm256_i64gather_pd(reinterpret_cast<double*>(data->data() + i), m_imag_idx_256i, 8);
            // Calculate power          
            // Square reals
            real = _mm256_mul_pd(real, real);
            // Square imags and add to squared reals to get power
            real = _mm256_fmadd_pd(imag, imag, real);
            // Multiply by normalization constant
            real = _mm256_mul_pd(real, m_norm_const_256);
            // Store result into psd
            _mm256_store_pd(psd->data() + i, real);
        }
        return psd;
    }

    [[gnu::target("avx512f")]]
    auto apply_multiplier_avx512(real_data_type* data) const -> void {
        for (auto i=0u; i < data->size(); i += STRIDE_512) {
            // Load data
            auto pd_data = _mm512_load_pd(data->data() + i);  
            // Apply constant
            pd_data = _mm512_mul_pd(pd_data, m_log_const_512);
            // Store result
            _mm512_store_pd(data->data() + i, pd_data);
        }
    }

    [[gnu::target("avx2")]]
    auto apply_multiplier_avx2(real_data_type* data) const -> void {
        for (auto i=0u; i < data->size(); i += STRIDE_256) {
            // Load data
            auto pd_data = _mm256_load_pd(data->data() + i);  
            // Apply constant
            pd_data = _mm256_mul_pd(pd_data, m_log_const_256);
            // Store result
            _mm256_store_pd(data->data() + i, pd_data);
        }
    }

    auto (work::*process_func)(cplx_data_type*) -> std::unique_ptr<real_data_type>;
    auto (work::*apply_multiplier_func)(real_data_type*) const -> void;
    double m_norm_const{1};
    __m256d m_norm_const_256;
    __m512d m_norm_const_512;
    __m256d m_log_const_256;
    __m512d m_log_const_512;
    __m256i m_real_idx_256i;
    __m512i m_real_idx_512i;
    __m256i m_imag_idx_256i;
    __m512i m_imag_idx_512i;

}; // class psd_work<double>
