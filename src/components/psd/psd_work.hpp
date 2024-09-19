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

#include "aligned_mem.hpp"

#include <algorithm>
#include <complex>
#include <numeric>
#include <immintrin.h>

template <typename T>
class psd_work {};

template <>
class psd_work<float> {
public:
    psd_work(float window_sum, float sample_rate) {
        m_window_sum_512 = _mm512_set1_ps(window_sum);
        m_fs_512 = _mm512_set1_ps(sample_rate);
        m_real_idx_512i = _mm512_set_epi32(30,28,26,24,22,20,18,16,14,12,10,8,6,4,2,0);
        m_imag_idx_512i = _mm512_set_epi32(31,29,27,25,23,21,19,17,15,13,11,9,7,5,3,1);
    }

    auto process(aligned::aligned_mem<std::complex<float>>* data) -> std::unique_ptr<aligned::aligned_mem<float>> {
        auto psd = aligned::make_aligned<float>(data->alignment(), data->size());
        for (auto i=0u; i < data->size(); i += 16) {
            // Load real and imag parts separately
            auto real_m512 = _mm512_i32gather_ps(m_real_idx_512i, data->data() + i, 4);
            auto imag_m512 = _mm512_i32gather_ps(m_imag_idx_512i, data->data() + i, 4);
            // Calculate power          
            // Square reals
            real_m512 = _mm512_mul_ps(real_m512, real_m512);
            // Square imags and add to squared reals to get power
            real_m512 = _mm512_fmadd_ps(imag_m512, imag_m512, real_m512);
            // Divide by Fs
            real_m512 = _mm512_div_ps(real_m512, m_fs_512);
            // Divide by window sum
            real_m512 = _mm512_div_ps(real_m512, m_window_sum_512);
            // Store result into psd
            _mm512_store_ps(psd->data() + i, real_m512);
        }
        return psd;
    }

private:
    __m512 m_window_sum_512;
    __m512 m_fs_512;
    __m512i m_real_idx_512i;
    __m512i m_imag_idx_512i;

}; // class psd_work<float>

template <>
class psd_work<double> {
public:
    psd_work(double window_sum, double sample_rate) {
        m_window_sum_512 = _mm512_set1_pd(window_sum);
        m_fs_512 = _mm512_set1_pd(sample_rate);
        m_real_idx_512i = _mm512_set_epi64(14,12,10,8,6,4,2,0);
        m_imag_idx_512i = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    }

    auto process(aligned::aligned_mem<std::complex<double>>* data) -> std::unique_ptr<aligned::aligned_mem<double>> {
        auto psd = aligned::make_aligned<double>(data->alignment(), data->size());
        for (auto i=0u; i < data->size(); i += 8) {
            // Load real and imag parts separately
            auto real_m512 = _mm512_i64gather_pd(m_real_idx_512i, data->data() + i, 8);
            auto imag_m512 = _mm512_i64gather_pd(m_imag_idx_512i, data->data() + i, 8);
            // Calculate power          
            // Square reals
            real_m512 = _mm512_mul_pd(real_m512, real_m512);
            // Square imags and add to squared reals to get power
            real_m512 = _mm512_fmadd_pd(imag_m512, imag_m512, real_m512);
            // Divide by Fs
            real_m512 = _mm512_div_pd(real_m512, m_fs_512);
            // Divide by window sum
            real_m512 = _mm512_div_pd(real_m512, m_window_sum_512);
            // Store result into psd
            _mm512_store_pd(psd->data() + i, real_m512);
        }
        return psd;
    }

private:
    __m512d m_window_sum_512;
    __m512d m_fs_512;
    __m512i m_real_idx_512i;
    __m512i m_imag_idx_512i;

}; // class psd_work<double>