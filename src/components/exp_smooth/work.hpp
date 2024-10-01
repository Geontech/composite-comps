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
    using psd_data_t = aligned::aligned_mem<float>;
public:
    explicit work(float alpha) {
        m_alpha_vec = _mm512_set1_ps(alpha);
        m_one_minus_alpha_vec = _mm512_set1_ps(1 - alpha);
    }

    auto process(psd_data_t* curr_psd, psd_data_t* prev_psd) const -> void {
        for (auto i=0u; i < curr_psd->size(); i += 16) {
            // Load data
            auto curr_data = _mm512_load_ps(curr_psd->data() + i);
            auto prev_data = _mm512_load_ps(prev_psd->data() + i);
            // Multiply current data by alpha
            curr_data = _mm512_mul_ps(curr_data, m_alpha_vec);
            // Multiply prev by (1-alpha) and add to current
            curr_data = _mm512_fmadd_ps(prev_data, m_one_minus_alpha_vec, curr_data);
            // Store result into psd
            _mm512_store_ps(curr_psd->data() + i, curr_data);
        }
    }

private:
    __m512 m_alpha_vec;
    __m512 m_one_minus_alpha_vec;

}; // class work<float>

template <>
class work<double> {
    using psd_data_t = aligned::aligned_mem<double>;
public:
    explicit work(double alpha) {
        m_alpha_vec = _mm512_set1_pd(alpha);
        m_one_minus_alpha_vec = _mm512_set1_pd(1 - alpha);
    }

    auto process(psd_data_t* curr_psd, psd_data_t* prev_psd) const -> void {
        for (auto i=0u; i < curr_psd->size(); i += 8) {
            // Load data
            auto curr_data = _mm512_load_pd(curr_psd->data() + i);
            auto prev_data = _mm512_load_pd(prev_psd->data() + i);
            // Multiply current data by alpha
            curr_data = _mm512_mul_pd(curr_data, m_alpha_vec);
            // Multiply prev by (1-alpha) and add to current
            curr_data = _mm512_fmadd_pd(prev_data, m_one_minus_alpha_vec, curr_data);
            // Store result into psd
            _mm512_store_pd(curr_psd->data() + i, curr_data);
        }
    }

private:
    __m512d m_alpha_vec;
    __m512d m_one_minus_alpha_vec;

}; // class work<double>
