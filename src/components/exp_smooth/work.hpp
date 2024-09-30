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
#include <memory>

template <typename T>
class work {};

template <>
class work<float> {
public:
    explicit work(float alpha) {
        m_alpha_vec = _mm512_set1_ps(alpha);
        m_one_minus_alpha_vec = _mm512_set1_ps(1 - alpha);
    }

    auto process(aligned::aligned_mem<float>* curr_psd) -> void {
        if (m_exp_data == nullptr) {
            m_exp_data = aligned::make_aligned<float>(64, curr_psd->size());
            std::copy(curr_psd->data(), curr_psd->data() + curr_psd->size(), m_exp_data->data());
            return;
        }
        for (auto i=0u; i < curr_psd->size(); i += 16) {
            // Load data
            auto curr_data = _mm512_load_ps(curr_psd->data() + i);
            auto exp_data = _mm512_load_ps(m_exp_data->data() + i);
            // Multiply current data by alpha
            curr_data = _mm512_mul_ps(curr_data, m_alpha_vec);
            // Multiply prev by (1-alpha) and add to current
            exp_data = _mm512_fmadd_ps(exp_data, m_one_minus_alpha_vec, curr_data);
            // Store result into psd
            _mm512_store_ps(curr_psd->data() + i, exp_data);
            // Store for feedback
            _mm512_store_ps(m_exp_data->data() + i, exp_data);
        }
    }

private:
    std::unique_ptr<aligned::aligned_mem<float>> m_exp_data{nullptr};
    __m512 m_alpha_vec;
    __m512 m_one_minus_alpha_vec;

}; // class work<float>

template <>
class work<double> {
public:
    explicit work(double alpha) {
        m_alpha_vec = _mm512_set1_pd(alpha);
        m_one_minus_alpha_vec = _mm512_set1_pd(1- alpha);
    }

    auto process(aligned::aligned_mem<double>* curr_psd) -> void {
        if (m_exp_data == nullptr) {
            m_exp_data = aligned::make_aligned<double>(64, curr_psd->size());
            std::copy(curr_psd->data(), curr_psd->data() + curr_psd->size(), m_exp_data->data());
            return;
        }
        for (auto i=0u; i < curr_psd->size(); i += 16) {
            // Load data
            auto curr_data = _mm512_load_pd(curr_psd->data() + i);
            auto exp_data = _mm512_load_pd(m_exp_data->data() + i);
            // Multiply current data by alpha
            curr_data = _mm512_mul_pd(curr_data, m_alpha_vec);
            // Multiply prev by (1-alpha) and add to current
            exp_data = _mm512_fmadd_pd(exp_data, m_one_minus_alpha_vec, curr_data);
            // Store result into psd
            _mm512_store_pd(curr_psd->data() + i, exp_data);
            // Store for feedback
            _mm512_store_pd(m_exp_data->data() + i, exp_data);
        }
    }

private:
    std::unique_ptr<aligned::aligned_mem<double>> m_exp_data{nullptr};
    __m512d m_alpha_vec;
    __m512d m_one_minus_alpha_vec;

}; // class work<float>

