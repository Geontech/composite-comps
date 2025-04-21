/*
 * Copyright (C) 2025 Geon Technologies, LLC
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

#include "window_buffer.hpp"

#include <complex>
#include <immintrin.h>
#include <vector>

namespace dotprod {

/**
 * @brief Scalar dot-product implementation
 */
 [[gnu::target("default")]]
inline 
auto process(const std::vector<float>& h, const window_buffer<std::complex<float>>& x) -> std::complex<float> {
    auto result = std::complex<float>{0.0f, 0.0f};
    auto i = std::size_t{};
    for (; i + 4 <= h.size(); i += 4) {
        result += h[i] * x[i];
        result += h[i+1] * x[i + 1];
        result += h[i+2] * x[i + 2];
        result += h[i+3] * x[i + 3];
    }
    for (; i < h.size(); ++i) {
        result += h[i] * x[i];
    }
    return result;
}

/**
 * @brief AVX2 dot product for 8 complex floats
 */
[[gnu::target("avx2,fma")]]
inline 
auto process(const std::vector<float>& h, const window_buffer<std::complex<float>>& x) -> std::complex<float> {
    // Init coefficients reshuffle index
    __m256i reshuffle_h = _mm256_set_epi32(7, 6, 3, 2, 5, 4, 1, 0);

    // Init accumulation registers
    __m256 acc_re = _mm256_setzero_ps();
    __m256 acc_im = _mm256_setzero_ps();

    // AVX loop
    auto i = std::size_t{};
    auto x_ptr = reinterpret_cast<const float*>(x.data());
    auto h_ptr = h.data();
    for (; i + 8 <= h.size(); i += 8) {
        // Calculate the float pointer offset for x[i]
        size_t x_offset = 2 * i;

        // Load 8 complex numbers = 16 floats from x, starting at index i
        __m256 a = _mm256_loadu_ps(x_ptr + x_offset);      // [r0, i0, r1, i1, r2, i2, r3, i3]
        __m256 b = _mm256_loadu_ps(x_ptr + x_offset + 8);  // [r4, i4, r5, i5, r6, i6, r7, i7]

       // Shuffle to split real/imag
        __m256 real = _mm256_shuffle_ps(a, b, 0x88); // real: [r0, r1, r4, r5, r2, r3, r6, r7]
        __m256 imag = _mm256_shuffle_ps(a, b, 0xDD); // imag: [i0, i1, i4, i5, i2, i3, i6, i7]

        // Load coefficients and reshuffle to align with split real/imag indices
        __m256 coeffs = _mm256_loadu_ps(h_ptr + i);
        coeffs = _mm256_permutevar8x32_ps(coeffs, reshuffle_h); // [h0, h1, h4, h5, h2, h3, h6, h7]

        // Multiply-accumulate
        acc_re = _mm256_fmadd_ps(coeffs, real, acc_re);
        acc_im = _mm256_fmadd_ps(coeffs, imag, acc_im);
    }

    // Reduce
    auto yr = float{};
    auto yi = float{};
    float re_arr[8], im_arr[8];
    _mm256_storeu_ps(re_arr, acc_re);
    _mm256_storeu_ps(im_arr, acc_im);
    for (int j = 0; j < 8; ++j) {
        yr += re_arr[j];
        yi += im_arr[j];
    }
    auto result = std::complex<float>{yr, yi};

    // Handle tail: process remaining taps using scalar method
    for (; i + 4 <= h.size(); i += 4) {
        result += h[i] * x[i];
        result += h[i+1] * x[i + 1];
        result += h[i+2] * x[i + 2];
        result += h[i+3] * x[i + 3];
    }
    for (; i < h.size(); ++i) {
        result += h[i] * x[i];
    }

    return result;
}

} // namespace dotprod