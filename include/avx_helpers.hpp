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

#include <concepts>
#include <immintrin.h>

namespace avx {

template <typename T>
constexpr bool avxable_ps =
    std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>;

template <typename T>
constexpr bool avxable_pd =
    std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t> ||
    std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>;

template <typename T>
inline auto cvt_pd(const T* data) -> __m512i {
    auto retval = __m512i{};
    // Load payload (type T))
    // Up-convert based on incoming type to 64-bit variant (signed or unsigned)
    if constexpr (std::is_same_v<T, uint16_t>) {
        auto loaded = _mm_loadu_epi16(data);
        retval = _mm512_cvtepu16_epu64(loaded);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        auto loaded = _mm_loadu_epi16(data);
        retval = _mm512_cvtepi16_epi64(loaded);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        auto loaded = _mm256_load_epi32(data);
        retval = _mm512_cvtepu32_epu64(loaded);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        auto loaded = _mm256_load_epi32(data);
        retval = _mm512_cvtepi32_epi64(loaded);
    } else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) {
        retval = _mm512_load_epi64(data);
    }
    return retval;
}

template <typename T>
inline auto cvt_ps(const T* data) -> __m512i {
    auto retval = __m512i{};
    // Load payload (type T))
    // Up-convert based on incoming type to 32-bit variant (signed or unsigned)
    if constexpr (std::is_same_v<T, uint16_t>) {
        auto loaded = _mm256_loadu_epi16(data);
        retval = _mm512_cvtepu16_epu32(loaded);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        auto loaded = _mm256_loadu_epi16(data);
        retval = _mm512_cvtepi16_epi32(loaded);
    } else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) {
        retval = _mm512_load_epi32(data);
    }
    return retval;
}

template <typename T>
requires avxable_ps<T>
inline auto window_cvt_i_ps(const T* data, const float* window, float* dst) -> void {
    // Load and convert
    auto data_m512i = cvt_ps(data);
    // Convert u/int64 to floats
    auto payload_m512 = __m512{};
    if constexpr (std::is_signed_v<T>) {
        payload_m512 = _mm512_cvtepi32_ps(data_m512i);
    } else {
        payload_m512 = _mm512_cvtepu32_ps(data_m512i);
    }
    if (window != nullptr) {
        // Load window doubles
        auto window_m512 = _mm512_load_ps(window);
        // Multiply payload by window
        payload_m512 = _mm512_mul_ps(payload_m512, window_m512);
    }
    // Stored result into dst
    _mm512_store_ps(dst, payload_m512);
}

template <typename T>
requires avxable_pd<T>
inline auto window_cvt_i_pd(const T* data, const double* window, double* dst) -> void {
    // Load and convert
    auto data_m512i = cvt_pd(data);
    // Convert u/int64 to doubles
    auto payload_m512d = __m512d{};
    if constexpr (std::is_signed_v<T>) {
        payload_m512d = _mm512_cvtepi64_pd(data_m512i);
    } else {
        payload_m512d = _mm512_cvtepu64_pd(data_m512i);
    }
    if (window != nullptr) {
        // Load window doubles
        auto window_m512d = _mm512_load_pd(window);
        // Multiply payload by window
        payload_m512d = _mm512_mul_pd(payload_m512d, window_m512d);
    }
    // Stored result into dst
    _mm512_store_pd(dst, payload_m512d);
}

} // namespace avx