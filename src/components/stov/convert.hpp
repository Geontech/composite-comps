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
#include <cstdint>
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

namespace swap {

const __m128i shuffle_u16_8 = _mm_set_epi8(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1);
const __m256i shuffle_u16_16 = _mm256_set_epi8(
    30,31,28,29,26,27,24,25,22,23,20,21,18,19,16,17,
    14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1
);
const __m256i shuffle_u32_8 = _mm256_set_epi8(
    28,29,30,31,24,25,26,27,20,21,22,23,16,17,18,19,
    12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3
);
const __m512i shuffle_u32_16 = _mm512_set_epi8(
    60,61,62,63,56,57,58,59,52,53,54,55,48,49,50,51,
    44,45,46,47,40,41,42,43,36,37,38,39,32,33,34,35,
    28,29,30,31,24,25,26,27,20,21,22,23,16,17,18,19,
    12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3
);
const __m512i shuffle_u64 = _mm512_set_epi8(
    56,57,58,59,60,61,62,63,48,49,50,51,52,53,54,55,
    40,41,42,43,44,45,46,47,32,33,34,35,36,37,38,39,
    24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23,
    8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7
);

} // namespace swap

template <typename T>
requires avxable_pd<T>
inline auto cvt_64(const T* data, bool byteswap) -> __m512i {
    auto retval = __m512i{};
    // Load payload (type T))
    // Up-convert based on incoming type to 64-bit variant (signed or unsigned)
    if constexpr (std::is_same_v<T, uint16_t>) {
        auto loaded = _mm_loadu_epi16(data);
        if (byteswap) {
            loaded = _mm_shuffle_epi8(loaded, swap::shuffle_u16_8);
        }
        retval = _mm512_cvtepu16_epu64(loaded);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        auto loaded = _mm_loadu_epi16(data);
        if (byteswap) {
            loaded = _mm_shuffle_epi8(loaded, swap::shuffle_u16_8);
        }
        retval = _mm512_cvtepi16_epi64(loaded);
    } else if constexpr (std::is_same_v<T, uint32_t>) {
        auto loaded = _mm256_loadu_epi32(data);
        if (byteswap) {
            loaded = _mm256_shuffle_epi8(loaded, swap::shuffle_u32_8);
        }
        retval = _mm512_cvtepu32_epu64(loaded);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        auto loaded = _mm256_loadu_epi32(data);
        if (byteswap) {
            loaded = _mm256_shuffle_epi8(loaded, swap::shuffle_u32_8);
        }
        retval = _mm512_cvtepi32_epi64(loaded);
    } else if constexpr (std::is_same_v<T, uint64_t> || std::is_same_v<T, int64_t>) {
        retval = _mm512_loadu_epi64(data);
        if (byteswap) {
            retval = _mm512_shuffle_epi8(retval, swap::shuffle_u64);
        }
    }
    return retval;
}

template <typename T>
requires avxable_ps<T>
inline auto cvt_32(const T* data, bool byteswap) -> __m512i {
    auto retval = __m512i{};
    // Load payload (type T))
    // Up-convert based on incoming type to 32-bit variant (signed or unsigned)
    if constexpr (std::is_same_v<T, uint16_t>) {
        auto loaded = _mm256_loadu_epi16(data);
        if (byteswap) {
            loaded = _mm256_shuffle_epi8(loaded, swap::shuffle_u16_16);
        }
        retval = _mm512_cvtepu16_epu32(loaded);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        auto loaded = _mm256_loadu_epi16(data);
        if (byteswap) {
            loaded = _mm256_shuffle_epi8(loaded, swap::shuffle_u16_16);
        }
        retval = _mm512_cvtepi16_epi32(loaded);
    } else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) {
        retval = _mm512_loadu_epi32(data);
        if (byteswap) {
            retval = _mm512_shuffle_epi8(retval, swap::shuffle_u32_16);
        }
    }
    return retval;
}

} // namespace avx

template <typename T>
auto convert(const T* data, float* dst, bool byteswap=false) -> void {
    // Load and convert
    auto data_m512i = avx::cvt_32(data, byteswap);
    // Convert u/int64 to floats
    auto payload_m512 = __m512{};
    if constexpr (std::is_signed_v<T>) {
        payload_m512 = _mm512_cvtepi32_ps(data_m512i);
    } else {
        payload_m512 = _mm512_cvtepu32_ps(data_m512i);
    }
    // Stored result into dst
    _mm512_store_ps(dst, payload_m512);
}

template <typename T>
auto convert(const T* data, double* dst, bool byteswap=false) -> void {
    // Load and convert
    auto data_m512i = avx::cvt_64(data, byteswap);
    // Convert u/int64 to floats
    auto payload_m512 = __m512d{};
    if constexpr (std::is_signed_v<T>) {
        payload_m512 = _mm512_cvtepi64_pd(data_m512i);
    } else {
        payload_m512 = _mm512_cvtepu64_pd(data_m512i);
    }
    // Stored result into dst
    _mm512_store_pd(dst, payload_m512);
}