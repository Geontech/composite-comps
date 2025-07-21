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

namespace swap {

__m128i shuffle_u16_8;
__m256i shuffle_u16_16;
__m256i shuffle_u32_8;
__m256i shuffle_u64_4;
__m512i shuffle_u16_32;
__m512i shuffle_u32_16;
__m512i shuffle_u64_8;

[[gnu::target("avx512f,avx2")]]
auto init_avx2() {
    shuffle_u16_8 = _mm_set_epi8(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1);
    shuffle_u16_16 = _mm256_set_epi8(
        14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1,
        14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1
    );
    shuffle_u32_8 = _mm256_set_epi8(
        12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3,
        12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3
    );
    shuffle_u64_4 = _mm256_set_epi8(
        8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,
        8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7
    );
}

[[gnu::target("avx512f")]]
auto init_avx512() {
    init_avx2();
    shuffle_u16_32 = _mm512_set_epi8(
        14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1,
        14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1,
        14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1,
        14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1
    );
    shuffle_u32_16 = _mm512_set_epi8(
        12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3,
        12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3,
        12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3,
        12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3
    );
    shuffle_u64_8 = _mm512_set_epi8(
        8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,
        8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,
        8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,
        8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7
    );
}

} // namespace swap
} // namespace avx

template <typename It, typename Ot>
class converter {};

template <typename T>
requires avx::avxable_ps<T>
class converter<T, float> {
public:
    converter() = default;

    explicit converter(bool byteswap) : m_byteswap(byteswap) {
        // Initialize the function pointer based on CPU features
        if (__builtin_cpu_supports("avx512f")) {
            avx::swap::init_avx512();
            process_func = &converter::process_avx512;
        } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            avx::swap::init_avx2();
            process_func = &converter::process_avx2;
        }
    }

    auto process(const T* data, float* dst) -> void {
        (this->*process_func)(data, dst);
    }

private:
    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    auto process_avx512(const T* data, float* dst) -> void {
        auto data_m512i = __m512i{};
        // Load payload (type T))
        // Up-convert based on incoming type to 32-bit variant (signed or unsigned)
        if constexpr (std::is_same_v<T, uint16_t>) {
            auto loaded = _mm256_loadu_epi16(data);
            if (m_byteswap) {
                loaded = _mm256_shuffle_epi8(loaded, avx::swap::shuffle_u16_16);
            }
            data_m512i = _mm512_cvtepu16_epi32(loaded);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            auto loaded = _mm256_loadu_epi16(data);
            if (m_byteswap) {
                loaded = _mm256_shuffle_epi8(loaded, avx::swap::shuffle_u16_16);
            }
            data_m512i = _mm512_cvtepi16_epi32(loaded);
        } else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) {
            data_m512i = _mm512_loadu_epi32(data);
            if (m_byteswap) {
                data_m512i = _mm512_shuffle_epi8(data_m512i, avx::swap::shuffle_u32_16);
            }
        }
        // Convert u/int32 to floats
        auto payload_m512 = __m512{};
        if constexpr (std::is_signed_v<T>) {
            payload_m512 = _mm512_cvtepi32_ps(data_m512i);
        } else {
            payload_m512 = _mm512_cvtepu32_ps(data_m512i);
        }
        // Stored result into dst
        _mm512_store_ps(dst, payload_m512);
    }

    [[gnu::target("avx2")]]
    auto process_avx2(const T* data, float* dst) -> void {
        auto data_m256i = __m256i{};
        // Load payload (type T))
        // Up-convert based on incoming type to 32-bit variant (signed or unsigned)
        if constexpr (std::is_same_v<T, uint16_t>) {
            auto loaded = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
            if (m_byteswap) {
                loaded = _mm_shuffle_epi8(loaded, avx::swap::shuffle_u16_8);
            }
            data_m256i = _mm256_cvtepu16_epi32(loaded);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            auto loaded = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
            if (m_byteswap) {
                loaded = _mm_shuffle_epi8(loaded, avx::swap::shuffle_u16_8);
            }
            data_m256i = _mm256_cvtepi16_epi32(loaded);
        } else if constexpr (std::is_same_v<T, uint32_t> || std::is_same_v<T, int32_t>) {
            data_m256i = _mm256_loadu_epi32(data);
            if (m_byteswap) {
                data_m256i = _mm256_shuffle_epi8(data_m256i, avx::swap::shuffle_u32_8);
            }
        }
        // Convert u/int32 to floats
        auto payload = __m256{};
        if constexpr (std::is_signed_v<T>) {
            payload = _mm256_cvtepi32_ps(data_m256i);
        } else {
            payload = _mm256_cvtepi32_ps(_mm256_and_si256(data_m256i, _mm256_set1_epi32(0xFFFFFFFF)));
        }
        // Stored result into dst
        _mm256_store_ps(dst, payload);
    }

    auto (converter::*process_func)(const T*, float*) -> void;
    bool m_byteswap{};

}; // class converter<T, float>

template <>
class converter<int16_t, int16_t> {
public:
    converter() = default;

    explicit converter(bool byteswap) : m_byteswap(byteswap) {
        // Initialize the function pointer based on CPU features
        if (__builtin_cpu_supports("avx512f")) {
            avx::swap::init_avx512();
            process_func = &converter::process_avx512;
        } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            avx::swap::init_avx2();
            process_func = &converter::process_avx2;
        }
    }

    auto process(const int16_t* data, int16_t* dst) -> void {
        (this->*process_func)(data, dst);
    }

private:
    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    auto process_avx512(const int16_t* data, int16_t* dst) -> void {
        auto data_m512i = _mm512_loadu_epi16(data);
        if (m_byteswap) {
            data_m512i = _mm512_shuffle_epi8(data_m512i, avx::swap::shuffle_u16_32);
        }
        // Stored result into dst
        _mm512_storeu_epi16(dst, data_m512i);
    }

    [[gnu::target("avx2")]]
    auto process_avx2(const int16_t* data, int16_t* dst) -> void {
        auto data_m256i = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
        if (m_byteswap) {
            data_m256i = _mm256_shuffle_epi8(data_m256i, avx::swap::shuffle_u16_16);
        }
        // Stored result into dst
        _mm256_store_si256(reinterpret_cast<__m256i*>(dst), data_m256i);
    }

    auto (converter::*process_func)(const int16_t*, int16_t*) -> void;
    bool m_byteswap{};

}; // class converter<int16_t, int16_t>
