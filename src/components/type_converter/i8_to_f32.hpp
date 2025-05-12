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

#pragma once

#include <cstdint>
#include <immintrin.h>
#include <span>

[[gnu::target("default")]]
auto convert(std::span<const int8_t> src, std::span<float> dst, bool is_complex=false) -> void {
    auto len = is_complex ? src.size() * 2 : src.size();
    for (auto i = std::size_t{}; i < len; ++i) {
        dst[i] = src[i];
    }
}

[[gnu::target("avx2")]]
auto convert(std::span<const int8_t> src, std::span<float> dst, bool is_complex=false) -> void {
    auto len = is_complex ? src.size() * 2 : src.size();
    auto i = std::size_t{};
    for (; i + 8 <= len; i += 16) {
        auto vi8 = _mm_loadu_si128((const __m128i*)&src[i]);
        auto vi32 = _mm256_cvtepi8_epi32(vi8);
        auto vf = _mm256_cvtepi32_ps(vi32);
        _mm256_store_ps(&dst[i], vf);
    }
    for (; i < len; ++i) {
        dst[i] = src[i];
    }
}

[[gnu::target("avx512f")]]
auto convert(std::span<const int8_t> src, std::span<float> dst, bool is_complex=false) -> void {
    auto len = is_complex ? src.size() * 2 : src.size();
    auto i = std::size_t{};
    for (; i + 16 <= len; i += 16) {
        auto vi8 = _mm_loadu_si128((const __m128i*)&src[i]);
        auto vi32 = _mm512_cvtepi8_epi32(vi8);
        auto vf = _mm512_cvtepi32_ps(vi32);
        _mm512_store_ps(&dst[i], vf);
    }
    for (; i < len; ++i) {
        dst[i] = src[i];
    }
}
