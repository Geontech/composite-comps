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
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <span>

[[gnu::target("default")]]
auto convert(std::span<const float> src, std::span<int16_t> dst, bool is_complex=false) -> void {
    auto len = is_complex ? src.size() * 2 : src.size();
    for (auto i = std::size_t{}; i < len; ++i) {
        auto val = std::round(src[i]);
        if (val > 32767.0f) { val = 32767.0f; }
        if (val < -32768.0f) { val = -32768.0f; }
        dst[i] = static_cast<int16_t>(val);
    }
}

// [[gnu::target("avx2")]]
// auto convert(std::span<const float> src, std::span<int16_t> dst, bool is_complex=false) -> void {
//     auto len = is_complex ? src.size() * 2 : src.size();
//     auto i = std::size_t{};
//     for (; i + 8 <= len; i += 8) {
//         auto vf = _mm256_loadu_ps(&src[i]);
//         auto vi32 = _mm256_cvtps_epi32(vf);
//         auto vi16 = _mm256_cvtsepi32_epi16(vi32);
//         _mm256_store_si256((__m256i*)(&dst[i]), vi16);
//     }
//     for (; i < len; ++i) {
//         auto val = std::round(src[i]);
//         if (val > 32767.0f) { val = 32767.0f; }
//         if (val < -32768.0f) { val = -32768.0f; }
//         dst[i] = static_cast<int16_t>(val);
//     }
// }

[[gnu::target("avx512f")]]
auto convert(std::span<const float> src, std::span<int16_t> dst, bool is_complex=false) -> void {
    auto len = is_complex ? src.size() * 2 : src.size();
    auto i = std::size_t{};
    for (; i + 16 <= len; i += 16) {
        auto vf = _mm512_loadu_ps(&src[i]);
        auto vi32 = _mm512_cvtps_epi32(vf);
        auto vi16 = _mm512_cvtsepi32_epi16(vi32);
        _mm256_store_si256((__m256i*)(&dst[i]), vi16);
    }
    for (; i < len; ++i) {
        auto val = std::round(src[i]);
        if (val > 32767.0f) { val = 32767.0f; }
        if (val < -32768.0f) { val = -32768.0f; }
        dst[i] = static_cast<int16_t>(val);
    }
}
