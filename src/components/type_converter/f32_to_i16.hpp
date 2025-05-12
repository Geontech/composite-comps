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

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <span>

[[gnu::target("default")]]
auto convert(std::span<const float> src, std::span<int16_t> dst, bool is_complex=false) -> void {
    const auto min_val = (float)SHRT_MIN;
    const auto max_val = (float)SHRT_MAX;
    auto len = is_complex ? src.size() * 2 : src.size();
    for (auto i = std::size_t{}; i < len; ++i) {
        auto val = std::round(src[i]);
        dst[i] = static_cast<int16_t>(std::clamp(val, min_val, max_val));
    }
}

[[gnu::target("avx2")]]
auto convert(std::span<const float> src, std::span<int16_t> dst, bool is_complex=false) -> void {
    auto len = is_complex ? src.size() * 2 : src.size();
    auto i = std::size_t{};
    // Requires: float -> int32 -> pack int32 -> int16
    const auto permute_mask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0); // For packing result
    for (; i + 16 <= len; i += 16) {
        // Load 16 floats (two 256-bit vectors)
        auto vf_1 = _mm256_loadu_ps(&src[i]);
        auto vf_2 = _mm256_loadu_ps(&src[i + 8]);

        // Convert float -> int32 (rounding towards zero - truncation)
        auto vi32_1 = _mm256_cvtps_epi32(vf_1); // 8x int32
        auto vi32_2 = _mm256_cvtps_epi32(vf_2); // 8x int32

        // Pack int32 -> int16 with saturation
        // Result layout is [a0..a3 | b0..b3 | a4..a7 | b4..b7] (16-bit elements)
        auto vi16 = _mm256_packs_epi32(vi32_1, vi32_2);

        // Permute to get the order [a0..a7 | b0..b7]
        vi16 = _mm256_permutevar8x32_epi32(vi16, permute_mask);

        // Store the 256-bit result (16 shorts)
        _mm256_store_si256((__m256i*)(&dst[i]), vi16);
    }
    const auto min_val = (float)SHRT_MIN;
    const auto max_val = (float)SHRT_MAX;
    for (; i < len; ++i) {
        dst[i] = static_cast<int16_t>(std::clamp(std::round(src[i]), min_val, max_val));
    }
}

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
    const auto min_val = (float)SHRT_MIN;
    const auto max_val = (float)SHRT_MAX;
    for (; i < len; ++i) {
        dst[i] = static_cast<int16_t>(std::clamp(std::round(src[i]), min_val, max_val));
    }
}
