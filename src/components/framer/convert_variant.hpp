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

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <type_traits>
#include <variant>

#include <composite/core/metadata.hpp>

#include "simd_fmv.hpp"

// ============================================================================
// Wire-format converters
// ============================================================================
//
// Each converter turns raw input bytes in one wire format into the framer's scalar type
// (float or int16_t), whole buffer at a time. SIMD selection uses GCC native function
// multiversioning (same-name run() overloads with [[gnu::target]]; see simd_fmv.hpp), so the
// dispatch costs one resolved call per BUFFER and the vector loop lives inside the selected
// kernel — not an indirect call per vector.
//
// Value semantics:
//  - Signed integers are widened numerically, NOT normalized (i8 -> [-128, 127]).
//  - Unsigned integers are offset binary (the wire convention for unsigned sample data, and
//    what the SDDS unsigned data modes carry): mid-scale is subtracted, so u8 -> [-128, 127]
//    and u16 -> [-32768, 32767], matching the signed formats' ranges. Widening them without
//    the offset would bias every sample by half full-scale and put a spurious DC spike in
//    downstream spectra. Bit-wise this is an XOR of the sign bit followed by the signed path.
//  - i32 is widened to float numerically; values beyond float's 24-bit mantissa round.
//  - f32 passes through bit-exact (modulo byte order).
//
// Byte order: converters take a `swap` flag (input endianness != native); 8-bit formats have
// no byte order. The swap branch is loop-invariant and predicted; swap masks are shared
// constants below.

namespace framer_swap_masks {
// pshufb control bytes for a byteswap, lowest byte first; the 16-byte pattern repeats per
// 128-bit lane, so one 64-byte constant serves the 128/256/512-bit loads as prefixes.
alignas(64) inline constexpr std::int8_t U16[64] = {
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
};
alignas(64) inline constexpr std::int8_t U32[64] = {
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
};
} // namespace framer_swap_masks

/// Read a possibly-misaligned value (packet payload slices land at arbitrary byte offsets;
/// a misaligned typed lvalue read is UB even where the hardware tolerates it).
template <typename V>
inline auto framer_load_unaligned(const V* p) -> V {
    V v;
    std::memcpy(&v, p, sizeof(V));
    return v;
}

template <typename InputT, typename OutputT>
struct converter;

// ---------------------------------------------------------------------------
// int8 -> float (numeric widening; single-byte, no byte order)
// ---------------------------------------------------------------------------
template <>
struct converter<int8_t, float> {
    converter() = default;
    explicit converter(bool /*swap: single byte*/) {}

    auto operator()(const uint8_t* input, float* output, std::size_t count) -> void {
        run(reinterpret_cast<const int8_t*>(input), output, count);
    }

private:
    static auto scalar_range(const int8_t* in, float* out, std::size_t i, std::size_t count) -> void {
        for (; i < count; ++i) {
            out[i] = static_cast<float>(in[i]);
        }
    }

    COMPS_FMV_DEFAULT
    static auto run(const int8_t* in, float* out, std::size_t count) -> void {
        scalar_range(in, out, 0, count);
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx2")]]
    static auto run(const int8_t* in, float* out, std::size_t count) -> void {
        std::size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            const auto v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + i));
            _mm256_storeu_ps(out + i, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)));
        }
        scalar_range(in, out, i, count);
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    static auto run(const int8_t* in, float* out, std::size_t count) -> void {
        std::size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            const auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
            _mm512_storeu_ps(out + i, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v)));
        }
        scalar_range(in, out, i, count);
    }
#endif
};

// ---------------------------------------------------------------------------
// uint8 (offset binary) -> float: XOR the sign bit == subtract 128, then widen
// ---------------------------------------------------------------------------
template <>
struct converter<uint8_t, float> {
    converter() = default;
    explicit converter(bool /*swap: single byte*/) {}

    auto operator()(const uint8_t* input, float* output, std::size_t count) -> void {
        run(input, output, count);
    }

private:
    static auto scalar_range(const uint8_t* in, float* out, std::size_t i, std::size_t count) -> void {
        for (; i < count; ++i) {
            out[i] = static_cast<float>(static_cast<int32_t>(in[i]) - 128);
        }
    }

    COMPS_FMV_DEFAULT
    static auto run(const uint8_t* in, float* out, std::size_t count) -> void {
        scalar_range(in, out, 0, count);
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx2")]]
    static auto run(const uint8_t* in, float* out, std::size_t count) -> void {
        const auto bias = _mm_set1_epi8(static_cast<char>(0x80));
        std::size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            auto v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in + i));
            v = _mm_xor_si128(v, bias);
            _mm256_storeu_ps(out + i, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)));
        }
        scalar_range(in, out, i, count);
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    static auto run(const uint8_t* in, float* out, std::size_t count) -> void {
        const auto bias = _mm_set1_epi8(static_cast<char>(0x80));
        std::size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
            v = _mm_xor_si128(v, bias);
            _mm512_storeu_ps(out + i, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v)));
        }
        scalar_range(in, out, i, count);
    }
#endif
};

// ---------------------------------------------------------------------------
// int16 -> float (numeric widening, optional byteswap)
// ---------------------------------------------------------------------------
template <>
struct converter<int16_t, float> {
    bool byteswap{false};

    converter() = default;
    explicit converter(bool swap) : byteswap(swap) {}

    auto operator()(const uint8_t* input, float* output, std::size_t count) -> void {
        run(reinterpret_cast<const int16_t*>(input), output, count, byteswap);
    }

private:
    static auto scalar_range(const int16_t* in, float* out, std::size_t i, std::size_t count, bool swap) -> void {
        for (; i < count; ++i) {
            auto v = framer_load_unaligned(in + i);
            if (swap) v = std::byteswap(v);
            out[i] = static_cast<float>(v);
        }
    }

    COMPS_FMV_DEFAULT
    static auto run(const int16_t* in, float* out, std::size_t count, bool swap) -> void {
        scalar_range(in, out, 0, count, swap);
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx2")]]
    static auto run(const int16_t* in, float* out, std::size_t count, bool swap) -> void {
        const auto mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(framer_swap_masks::U16));
        std::size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
            if (swap) v = _mm_shuffle_epi8(v, mask);
            _mm256_storeu_ps(out + i, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v)));
        }
        scalar_range(in, out, i, count, swap);
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    static auto run(const int16_t* in, float* out, std::size_t count, bool swap) -> void {
        const auto mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(framer_swap_masks::U16));
        std::size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
            if (swap) v = _mm256_shuffle_epi8(v, mask);
            _mm512_storeu_ps(out + i, _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v)));
        }
        scalar_range(in, out, i, count, swap);
    }
#endif
};

// ---------------------------------------------------------------------------
// uint16 (offset binary) -> float: optional byteswap, XOR the sign bit, widen
// ---------------------------------------------------------------------------
template <>
struct converter<uint16_t, float> {
    bool byteswap{false};

    converter() = default;
    explicit converter(bool swap) : byteswap(swap) {}

    auto operator()(const uint8_t* input, float* output, std::size_t count) -> void {
        run(reinterpret_cast<const uint16_t*>(input), output, count, byteswap);
    }

private:
    static auto scalar_range(const uint16_t* in, float* out, std::size_t i, std::size_t count, bool swap) -> void {
        for (; i < count; ++i) {
            auto v = framer_load_unaligned(in + i);
            if (swap) v = std::byteswap(v);
            out[i] = static_cast<float>(static_cast<int32_t>(v) - 32768);
        }
    }

    COMPS_FMV_DEFAULT
    static auto run(const uint16_t* in, float* out, std::size_t count, bool swap) -> void {
        scalar_range(in, out, 0, count, swap);
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx2")]]
    static auto run(const uint16_t* in, float* out, std::size_t count, bool swap) -> void {
        const auto mask = _mm_loadu_si128(reinterpret_cast<const __m128i*>(framer_swap_masks::U16));
        const auto bias = _mm_set1_epi16(static_cast<short>(0x8000));
        std::size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
            if (swap) v = _mm_shuffle_epi8(v, mask);
            v = _mm_xor_si128(v, bias);
            _mm256_storeu_ps(out + i, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(v)));
        }
        scalar_range(in, out, i, count, swap);
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    static auto run(const uint16_t* in, float* out, std::size_t count, bool swap) -> void {
        const auto mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(framer_swap_masks::U16));
        const auto bias = _mm256_set1_epi16(static_cast<short>(0x8000));
        std::size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
            if (swap) v = _mm256_shuffle_epi8(v, mask);
            v = _mm256_xor_si256(v, bias);
            _mm512_storeu_ps(out + i, _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(v)));
        }
        scalar_range(in, out, i, count, swap);
    }
#endif
};

// ---------------------------------------------------------------------------
// int32 -> float (numeric widening; values beyond 24 mantissa bits round)
// ---------------------------------------------------------------------------
template <>
struct converter<int32_t, float> {
    bool byteswap{false};

    converter() = default;
    explicit converter(bool swap) : byteswap(swap) {}

    auto operator()(const uint8_t* input, float* output, std::size_t count) -> void {
        run(reinterpret_cast<const int32_t*>(input), output, count, byteswap);
    }

private:
    static auto scalar_range(const int32_t* in, float* out, std::size_t i, std::size_t count, bool swap) -> void {
        for (; i < count; ++i) {
            auto v = framer_load_unaligned(in + i);
            if (swap) v = std::byteswap(v);
            out[i] = static_cast<float>(v);
        }
    }

    COMPS_FMV_DEFAULT
    static auto run(const int32_t* in, float* out, std::size_t count, bool swap) -> void {
        scalar_range(in, out, 0, count, swap);
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx2")]]
    static auto run(const int32_t* in, float* out, std::size_t count, bool swap) -> void {
        const auto mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(framer_swap_masks::U32));
        std::size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
            if (swap) v = _mm256_shuffle_epi8(v, mask);
            _mm256_storeu_ps(out + i, _mm256_cvtepi32_ps(v));
        }
        scalar_range(in, out, i, count, swap);
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    static auto run(const int32_t* in, float* out, std::size_t count, bool swap) -> void {
        const auto mask = _mm512_loadu_si512(framer_swap_masks::U32);
        std::size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            auto v = _mm512_loadu_si512(in + i);
            if (swap) v = _mm512_shuffle_epi8(v, mask);
            _mm512_storeu_ps(out + i, _mm512_cvtepi32_ps(v));
        }
        scalar_range(in, out, i, count, swap);
    }
#endif
};

// ---------------------------------------------------------------------------
// f32 passthrough (keyed as uint32 so the optional byteswap is integer work)
// ---------------------------------------------------------------------------
template <>
struct converter<uint32_t, float> {
    bool byteswap{false};

    converter() = default;
    explicit converter(bool swap) : byteswap(swap) {}

    auto operator()(const uint8_t* input, float* output, std::size_t count) -> void {
        if (!byteswap) {
            std::memcpy(output, input, count * sizeof(float));
            return;
        }
        run(reinterpret_cast<const uint32_t*>(input), output, count);
    }

private:
    // Kernels handle the byteswapping case only; the straight copy short-circuits above.
    static auto scalar_range(const uint32_t* in, float* out, std::size_t i, std::size_t count) -> void {
        for (; i < count; ++i) {
            out[i] = std::bit_cast<float>(std::byteswap(framer_load_unaligned(in + i)));
        }
    }

    COMPS_FMV_DEFAULT
    static auto run(const uint32_t* in, float* out, std::size_t count) -> void {
        scalar_range(in, out, 0, count);
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx2")]]
    static auto run(const uint32_t* in, float* out, std::size_t count) -> void {
        const auto mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(framer_swap_masks::U32));
        std::size_t i = 0;
        for (; i + 8 <= count; i += 8) {
            auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
            v = _mm256_shuffle_epi8(v, mask);
            _mm256_storeu_ps(out + i, _mm256_castsi256_ps(v));
        }
        scalar_range(in, out, i, count);
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    static auto run(const uint32_t* in, float* out, std::size_t count) -> void {
        const auto mask = _mm512_loadu_si512(framer_swap_masks::U32);
        std::size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            auto v = _mm512_loadu_si512(in + i);
            v = _mm512_shuffle_epi8(v, mask);
            _mm512_storeu_ps(out + i, _mm512_castsi512_ps(v));
        }
        scalar_range(in, out, i, count);
    }
#endif
};

// ---------------------------------------------------------------------------
// int8 -> int16 (numeric widening; single-byte, no byte order)
// ---------------------------------------------------------------------------
template <>
struct converter<int8_t, int16_t> {
    converter() = default;
    explicit converter(bool /*swap: single byte*/) {}

    auto operator()(const uint8_t* input, int16_t* output, std::size_t count) -> void {
        run(reinterpret_cast<const int8_t*>(input), output, count);
    }

private:
    static auto scalar_range(const int8_t* in, int16_t* out, std::size_t i, std::size_t count) -> void {
        for (; i < count; ++i) {
            out[i] = static_cast<int16_t>(in[i]);
        }
    }

    COMPS_FMV_DEFAULT
    static auto run(const int8_t* in, int16_t* out, std::size_t count) -> void {
        scalar_range(in, out, 0, count);
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx2")]]
    static auto run(const int8_t* in, int16_t* out, std::size_t count) -> void {
        std::size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            const auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_cvtepi8_epi16(v));
        }
        scalar_range(in, out, i, count);
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    static auto run(const int8_t* in, int16_t* out, std::size_t count) -> void {
        std::size_t i = 0;
        for (; i + 32 <= count; i += 32) {
            const auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
            _mm512_storeu_si512(out + i, _mm512_cvtepi8_epi16(v));
        }
        scalar_range(in, out, i, count);
    }
#endif
};

// ---------------------------------------------------------------------------
// uint8 (offset binary) -> int16: XOR the sign bit == subtract 128, then widen
// ---------------------------------------------------------------------------
template <>
struct converter<uint8_t, int16_t> {
    converter() = default;
    explicit converter(bool /*swap: single byte*/) {}

    auto operator()(const uint8_t* input, int16_t* output, std::size_t count) -> void {
        run(input, output, count);
    }

private:
    static auto scalar_range(const uint8_t* in, int16_t* out, std::size_t i, std::size_t count) -> void {
        for (; i < count; ++i) {
            out[i] = static_cast<int16_t>(static_cast<int32_t>(in[i]) - 128);
        }
    }

    COMPS_FMV_DEFAULT
    static auto run(const uint8_t* in, int16_t* out, std::size_t count) -> void {
        scalar_range(in, out, 0, count);
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx2")]]
    static auto run(const uint8_t* in, int16_t* out, std::size_t count) -> void {
        const auto bias = _mm_set1_epi8(static_cast<char>(0x80));
        std::size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i));
            v = _mm_xor_si128(v, bias);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), _mm256_cvtepi8_epi16(v));
        }
        scalar_range(in, out, i, count);
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    static auto run(const uint8_t* in, int16_t* out, std::size_t count) -> void {
        const auto bias = _mm256_set1_epi8(static_cast<char>(0x80));
        std::size_t i = 0;
        for (; i + 32 <= count; i += 32) {
            auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
            v = _mm256_xor_si256(v, bias);
            _mm512_storeu_si512(out + i, _mm512_cvtepi8_epi16(v));
        }
        scalar_range(in, out, i, count);
    }
#endif
};

// ---------------------------------------------------------------------------
// int16 -> int16 (passthrough; optional byteswap, straight copy short-circuits)
// ---------------------------------------------------------------------------
template <>
struct converter<int16_t, int16_t> {
    bool byteswap{false};

    converter() = default;
    explicit converter(bool swap) : byteswap(swap) {}

    auto operator()(const uint8_t* input, int16_t* output, std::size_t count) -> void {
        if (!byteswap) {
            std::memcpy(output, input, count * sizeof(int16_t));
            return;
        }
        run(reinterpret_cast<const int16_t*>(input), output, count);
    }

private:
    // Kernels handle the byteswapping case only; the straight copy short-circuits above.
    static auto scalar_range(const int16_t* in, int16_t* out, std::size_t i, std::size_t count) -> void {
        for (; i < count; ++i) {
            out[i] = std::byteswap(framer_load_unaligned(in + i));
        }
    }

    COMPS_FMV_DEFAULT
    static auto run(const int16_t* in, int16_t* out, std::size_t count) -> void {
        scalar_range(in, out, 0, count);
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx2")]]
    static auto run(const int16_t* in, int16_t* out, std::size_t count) -> void {
        const auto mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(framer_swap_masks::U16));
        std::size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
            v = _mm256_shuffle_epi8(v, mask);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), v);
        }
        scalar_range(in, out, i, count);
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    static auto run(const int16_t* in, int16_t* out, std::size_t count) -> void {
        const auto mask = _mm512_loadu_si512(framer_swap_masks::U16);
        std::size_t i = 0;
        for (; i + 32 <= count; i += 32) {
            auto v = _mm512_loadu_si512(in + i);
            v = _mm512_shuffle_epi8(v, mask);
            _mm512_storeu_si512(out + i, v);
        }
        scalar_range(in, out, i, count);
    }
#endif
};

// ---------------------------------------------------------------------------
// uint16 (offset binary) -> int16: optional byteswap, then XOR the sign bit
// ---------------------------------------------------------------------------
template <>
struct converter<uint16_t, int16_t> {
    bool byteswap{false};

    converter() = default;
    explicit converter(bool swap) : byteswap(swap) {}

    auto operator()(const uint8_t* input, int16_t* output, std::size_t count) -> void {
        run(reinterpret_cast<const uint16_t*>(input), output, count, byteswap);
    }

private:
    static auto scalar_range(const uint16_t* in, int16_t* out, std::size_t i, std::size_t count, bool swap) -> void {
        for (; i < count; ++i) {
            auto v = framer_load_unaligned(in + i);
            if (swap) v = std::byteswap(v);
            out[i] = static_cast<int16_t>(v ^ 0x8000u);
        }
    }

    COMPS_FMV_DEFAULT
    static auto run(const uint16_t* in, int16_t* out, std::size_t count, bool swap) -> void {
        scalar_range(in, out, 0, count, swap);
    }

#if COMPS_FMV_ENABLED
    [[gnu::target("avx2")]]
    static auto run(const uint16_t* in, int16_t* out, std::size_t count, bool swap) -> void {
        const auto mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(framer_swap_masks::U16));
        const auto bias = _mm256_set1_epi16(static_cast<short>(0x8000));
        std::size_t i = 0;
        for (; i + 16 <= count; i += 16) {
            auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i));
            if (swap) v = _mm256_shuffle_epi8(v, mask);
            v = _mm256_xor_si256(v, bias);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), v);
        }
        scalar_range(in, out, i, count, swap);
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    static auto run(const uint16_t* in, int16_t* out, std::size_t count, bool swap) -> void {
        const auto mask = _mm512_loadu_si512(framer_swap_masks::U16);
        const auto bias = _mm512_set1_epi16(static_cast<short>(0x8000));
        std::size_t i = 0;
        for (; i + 32 <= count; i += 32) {
            auto v = _mm512_loadu_si512(in + i);
            if (swap) v = _mm512_shuffle_epi8(v, mask);
            v = _mm512_xor_si512(v, bias);
            _mm512_storeu_si512(out + i, v);
        }
        scalar_range(in, out, i, count, swap);
    }
#endif
};

// ============================================================================
// Real -> complex in-place expansion
// ============================================================================
//
// The framer converts a real input chunk into the BACK half of its complex destination
// (scalar slots [count, 2*count)), then this kernel rewrites forward as (re, 0) pairs filling
// [0, 2*count). Forward iteration is aliasing-safe: iteration i reads slot count+i before
// writing slots 2i and 2i+1, and 2i+1 < count+j for every not-yet-read j > i. The vector
// versions satisfy the same bound (store end 2i+2w <= count+i+w whenever the loop admits
// iteration i), and zero-extension IS the interleave: u32->u64 makes (f32, 0.0f) pairs,
// u16->u32 makes (i16, 0) pairs.

namespace framer_expand_detail {

inline auto scalar_range_f32(float* buf, std::size_t i, std::size_t count) -> void {
    for (; i < count; ++i) {
        const auto value = buf[count + i];
        buf[2 * i] = value;
        buf[2 * i + 1] = 0.0f;
    }
}

COMPS_FMV_DEFAULT
inline auto expand_f32(float* buf, std::size_t count) -> void {
    scalar_range_f32(buf, 0, count);
}

#if COMPS_FMV_ENABLED
[[gnu::target("avx2")]]
inline auto expand_f32(float* buf, std::size_t count) -> void {
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf + count + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf + 2 * i), _mm256_cvtepu32_epi64(v));
    }
    scalar_range_f32(buf, i, count);
}

[[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
inline auto expand_f32(float* buf, std::size_t count) -> void {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + count + i));
        _mm512_storeu_si512(buf + 2 * i, _mm512_cvtepu32_epi64(v));
    }
    scalar_range_f32(buf, i, count);
}
#endif

inline auto scalar_range_i16(int16_t* buf, std::size_t i, std::size_t count) -> void {
    for (; i < count; ++i) {
        const auto value = buf[count + i];
        buf[2 * i] = value;
        buf[2 * i + 1] = 0;
    }
}

COMPS_FMV_DEFAULT
inline auto expand_i16(int16_t* buf, std::size_t count) -> void {
    scalar_range_i16(buf, 0, count);
}

#if COMPS_FMV_ENABLED
[[gnu::target("avx2")]]
inline auto expand_i16(int16_t* buf, std::size_t count) -> void {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf + count + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf + 2 * i), _mm256_cvtepu16_epi32(v));
    }
    scalar_range_i16(buf, i, count);
}

[[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
inline auto expand_i16(int16_t* buf, std::size_t count) -> void {
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + count + i));
        _mm512_storeu_si512(buf + 2 * i, _mm512_cvtepu16_epi32(v));
    }
    scalar_range_i16(buf, i, count);
}
#endif

} // namespace framer_expand_detail

/// Expand `count` real scalars staged at buf[count .. 2*count) into (re, 0) complex pairs
/// filling buf[0 .. 2*count), in place.
template <typename ScalarT>
inline auto expand_real_to_complex(ScalarT* buf, std::size_t count) -> void {
    if constexpr (std::is_same_v<ScalarT, float>) {
        framer_expand_detail::expand_f32(buf, count);
    } else {
        static_assert(std::is_same_v<ScalarT, int16_t>);
        framer_expand_detail::expand_i16(buf, count);
    }
}

// ============================================================================
// Variant type holding all converters - specialized by output type
// ============================================================================

template <typename OutputT>
struct converter_variant_traits;

// Float output: every supported wire format
template <>
struct converter_variant_traits<float> {
    using type = std::variant<
        converter<int8_t, float>,
        converter<uint8_t, float>,
        converter<int16_t, float>,
        converter<uint16_t, float>,
        converter<int32_t, float>,
        converter<uint32_t, float>
    >;
};

// int16_t output: 8/16-bit integer inputs only (no narrowing paths)
template <>
struct converter_variant_traits<int16_t> {
    using type = std::variant<
        converter<int8_t, int16_t>,
        converter<uint8_t, int16_t>,
        converter<int16_t, int16_t>,
        converter<uint16_t, int16_t>
    >;
};

template <typename OutputT>
using converter_variant = typename converter_variant_traits<OutputT>::type;

// ============================================================================
// Visitor for invoking converters
// ============================================================================

template <typename OutputT>
struct convert_visitor {
    const uint8_t* input;
    OutputT* output;
    std::size_t count;

    auto operator()(auto& conv) const -> void {
        conv(input, output, count);
    }
};

// Convenience function for conversion
template <typename OutputT>
inline auto convert(converter_variant<OutputT>& converter_var,
                   const uint8_t* input, OutputT* output, std::size_t count) -> void {
    std::visit(convert_visitor<OutputT>{input, output, count}, converter_var);
}

// ============================================================================
// Supported-input-format table - the single source of truth per output type
// ============================================================================
//
// Whether a format is supported, its byte stride, and which converter handles it are all
// answered by ONE table entry, so they cannot disagree (a format that "passes the support
// check" but has no converter is impossible by construction). Keyed on the fields that
// identify a wire format (complexity, sample type, bit width); endianness is handled
// uniformly by the converters' optional byteswap, so it is not part of the key.

template <typename OutputT>
struct input_format_support {
    bool is_complex;
    composite::data_type type;
    uint32_t bit_width;
    std::size_t bytes_per_sample;                     // I+Q together for complex formats
    converter_variant<OutputT> (*make)(bool swap);    // converter factory
};

template <typename OutputT>
inline constexpr std::array<input_format_support<OutputT>, 0> supported_input_formats{};

// Float output: integer inputs are widened numerically (NOT normalized) with unsigned formats
// read as offset binary (see the converter block comment); f32 passes through.
template <>
inline constexpr auto supported_input_formats<float> = std::to_array<input_format_support<float>>({
    {false, composite::data_type::signed_integer, 8, 1,
     [](bool swap) -> converter_variant<float> { return converter<int8_t, float>(swap); }},
    {true, composite::data_type::signed_integer, 8, 2,
     [](bool swap) -> converter_variant<float> { return converter<int8_t, float>(swap); }},
    {false, composite::data_type::unsigned_integer, 8, 1,
     [](bool swap) -> converter_variant<float> { return converter<uint8_t, float>(swap); }},
    {true, composite::data_type::unsigned_integer, 8, 2,
     [](bool swap) -> converter_variant<float> { return converter<uint8_t, float>(swap); }},
    {false, composite::data_type::signed_integer, 16, 2,
     [](bool swap) -> converter_variant<float> { return converter<int16_t, float>(swap); }},
    {true, composite::data_type::signed_integer, 16, 4,
     [](bool swap) -> converter_variant<float> { return converter<int16_t, float>(swap); }},
    {false, composite::data_type::unsigned_integer, 16, 2,
     [](bool swap) -> converter_variant<float> { return converter<uint16_t, float>(swap); }},
    {true, composite::data_type::unsigned_integer, 16, 4,
     [](bool swap) -> converter_variant<float> { return converter<uint16_t, float>(swap); }},
    {false, composite::data_type::signed_integer, 32, 4,
     [](bool swap) -> converter_variant<float> { return converter<int32_t, float>(swap); }},
    {true, composite::data_type::signed_integer, 32, 8,
     [](bool swap) -> converter_variant<float> { return converter<int32_t, float>(swap); }},
    {false, composite::data_type::floating_point, 32, 4,
     [](bool swap) -> converter_variant<float> { return converter<uint32_t, float>(swap); }},
    {true, composite::data_type::floating_point, 32, 8,
     [](bool swap) -> converter_variant<float> { return converter<uint32_t, float>(swap); }},
});

// int16_t output: 8/16-bit integer inputs only (no narrowing paths), unsigned read as
// offset binary.
template <>
inline constexpr auto supported_input_formats<int16_t> = std::to_array<input_format_support<int16_t>>({
    {false, composite::data_type::signed_integer, 8, 1,
     [](bool swap) -> converter_variant<int16_t> { return converter<int8_t, int16_t>(swap); }},
    {true, composite::data_type::signed_integer, 8, 2,
     [](bool swap) -> converter_variant<int16_t> { return converter<int8_t, int16_t>(swap); }},
    {false, composite::data_type::unsigned_integer, 8, 1,
     [](bool swap) -> converter_variant<int16_t> { return converter<uint8_t, int16_t>(swap); }},
    {true, composite::data_type::unsigned_integer, 8, 2,
     [](bool swap) -> converter_variant<int16_t> { return converter<uint8_t, int16_t>(swap); }},
    {false, composite::data_type::signed_integer, 16, 2,
     [](bool swap) -> converter_variant<int16_t> { return converter<int16_t, int16_t>(swap); }},
    {true, composite::data_type::signed_integer, 16, 4,
     [](bool swap) -> converter_variant<int16_t> { return converter<int16_t, int16_t>(swap); }},
    {false, composite::data_type::unsigned_integer, 16, 2,
     [](bool swap) -> converter_variant<int16_t> { return converter<uint16_t, int16_t>(swap); }},
    {true, composite::data_type::unsigned_integer, 16, 4,
     [](bool swap) -> converter_variant<int16_t> { return converter<uint16_t, int16_t>(swap); }},
});

/// Look up the support entry for @p fmt, or nullptr if the (input format, OutputT)
/// combination is not convertible.
template <typename OutputT>
inline auto find_input_format(const composite::data_format& fmt)
    -> const input_format_support<OutputT>* {
    for (const auto& entry : supported_input_formats<OutputT>) {
        if (entry.is_complex == fmt.is_complex && entry.type == fmt.type &&
            entry.bit_width == fmt.bit_width) {
            return &entry;
        }
    }
    return nullptr;
}
