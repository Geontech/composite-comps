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
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <mutex>
#include <type_traits>
#include <variant>

#include <composite/core/metadata.hpp>

namespace avx {

template <typename T>
constexpr bool avxable_ps =
    std::is_same_v<T, int16_t> || std::is_same_v<T, uint16_t> ||
    std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>;

namespace swap {

inline __m128i shuffle_u16_8;
inline __m256i shuffle_u16_16;
inline __m256i shuffle_u32_8;
inline __m256i shuffle_u64_4;
inline __m512i shuffle_u16_32;
inline __m512i shuffle_u32_16;
inline __m512i shuffle_u64_8;

inline std::once_flag avx2_once_flag;
inline std::once_flag avx512_once_flag;

[[gnu::target("avx2")]]
inline auto init_avx2() {
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
inline auto init_avx512() {
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

/// SIMD tier selected at runtime for the converters below.
enum class simd_level { scalar, avx2, avx512 };

/// One CPU-feature probe shared by every converter. The avx512 tier requires BW/VL/DQ in
/// addition to F because the converters use 8/16-bit shuffles, epi8/epi16 widening
/// conversions, and 256-bit EVEX loads — gating on avx512f alone would dispatch illegal
/// instructions on an F-only CPU. Also runs the one-time shuffle-mask initialization for
/// the selected tier.
inline auto detect_simd_level() -> simd_level {
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq")) {
        std::call_once(swap::avx512_once_flag, swap::init_avx512);
        return simd_level::avx512;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        std::call_once(swap::avx2_once_flag, swap::init_avx2);
        return simd_level::avx2;
    }
    return simd_level::scalar;
}

} // namespace avx

// ============================================================================
// Converter callable structs - eliminates virtual function overhead
// ============================================================================

// Forward declarations for all converter types
template <typename InputT, typename OutputT>
struct converter;

// Specialization: int8_t -> float
template <>
struct converter<int8_t, float> {
    converter() {
        init_cpu_features();
    }

    explicit converter([[maybe_unused]] bool swap) {
        // int8_t doesn't need byteswap (single byte)
        init_cpu_features();
    }

    auto operator()(const uint8_t* input, float* output, std::size_t count) -> void {
        const int8_t* data = reinterpret_cast<const int8_t*>(input);
        std::size_t processed = 0;

        // Process full vectors using CPU-specific implementation
        for (; processed + samples_per_vector <= count; processed += samples_per_vector) {
            (this->*process_func)(data + processed, output + processed);
        }

        // Process remaining samples scalar
        for (; processed < count; ++processed) {
            output[processed] = static_cast<float>(data[processed]);
        }
    }

private:
    auto init_cpu_features() -> void {
        switch (avx::detect_simd_level()) {
            case avx::simd_level::avx512:
                process_func = &converter::process_avx512;
                samples_per_vector = 16;
                break;
            case avx::simd_level::avx2:
                process_func = &converter::process_avx2;
                samples_per_vector = 8;
                break;
            case avx::simd_level::scalar:
                process_func = &converter::process_single;
                samples_per_vector = 1;
                break;
        }
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    auto process_avx512(const int8_t* data, float* dst) -> void {
        auto loaded = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
        auto data_m512i = _mm512_cvtepi8_epi32(loaded);
        auto payload_m512 = _mm512_cvtepi32_ps(data_m512i);
        _mm512_storeu_ps(dst, payload_m512);  // Use unaligned store for safety
    }

    [[gnu::target("avx2")]]
    auto process_avx2(const int8_t* data, float* dst) -> void {
        auto loaded = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(data));
        auto data_m256i = _mm256_cvtepi8_epi32(loaded);
        auto payload = _mm256_cvtepi32_ps(data_m256i);
        _mm256_storeu_ps(dst, payload);  // Use unaligned store for safety
    }

    // Process single sample (for systems without AVX)
    auto process_single(const int8_t* data, float* dst) -> void {
        dst[0] = static_cast<float>(data[0]);
    }

    auto (converter::*process_func)(const int8_t*, float*) -> void = &converter::process_single;
    std::size_t samples_per_vector{1};
};

// Specialization: int16_t -> float
template <>
struct converter<int16_t, float> {
    bool byteswap{false};

    converter() {
        init_cpu_features();
    }

    explicit converter(bool swap) : byteswap(swap) {
        init_cpu_features();
    }

    auto operator()(const uint8_t* input, float* output, std::size_t count) -> void {
        const int16_t* data = reinterpret_cast<const int16_t*>(input);
        std::size_t processed = 0;

        // Process full vectors using CPU-specific implementation
        for (; processed + samples_per_vector <= count; processed += samples_per_vector) {
            (this->*process_func)(data + processed, output + processed);
        }

        // Process remaining samples scalar
        for (; processed < count; ++processed) {
            int16_t value = data[processed];
            if (byteswap) value = std::byteswap(value);
            output[processed] = static_cast<float>(value);
        }
    }

private:
    auto init_cpu_features() -> void {
        switch (avx::detect_simd_level()) {
            case avx::simd_level::avx512:
                process_func = &converter::process_avx512;
                samples_per_vector = 16;
                break;
            case avx::simd_level::avx2:
                process_func = &converter::process_avx2;
                samples_per_vector = 8;
                break;
            case avx::simd_level::scalar:
                process_func = &converter::process_single;
                samples_per_vector = 1;
                break;
        }
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    auto process_avx512(const int16_t* data, float* dst) -> void {
        auto loaded = _mm256_loadu_epi16(data);
        if (byteswap) {
            loaded = _mm256_shuffle_epi8(loaded, avx::swap::shuffle_u16_16);
        }
        auto data_m512i = _mm512_cvtepi16_epi32(loaded);
        auto payload_m512 = _mm512_cvtepi32_ps(data_m512i);
        _mm512_storeu_ps(dst, payload_m512);  // Use unaligned store for safety
    }

    [[gnu::target("avx2")]]
    auto process_avx2(const int16_t* data, float* dst) -> void {
        auto loaded = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
        if (byteswap) {
            loaded = _mm_shuffle_epi8(loaded, avx::swap::shuffle_u16_8);
        }
        auto data_m256i = _mm256_cvtepi16_epi32(loaded);
        auto payload = _mm256_cvtepi32_ps(data_m256i);
        _mm256_storeu_ps(dst, payload);  // Use unaligned store for safety
    }

    // Process single sample (for systems without AVX)
    auto process_single(const int16_t* data, float* dst) -> void {
        int16_t value = data[0];
        if (byteswap) value = std::byteswap(value);
        dst[0] = static_cast<float>(value);
    }

    auto (converter::*process_func)(const int16_t*, float*) -> void = &converter::process_single;
    std::size_t samples_per_vector{1};
};

// Specialization: uint32_t -> float (for complex_cf32 passthrough with byteswap)
template <>
struct converter<uint32_t, float> {
    bool byteswap{false};

    converter() {
        init_cpu_features();
    }

    explicit converter(bool swap) : byteswap(swap) {
        init_cpu_features();
    }

    auto operator()(const uint8_t* input, float* output, std::size_t count) -> void {
        const uint32_t* data = reinterpret_cast<const uint32_t*>(input);
        std::size_t processed = 0;

        // Process full vectors using CPU-specific implementation
        for (; processed + samples_per_vector <= count; processed += samples_per_vector) {
            (this->*process_func)(data + processed, output + processed);
        }

        // Process remaining samples scalar
        for (; processed < count; ++processed) {
            uint32_t raw = data[processed];
            if (byteswap) raw = std::byteswap(raw);
            output[processed] = std::bit_cast<float>(raw);
        }
    }

private:
    auto init_cpu_features() -> void {
        switch (avx::detect_simd_level()) {
            case avx::simd_level::avx512:
                process_func = &converter::process_avx512;
                samples_per_vector = 16;
                break;
            case avx::simd_level::avx2:
                process_func = &converter::process_avx2;
                samples_per_vector = 8;
                break;
            case avx::simd_level::scalar:
                process_func = &converter::process_single;
                samples_per_vector = 1;
                break;
        }
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    auto process_avx512(const uint32_t* data, float* dst) -> void {
        auto data_m512i = _mm512_loadu_epi32(data);
        if (byteswap) {
            data_m512i = _mm512_shuffle_epi8(data_m512i, avx::swap::shuffle_u32_16);
        }
        _mm512_storeu_ps(dst, _mm512_castsi512_ps(data_m512i));  // Use unaligned store for safety
    }

    [[gnu::target("avx2")]]
    auto process_avx2(const uint32_t* data, float* dst) -> void {
        auto data_m256i = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
        if (byteswap) {
            data_m256i = _mm256_shuffle_epi8(data_m256i, avx::swap::shuffle_u32_8);
        }
        _mm256_storeu_ps(dst, _mm256_castsi256_ps(data_m256i));  // Use unaligned store for safety
    }

    // Process single sample (for systems without AVX)
    auto process_single(const uint32_t* data, float* dst) -> void {
        uint32_t raw = data[0];
        if (byteswap) raw = std::byteswap(raw);
        dst[0] = std::bit_cast<float>(raw);
    }

    auto (converter::*process_func)(const uint32_t*, float*) -> void = &converter::process_single;
    std::size_t samples_per_vector{1};
};

// Specialization: int8_t -> int16_t
template <>
struct converter<int8_t, int16_t> {
    converter() {
        init_cpu_features();
    }

    explicit converter([[maybe_unused]] bool swap) {
        // int8_t doesn't need byteswap (single byte)
        init_cpu_features();
    }

    auto operator()(const uint8_t* input, int16_t* output, std::size_t count) -> void {
        const int8_t* data = reinterpret_cast<const int8_t*>(input);
        std::size_t processed = 0;

        // Process full vectors using CPU-specific implementation
        for (; processed + samples_per_vector <= count; processed += samples_per_vector) {
            (this->*process_func)(data + processed, output + processed);
        }

        // Process remaining samples scalar
        for (; processed < count; ++processed) {
            output[processed] = static_cast<int16_t>(data[processed]);
        }
    }

private:
    auto init_cpu_features() -> void {
        switch (avx::detect_simd_level()) {
            case avx::simd_level::avx512:
                process_func = &converter::process_avx512;
                samples_per_vector = 32;
                break;
            case avx::simd_level::avx2:
                process_func = &converter::process_avx2;
                samples_per_vector = 16;
                break;
            case avx::simd_level::scalar:
                process_func = &converter::process_single;
                samples_per_vector = 1;
                break;
        }
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    auto process_avx512(const int8_t* data, int16_t* dst) -> void {
        auto loaded = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
        auto data_m512i = _mm512_cvtepi8_epi16(loaded);
        _mm512_storeu_epi16(dst, data_m512i);  // Already uses unaligned store
    }

    [[gnu::target("avx2")]]
    auto process_avx2(const int8_t* data, int16_t* dst) -> void {
        auto loaded = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
        auto data_m256i = _mm256_cvtepi8_epi16(loaded);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), data_m256i);  // Already uses unaligned store
    }

    // Process single sample (for systems without AVX)
    auto process_single(const int8_t* data, int16_t* dst) -> void {
        dst[0] = static_cast<int16_t>(data[0]);
    }

    auto (converter::*process_func)(const int8_t*, int16_t*) -> void = &converter::process_single;
    std::size_t samples_per_vector{1};
};

// Specialization: int16_t -> int16_t (passthrough with optional byteswap)
template <>
struct converter<int16_t, int16_t> {
    bool byteswap{false};

    converter() {
        init_cpu_features();
    }

    explicit converter(bool swap) : byteswap(swap) {
        init_cpu_features();
    }

    auto operator()(const uint8_t* input, int16_t* output, std::size_t count) -> void {
        const int16_t* data = reinterpret_cast<const int16_t*>(input);
        std::size_t processed = 0;

        // Process full vectors using CPU-specific implementation
        for (; processed + samples_per_vector <= count; processed += samples_per_vector) {
            (this->*process_func)(data + processed, output + processed);
        }

        // Process remaining samples scalar
        for (; processed < count; ++processed) {
            int16_t value = data[processed];
            if (byteswap) value = std::byteswap(value);
            output[processed] = value;
        }
    }

private:
    auto init_cpu_features() -> void {
        switch (avx::detect_simd_level()) {
            case avx::simd_level::avx512:
                process_func = &converter::process_avx512;
                samples_per_vector = 32;
                break;
            case avx::simd_level::avx2:
                process_func = &converter::process_avx2;
                samples_per_vector = 16;
                break;
            case avx::simd_level::scalar:
                process_func = &converter::process_single;
                samples_per_vector = 1;
                break;
        }
    }

    [[gnu::target("avx512f,avx512bw,avx512vl,avx512dq")]]
    auto process_avx512(const int16_t* data, int16_t* dst) -> void {
        auto data_m512i = _mm512_loadu_epi16(data);
        if (byteswap) {
            data_m512i = _mm512_shuffle_epi8(data_m512i, avx::swap::shuffle_u16_32);
        }
        _mm512_storeu_epi16(dst, data_m512i);  // Already uses unaligned store
    }

    [[gnu::target("avx2")]]
    auto process_avx2(const int16_t* data, int16_t* dst) -> void {
        auto data_m256i = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
        if (byteswap) {
            data_m256i = _mm256_shuffle_epi8(data_m256i, avx::swap::shuffle_u16_16);
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), data_m256i);  // Use unaligned store for safety
    }

    // Process single sample (for systems without AVX)
    auto process_single(const int16_t* data, int16_t* dst) -> void {
        int16_t value = data[0];
        if (byteswap) value = std::byteswap(value);
        dst[0] = value;
    }

    auto (converter::*process_func)(const int16_t*, int16_t*) -> void = &converter::process_single;
    std::size_t samples_per_vector{1};
};

// ============================================================================
// Variant type holding all converters - specialized by output type
// ============================================================================

template <typename OutputT>
struct converter_variant_traits;

// Float output supports all input types
template <>
struct converter_variant_traits<float> {
    using type = std::variant<
        converter<int8_t, float>,
        converter<int16_t, float>,
        converter<uint32_t, float>
    >;
};

// int16_t output only supports int8_t and int16_t inputs (no cf32->ci16)
template <>
struct converter_variant_traits<int16_t> {
    using type = std::variant<
        converter<int8_t, int16_t>,
        converter<int16_t, int16_t>
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

// Float output: integer inputs are widened numerically (NOT normalized), cf32 passes through.
template <>
inline constexpr auto supported_input_formats<float> = std::to_array<input_format_support<float>>({
    {false, composite::data_type::signed_integer, 8, 1,
     [](bool swap) -> converter_variant<float> { return converter<int8_t, float>(swap); }},
    {false, composite::data_type::signed_integer, 16, 2,
     [](bool swap) -> converter_variant<float> { return converter<int16_t, float>(swap); }},
    {true, composite::data_type::signed_integer, 8, 2,
     [](bool swap) -> converter_variant<float> { return converter<int8_t, float>(swap); }},
    {true, composite::data_type::signed_integer, 16, 4,
     [](bool swap) -> converter_variant<float> { return converter<int16_t, float>(swap); }},
    {true, composite::data_type::floating_point, 32, 8,
     [](bool swap) -> converter_variant<float> { return converter<uint32_t, float>(swap); }},
});

// int16_t output: integer inputs only (no float -> int16 narrowing path).
template <>
inline constexpr auto supported_input_formats<int16_t> = std::to_array<input_format_support<int16_t>>({
    {false, composite::data_type::signed_integer, 8, 1,
     [](bool swap) -> converter_variant<int16_t> { return converter<int8_t, int16_t>(swap); }},
    {false, composite::data_type::signed_integer, 16, 2,
     [](bool swap) -> converter_variant<int16_t> { return converter<int16_t, int16_t>(swap); }},
    {true, composite::data_type::signed_integer, 8, 2,
     [](bool swap) -> converter_variant<int16_t> { return converter<int8_t, int16_t>(swap); }},
    {true, composite::data_type::signed_integer, 16, 4,
     [](bool swap) -> converter_variant<int16_t> { return converter<int16_t, int16_t>(swap); }},
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
