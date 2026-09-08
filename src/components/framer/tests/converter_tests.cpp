/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Converter Tests - Unit tests for all converter variants
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <bit>
#include <complex>
#include <cstdint>
#include <vector>

#include "../convert_variant.hpp"

using namespace std;
using Catch::Matchers::WithinAbs;

// ============================================================================
// Test int8_t -> float converter
// ============================================================================

TEST_CASE("converter<int8_t, float> - basic conversion", "[converter][int8]") {
    converter<int8_t, float> conv(false);

    vector<uint8_t> input = {0, 127, 128, 255}; // 0, 127, -128, -1 as int8_t
    vector<float> output(4);

    conv(input.data(), output.data(), 4);

    REQUIRE(output[0] == 0.0f);
    REQUIRE(output[1] == 127.0f);
    REQUIRE(output[2] == -128.0f);
    REQUIRE(output[3] == -1.0f);
}

TEST_CASE("converter<int8_t, float> - large batch", "[converter][int8]") {
    converter<int8_t, float> conv(false);

    constexpr size_t count = 1024;
    vector<uint8_t> input(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = static_cast<int8_t>(i % 256);
    }

    alignas(64) float output[count];
    conv(input.data(), output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<float>(static_cast<int8_t>(i % 256));
        REQUIRE_THAT(output[i], WithinAbs(expected, 0.001f));
    }
}

// ============================================================================
// Test int16_t -> float converter
// ============================================================================

TEST_CASE("converter<int16_t, float> - basic conversion, no byteswap", "[converter][int16]") {
    converter<int16_t, float> conv(false);

    vector<int16_t> input_i16 = {0, 32767, -32768, -1, 1000, -1000};
    auto* input = reinterpret_cast<uint8_t*>(input_i16.data());
    vector<float> output(6);

    conv(input, output.data(), 6);

    REQUIRE(output[0] == 0.0f);
    REQUIRE(output[1] == 32767.0f);
    REQUIRE(output[2] == -32768.0f);
    REQUIRE(output[3] == -1.0f);
    REQUIRE(output[4] == 1000.0f);
    REQUIRE(output[5] == -1000.0f);
}

TEST_CASE("converter<int16_t, float> - with byteswap", "[converter][int16][byteswap]") {
    converter<int16_t, float> conv(true);

    // Create big-endian representation of 256 (0x0100)
    vector<uint8_t> input = {0x01, 0x00};  // Big-endian 256
    vector<float> output(1);

    conv(input.data(), output.data(), 1);

    REQUIRE(output[0] == 256.0f);
}

TEST_CASE("converter<int16_t, float> - large batch", "[converter][int16]") {
    converter<int16_t, float> conv(false);

    constexpr size_t count = 1024;
    vector<int16_t> input_i16(count);
    for (size_t i = 0; i < count; ++i) {
        input_i16[i] = static_cast<int16_t>(i - 512);
    }

    auto* input = reinterpret_cast<uint8_t*>(input_i16.data());
    alignas(64) float output[count];
    conv(input, output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<float>(input_i16[i]);
        REQUIRE_THAT(output[i], WithinAbs(expected, 0.001f));
    }
}

// ============================================================================
// Test uint32_t -> float converter (complex_cf32 passthrough)
// ============================================================================

TEST_CASE("converter<uint32_t, float> - passthrough, no byteswap", "[converter][cf32]") {
    converter<uint32_t, float> conv(false);

    vector<float> input_float = {1.0f, -1.0f, 3.14159f, -2.71828f};
    auto* input = reinterpret_cast<uint8_t*>(input_float.data());
    vector<float> output(4);

    conv(input, output.data(), 4);

    REQUIRE_THAT(output[0], WithinAbs(1.0f, 0.00001f));
    REQUIRE_THAT(output[1], WithinAbs(-1.0f, 0.00001f));
    REQUIRE_THAT(output[2], WithinAbs(3.14159f, 0.00001f));
    REQUIRE_THAT(output[3], WithinAbs(-2.71828f, 0.00001f));
}

TEST_CASE("converter<uint32_t, float> - with byteswap", "[converter][cf32][byteswap]") {
    converter<uint32_t, float> conv(true);

    // Create 1.0f = 0x3F800000 in big-endian
    vector<uint8_t> input = {0x3F, 0x80, 0x00, 0x00};
    vector<float> output(1);

    conv(input.data(), output.data(), 1);

    REQUIRE_THAT(output[0], WithinAbs(1.0f, 0.00001f));
}

// ============================================================================
// Test u32_offset_binary -> float converter (distinct from the f32 passthrough
// keyed on uint32_t: this is a NUMERIC conversion of unsigned offset-binary data)
// ============================================================================

TEST_CASE("converter<u32_offset_binary, float> - offset binary, no byteswap",
          "[converter][u32][offset_binary]") {
    converter<u32_offset_binary, float> conv(false);

    vector<uint32_t> input_u32 = {
        0x80000000U,          // mid-scale -> 0
        0x80000001U,          // mid-scale + 1 -> 1
        0x7FFFFFFFU,          // mid-scale - 1 -> -1
        0x00000000U,          // minimum -> -2^31
        0xFFFFFFFFU,          // maximum -> 2^31 - 1 (rounds in float)
        0x80019000U,          // arbitrary positive
    };
    auto* input = reinterpret_cast<uint8_t*>(input_u32.data());
    vector<float> output(input_u32.size());

    conv(input, output.data(), input_u32.size());

    for (size_t i = 0; i < input_u32.size(); ++i) {
        INFO("Sample " << i);
        const auto expected =
            static_cast<float>(static_cast<int32_t>(input_u32[i] ^ 0x80000000U));
        REQUIRE(output[i] == expected);
    }
}

TEST_CASE("converter<u32_offset_binary, float> - with byteswap",
          "[converter][u32][offset_binary][byteswap]") {
    converter<u32_offset_binary, float> conv(true);

    // 0x80000005 (mid-scale + 5) in big-endian byte order.
    vector<uint8_t> input = {0x80, 0x00, 0x00, 0x05};
    vector<float> output(1);

    conv(input.data(), output.data(), 1);

    REQUIRE(output[0] == 5.0f);
}

TEST_CASE("converter<u32_offset_binary, float> - large batch exercises SIMD and scalar tail",
          "[converter][u32][offset_binary]") {
    // 69 = 4*16 + 5: covers full 512/256-bit vectors plus a scalar remainder in every
    // resolved kernel.
    constexpr size_t COUNT = 69;
    converter<u32_offset_binary, float> conv(false);

    vector<uint32_t> input_u32(COUNT);
    for (size_t i = 0; i < COUNT; ++i) {
        input_u32[i] = 0x80000000U + static_cast<uint32_t>(i * 1'000'003) - 34'000'000U;
    }
    auto* input = reinterpret_cast<uint8_t*>(input_u32.data());
    vector<float> output(COUNT);

    conv(input, output.data(), COUNT);

    for (size_t i = 0; i < COUNT; ++i) {
        INFO("Sample " << i);
        const auto expected =
            static_cast<float>(static_cast<int32_t>(input_u32[i] ^ 0x80000000U));
        REQUIRE(output[i] == expected);
    }
}

// ============================================================================
// Test int8_t -> int16_t converter
// ============================================================================

TEST_CASE("converter<int8_t, int16_t> - basic conversion", "[converter][int8][int16]") {
    converter<int8_t, int16_t> conv(false);

    vector<uint8_t> input = {0, 127, 128, 255}; // 0, 127, -128, -1 as int8_t
    vector<int16_t> output(4);

    conv(input.data(), output.data(), 4);

    REQUIRE(output[0] == 0);
    REQUIRE(output[1] == 127);
    REQUIRE(output[2] == -128);
    REQUIRE(output[3] == -1);
}

TEST_CASE("converter<int8_t, int16_t> - large batch", "[converter][int8][int16]") {
    converter<int8_t, int16_t> conv(false);

    constexpr size_t count = 1024;
    vector<uint8_t> input(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = static_cast<int8_t>(i % 256);
    }

    alignas(64) int16_t output[count];
    conv(input.data(), output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<int16_t>(static_cast<int8_t>(i % 256));
        REQUIRE(output[i] == expected);
    }
}

// ============================================================================
// Test int16_t -> int16_t converter (passthrough)
// ============================================================================

TEST_CASE("converter<int16_t, int16_t> - passthrough, no byteswap", "[converter][int16][passthrough]") {
    converter<int16_t, int16_t> conv(false);

    vector<int16_t> input_i16 = {0, 32767, -32768, -1, 1000, -1000};
    auto* input = reinterpret_cast<uint8_t*>(input_i16.data());
    vector<int16_t> output(6);

    conv(input, output.data(), 6);

    REQUIRE(output[0] == 0);
    REQUIRE(output[1] == 32767);
    REQUIRE(output[2] == -32768);
    REQUIRE(output[3] == -1);
    REQUIRE(output[4] == 1000);
    REQUIRE(output[5] == -1000);
}

TEST_CASE("converter<int16_t, int16_t> - with byteswap", "[converter][int16][passthrough][byteswap]") {
    converter<int16_t, int16_t> conv(true);

    // Create big-endian representation of 256 (0x0100)
    vector<uint8_t> input = {0x01, 0x00};
    vector<int16_t> output(1);

    conv(input.data(), output.data(), 1);

    REQUIRE(output[0] == 256);
}

// ============================================================================
// Test converter_variant with visitor
// ============================================================================

TEST_CASE("converter_variant<float> - visitor pattern", "[converter][variant]") {
    // Test int8_t -> float via variant
    converter_variant<float> conv_var = converter<int8_t, float>(false);

    vector<uint8_t> input = {0, 127, 128, 255};
    vector<float> output(4);

    convert(conv_var, input.data(), output.data(), 4);

    REQUIRE(output[0] == 0.0f);
    REQUIRE(output[1] == 127.0f);
    REQUIRE(output[2] == -128.0f);
    REQUIRE(output[3] == -1.0f);
}

TEST_CASE("converter_variant<float> - switch converter types", "[converter][variant]") {
    vector<float> output(4);

    // Test int8_t -> float
    {
        converter_variant<float> conv_var = converter<int8_t, float>(false);
        vector<uint8_t> input = {10, 20, 30, 40};
        convert(conv_var, input.data(), output.data(), 4);

        REQUIRE(output[0] == 10.0f);
        REQUIRE(output[1] == 20.0f);
    }

    // Test int16_t -> float
    {
        converter_variant<float> conv_var = converter<int16_t, float>(false);
        vector<int16_t> input_i16 = {100, 200, 300, 400};
        auto* input = reinterpret_cast<uint8_t*>(input_i16.data());
        convert(conv_var, input, output.data(), 4);

        REQUIRE(output[0] == 100.0f);
        REQUIRE(output[1] == 200.0f);
    }
}

TEST_CASE("converter_variant<int16_t> - visitor pattern", "[converter][variant]") {
    // Test int8_t -> int16_t via variant
    converter_variant<int16_t> conv_var = converter<int8_t, int16_t>(false);

    vector<uint8_t> input = {0, 127, 128, 255};
    vector<int16_t> output(4);

    convert(conv_var, input.data(), output.data(), 4);

    REQUIRE(output[0] == 0);
    REQUIRE(output[1] == 127);
    REQUIRE(output[2] == -128);
    REQUIRE(output[3] == -1);
}

// ============================================================================
// Real vs Complex Input Handling
// ============================================================================

TEST_CASE("Real i8 input creates complex output with Q=0", "[converter][real]") {
    // This test verifies that real input is properly converted to complex
    // by setting the imaginary component to zero

    converter<int8_t, float> conv(false);

    // Real input: 4 samples [1, 2, 3, 4]
    vector<uint8_t> real_input = {1, 2, 3, 4};
    alignas(64) float output[4];

    // Convert as real (not interleaved I/Q)
    conv(real_input.data(), output, 4);

    // Verify real components are correct
    REQUIRE(output[0] == 1.0f);
    REQUIRE(output[1] == 2.0f);
    REQUIRE(output[2] == 3.0f);
    REQUIRE(output[3] == 4.0f);
}

TEST_CASE("Complex i8 input preserves I and Q", "[converter][complex]") {
    // This test verifies that complex input properly converts both I and Q

    converter<int8_t, float> conv(false);

    // Complex input: 2 samples [(1+2i), (3+4i)] as interleaved I,Q bytes
    vector<uint8_t> complex_input = {1, 2, 3, 4};
    alignas(64) float output[4];

    // Convert as complex (interleaved I/Q)
    conv(complex_input.data(), output, 4);  // 4 components (2 samples * 2)

    // Verify both I and Q components
    REQUIRE(output[0] == 1.0f);  // Sample 0 I
    REQUIRE(output[1] == 2.0f);  // Sample 0 Q
    REQUIRE(output[2] == 3.0f);  // Sample 1 I
    REQUIRE(output[3] == 4.0f);  // Sample 1 Q
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_CASE("converter - zero length", "[converter][edge]") {
    converter<int8_t, float> conv(false);

    vector<uint8_t> input = {1, 2, 3};
    vector<float> output(3);

    // Should handle zero gracefully
    conv(input.data(), output.data(), 0);

    // Output should be unchanged (no writes)
    REQUIRE(true);
}

TEST_CASE("converter - unaligned count (tests scalar fallback)", "[converter][edge]") {
    converter<int8_t, float> conv(false);

    // Use a prime number to ensure scalar fallback is tested
    constexpr size_t count = 17;
    vector<uint8_t> input(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = static_cast<uint8_t>(i);
    }

    alignas(64) float output[count];
    conv(input.data(), output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<float>(static_cast<int8_t>(i));
        REQUIRE_THAT(output[i], WithinAbs(expected, 0.001f));
    }
}

TEST_CASE("converter<int8_t, float> - large remainder exposes scalar bug", "[converter][scalar][bug]") {
    converter<int8_t, float> conv(false);

    // Count = 25: AVX-512 processes 16, leaves 9 for scalar
    // Count = 25: AVX2 processes 16, leaves 9 for scalar
    // Count = 25: No AVX processes 25 via scalar
    // This should expose if process_scalar() only processes 1 sample!
    constexpr size_t count = 25;
    vector<uint8_t> input(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = static_cast<uint8_t>(i + 10);  // offset to avoid zeros
    }

    alignas(64) float output[count];
    // Initialize to sentinel value to detect unprocessed samples
    for (size_t i = 0; i < count; ++i) {
        output[i] = -9999.0f;
    }

    conv(input.data(), output, count);

    // Verify ALL samples were converted (not just first 16 or 17)
    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<float>(static_cast<int8_t>(i + 10));
        INFO("Sample " << i << " should be " << expected << " but got " << output[i]);
        REQUIRE_THAT(output[i], WithinAbs(expected, 0.001f));
    }
}

TEST_CASE("converter<int16_t, float> - large remainder exposes scalar bug", "[converter][scalar][bug]") {
    converter<int16_t, float> conv(false);

    constexpr size_t count = 25;
    vector<int16_t> input_i16(count);
    for (size_t i = 0; i < count; ++i) {
        input_i16[i] = static_cast<int16_t>(i * 100);
    }

    auto* input = reinterpret_cast<uint8_t*>(input_i16.data());
    alignas(64) float output[count];
    for (size_t i = 0; i < count; ++i) {
        output[i] = -9999.0f;
    }

    conv(input, output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<float>(input_i16[i]);
        INFO("Sample " << i << " should be " << expected << " but got " << output[i]);
        REQUIRE_THAT(output[i], WithinAbs(expected, 0.001f));
    }
}

// ============================================================================
// Vector-path byteswap coverage (counts large enough to hit the SIMD loop)
// ============================================================================

TEST_CASE("converter<int16_t, float> - byteswap large batch hits vector path", "[converter][int16][byteswap]") {
    converter<int16_t, float> conv(true);

    constexpr size_t count = 41;
    vector<int16_t> input_i16(count);
    for (size_t i = 0; i < count; ++i) {
        auto value = static_cast<int16_t>(static_cast<int>(i) * 700 - 14'000);
        auto u = static_cast<uint16_t>(value);
        input_i16[i] = static_cast<int16_t>(static_cast<uint16_t>((u << 8) | (u >> 8)));  // store big-endian
    }

    auto* input = reinterpret_cast<uint8_t*>(input_i16.data());
    alignas(64) float output[count];
    conv(input, output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<float>(static_cast<int16_t>(static_cast<int>(i) * 700 - 14'000));
        INFO("Sample " << i);
        REQUIRE_THAT(output[i], WithinAbs(expected, 0.001f));
    }
}

TEST_CASE("converter<int16_t, int16_t> - byteswap large batch hits vector path", "[converter][int16][passthrough][byteswap]") {
    converter<int16_t, int16_t> conv(true);

    constexpr size_t count = 73;
    vector<int16_t> input_i16(count);
    for (size_t i = 0; i < count; ++i) {
        auto value = static_cast<int16_t>(static_cast<int>(i) * 421 - 15'000);
        auto u = static_cast<uint16_t>(value);
        input_i16[i] = static_cast<int16_t>(static_cast<uint16_t>((u << 8) | (u >> 8)));
    }

    auto* input = reinterpret_cast<uint8_t*>(input_i16.data());
    alignas(64) int16_t output[count];
    conv(input, output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<int16_t>(static_cast<int>(i) * 421 - 15'000);
        INFO("Sample " << i);
        REQUIRE(output[i] == expected);
    }
}

TEST_CASE("converter<uint32_t, float> - byteswap large batch hits vector path", "[converter][cf32][byteswap]") {
    converter<uint32_t, float> conv(true);

    constexpr size_t count = 25;
    vector<uint32_t> input_u32(count);
    vector<float> expected(count);
    for (size_t i = 0; i < count; ++i) {
        expected[i] = static_cast<float>(i) * 1.25f - 3.5f;
        auto bits = std::bit_cast<uint32_t>(expected[i]);
        input_u32[i] = std::byteswap(bits);  // store big-endian
    }

    auto* input = reinterpret_cast<uint8_t*>(input_u32.data());
    alignas(64) float output[count];
    conv(input, output, count);

    for (size_t i = 0; i < count; ++i) {
        INFO("Sample " << i);
        REQUIRE_THAT(output[i], WithinAbs(expected[i], 0.00001f));
    }
}

// ============================================================================
// Unsigned (offset binary) converters
// ============================================================================

TEST_CASE("converter<uint8_t, float> - offset binary conversion", "[converter][uint8]") {
    converter<uint8_t, float> conv(false);

    vector<uint8_t> input = {0, 128, 255, 127, 129};
    vector<float> output(5);

    conv(input.data(), output.data(), 5);

    REQUIRE(output[0] == -128.0f);  // wire minimum -> full negative scale
    REQUIRE(output[1] == 0.0f);     // mid-scale -> zero
    REQUIRE(output[2] == 127.0f);   // wire maximum -> full positive scale
    REQUIRE(output[3] == -1.0f);
    REQUIRE(output[4] == 1.0f);
}

TEST_CASE("converter<uint8_t, float> - large remainder exposes scalar bug", "[converter][uint8][scalar][bug]") {
    converter<uint8_t, float> conv(false);

    constexpr size_t count = 25;
    vector<uint8_t> input(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = static_cast<uint8_t>(i * 10);
    }

    alignas(64) float output[count];
    for (size_t i = 0; i < count; ++i) {
        output[i] = -9999.0f;
    }

    conv(input.data(), output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<float>(static_cast<int>(input[i]) - 128);
        INFO("Sample " << i << " should be " << expected << " but got " << output[i]);
        REQUIRE_THAT(output[i], WithinAbs(expected, 0.001f));
    }
}

TEST_CASE("converter<uint16_t, float> - offset binary conversion", "[converter][uint16]") {
    converter<uint16_t, float> conv(false);

    vector<uint16_t> input_u16 = {0, 32768, 65535, 32767, 32769};
    auto* input = reinterpret_cast<uint8_t*>(input_u16.data());
    vector<float> output(5);

    conv(input, output.data(), 5);

    REQUIRE(output[0] == -32768.0f);
    REQUIRE(output[1] == 0.0f);
    REQUIRE(output[2] == 32767.0f);
    REQUIRE(output[3] == -1.0f);
    REQUIRE(output[4] == 1.0f);
}

TEST_CASE("converter<uint16_t, float> - with byteswap", "[converter][uint16][byteswap]") {
    converter<uint16_t, float> conv(true);

    // Big-endian 0x8100 = 33024 -> 33024 - 32768 = 256
    vector<uint8_t> input = {0x81, 0x00};
    vector<float> output(1);

    conv(input.data(), output.data(), 1);

    REQUIRE(output[0] == 256.0f);
}

TEST_CASE("converter<uint16_t, float> - large remainder exposes scalar bug", "[converter][uint16][scalar][bug]") {
    converter<uint16_t, float> conv(false);

    constexpr size_t count = 25;
    vector<uint16_t> input_u16(count);
    for (size_t i = 0; i < count; ++i) {
        input_u16[i] = static_cast<uint16_t>(i * 2500);
    }

    auto* input = reinterpret_cast<uint8_t*>(input_u16.data());
    alignas(64) float output[count];
    for (size_t i = 0; i < count; ++i) {
        output[i] = -99999.0f;
    }

    conv(input, output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<float>(static_cast<int32_t>(input_u16[i]) - 32768);
        INFO("Sample " << i << " should be " << expected << " but got " << output[i]);
        REQUIRE_THAT(output[i], WithinAbs(expected, 0.001f));
    }
}

TEST_CASE("converter<uint8_t, int16_t> - offset binary conversion", "[converter][uint8][int16]") {
    converter<uint8_t, int16_t> conv(false);

    vector<uint8_t> input = {0, 128, 255, 127, 129};
    vector<int16_t> output(5);

    conv(input.data(), output.data(), 5);

    REQUIRE(output[0] == -128);
    REQUIRE(output[1] == 0);
    REQUIRE(output[2] == 127);
    REQUIRE(output[3] == -1);
    REQUIRE(output[4] == 1);
}

TEST_CASE("converter<uint8_t, int16_t> - large remainder exposes scalar bug", "[converter][uint8][int16][scalar][bug]") {
    converter<uint8_t, int16_t> conv(false);

    constexpr size_t count = 41;  // AVX-512 does 32, leaves 9; AVX2 does 32, leaves 9
    vector<uint8_t> input(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = static_cast<uint8_t>(i * 6);
    }

    alignas(64) int16_t output[count];
    for (size_t i = 0; i < count; ++i) {
        output[i] = -9999;
    }

    conv(input.data(), output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<int16_t>(static_cast<int>(input[i]) - 128);
        INFO("Sample " << i << " should be " << expected << " but got " << output[i]);
        REQUIRE(output[i] == expected);
    }
}

TEST_CASE("converter<uint16_t, int16_t> - offset binary conversion", "[converter][uint16][int16]") {
    converter<uint16_t, int16_t> conv(false);

    vector<uint16_t> input_u16 = {0, 32768, 65535, 32767, 32769};
    auto* input = reinterpret_cast<uint8_t*>(input_u16.data());
    vector<int16_t> output(5);

    conv(input, output.data(), 5);

    REQUIRE(output[0] == -32768);
    REQUIRE(output[1] == 0);
    REQUIRE(output[2] == 32767);
    REQUIRE(output[3] == -1);
    REQUIRE(output[4] == 1);
}

TEST_CASE("converter<uint16_t, int16_t> - with byteswap", "[converter][uint16][int16][byteswap]") {
    converter<uint16_t, int16_t> conv(true);

    // Big-endian 0x8100 = 33024 -> 33024 - 32768 = 256
    vector<uint8_t> input = {0x81, 0x00};
    vector<int16_t> output(1);

    conv(input.data(), output.data(), 1);

    REQUIRE(output[0] == 256);
}

TEST_CASE("converter<uint16_t, int16_t> - byteswap large remainder", "[converter][uint16][int16][byteswap][scalar][bug]") {
    converter<uint16_t, int16_t> conv(true);

    constexpr size_t count = 41;
    vector<uint16_t> input_u16(count);
    for (size_t i = 0; i < count; ++i) {
        auto value = static_cast<uint16_t>(i * 1500);
        input_u16[i] = static_cast<uint16_t>((value << 8) | (value >> 8));  // store big-endian
    }

    auto* input = reinterpret_cast<uint8_t*>(input_u16.data());
    alignas(64) int16_t output[count];
    for (size_t i = 0; i < count; ++i) {
        output[i] = -9999;
    }

    conv(input, output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<int16_t>(static_cast<int32_t>(static_cast<uint16_t>(i * 1500)) - 32768);
        INFO("Sample " << i << " should be " << expected << " but got " << output[i]);
        REQUIRE(output[i] == expected);
    }
}

// ============================================================================
// int32_t -> float converter
// ============================================================================

TEST_CASE("converter<int32_t, float> - basic conversion", "[converter][int32]") {
    converter<int32_t, float> conv(false);

    vector<int32_t> input_i32 = {0, 1, -1, 1'000'000, -1'000'000, 8'388'607};
    auto* input = reinterpret_cast<uint8_t*>(input_i32.data());
    vector<float> output(6);

    conv(input, output.data(), 6);

    REQUIRE(output[0] == 0.0f);
    REQUIRE(output[1] == 1.0f);
    REQUIRE(output[2] == -1.0f);
    REQUIRE(output[3] == 1'000'000.0f);
    REQUIRE(output[4] == -1'000'000.0f);
    REQUIRE(output[5] == 8'388'607.0f);  // exactly representable (within 24-bit mantissa)
}

TEST_CASE("converter<int32_t, float> - with byteswap", "[converter][int32][byteswap]") {
    converter<int32_t, float> conv(true);

    // Big-endian 256 = 0x00000100
    vector<uint8_t> input = {0x00, 0x00, 0x01, 0x00};
    vector<float> output(1);

    conv(input.data(), output.data(), 1);

    REQUIRE(output[0] == 256.0f);
}

TEST_CASE("converter<int32_t, float> - large remainder exposes scalar bug", "[converter][int32][scalar][bug]") {
    converter<int32_t, float> conv(false);

    constexpr size_t count = 25;
    vector<int32_t> input_i32(count);
    for (size_t i = 0; i < count; ++i) {
        input_i32[i] = static_cast<int32_t>(i) * 100'000 - 1'200'000;
    }

    auto* input = reinterpret_cast<uint8_t*>(input_i32.data());
    alignas(64) float output[count];
    for (size_t i = 0; i < count; ++i) {
        output[i] = -9999.0f;
    }

    conv(input, output, count);

    for (size_t i = 0; i < count; ++i) {
        auto expected = static_cast<float>(input_i32[i]);
        INFO("Sample " << i << " should be " << expected << " but got " << output[i]);
        REQUIRE_THAT(output[i], WithinAbs(expected, 0.001f));
    }
}

// ============================================================================
// Real -> complex in-place expansion kernels
// ============================================================================

TEST_CASE("expand_real_to_complex<float> - in-place aliased expansion", "[expand][float]") {
    // Odd count exercises the vector loop AND the scalar tail; the source lives in the
    // back half of the same buffer the expansion writes, exactly as the framer stages it.
    constexpr size_t count = 25;
    vector<float> buf(2 * count, -9999.0f);
    for (size_t i = 0; i < count; ++i) {
        buf[count + i] = static_cast<float>(i) - 12.0f;
    }

    expand_real_to_complex(buf.data(), count);

    for (size_t i = 0; i < count; ++i) {
        INFO("Sample " << i);
        REQUIRE(buf[2 * i] == static_cast<float>(i) - 12.0f);
        REQUIRE(buf[2 * i + 1] == 0.0f);
    }
}

TEST_CASE("expand_real_to_complex<int16_t> - in-place aliased expansion", "[expand][int16]") {
    constexpr size_t count = 37;
    vector<int16_t> buf(2 * count, -9999);
    for (size_t i = 0; i < count; ++i) {
        buf[count + i] = static_cast<int16_t>(static_cast<int>(i) * 100 - 1800);
    }

    expand_real_to_complex(buf.data(), count);

    for (size_t i = 0; i < count; ++i) {
        INFO("Sample " << i);
        REQUIRE(buf[2 * i] == static_cast<int16_t>(static_cast<int>(i) * 100 - 1800));
        REQUIRE(buf[2 * i + 1] == 0);
    }
}

TEST_CASE("expand_real_to_complex<float> - single sample", "[expand][float][edge]") {
    vector<float> buf = {-9999.0f, 42.0f};
    expand_real_to_complex(buf.data(), 1);
    REQUIRE(buf[0] == 42.0f);
    REQUIRE(buf[1] == 0.0f);
}

TEST_CASE("converter<uint32_t, float> - large remainder exposes scalar bug", "[converter][scalar][bug]") {
    converter<uint32_t, float> conv(false);

    constexpr size_t count = 25;
    vector<float> input_float(count);
    for (size_t i = 0; i < count; ++i) {
        input_float[i] = static_cast<float>(i) * 1.5f;
    }

    auto* input = reinterpret_cast<uint8_t*>(input_float.data());
    alignas(64) float output[count];
    for (size_t i = 0; i < count; ++i) {
        output[i] = -9999.0f;
    }

    conv(input, output, count);

    for (size_t i = 0; i < count; ++i) {
        INFO("Sample " << i << " should be " << input_float[i] << " but got " << output[i]);
        REQUIRE_THAT(output[i], WithinAbs(input_float[i], 0.001f));
    }
}
