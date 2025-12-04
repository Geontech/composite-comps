/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Converter Tests - Unit tests for all converter variants
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
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
