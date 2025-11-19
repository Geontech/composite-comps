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

#include "../convert.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>
#include <array>

using Catch::Approx;

TEST_CASE("Converter int8_t to float - basic conversion", "[conversion]") {
    converter<int8_t, float> conv;

    std::array<int8_t, 8> input = {0, 1, -1, 127, -128, 64, -64, 42};
    std::array<float, 8> output{};

    conv.process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 8);

    REQUIRE(output[0] == Approx(0.0f));
    REQUIRE(output[1] == Approx(1.0f));
    REQUIRE(output[2] == Approx(-1.0f));
    REQUIRE(output[3] == Approx(127.0f));
    REQUIRE(output[4] == Approx(-128.0f));
    REQUIRE(output[5] == Approx(64.0f));
    REQUIRE(output[6] == Approx(-64.0f));
    REQUIRE(output[7] == Approx(42.0f));
}

TEST_CASE("Converter int8_t to int16_t - basic conversion", "[conversion]") {
    converter<int8_t, int16_t> conv;

    std::array<int8_t, 8> input = {0, 1, -1, 127, -128, 64, -64, 42};
    std::array<int16_t, 8> output{};

    conv.process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 8);

    REQUIRE(output[0] == 0);
    REQUIRE(output[1] == 1);
    REQUIRE(output[2] == -1);
    REQUIRE(output[3] == 127);
    REQUIRE(output[4] == -128);
    REQUIRE(output[5] == 64);
    REQUIRE(output[6] == -64);
    REQUIRE(output[7] == 42);
}

TEST_CASE("Converter int16_t to float - basic conversion", "[conversion]") {
    converter<int16_t, float> conv;

    std::array<int16_t, 8> input = {0, 100, -100, 32767, -32768, 16384, -16384, 12345};
    std::array<float, 8> output{};

    conv.process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 8);

    REQUIRE(output[0] == Approx(0.0f));
    REQUIRE(output[1] == Approx(100.0f));
    REQUIRE(output[2] == Approx(-100.0f));
    REQUIRE(output[3] == Approx(32767.0f));
    REQUIRE(output[4] == Approx(-32768.0f));
    REQUIRE(output[5] == Approx(16384.0f));
    REQUIRE(output[6] == Approx(-16384.0f));
    REQUIRE(output[7] == Approx(12345.0f));
}

TEST_CASE("Converter int16_t to int16_t - identity without byteswap", "[conversion]") {
    converter<int16_t, int16_t> conv(false);

    std::array<int16_t, 8> input = {0, 100, -100, 32767, -32768, 16384, -16384, 12345};
    std::array<int16_t, 8> output{};

    conv.process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 8);

    for (size_t i = 0; i < 8; ++i) {
        REQUIRE(output[i] == input[i]);
    }
}

TEST_CASE("Converter int16_t to int16_t - byteswap", "[conversion][byteswap]") {
    converter<int16_t, int16_t> conv(true);

    // Use values that change when byteswapped
    std::array<int16_t, 4> input = {0x0102, 0x1234, static_cast<int16_t>(0xABCD), 0x00FF};
    std::array<int16_t, 4> output{};

    conv.process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 4);

    REQUIRE(output[0] == 0x0201);
    REQUIRE(output[1] == 0x3412);
    REQUIRE(output[2] == static_cast<int16_t>(0xCDAB));
    REQUIRE(output[3] == static_cast<int16_t>(0xFF00));
}

TEST_CASE("Converter int16_t to float - byteswap", "[conversion][byteswap]") {
    converter<int16_t, float> conv(true);

    // Test with values where byteswap makes a difference
    std::array<int16_t, 4> input = {0x0100, 0x0200, 0x0001, 0x0002};
    std::array<float, 4> output{};

    conv.process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 4);

    // After byteswap: 0x0001 (1), 0x0002 (2), 0x0100 (256), 0x0200 (512)
    REQUIRE(output[0] == Approx(1.0f));
    REQUIRE(output[1] == Approx(2.0f));
    REQUIRE(output[2] == Approx(256.0f));
    REQUIRE(output[3] == Approx(512.0f));
}

TEST_CASE("Converter uint32_t to float - bit_cast conversion", "[conversion]") {
    converter<uint32_t, float> conv;

    // Create floats, then bit_cast to uint32_t for input
    std::array<float, 4> expected = {1.0f, -1.0f, 3.14159f, 0.0f};
    std::array<uint32_t, 4> input;
    for (size_t i = 0; i < 4; ++i) {
        input[i] = std::bit_cast<uint32_t>(expected[i]);
    }

    std::array<float, 4> output{};
    conv.process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 4);

    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(output[i] == Approx(expected[i]));
    }
}

TEST_CASE("Converter uint32_t to float - byteswap", "[conversion][byteswap]") {
    converter<uint32_t, float> conv(true);

    // Create a float value in big-endian format
    float original = 1.0f;
    uint32_t big_endian = std::byteswap(std::bit_cast<uint32_t>(original));

    std::array<uint32_t, 1> input = {big_endian};
    std::array<float, 1> output{};

    conv.process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 1);

    REQUIRE(output[0] == Approx(original));
}

TEST_CASE("Converter int8_t to float - large batch processing", "[conversion][batch]") {
    converter<int8_t, float> conv;

    constexpr size_t SIZE = 1024;
    alignas(64) int8_t input[SIZE];
    alignas(64) float output[SIZE];

    // Fill with pattern
    for (size_t i = 0; i < SIZE; ++i) {
        input[i] = static_cast<int8_t>(i % 256 - 128);
    }

    conv.process(reinterpret_cast<const uint8_t*>(input), output, SIZE);

    // Verify all conversions
    for (size_t i = 0; i < SIZE; ++i) {
        REQUIRE(output[i] == Approx(static_cast<float>(input[i])));
    }
}

TEST_CASE("Converter int16_t to float - large batch with non-vector-aligned size", "[conversion][batch]") {
    converter<int16_t, float> conv;

    // Use a size that's not a multiple of any vector size (8, 16, 32)
    constexpr size_t SIZE = 1019;  // Prime number
    alignas(64) int16_t input[SIZE];
    alignas(64) float output[SIZE];

    // Fill with pattern
    for (size_t i = 0; i < SIZE; ++i) {
        input[i] = static_cast<int16_t>((i * 17) % 32768 - 16384);
    }

    conv.process(reinterpret_cast<const uint8_t*>(input), output, SIZE);

    // Verify all conversions including remainder
    for (size_t i = 0; i < SIZE; ++i) {
        REQUIRE(output[i] == Approx(static_cast<float>(input[i])));
    }
}

TEST_CASE("Converter int8_t to int16_t - large batch processing", "[conversion][batch]") {
    converter<int8_t, int16_t> conv;

    constexpr size_t SIZE = 2048;
    alignas(64) int8_t input[SIZE];
    alignas(64) int16_t output[SIZE];

    // Fill with full range
    for (size_t i = 0; i < SIZE; ++i) {
        input[i] = static_cast<int8_t>(i % 256 - 128);
    }

    conv.process(reinterpret_cast<const uint8_t*>(input), output, SIZE);

    // Verify all conversions
    for (size_t i = 0; i < SIZE; ++i) {
        REQUIRE(output[i] == static_cast<int16_t>(input[i]));
    }
}

TEST_CASE("Converter edge cases - single sample", "[conversion][edge]") {
    SECTION("int8_t to float") {
        converter<int8_t, float> conv;
        int8_t input = 42;
        float output = 0.0f;

        conv.process(reinterpret_cast<const uint8_t*>(&input), &output, 1);
        REQUIRE(output == Approx(42.0f));
    }

    SECTION("int16_t to float") {
        converter<int16_t, float> conv;
        int16_t input = 12345;
        float output = 0.0f;

        conv.process(reinterpret_cast<const uint8_t*>(&input), &output, 1);
        REQUIRE(output == Approx(12345.0f));
    }
}

TEST_CASE("Converter edge cases - zero samples", "[conversion][edge]") {
    converter<int8_t, float> conv;
    int8_t input = 0;
    float output = 999.0f;  // Sentinel value

    conv.process(reinterpret_cast<const uint8_t*>(&input), &output, 0);

    // Output should be unchanged
    REQUIRE(output == Approx(999.0f));
}

TEST_CASE("Converter samples_per_vector - verify interface", "[conversion]") {
    SECTION("int8_t to float") {
        converter<int8_t, float> conv;
        auto spv = conv.samples_per_vector();
        // Should be 1, 8, or 16 depending on CPU features
        REQUIRE((spv == 1 || spv == 8 || spv == 16));
    }

    SECTION("int8_t to int16_t") {
        converter<int8_t, int16_t> conv;
        auto spv = conv.samples_per_vector();
        // Should be 1, 16, or 32 depending on CPU features
        REQUIRE((spv == 1 || spv == 16 || spv == 32));
    }

    SECTION("int16_t to float") {
        converter<int16_t, float> conv;
        auto spv = conv.samples_per_vector();
        // Should be 1, 8, or 16 depending on CPU features
        REQUIRE((spv == 1 || spv == 8 || spv == 16));
    }

    SECTION("int16_t to int16_t") {
        converter<int16_t, int16_t> conv;
        auto spv = conv.samples_per_vector();
        // Should be 1, 16, or 32 depending on CPU features
        REQUIRE((spv == 1 || spv == 16 || spv == 32));
    }

    SECTION("uint32_t to float") {
        converter<uint32_t, float> conv;
        auto spv = conv.samples_per_vector();
        // Should be 1, 8, or 16 depending on CPU features
        REQUIRE((spv == 1 || spv == 8 || spv == 16));
    }
}

TEST_CASE("Converter int16_t to float - various sizes for remainder handling", "[conversion][remainder]") {
    converter<int16_t, float> conv;
    auto spv = conv.samples_per_vector();

    // Test sizes around vector boundaries (avoid underflow if spv = 1)
    std::vector<size_t> test_sizes;
    if (spv > 1) test_sizes.push_back(spv - 1);
    test_sizes.insert(test_sizes.end(), {spv, spv + 1});
    if (2 * spv > 1) test_sizes.push_back(2 * spv - 1);
    test_sizes.insert(test_sizes.end(), {2 * spv, 2 * spv + 1});

    for (size_t test_size : test_sizes) {
        // Use aligned memory for AVX/AVX-512 store instructions
        alignas(64) int16_t input[256];
        alignas(64) float output[256];

        REQUIRE(test_size <= 256);  // Ensure we don't overflow

        for (size_t i = 0; i < test_size; ++i) {
            input[i] = static_cast<int16_t>(i * 11);
        }

        conv.process(reinterpret_cast<const uint8_t*>(input), output, test_size);

        for (size_t i = 0; i < test_size; ++i) {
            REQUIRE(output[i] == Approx(static_cast<float>(input[i])));
        }
    }
}

TEST_CASE("Converter int8_t to int16_t - sign extension verification", "[conversion]") {
    converter<int8_t, int16_t> conv;

    // Test negative values to ensure proper sign extension
    std::array<int8_t, 8> input = {-128, -127, -64, -1, 0, 1, 64, 127};
    std::array<int16_t, 8> output{};

    conv.process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 8);

    REQUIRE(output[0] == -128);
    REQUIRE(output[1] == -127);
    REQUIRE(output[2] == -64);
    REQUIRE(output[3] == -1);
    REQUIRE(output[4] == 0);
    REQUIRE(output[5] == 1);
    REQUIRE(output[6] == 64);
    REQUIRE(output[7] == 127);
}

TEST_CASE("Converter base pointer interface - polymorphism", "[conversion]") {
    SECTION("int8_t to float via base pointer") {
        std::unique_ptr<converter_base<float>> conv =
            std::make_unique<converter<int8_t, float>>();

        std::array<int8_t, 4> input = {-10, 0, 10, 100};
        std::array<float, 4> output{};

        conv->process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 4);

        REQUIRE(output[0] == Approx(-10.0f));
        REQUIRE(output[1] == Approx(0.0f));
        REQUIRE(output[2] == Approx(10.0f));
        REQUIRE(output[3] == Approx(100.0f));
        REQUIRE(conv->samples_per_vector() >= 1);
    }

    SECTION("int16_t to int16_t via base pointer") {
        std::unique_ptr<converter_base<int16_t>> conv =
            std::make_unique<converter<int16_t, int16_t>>(true);  // With byteswap

        std::array<int16_t, 2> input = {0x1234, 0x5678};
        std::array<int16_t, 2> output{};

        conv->process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 2);

        REQUIRE(output[0] == 0x3412);
        REQUIRE(output[1] == 0x7856);
        REQUIRE(conv->samples_per_vector() >= 1);
    }
}

TEST_CASE("Converter uint32_t to float - special float values", "[conversion]") {
    converter<uint32_t, float> conv;

    // Test special IEEE 754 float values
    std::array<float, 6> special_floats = {
        0.0f,
        -0.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::min(),
        std::numeric_limits<float>::max()
    };

    std::array<uint32_t, 6> input;
    for (size_t i = 0; i < 6; ++i) {
        input[i] = std::bit_cast<uint32_t>(special_floats[i]);
    }

    std::array<float, 6> output{};
    conv.process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), 6);

    REQUIRE(output[0] == 0.0f);
    REQUIRE(output[1] == -0.0f);
    REQUIRE(std::isinf(output[2]));
    REQUIRE(std::isinf(output[3]));
    REQUIRE(output[4] == Approx(special_floats[4]));
    REQUIRE(output[5] == Approx(special_floats[5]));
}

TEST_CASE("Converter int16_t byteswap - round trip", "[conversion][byteswap]") {
    // Byteswap twice should give original value
    converter<int16_t, int16_t> conv_swap(true);

    std::array<int16_t, 4> input = {0x1234, 0x5678, static_cast<int16_t>(0xABCD), 0x00FF};
    std::array<int16_t, 4> temp{};
    std::array<int16_t, 4> output{};

    // First swap
    conv_swap.process(reinterpret_cast<const uint8_t*>(input.data()), temp.data(), 4);
    // Second swap
    conv_swap.process(reinterpret_cast<const uint8_t*>(temp.data()), output.data(), 4);

    // Should match original
    for (size_t i = 0; i < 4; ++i) {
        REQUIRE(output[i] == input[i]);
    }
}

TEST_CASE("Converter large buffer processing", "[conversion][alignment]") {
    converter<int16_t, float> conv;

    // Process a reasonably large aligned buffer
    constexpr size_t SIZE = 256;
    // Use 64-byte alignment for AVX-512
    alignas(64) std::array<int16_t, SIZE> input;
    alignas(64) std::array<float, SIZE> output;

    for (size_t i = 0; i < SIZE; ++i) {
        input[i] = static_cast<int16_t>(i * 123);
    }

    conv.process(reinterpret_cast<const uint8_t*>(input.data()), output.data(), SIZE);

    for (size_t i = 0; i < SIZE; ++i) {
        REQUIRE(output[i] == Approx(static_cast<float>(input[i])));
    }
}
