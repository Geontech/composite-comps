/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Framer Pool Basic Tests - Core functionality validation
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <complex>
#include <memory>
#include <vector>

#include "../framer_pool.hpp"
#include "../convert_variant.hpp"

using namespace std;
using Catch::Matchers::WithinAbs;

constexpr size_t FRAME_SIZE = 2048;
constexpr size_t OVERLAP = 512;
constexpr size_t FRAME_COUNT = 64;
constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;

// ============================================================================
// T1: Pool Basic Operations
// ============================================================================

TEST_CASE("T1.1: Pool creation with valid parameters", "[pool][basic]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);

    REQUIRE(pool != nullptr);
    REQUIRE(pool->frame_size() == FRAME_SIZE);
    REQUIRE(pool->hop_size() == HOP_SIZE);
    REQUIRE(pool->head() == 0);
}

TEST_CASE("T1.2: Pool creation with invalid parameters", "[pool][basic]") {
    SECTION("frame_size = 0") {
        REQUIRE_THROWS_AS(
            make_shared<framer_pool<complex<float>>>(0, 512, 64),
            std::invalid_argument
        );
    }

    SECTION("overlap >= frame_size") {
        REQUIRE_THROWS_AS(
            make_shared<framer_pool<complex<float>>>(1024, 1024, 64),
            std::invalid_argument
        );
        REQUIRE_THROWS_AS(
            make_shared<framer_pool<complex<float>>>(1024, 2048, 64),
            std::invalid_argument
        );
    }

    SECTION("frame_count = 0") {
        REQUIRE_THROWS_AS(
            make_shared<framer_pool<complex<float>>>(1024, 512, 0),
            std::invalid_argument
        );
    }

    SECTION("hop_size too small (overlap too large)") {
        // frame_size = 64, overlap = 64 → hop_size = 0 (invalid, caught by overlap >= frame_size check)
        REQUIRE_THROWS_AS(
            make_shared<framer_pool<complex<float>>>(64, 64, 4),
            std::invalid_argument
        );
    }
}

TEST_CASE("T1.3: Ring size calculation", "[pool][basic]") {
    // ring_size = frame_count * hop_size + overlap
    size_t expected_ring_size = FRAME_COUNT * HOP_SIZE + OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Write exactly ring_size samples
    vector<uint8_t> data(expected_ring_size * 2, 0);
    bool write_ok = pool->write_samples(data.data(), expected_ring_size, &conv, 2, true);

    REQUIRE(write_ok);
    REQUIRE(pool->head() == expected_ring_size);
}

TEST_CASE("T1.4: Slot offset calculation", "[pool][basic]") {
    constexpr size_t TEST_FRAME_SIZE = 128;  // Aligned, meets minimum
    constexpr size_t TEST_OVERLAP = 32;      // Aligned
    constexpr size_t TEST_FRAME_COUNT = 4;
    constexpr size_t TEST_HOP = TEST_FRAME_SIZE - TEST_OVERLAP; // 96

    auto pool = make_shared<framer_pool<complex<float>>>(
        TEST_FRAME_SIZE, TEST_OVERLAP, TEST_FRAME_COUNT
    );
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Fill ring buffer with identifiable data
    size_t ring_size = TEST_FRAME_COUNT * TEST_HOP + TEST_OVERLAP; // 416
    vector<uint8_t> data(ring_size * 2);
    for (size_t i = 0; i < ring_size; ++i) {
        data[i*2] = static_cast<int8_t>(i % 128);
        data[i*2+1] = static_cast<int8_t>(i % 128);
    }
    pool->write_samples(data.data(), ring_size, &conv, 2, true);

    // Verify each slot has correct offset
    // Slot 0: offset 0, samples [0, 128)
    // Slot 1: offset 96, samples [96, 224)
    // Slot 2: offset 192, samples [192, 320)
    // Slot 3: offset 288, samples [288, 416)

    for (size_t i = 0; i < TEST_FRAME_COUNT; ++i) {
        auto frame = pool->try_emit_frame(i * TEST_HOP);
        REQUIRE(frame.has_value());

        auto data_ptr = frame->data();
        size_t expected_start = i * TEST_HOP;

        // Check first sample
        float expected_val = static_cast<float>(static_cast<int8_t>(expected_start % 128));
        REQUIRE_THAT(data_ptr[0].real(), WithinAbs(expected_val, 0.01f));
    }
}

// ============================================================================
// T2: Single Frame Operations
// ============================================================================

TEST_CASE("T2.1: Write and emit single frame", "[pool][single]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    vector<uint8_t> data(FRAME_SIZE * 2, 42);

    bool write_ok = pool->write_samples(data.data(), FRAME_SIZE, &conv, 2, true);
    REQUIRE(write_ok);
    REQUIRE(pool->head() == FRAME_SIZE);

    auto frame = pool->try_emit_frame(0);
    REQUIRE(frame.has_value());
    REQUIRE(frame->size() == FRAME_SIZE);
}

TEST_CASE("T2.2: Data integrity verification", "[pool][single]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Create known pattern
    vector<uint8_t> data(FRAME_SIZE * 2);
    for (size_t i = 0; i < FRAME_SIZE; ++i) {
        data[i*2] = static_cast<int8_t>(i % 128);      // I
        data[i*2+1] = static_cast<int8_t>((i+1) % 128); // Q
    }

    pool->write_samples(data.data(), FRAME_SIZE, &conv, 2, true);
    auto frame = pool->try_emit_frame(0);
    REQUIRE(frame.has_value());

    auto frame_data = frame->data();

    // Verify first, middle, and last samples
    REQUIRE_THAT(frame_data[0].real(), WithinAbs(0.0f, 0.01f));
    REQUIRE_THAT(frame_data[0].imag(), WithinAbs(1.0f, 0.01f));

    REQUIRE_THAT(frame_data[100].real(), WithinAbs(100.0f, 0.01f));
    REQUIRE_THAT(frame_data[100].imag(), WithinAbs(101.0f, 0.01f));

    size_t last = FRAME_SIZE - 1;
    float expected_i = static_cast<float>(static_cast<int8_t>(last % 128));
    float expected_q = static_cast<float>(static_cast<int8_t>((last+1) % 128));
    REQUIRE_THAT(frame_data[last].real(), WithinAbs(expected_i, 0.01f));
    REQUIRE_THAT(frame_data[last].imag(), WithinAbs(expected_q, 0.01f));
}

TEST_CASE("T2.3: Frame size correctness", "[pool][single]") {
    constexpr size_t sizes[] = {128, 512, 1024, 2048, 4096};

    for (auto frame_size : sizes) {
        auto pool = make_shared<framer_pool<complex<float>>>(frame_size, frame_size/4, 16);
        converter_variant<float> conv = converter<int8_t, float>(false);

        vector<uint8_t> data(frame_size * 2, 0);
        pool->write_samples(data.data(), frame_size, &conv, 2, true);

        auto frame = pool->try_emit_frame(0);
        REQUIRE(frame.has_value());
        REQUIRE(frame->size() == frame_size);
    }
}

TEST_CASE("T2.4: Slot acquisition and release", "[pool][single]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    vector<uint8_t> data(FRAME_SIZE * 2, 0);
    pool->write_samples(data.data(), FRAME_SIZE, &conv, 2, true);

    // Acquire frame
    auto frame1 = pool->try_emit_frame(0);
    REQUIRE(frame1.has_value());

    // Try to acquire same slot again - should fail
    auto frame2 = pool->try_emit_frame(0);
    REQUIRE_FALSE(frame2.has_value());

    // Release frame
    frame1.reset();

    // Write more data to wrap and reuse slot 0
    size_t ring_size = FRAME_COUNT * HOP_SIZE + OVERLAP;
    for (size_t written = FRAME_SIZE; written < ring_size + HOP_SIZE; written += HOP_SIZE) {
        pool->write_samples(data.data(), HOP_SIZE, &conv, 2, true);
    }

    // Now slot 0 should be available for reuse
    auto frame3 = pool->try_emit_frame(FRAME_COUNT * HOP_SIZE);
    REQUIRE(frame3.has_value());
}

TEST_CASE("T2.5: Failed acquisition when slot busy", "[pool][single]") {
    auto pool = make_shared<framer_pool<complex<float>>>(1024, 256, 4);
    converter_variant<float> conv = converter<int8_t, float>(false);

    vector<uint8_t> data(1024 * 2, 0);
    pool->write_samples(data.data(), 1024, &conv, 2, true);

    // Acquire slot 0
    auto frame = pool->try_emit_frame(0);
    REQUIRE(frame.has_value());

    // Try to acquire again - should fail immediately
    auto frame2 = pool->try_emit_frame(0);
    REQUIRE_FALSE(frame2.has_value());
}
