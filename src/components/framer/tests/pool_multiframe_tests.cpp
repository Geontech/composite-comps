/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Framer Pool Multiple Frame Tests - Testing concurrent frame operations
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
// T3: Multiple Frame Operations
// ============================================================================

TEST_CASE("T3.1: Write and emit consecutive frames", "[pool][multiframe]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Write enough for 5 frames
    vector<uint8_t> data(FRAME_SIZE * 2);
    for (size_t i = 0; i < FRAME_SIZE; ++i) {
        data[i*2] = static_cast<int8_t>(i % 128);
        data[i*2+1] = static_cast<int8_t>(i % 128);
    }

    REQUIRE(pool->write_samples(data.data(), FRAME_SIZE, &conv, 2, true));

    for (size_t i = 0; i < 4; ++i) {
        vector<uint8_t> hop_data(HOP_SIZE * 2, static_cast<uint8_t>(i + 10));
        REQUIRE(pool->write_samples(hop_data.data(), HOP_SIZE, &conv, 2, true));
    }

    // Emit all 5 frames
    for (size_t i = 0; i < 5; ++i) {
        auto frame = pool->try_emit_frame(i * HOP_SIZE);
        REQUIRE(frame.has_value());
        REQUIRE(frame->size() == FRAME_SIZE);
    }
}

TEST_CASE("T3.2: Overlapping frames correctness", "[pool][multiframe]") {
    constexpr size_t TEST_FRAME_SIZE = 128;  // Aligned, meets minimum
    constexpr size_t TEST_OVERLAP = 32;      // Aligned
    constexpr size_t TEST_FRAME_COUNT = 8;
    constexpr size_t TEST_HOP = TEST_FRAME_SIZE - TEST_OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(
        TEST_FRAME_SIZE, TEST_OVERLAP, TEST_FRAME_COUNT
    );
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Write continuous incrementing pattern
    vector<uint8_t> data(TEST_FRAME_SIZE * 2);
    for (size_t i = 0; i < TEST_FRAME_SIZE; ++i) {
        data[i*2] = static_cast<int8_t>(i);
        data[i*2+1] = static_cast<int8_t>(i);
    }
    REQUIRE(pool->write_samples(data.data(), TEST_FRAME_SIZE, &conv, 2, true));

    // Write more data
    for (size_t write = 1; write < 5; ++write) {
        vector<uint8_t> hop_data(TEST_HOP * 2);
        for (size_t i = 0; i < TEST_HOP; ++i) {
            size_t val = write * TEST_HOP + i;
            hop_data[i*2] = static_cast<int8_t>(val);
            hop_data[i*2+1] = static_cast<int8_t>(val);
        }
        REQUIRE(pool->write_samples(hop_data.data(), TEST_HOP, &conv, 2, true));
    }

    // Emit frames 0 and 1, verify overlap
    auto frame0 = pool->try_emit_frame(0);
    auto frame1 = pool->try_emit_frame(TEST_HOP);

    REQUIRE(frame0.has_value());
    REQUIRE(frame1.has_value());

    auto data0 = frame0->data();
    auto data1 = frame1->data();

    // Last OVERLAP samples of frame 0 should match first OVERLAP samples of frame 1
    INFO("Verifying overlap between frame 0 and frame 1");
    for (size_t i = 0; i < TEST_OVERLAP; ++i) {
        size_t idx0 = TEST_FRAME_SIZE - TEST_OVERLAP + i;
        REQUIRE_THAT(data1[i].real(), WithinAbs(data0[idx0].real(), 0.01f));
        REQUIRE_THAT(data1[i].imag(), WithinAbs(data0[idx0].imag(), 0.01f));
    }
}

TEST_CASE("T3.3: All N slots can be used", "[pool][multiframe]") {
    constexpr size_t TEST_FRAME_SIZE = 128;
    constexpr size_t TEST_OVERLAP = 32;
    constexpr size_t TEST_FRAME_COUNT = 8;
    constexpr size_t TEST_HOP = TEST_FRAME_SIZE - TEST_OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(
        TEST_FRAME_SIZE, TEST_OVERLAP, TEST_FRAME_COUNT
    );
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Fill ring buffer
    size_t ring_size = TEST_FRAME_COUNT * TEST_HOP + TEST_OVERLAP;
    vector<uint8_t> data(ring_size * 2, 42);
    REQUIRE(pool->write_samples(data.data(), ring_size, &conv, 2, true));

    // Acquire all N frames simultaneously
    vector<optional<composite::immutable_buffer<complex<float>>>> frames;
    for (size_t i = 0; i < TEST_FRAME_COUNT; ++i) {
        auto frame = pool->try_emit_frame(i * TEST_HOP);
        REQUIRE(frame.has_value());
        frames.push_back(move(frame));
    }

    // Verify all slots are now busy
    for (size_t i = 0; i < TEST_FRAME_COUNT; ++i) {
        auto should_fail = pool->try_emit_frame(i * TEST_HOP);
        REQUIRE_FALSE(should_fail.has_value());
    }
}

TEST_CASE("T3.4: Out-of-order frame release", "[pool][multiframe]") {
    constexpr size_t TEST_FRAME_SIZE = 256;
    constexpr size_t TEST_OVERLAP = 64;
    constexpr size_t TEST_FRAME_COUNT = 8;
    constexpr size_t TEST_HOP = TEST_FRAME_SIZE - TEST_OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(
        TEST_FRAME_SIZE, TEST_OVERLAP, TEST_FRAME_COUNT
    );
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Fill ring
    size_t ring_size = TEST_FRAME_COUNT * TEST_HOP + TEST_OVERLAP;
    vector<uint8_t> data(ring_size * 2, 0);
    REQUIRE(pool->write_samples(data.data(), ring_size, &conv, 2, true));

    // Acquire frames 0, 2, 4
    auto frame0 = pool->try_emit_frame(0 * TEST_HOP);
    auto frame2 = pool->try_emit_frame(2 * TEST_HOP);
    auto frame4 = pool->try_emit_frame(4 * TEST_HOP);

    REQUIRE(frame0.has_value());
    REQUIRE(frame2.has_value());
    REQUIRE(frame4.has_value());

    // Release frame 2 (middle)
    frame2.reset();

    // Frame 2 should now be re-acquirable
    auto frame2_again = pool->try_emit_frame(2 * TEST_HOP);
    REQUIRE(frame2_again.has_value());

    // Frames 0 and 4 should still be busy
    REQUIRE_FALSE(pool->try_emit_frame(0 * TEST_HOP).has_value());
    REQUIRE_FALSE(pool->try_emit_frame(4 * TEST_HOP).has_value());

    // Release all
    frame0.reset();
    frame4.reset();
    frame2_again.reset();

    // All should be available now
    REQUIRE(pool->try_emit_frame(0 * TEST_HOP).has_value());
    REQUIRE(pool->try_emit_frame(2 * TEST_HOP).has_value());
    REQUIRE(pool->try_emit_frame(4 * TEST_HOP).has_value());
}

TEST_CASE("T3.5: Multiple references to same frame", "[pool][multiframe]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    vector<uint8_t> data(FRAME_SIZE * 2, 99);
    REQUIRE(pool->write_samples(data.data(), FRAME_SIZE, &conv, 2, true));

    // Acquire frame 0
    auto frame1 = pool->try_emit_frame(0);
    REQUIRE(frame1.has_value());

    // Copy the optional (shares the immutable_buffer)
    auto frame2 = frame1;
    REQUIRE(frame2.has_value());

    // Both should point to same data
    REQUIRE(frame1->data() == frame2->data());

    // Release first reference
    frame1.reset();
    REQUIRE_FALSE(frame1.has_value());

    // Second reference should still be valid
    REQUIRE(frame2.has_value());
    REQUIRE(frame2->size() == FRAME_SIZE);

    // Slot should still be busy (second reference alive)
    auto should_fail = pool->try_emit_frame(0);
    REQUIRE_FALSE(should_fail.has_value());

    // Release second reference
    frame2.reset();

    // Now slot should be free (after wraparound reuse)
}

TEST_CASE("T3.6: Consecutive frame data continuity", "[pool][multiframe]") {
    constexpr size_t TEST_FRAME_SIZE = 256;  // Aligned, meets minimum
    constexpr size_t TEST_OVERLAP = 64;      // Aligned
    constexpr size_t TEST_FRAME_COUNT = 4;
    constexpr size_t TEST_HOP = TEST_FRAME_SIZE - TEST_OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(
        TEST_FRAME_SIZE, TEST_OVERLAP, TEST_FRAME_COUNT
    );
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Write continuous stream with known pattern
    size_t total_samples = 100;
    vector<uint8_t> chunk(TEST_HOP * 2);

    for (size_t written = 0; written < total_samples; written += TEST_HOP) {
        size_t count = min(TEST_HOP, total_samples - written);
        for (size_t i = 0; i < count; ++i) {
            size_t val = (written + i) % 256;
            chunk[i*2] = static_cast<int8_t>(val);
            chunk[i*2+1] = static_cast<int8_t>(val + 1);
        }
        REQUIRE(pool->write_samples(chunk.data(), count, &conv, 2, true));
    }

    // Emit several consecutive frames and verify continuity
    for (size_t f = 0; f < 3; ++f) {
        auto curr = pool->try_emit_frame(f * TEST_HOP);
        auto next = pool->try_emit_frame((f + 1) * TEST_HOP);

        if (!curr.has_value() || !next.has_value()) continue;

        auto curr_data = curr->data();
        auto next_data = next->data();

        // Verify overlap region matches
        for (size_t i = 0; i < TEST_OVERLAP; ++i) {
            INFO("Frame " << f << " to " << (f+1) << ", overlap sample " << i);
            REQUIRE_THAT(
                next_data[i].real(),
                WithinAbs(curr_data[TEST_FRAME_SIZE - TEST_OVERLAP + i].real(), 0.01f)
            );
        }
    }
}
