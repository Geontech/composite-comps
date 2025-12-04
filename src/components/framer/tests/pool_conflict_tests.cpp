/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Framer Pool Conflict Detection Tests - CRITICAL: Tests for conflict logic bugs
 */

#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <complex>
#include <memory>
#include <thread>
#include <vector>

#include "../framer_pool.hpp"
#include "../convert_variant.hpp"

using namespace std;

constexpr size_t FRAME_SIZE = 2048;
constexpr size_t OVERLAP = 512;
constexpr size_t FRAME_COUNT = 64;
constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;

// ============================================================================
// T5: Conflict Detection Tests
// ============================================================================

TEST_CASE("T5.1: Write blocked when would overwrite held frame", "[pool][conflict][critical]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    vector<uint8_t> data(FRAME_SIZE * 2, 0);

    // Write and acquire frame 0
    REQUIRE(pool->write_samples(data.data(), FRAME_SIZE, &conv, 2, true));
    auto frame = pool->try_emit_frame(0);
    REQUIRE(frame.has_value());

    // Calculate ring size
    size_t ring_size = FRAME_COUNT * HOP_SIZE + OVERLAP;

    // Write until we would wrap and conflict with frame 0
    size_t samples_needed = ring_size - FRAME_SIZE;
    size_t writes_needed = (samples_needed + HOP_SIZE - 1) / HOP_SIZE;

    // Write up to the point of conflict
    for (size_t i = 0; i < writes_needed; ++i) {
        bool write_ok = pool->write_samples(data.data(), HOP_SIZE, &conv, 2, true);
        REQUIRE(write_ok);
    }

    // Next write should block (conflict with frame 0)
    atomic<bool> write_blocked{true};
    thread writer([&]() {
        pool->write_samples(data.data(), HOP_SIZE, &conv, 2, true);
        write_blocked = false;
    });

    // Give writer thread time to block
    this_thread::sleep_for(chrono::milliseconds(100));
    REQUIRE(write_blocked == true);  // Should still be blocked

    // Release frame - this should unblock the writer
    frame.reset();

    // Wait for write to complete
    writer.join();
    REQUIRE(write_blocked == false);  // Should have completed
}

TEST_CASE("T5.2: Write succeeds when no conflicts", "[pool][conflict]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    vector<uint8_t> data(FRAME_SIZE * 2, 0);

    // Write first frame's worth of data
    REQUIRE(pool->write_samples(data.data(), FRAME_SIZE, &conv, 2, true));

    // Write and immediately release frames
    for (size_t i = 0; i < 20; ++i) {
        auto frame = pool->try_emit_frame(i * HOP_SIZE);
        REQUIRE(frame.has_value());
        // Frame released at end of scope

        // Write next hop
        REQUIRE(pool->write_samples(data.data(), HOP_SIZE, &conv, 2, true));
    }
}

TEST_CASE("T5.3: Conflict detection with wrapping write", "[pool][conflict][critical]") {
    constexpr size_t TEST_FRAME_SIZE = 1024;
    constexpr size_t TEST_OVERLAP = 256;
    constexpr size_t TEST_FRAME_COUNT = 8;
    constexpr size_t TEST_HOP = TEST_FRAME_SIZE - TEST_OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(
        TEST_FRAME_SIZE, TEST_OVERLAP, TEST_FRAME_COUNT
    );
    converter_variant<float> conv = converter<int8_t, float>(false);

    size_t ring_size = TEST_FRAME_COUNT * TEST_HOP + TEST_OVERLAP;
    vector<uint8_t> data(TEST_HOP * 2, 0);

    // Fill most of ring
    size_t samples_before_wrap = ring_size - TEST_HOP;
    for (size_t written = 0; written < samples_before_wrap; written += TEST_HOP) {
        REQUIRE(pool->write_samples(data.data(), TEST_HOP, &conv, 2, true));
    }

    // Acquire frame 0
    auto frame0 = pool->try_emit_frame(0);
    REQUIRE(frame0.has_value());

    // Write that wraps (goes past ring_size, wraps to beginning)
    // This should block (conflict with held frame 0)
    atomic<bool> write_blocked{true};
    thread writer([&]() {
        pool->write_samples(data.data(), TEST_HOP, &conv, 2, true);
        write_blocked = false;
    });

    // Give writer time to block
    this_thread::sleep_for(chrono::milliseconds(100));
    REQUIRE(write_blocked == true);

    // Release frame - unblocks writer
    frame0.reset();
    writer.join();
    REQUIRE(write_blocked == false);
}

TEST_CASE("T5.4: Backpressure - write blocks then proceeds", "[pool][conflict][critical]") {
    // Test that writes block when ring is full, then proceed when space is available
    constexpr size_t TEST_FRAME_SIZE = 128;  // Aligned, meets minimum
    constexpr size_t TEST_OVERLAP = 32;      // Aligned
    constexpr size_t TEST_FRAME_COUNT = 4;
    constexpr size_t TEST_HOP = TEST_FRAME_SIZE - TEST_OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(
        TEST_FRAME_SIZE, TEST_OVERLAP, TEST_FRAME_COUNT
    );
    converter_variant<float> conv = converter<int8_t, float>(false);

    vector<uint8_t> data(TEST_HOP * 2, 0);

    // Fill ring
    size_t ring_size = TEST_FRAME_COUNT * TEST_HOP + TEST_OVERLAP;
    for (size_t written = 0; written < ring_size; written += TEST_HOP) {
        REQUIRE(pool->write_samples(data.data(), TEST_HOP, &conv, 2, true));
    }

    // Hold all frames - ring is now full
    vector<optional<composite::immutable_buffer<complex<float>>>> frames;
    for (size_t i = 0; i < TEST_FRAME_COUNT; ++i) {
        auto frame = pool->try_emit_frame(i * TEST_HOP);
        REQUIRE(frame.has_value());
        frames.push_back(move(frame));
    }

    // Start write in background thread - will block due to backpressure
    atomic<bool> write_completed{false};
    thread writer([&]() {
        pool->write_samples(data.data(), TEST_HOP, &conv, 2, true);
        write_completed = true;
    });

    // Write should be blocked
    this_thread::sleep_for(chrono::milliseconds(50));
    REQUIRE_FALSE(write_completed);

    // Release one frame - should unblock write
    frames[0].reset();

    // Wait for write to complete
    writer.join();
    REQUIRE(write_completed);
}

TEST_CASE("T5.5: Backpressure stress - multiple releases", "[pool][conflict]") {
    // Test backpressure with multiple blocked writes
    constexpr size_t TEST_FRAME_SIZE = 128;
    constexpr size_t TEST_OVERLAP = 32;
    constexpr size_t TEST_FRAME_COUNT = 8;
    constexpr size_t TEST_HOP = TEST_FRAME_SIZE - TEST_OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(
        TEST_FRAME_SIZE, TEST_OVERLAP, TEST_FRAME_COUNT
    );
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Fill ring
    size_t ring_size = TEST_FRAME_COUNT * TEST_HOP + TEST_OVERLAP;
    vector<uint8_t> data(TEST_HOP * 2, 0);

    size_t total_samples = ring_size + TEST_HOP;
    for (size_t written = 0; written < total_samples; written += TEST_HOP) {
        REQUIRE(pool->write_samples(data.data(), TEST_HOP, &conv, 2, true));
    }

    // Hold all frames
    vector<optional<composite::immutable_buffer<complex<float>>>> frames(TEST_FRAME_COUNT);
    for (size_t i = 0; i < TEST_FRAME_COUNT; ++i) {
        frames[i] = pool->try_emit_frame(i * TEST_HOP);
        REQUIRE(frames[i].has_value());
    }

    // Start write in background - will block
    atomic<int> writes_completed{0};
    thread writer([&]() {
        pool->write_samples(data.data(), TEST_HOP, &conv, 2, true);
        writes_completed++;
    });

    // Write should be blocked
    this_thread::sleep_for(chrono::milliseconds(50));
    REQUIRE(writes_completed == 0);

    // Release frames one by one
    for (size_t i = 0; i < 3; ++i) {
        frames[i].reset();
        this_thread::sleep_for(chrono::milliseconds(10));
    }

    // Wait for write to complete
    writer.join();
    REQUIRE(writes_completed == 1);
}

TEST_CASE("T5.6: Write at gap between frames", "[pool][conflict]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    vector<uint8_t> data(FRAME_SIZE * 2, 0);

    // Write enough for multiple frames
    pool->write_samples(data.data(), FRAME_SIZE + 4 * HOP_SIZE, &conv, 2, true);

    // Hold frames 0 and 4, but not 1-3
    auto frame0 = pool->try_emit_frame(0);
    auto frame4 = pool->try_emit_frame(4 * HOP_SIZE);

    REQUIRE(frame0.has_value());
    REQUIRE(frame4.has_value());

    // Writing in the gap (frames 1-3 region) should succeed
    vector<uint8_t> new_data(HOP_SIZE * 2, 99);
    REQUIRE(pool->write_samples(new_data.data(), HOP_SIZE, &conv, 2, true));
}

TEST_CASE("T5.7: Multiple held frames conflict detection", "[pool][conflict]") {
    constexpr size_t TEST_FRAME_SIZE = 512;
    constexpr size_t TEST_OVERLAP = 128;
    constexpr size_t TEST_FRAME_COUNT = 16;
    constexpr size_t TEST_HOP = TEST_FRAME_SIZE - TEST_OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(
        TEST_FRAME_SIZE, TEST_OVERLAP, TEST_FRAME_COUNT
    );
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Fill ring
    size_t ring_size = TEST_FRAME_COUNT * TEST_HOP + TEST_OVERLAP;
    vector<uint8_t> data(TEST_HOP * 2, 0);

    for (size_t written = 0; written < ring_size + TEST_HOP; written += TEST_HOP) {
        REQUIRE(pool->write_samples(data.data(), TEST_HOP, &conv, 2, true));
    }

    // Hold every other frame
    vector<optional<composite::immutable_buffer<complex<float>>>> frames;
    for (size_t i = 0; i < TEST_FRAME_COUNT; i += 2) {
        auto frame = pool->try_emit_frame(i * TEST_HOP);
        if (frame.has_value()) {
            frames.push_back(move(frame));
        }
    }

    REQUIRE(frames.size() > 0);

    // Continue writing - should eventually block due to conflict
    atomic<bool> write_blocked{false};
    atomic<size_t> writes_completed{0};
    thread writer([&]() {
        for (size_t i = 0; i < 50; ++i) {
            pool->write_samples(data.data(), TEST_HOP, &conv, 2, true);
            writes_completed++;
        }
        write_blocked = false;
    });

    // Give writer time to fill ring and block
    this_thread::sleep_for(chrono::milliseconds(200));

    // Writer should have blocked before completing all writes
    size_t completed = writes_completed.load();
    REQUIRE(completed < 50);  // Should have blocked partway through

    // Release all frames - unblocks writer
    frames.clear();
    writer.join();

    // Now all writes should have completed
    REQUIRE(writes_completed.load() == 50);
}
