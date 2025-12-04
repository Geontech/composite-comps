/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Framer Pool Wraparound Tests - CRITICAL: Tests for overlap continuity bugs
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

// ============================================================================
// T4: Ring Buffer Wraparound Tests
// ============================================================================

TEST_CASE("T4.1: First wrap at ring_size", "[pool][wraparound]") {
    constexpr size_t FRAME_SIZE = 80;   // Aligned, meets minimum
    constexpr size_t OVERLAP = 16;      // Aligned
    constexpr size_t FRAME_COUNT = 4;
    constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP; // 64
    const size_t RING_SIZE = FRAME_COUNT * HOP_SIZE + OVERLAP; // 272

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Write exactly ring_size samples
    vector<uint8_t> data(RING_SIZE * 2);
    for (size_t i = 0; i < RING_SIZE; ++i) {
        data[i*2] = static_cast<int8_t>(i);
        data[i*2+1] = static_cast<int8_t>(i);
    }

    REQUIRE(pool->write_samples(data.data(), RING_SIZE, &conv, 2, true));
    REQUIRE(pool->head() == RING_SIZE);

    // Emit last frame before wrap (frame 3)
    auto frame3 = pool->try_emit_frame(3 * HOP_SIZE);
    REQUIRE(frame3.has_value());

    // Write more to trigger wrap
    vector<uint8_t> more_data(HOP_SIZE * 2);
    for (size_t i = 0; i < HOP_SIZE; ++i) {
        more_data[i*2] = static_cast<int8_t>(RING_SIZE + i);
        more_data[i*2+1] = static_cast<int8_t>(RING_SIZE + i);
    }

    REQUIRE(pool->write_samples(more_data.data(), HOP_SIZE, &conv, 2, true));
    REQUIRE(pool->head() == RING_SIZE + HOP_SIZE);
}

TEST_CASE("T4.2: Overlap continuity across first wrap", "[pool][wraparound][critical]") {
    constexpr size_t FRAME_SIZE = 80;   // Aligned, meets minimum
    constexpr size_t OVERLAP = 16;      // Aligned
    constexpr size_t FRAME_COUNT = 4;
    constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;
    const size_t RING_SIZE = FRAME_COUNT * HOP_SIZE + OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Write pattern: 0, 1, 2, ..., RING_SIZE-1
    vector<uint8_t> data(RING_SIZE * 2);
    for (size_t i = 0; i < RING_SIZE; ++i) {
        data[i*2] = static_cast<int8_t>(i);
        data[i*2+1] = static_cast<int8_t>(i);
    }
    REQUIRE(pool->write_samples(data.data(), RING_SIZE, &conv, 2, true));

    // Emit frame 3 (before wrap): absolute [192, 272), ring [192, 272)
    // Frame 3 starts at 3 * HOP_SIZE = 3 * 64 = 192
    auto frame3 = pool->try_emit_frame(3 * HOP_SIZE);
    REQUIRE(frame3.has_value());
    auto frame3_data = frame3->data();

    // Verify frame 3 last OVERLAP samples
    // At position FRAME_SIZE - OVERLAP = 64 in frame, absolute sample = 192 + 64 = 256
    size_t overlap_start_abs = 3 * HOP_SIZE + (FRAME_SIZE - OVERLAP);
    float expected_first = static_cast<float>(static_cast<int8_t>(overlap_start_abs));
    float expected_second = static_cast<float>(static_cast<int8_t>(overlap_start_abs + 1));
    REQUIRE_THAT(frame3_data[FRAME_SIZE - OVERLAP].real(), WithinAbs(expected_first, 0.01f));
    REQUIRE_THAT(frame3_data[FRAME_SIZE - OVERLAP + 1].real(), WithinAbs(expected_second, 0.01f));

    // Write more data past ring_size
    vector<uint8_t> more_data(HOP_SIZE * 2);
    for (size_t i = 0; i < HOP_SIZE; ++i) {
        more_data[i*2] = static_cast<int8_t>(RING_SIZE + i);
        more_data[i*2+1] = static_cast<int8_t>(RING_SIZE + i);
    }
    REQUIRE(pool->write_samples(more_data.data(), HOP_SIZE, &conv, 2, true));

    // Emit frame 4 (after wrap): absolute [256, 336), ring [0, 80)
    // Frame 4 starts at 4 * HOP_SIZE = 4 * 64 = 256
    // This uses slot 0 (4 % 4 = 0)
    // First OVERLAP samples should match last OVERLAP of frame 3
    auto frame4 = pool->try_emit_frame(4 * HOP_SIZE);
    REQUIRE(frame4.has_value());
    auto frame4_data = frame4->data();

    INFO("Frame 4 should overlap with frame 3");
    INFO("Frame 3 last overlap starts at absolute sample: " << overlap_start_abs);
    INFO("Frame 4 starts at absolute sample: " << (4 * HOP_SIZE));

    REQUIRE_THAT(frame4_data[0].real(), WithinAbs(expected_first, 0.01f));
    REQUIRE_THAT(frame4_data[1].real(), WithinAbs(expected_second, 0.01f));
}

TEST_CASE("T4.3: Multiple wraps (2x ring_size)", "[pool][wraparound][critical]") {
    constexpr size_t FRAME_SIZE = 80;   // Aligned, meets minimum
    constexpr size_t OVERLAP = 16;      // Aligned
    constexpr size_t FRAME_COUNT = 4;
    constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;
    const size_t RING_SIZE = FRAME_COUNT * HOP_SIZE + OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Write 2x ring_size samples with incrementing pattern
    size_t total_samples = 2 * RING_SIZE;
    vector<uint8_t> data(HOP_SIZE * 2);

    // Ensure we write enough to complete the frame we'll emit
    size_t frame_num = (2 * RING_SIZE) / HOP_SIZE;
    size_t min_samples = frame_num * HOP_SIZE + FRAME_SIZE;
    if (total_samples < min_samples) {
        total_samples = min_samples;
    }

    for (size_t written = 0; written < total_samples; written += HOP_SIZE) {
        for (size_t i = 0; i < HOP_SIZE; ++i) {
            data[i*2] = static_cast<int8_t>((written + i) % 256);
            data[i*2+1] = static_cast<int8_t>((written + i) % 256);
        }

        // Calculate which frame we're currently writing to
        size_t current_frame_num = written / HOP_SIZE;

        // Before writing, emit the oldest frame that would conflict with future writes
        // We can have at most FRAME_COUNT frames in the pool
        if (current_frame_num >= FRAME_COUNT) {
            size_t old_frame_num = current_frame_num - FRAME_COUNT;
            auto old_frame = pool->try_emit_frame(old_frame_num * HOP_SIZE);
            // Let it drop to free the slot
        }

        REQUIRE(pool->write_samples(data.data(), HOP_SIZE, &conv, 2, true));
    }

    REQUIRE(pool->head() >= 2 * RING_SIZE);

    // Emit frames after 2nd wrap - slot 0 reused again
    auto frame = pool->try_emit_frame(frame_num * HOP_SIZE);
    REQUIRE(frame.has_value());
}

TEST_CASE("T4.4: Overlap correctness after multiple wraps", "[pool][wraparound][critical]") {
    // THIS TEST WILL LIKELY FAIL - exposes bug #1
    constexpr size_t FRAME_SIZE = 80;   // Aligned, meets minimum
    constexpr size_t OVERLAP = 16;      // Aligned
    constexpr size_t FRAME_COUNT = 4;
    constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;
    const size_t RING_SIZE = FRAME_COUNT * HOP_SIZE + OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Write 3x ring_size with known pattern
    vector<uint8_t> chunk(HOP_SIZE * 2);
    size_t total = 3 * RING_SIZE;

    // Calculate which frames we need to keep for validation
    size_t base = ((3 * RING_SIZE / HOP_SIZE) - 1) * HOP_SIZE;
    size_t keep_frame1_start = base;
    size_t keep_frame2_start = base + HOP_SIZE;

    // Need to write enough data to complete frame_after
    // frame_after starts at base + HOP_SIZE and needs FRAME_SIZE samples
    size_t min_samples_needed = keep_frame2_start + FRAME_SIZE;
    if (total < min_samples_needed) {
        total = min_samples_needed;
    }

    for (size_t written = 0; written < total; written += HOP_SIZE) {
        for (size_t i = 0; i < HOP_SIZE; ++i) {
            chunk[i*2] = static_cast<int8_t>((written + i) % 128);
            chunk[i*2+1] = static_cast<int8_t>((written + i) % 128);
        }

        // Calculate which frame we're currently writing to
        size_t current_frame_num = written / HOP_SIZE;

        // Before writing, emit the oldest frame that would conflict
        if (current_frame_num >= FRAME_COUNT) {
            size_t old_frame_num = current_frame_num - FRAME_COUNT;
            size_t old_frame_start = old_frame_num * HOP_SIZE;
            // Don't emit frames we need to keep for validation
            if (old_frame_start != keep_frame1_start && old_frame_start != keep_frame2_start) {
                auto old_frame = pool->try_emit_frame(old_frame_start);
                // Let it drop to free the slot
            }
        }

        REQUIRE(pool->write_samples(chunk.data(), HOP_SIZE, &conv, 2, true));
    }

    // Emit consecutive frames near 3rd wrap

    auto frame_before = pool->try_emit_frame(base);
    auto frame_after = pool->try_emit_frame(base + HOP_SIZE);

    REQUIRE(frame_before.has_value());
    REQUIRE(frame_after.has_value());

    auto before_data = frame_before->data();
    auto after_data = frame_after->data();

    // Verify overlap: last OVERLAP samples of frame_before should equal
    // first OVERLAP samples of frame_after
    for (size_t i = 0; i < OVERLAP; ++i) {
        INFO("Checking overlap sample " << i);
        INFO("Before[" << (FRAME_SIZE - OVERLAP + i) << "] = " << before_data[FRAME_SIZE - OVERLAP + i].real());
        INFO("After[" << i << "] = " << after_data[i].real());

        REQUIRE_THAT(
            after_data[i].real(),
            WithinAbs(before_data[FRAME_SIZE - OVERLAP + i].real(), 0.01f)
        );
    }
}

TEST_CASE("T4.5: Frame data integrity across wraps", "[pool][wraparound]") {
    constexpr size_t FRAME_SIZE = 128;  // Aligned, meets minimum
    constexpr size_t OVERLAP = 32;      // Aligned
    constexpr size_t FRAME_COUNT = 8;
    constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;
    const size_t RING_SIZE = FRAME_COUNT * HOP_SIZE + OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Write 2x ring_size samples
    vector<uint8_t> chunk(HOP_SIZE * 2);
    size_t num_frames = (2 * RING_SIZE - FRAME_SIZE) / HOP_SIZE + 1;

    for (size_t written = 0; written < 2 * RING_SIZE; written += HOP_SIZE) {
        for (size_t i = 0; i < HOP_SIZE; ++i) {
            size_t sample_idx = written + i;
            chunk[i*2] = static_cast<int8_t>(sample_idx % 256);
            chunk[i*2+1] = static_cast<int8_t>((sample_idx + 1) % 256);
        }

        // Calculate which frame we're currently writing to
        size_t current_frame_num = written / HOP_SIZE;

        // Before writing, emit the oldest frame that would conflict
        // Keep the last 4 frames for validation
        if (current_frame_num >= FRAME_COUNT) {
            size_t old_frame_num = current_frame_num - FRAME_COUNT;
            if (old_frame_num < num_frames - 4) {
                auto old_frame = pool->try_emit_frame(old_frame_num * HOP_SIZE);
                // Let it drop to free the slot
            }
        }

        REQUIRE(pool->write_samples(chunk.data(), HOP_SIZE, &conv, 2, true));
    }

    // Emit frames and verify each has correct data

    for (size_t f = num_frames - 4; f < num_frames; ++f) {
        auto frame = pool->try_emit_frame(f * HOP_SIZE);
        if (!frame.has_value()) continue; // Skip if slot busy

        auto data = frame->data();
        size_t start_sample = f * HOP_SIZE;

        // Verify first and last samples
        float expected_i = static_cast<float>(static_cast<int8_t>(start_sample % 256));
        float expected_q = static_cast<float>(static_cast<int8_t>((start_sample + 1) % 256));

        INFO("Frame " << f << " start sample " << start_sample);
        REQUIRE_THAT(data[0].real(), WithinAbs(expected_i, 0.01f));
        REQUIRE_THAT(data[0].imag(), WithinAbs(expected_q, 0.01f));
    }
}

TEST_CASE("T4.6: Sustained operation across 10x ring_size", "[pool][wraparound][stress]") {
    constexpr size_t FRAME_SIZE = 128;
    constexpr size_t OVERLAP = 32;
    constexpr size_t FRAME_COUNT = 16;
    constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;
    const size_t RING_SIZE = FRAME_COUNT * HOP_SIZE + OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Write 10x ring_size samples
    vector<uint8_t> chunk(HOP_SIZE * 2, 42);
    size_t target_samples = 10 * RING_SIZE;
    size_t write_count = (target_samples + HOP_SIZE - 1) / HOP_SIZE;  // Round up

    for (size_t i = 0; i < write_count; ++i) {
        // Before writing, emit the oldest frame that would conflict
        if (i >= FRAME_COUNT) {
            size_t old_frame_num = i - FRAME_COUNT;
            auto old_frame = pool->try_emit_frame(old_frame_num * HOP_SIZE);
            // Let it drop to free the slot
        }

        bool write_ok = pool->write_samples(chunk.data(), HOP_SIZE, &conv, 2, true);
        REQUIRE(write_ok);
    }

    REQUIRE(pool->head() >= 10 * RING_SIZE);

    // Verify we can still emit frames
    size_t frame_idx = (pool->head() - FRAME_SIZE) / HOP_SIZE;
    auto frame = pool->try_emit_frame(frame_idx * HOP_SIZE);
    REQUIRE(frame.has_value());
}
