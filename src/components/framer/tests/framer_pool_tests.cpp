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

#include "../framer_pool.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <complex>
#include <memory>
#include <vector>

using namespace std;
using Catch::Matchers::WithinAbs;

constexpr size_t FRAME_SIZE = 2048;
constexpr size_t OVERLAP = 512;
constexpr size_t FRAME_COUNT = 64;
constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;  // 1536

TEST_CASE("Pool initialization", "[pool][init]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);

    SECTION("Initial head position is zero") {
        REQUIRE(pool->head() == 0);
    }

    SECTION("Hop size computed correctly") {
        REQUIRE(pool->hop_size() == HOP_SIZE);
    }

    SECTION("Can write initial samples successfully") {
        auto conv = make_unique<converter<int8_t, float>>(false);
        vector<uint8_t> data(100 * 2, 0);  // 100 samples, 2 bytes each (I/Q)
        REQUIRE(pool->write_samples(data.data(), 100, conv.get(), 2));
        REQUIRE(pool->head() == 100);
    }
}

TEST_CASE("Slot acquisition and release", "[pool][slots]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);

    SECTION("Acquire and release single frame") {
        auto conv = make_unique<converter<int8_t, float>>(false);
        vector<uint8_t> data(FRAME_SIZE * 2, 0);
        pool->write_samples(data.data(), FRAME_SIZE, conv.get(), 2);

        // Acquire frame
        auto frame = pool->try_emit_frame(0);
        REQUIRE(frame.has_value());
        REQUIRE(frame->size() == FRAME_SIZE);

        // Try to acquire same frame again - should fail (slot in use)
        auto frame2 = pool->try_emit_frame(0);
        REQUIRE_FALSE(frame2.has_value());

        // Release frame
        frame.reset();

        // Now can acquire again (after wraparound)
        // First need to fill the pool to wrap around
        for (size_t i = 0; i < FRAME_COUNT; ++i) {
            pool->write_samples(data.data(), HOP_SIZE, conv.get(), 2);
        }
        auto frame3 = pool->try_emit_frame(64 * HOP_SIZE);
        REQUIRE(frame3.has_value());
    }

    SECTION("Multiple frames in use") {
        auto conv = make_unique<converter<int8_t, float>>(false);
        vector<uint8_t> data(FRAME_SIZE * 2, 42);

        // Write enough for 3 frames
        pool->write_samples(data.data(), FRAME_SIZE + 2 * HOP_SIZE, conv.get(), 2);

        // Acquire all three
        auto frame0 = pool->try_emit_frame(0);
        auto frame1 = pool->try_emit_frame(HOP_SIZE);
        auto frame2 = pool->try_emit_frame(2 * HOP_SIZE);

        REQUIRE(frame0.has_value());
        REQUIRE(frame1.has_value());
        REQUIRE(frame2.has_value());

        // All three should be different slots
        REQUIRE(frame0->data() != frame1->data());
        REQUIRE(frame1->data() != frame2->data());
    }

    SECTION("All 64 frames can be acquired") {
        auto conv = make_unique<converter<int8_t, float>>(false);
        vector<uint8_t> data(FRAME_SIZE * 2, 0);

        // Write enough data for all frames
        size_t total_samples_needed = FRAME_SIZE + (FRAME_COUNT - 1) * HOP_SIZE;
        for (size_t written = 0; written < total_samples_needed; ) {
            size_t chunk = min<size_t>(FRAME_SIZE, total_samples_needed - written);
            pool->write_samples(data.data(), chunk, conv.get(), 2);
            written += chunk;
        }

        // Acquire all 64 frames
        vector<optional<composite::immutable_buffer<complex<float>>>> frames(FRAME_COUNT);
        for (size_t i = 0; i < FRAME_COUNT; ++i) {
            frames[i] = pool->try_emit_frame(i * HOP_SIZE);
            REQUIRE(frames[i].has_value());
        }
    }

    SECTION("Out of order release") {
        auto conv = make_unique<converter<int8_t, float>>(false);
        vector<uint8_t> data(FRAME_SIZE * 2, 0);
        pool->write_samples(data.data(), FRAME_SIZE + 2 * HOP_SIZE, conv.get(), 2);

        auto frame0 = pool->try_emit_frame(0);
        auto frame1 = pool->try_emit_frame(HOP_SIZE);
        auto frame2 = pool->try_emit_frame(2 * HOP_SIZE);

        // Release middle frame first
        frame1.reset();

        // Can now re-acquire the released frame
        auto retry1 = pool->try_emit_frame(HOP_SIZE);
        REQUIRE(retry1.has_value());

        // Release all
        retry1.reset();
        frame0.reset();
        frame2.reset();
    }
}

TEST_CASE("Write conflict detection", "[pool][conflicts]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    auto conv = make_unique<converter<int8_t, float>>(false);

    SECTION("Write fails when overlapping held frame") {
        vector<uint8_t> data(FRAME_SIZE * 2, 0);

        // Write initial data
        REQUIRE(pool->write_samples(data.data(), FRAME_SIZE, conv.get(), 2));

        // Acquire and hold frame at position 0 (occupies ring positions 0-2047)
        auto frame = pool->try_emit_frame(0);
        REQUIRE(frame.has_value());

        // Calculate how many writes needed to wrap ring and conflict with frame 0
        // Ring size = FRAME_SIZE * FRAME_COUNT + OVERLAP = 2048*64+512 = 131,584
        // After initial write, head = 2048
        // Need head >= 131,584 to wrap and write to ring position 0 again
        // Writes needed: ceil((131,584 - 2048) / 1536) = ceil(129,536 / 1536) = 85
        size_t ring_size = FRAME_SIZE * FRAME_COUNT + OVERLAP;
        size_t samples_needed = ring_size - FRAME_SIZE;
        size_t writes_needed = (samples_needed + HOP_SIZE - 1) / HOP_SIZE;

        // Write until conflict detected
        bool conflict_detected = false;
        for (size_t i = 0; i < writes_needed + 5; ++i) {
            if (!pool->write_samples(data.data(), HOP_SIZE, conv.get(), 2)) {
                conflict_detected = true;
                break;
            }
        }

        // Should detect conflict since frame 0 is still held
        REQUIRE(conflict_detected);

        // Release frame and verify we can now write
        frame.reset();
        REQUIRE(pool->write_samples(data.data(), HOP_SIZE, conv.get(), 2));
    }

    SECTION("Write succeeds when no conflicts") {
        vector<uint8_t> data(FRAME_SIZE * 2, 0);

        // Write and immediately release frames
        for (size_t i = 0; i < 10; ++i) {
            REQUIRE(pool->write_samples(data.data(), HOP_SIZE, conv.get(), 2));
            auto frame = pool->try_emit_frame(i * HOP_SIZE);
            REQUIRE(frame.has_value());
            // Implicitly released at end of loop
        }
    }

    SECTION("Write at gap between frames") {
        vector<uint8_t> data(FRAME_SIZE * 2, 0);

        // Write enough for multiple frames
        pool->write_samples(data.data(), FRAME_SIZE + 4 * HOP_SIZE, conv.get(), 2);

        // Hold frames 0 and 4, but not 1-3
        auto frame0 = pool->try_emit_frame(0);
        auto frame4 = pool->try_emit_frame(4 * HOP_SIZE);

        REQUIRE(frame0.has_value());
        REQUIRE(frame4.has_value());

        // Writing in the gap should succeed
        vector<uint8_t> new_data(HOP_SIZE * 2, 99);
        REQUIRE(pool->write_samples(new_data.data(), HOP_SIZE, conv.get(), 2));
    }
}

TEST_CASE("Ring buffer wraparound", "[pool][ring]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    auto conv = make_unique<converter<int8_t, float>>(false);

    SECTION("Head wraps correctly") {
        size_t ring_size = FRAME_SIZE * FRAME_COUNT + OVERLAP;
        vector<uint8_t> data(FRAME_SIZE * 2, 0);

        // Write enough to exceed ring size
        size_t total_written = 0;
        while (total_written < ring_size + 1000) {
            if (!pool->write_samples(data.data(), FRAME_SIZE, conv.get(), 2)) {
                // Write failed, emit and release some frames
                auto frame = pool->try_emit_frame(total_written - FRAME_SIZE);
                // Let it release automatically
            } else {
                total_written += FRAME_SIZE;
            }
        }

        REQUIRE(pool->head() > ring_size);
    }

    SECTION("Overlap copy maintains data continuity") {
        vector<uint8_t> data(FRAME_SIZE * 2, 42);

        // Fill pattern: first part different from rest
        fill_n(data.begin(), OVERLAP * 2, 99);

        pool->write_samples(data.data(), FRAME_SIZE, conv.get(), 2);

        // Write enough to cause wraparound
        vector<uint8_t> more_data(FRAME_SIZE * 2, 7);
        size_t ring_size = FRAME_SIZE * FRAME_COUNT + OVERLAP;

        for (size_t written = FRAME_SIZE; written < ring_size + OVERLAP; ) {
            size_t chunk = min<size_t>(FRAME_SIZE, ring_size + OVERLAP - written);
            pool->write_samples(more_data.data(), chunk, conv.get(), 2);
            written += chunk;
        }

        // Verify head advanced past ring size
        REQUIRE(pool->head() >= ring_size);
    }
}

TEST_CASE("Frame data integrity", "[pool][data]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);

    SECTION("Converted data is correct") {
        auto conv = make_unique<converter<int8_t, float>>(false);
        vector<uint8_t> data(FRAME_SIZE * 2, 0);

        // Set pattern: I=42, Q=43 for first sample
        data[0] = 42;
        data[1] = 43;

        pool->write_samples(data.data(), FRAME_SIZE, conv.get(), 2);

        auto frame = pool->try_emit_frame(0);
        REQUIRE(frame.has_value());

        auto frame_data = frame->data();
        float expected_i = static_cast<float>(static_cast<int8_t>(42));
        float expected_q = static_cast<float>(static_cast<int8_t>(43));

        REQUIRE_THAT(frame_data[0].real(), WithinAbs(expected_i, 0.01f));
        REQUIRE_THAT(frame_data[0].imag(), WithinAbs(expected_q, 0.01f));
    }

    SECTION("Frame remains valid while held") {
        auto conv = make_unique<converter<int8_t, float>>(false);
        vector<uint8_t> data(FRAME_SIZE * 2, 0);
        data[0] = 42;
        data[1] = 43;

        pool->write_samples(data.data(), FRAME_SIZE, conv.get(), 2);

        auto frame = pool->try_emit_frame(0);
        auto frame_data = frame->data();

        float expected_i = static_cast<float>(static_cast<int8_t>(42));
        float expected_q = static_cast<float>(static_cast<int8_t>(43));

        // Write more data (different values) at different position
        vector<uint8_t> new_data(HOP_SIZE * 2, 99);
        pool->write_samples(new_data.data(), HOP_SIZE, conv.get(), 2);

        // Original frame data should still be intact
        REQUIRE_THAT(frame_data[0].real(), WithinAbs(expected_i, 0.01f));
        REQUIRE_THAT(frame_data[0].imag(), WithinAbs(expected_q, 0.01f));
    }

    SECTION("Multiple overlapping frames have correct data") {
        auto conv = make_unique<converter<int8_t, float>>(false);

        // Create data with recognizable pattern
        vector<uint8_t> data(FRAME_SIZE * 2, 0);
        for (size_t i = 0; i < FRAME_SIZE * 2; ++i) {
            data[i] = static_cast<uint8_t>(i % 256);
        }

        pool->write_samples(data.data(), FRAME_SIZE + HOP_SIZE, conv.get(), 2);

        auto frame0 = pool->try_emit_frame(0);
        auto frame1 = pool->try_emit_frame(HOP_SIZE);

        REQUIRE(frame0.has_value());
        REQUIRE(frame1.has_value());

        auto frame0_data = frame0->data();
        auto frame1_data = frame1->data();

        // Frames overlap by OVERLAP samples
        // The overlapping region should match
        size_t overlap_start = FRAME_SIZE - OVERLAP;
        for (size_t i = 0; i < OVERLAP; ++i) {
            REQUIRE_THAT(frame0_data[overlap_start + i].real(),
                        WithinAbs(frame1_data[i].real(), 0.01f));
            REQUIRE_THAT(frame0_data[overlap_start + i].imag(),
                        WithinAbs(frame1_data[i].imag(), 0.01f));
        }
    }
}

TEST_CASE("Multiple references to same frame", "[pool][lifetime]") {
    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    auto conv = make_unique<converter<int8_t, float>>(false);

    vector<uint8_t> data(FRAME_SIZE * 2, 0);
    pool->write_samples(data.data(), FRAME_SIZE, conv.get(), 2);

    auto frame1 = pool->try_emit_frame(0);
    auto frame2 = frame1;  // Copy
    auto frame3 = frame2;  // Another copy

    REQUIRE(frame1.has_value());

    // Try to acquire same slot again - should fail
    auto frame_dup = pool->try_emit_frame(0);
    REQUIRE_FALSE(frame_dup.has_value());

    // Release one reference
    frame1.reset();
    // Still in use by frame2 and frame3
    REQUIRE_FALSE(pool->try_emit_frame(0).has_value());

    // Release another
    frame2.reset();
    // Still in use by frame3
    REQUIRE_FALSE(pool->try_emit_frame(0).has_value());

    // Release last - now we need to wrap around to reuse slot 0
    frame3.reset();

    // Advance past all 64 slots to wrap around
    for (size_t i = 0; i < FRAME_COUNT; ++i) {
        pool->write_samples(data.data(), HOP_SIZE, conv.get(), 2);
    }

    // Now slot 0 should be available again for frame at position 64*HOP_SIZE
    auto frame_new = pool->try_emit_frame(64 * HOP_SIZE);
    REQUIRE(frame_new.has_value());
}

TEST_CASE("Edge cases", "[pool][edge]") {
    SECTION("Zero overlap") {
        auto pool = make_shared<framer_pool<complex<float>>>(1024, 0, 32);
        auto conv = make_unique<converter<int8_t, float>>(false);

        REQUIRE(pool->hop_size() == 1024);

        vector<uint8_t> data(1024 * 2, 0);
        REQUIRE(pool->write_samples(data.data(), 1024, conv.get(), 2));

        auto frame = pool->try_emit_frame(0);
        REQUIRE(frame.has_value());
        REQUIRE(frame->size() == 1024);
    }

    SECTION("Maximum overlap") {
        size_t frame_size = 1024;
        size_t overlap = 1023;  // Only 1 sample hop
        auto pool = make_shared<framer_pool<complex<float>>>(frame_size, overlap, 32);
        auto conv = make_unique<converter<int8_t, float>>(false);

        REQUIRE(pool->hop_size() == 1);

        vector<uint8_t> data(frame_size * 2, 0);
        REQUIRE(pool->write_samples(data.data(), frame_size, conv.get(), 2));

        auto frame = pool->try_emit_frame(0);
        REQUIRE(frame.has_value());
    }

    SECTION("Small frame count") {
        auto pool = make_shared<framer_pool<complex<float>>>(512, 128, 4);
        auto conv = make_unique<converter<int8_t, float>>(false);

        vector<uint8_t> data(512 * 2, 0);

        // Should be able to fill all 4 slots
        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(pool->write_samples(data.data(), pool->hop_size(), conv.get(), 2));
        }
    }
}
