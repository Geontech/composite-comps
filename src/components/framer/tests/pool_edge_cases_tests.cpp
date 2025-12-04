/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Framer Pool Edge Case Tests - Boundary conditions and error handling
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
// T7: Edge Cases
// ============================================================================

TEST_CASE("T7.1: Zero overlap", "[pool][edge]") {
    constexpr size_t FRAME_SIZE = 1024;
    constexpr size_t OVERLAP = 0;  // No overlap
    constexpr size_t FRAME_COUNT = 8;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    REQUIRE(pool->frame_size() == 1024);  // May be rounded for alignment
    REQUIRE(pool->hop_size() == pool->frame_size());  // hop_size == frame_size when overlap = 0

    // Write 3 frames worth of data
    vector<uint8_t> data(FRAME_SIZE * 2, 42);
    for (size_t i = 0; i < 3; ++i) {
        REQUIRE(pool->write_samples(data.data(), FRAME_SIZE, &conv, 2, true));
    }

    // Emit frames (no overlap, so consecutive frames are completely independent)
    auto frame0 = pool->try_emit_frame(0);
    auto frame1 = pool->try_emit_frame(FRAME_SIZE);
    auto frame2 = pool->try_emit_frame(2 * FRAME_SIZE);

    REQUIRE(frame0.has_value());
    REQUIRE(frame1.has_value());
    REQUIRE(frame2.has_value());
}

TEST_CASE("T7.2: Maximum overlap (frame_size - hop_size = 1)", "[pool][edge]") {
    // Note: Due to alignment, actual overlap may be adjusted
    constexpr size_t FRAME_SIZE = 64;   // Will be aligned to 64 (already aligned)
    constexpr size_t OVERLAP = 56;      // Will be rounded down to 56 (7*8)
    constexpr size_t FRAME_COUNT = 4;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Verify alignment adjustments
    INFO("frame_size after alignment: " << pool->frame_size());
    INFO("overlap after alignment: " << pool->frame_size() - pool->hop_size());
    INFO("hop_size: " << pool->hop_size());

    REQUIRE(pool->hop_size() >= 8);  // Minimum hop after alignment

    // Write data
    vector<uint8_t> data(FRAME_SIZE * 2, 0);
    for (size_t i = 0; i < FRAME_SIZE; ++i) {
        data[i*2] = static_cast<int8_t>(i);
        data[i*2+1] = static_cast<int8_t>(i);
    }
    REQUIRE(pool->write_samples(data.data(), FRAME_SIZE, &conv, 2, true));

    // Write hop_size more samples
    vector<uint8_t> hop_data(pool->hop_size() * 2, 99);
    REQUIRE(pool->write_samples(hop_data.data(), pool->hop_size(), &conv, 2, true));

    // Emit both frames
    auto frame0 = pool->try_emit_frame(0);
    auto frame1 = pool->try_emit_frame(pool->hop_size());

    REQUIRE(frame0.has_value());
    REQUIRE(frame1.has_value());

    // Verify massive overlap
    auto data0 = frame0->data();
    auto data1 = frame1->data();

    size_t actual_overlap = pool->frame_size() - pool->hop_size();
    for (size_t i = 0; i < actual_overlap; ++i) {
        REQUIRE_THAT(
            data1[i].real(),
            WithinAbs(data0[pool->hop_size() + i].real(), 0.01f)
        );
    }
}

TEST_CASE("T7.3: Small frame count (2)", "[pool][edge]") {
    constexpr size_t FRAME_SIZE = 512;
    constexpr size_t OVERLAP = 128;
    constexpr size_t FRAME_COUNT = 2;  // Minimum useful count
    constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    size_t ring_size = FRAME_COUNT * HOP_SIZE + OVERLAP;
    vector<uint8_t> data(ring_size * 2, 42);

    REQUIRE(pool->write_samples(data.data(), ring_size, &conv, 2, true));

    // Should be able to emit both frames
    auto frame0 = pool->try_emit_frame(0);
    auto frame1 = pool->try_emit_frame(HOP_SIZE);

    REQUIRE(frame0.has_value());
    REQUIRE(frame1.has_value());

    // Both slots busy, next write should wrap and potentially conflict
    vector<uint8_t> more(HOP_SIZE * 2, 1);
    bool write_ok = pool->write_samples(more.data(), HOP_SIZE, &conv, 2, true);

    // Should conflict since both slots held
    REQUIRE_FALSE(write_ok);
}

TEST_CASE("T7.4: Large frame count (128)", "[pool][edge]") {
    constexpr size_t FRAME_SIZE = 128;
    constexpr size_t OVERLAP = 32;
    constexpr size_t FRAME_COUNT = 128;  // Large count
    constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Fill entire ring
    size_t ring_size = FRAME_COUNT * HOP_SIZE + OVERLAP;
    INFO("Ring size: " << ring_size << " samples");

    vector<uint8_t> chunk(HOP_SIZE * 2, 0);
    for (size_t written = 0; written < ring_size; written += HOP_SIZE) {
        size_t count = min(HOP_SIZE, ring_size - written);
        REQUIRE(pool->write_samples(chunk.data(), count, &conv, 2, true));
    }

    // Should be able to acquire many frames
    vector<optional<composite::immutable_buffer<complex<float>>>> frames;
    for (size_t i = 0; i < min(size_t(64), FRAME_COUNT); ++i) {
        auto frame = pool->try_emit_frame(i * HOP_SIZE);
        if (frame.has_value()) {
            frames.push_back(move(frame));
        }
    }

    REQUIRE(frames.size() > 32);  // Should get a decent number
}

TEST_CASE("T7.5: Misaligned absolute_start", "[pool][edge][critical]") {
    // THIS TEST MAY FAIL - exposes bug #3 (no frame alignment validation)
    constexpr size_t FRAME_SIZE = 128;
    constexpr size_t OVERLAP = 32;
    constexpr size_t FRAME_COUNT = 8;
    constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Fill ring
    size_t ring_size = FRAME_COUNT * HOP_SIZE + OVERLAP;
    vector<uint8_t> data(ring_size * 2, 0);
    REQUIRE(pool->write_samples(data.data(), ring_size, &conv, 2, true));

    // Try to emit frame at misaligned position (not a multiple of hop_size)
    size_t misaligned_start = HOP_SIZE + 5;  // Off by 5 samples

    // FIXED: Now throws exception for misaligned absolute_start
    REQUIRE_THROWS_AS(
        pool->try_emit_frame(misaligned_start),
        std::logic_error
    );
}

TEST_CASE("T7.6: Frame request beyond head", "[pool][edge]") {
    constexpr size_t FRAME_SIZE = 256;
    constexpr size_t OVERLAP = 64;
    constexpr size_t FRAME_COUNT = 8;
    constexpr size_t HOP_SIZE = FRAME_SIZE - OVERLAP;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Write only 2 frames worth
    vector<uint8_t> data(FRAME_SIZE * 2, 0);
    REQUIRE(pool->write_samples(data.data(), FRAME_SIZE, &conv, 2, true));
    REQUIRE(pool->write_samples(data.data(), HOP_SIZE, &conv, 2, true));

    size_t head = pool->head();
    INFO("Current head: " << head);

    // Try to emit frame beyond head (must be aligned to hop_size)
    size_t beyond_head = ((head + FRAME_SIZE) / HOP_SIZE) * HOP_SIZE;
    INFO("Requesting frame at: " << beyond_head << " (aligned)");
    auto frame = pool->try_emit_frame(beyond_head);

    // Should fail gracefully
    REQUIRE_FALSE(frame.has_value());
}

TEST_CASE("T7.7: Write with zero samples", "[pool][edge]") {
    constexpr size_t FRAME_SIZE = 1024;
    constexpr size_t OVERLAP = 256;
    constexpr size_t FRAME_COUNT = 16;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    vector<uint8_t> data(100, 0);

    // Write zero samples - should succeed but do nothing
    bool result = pool->write_samples(data.data(), 0, &conv, 2, true);
    REQUIRE(result);
    REQUIRE(pool->head() == 0);
}

TEST_CASE("T7.8: Null converter pointer", "[pool][edge]") {
    constexpr size_t FRAME_SIZE = 1024;
    constexpr size_t OVERLAP = 256;
    constexpr size_t FRAME_COUNT = 16;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);

    vector<uint8_t> data(FRAME_SIZE * 2, 0);

    // Pass null converter - should fail gracefully
    bool result = pool->write_samples(data.data(), FRAME_SIZE, nullptr, 2, true);
    REQUIRE_FALSE(result);
    REQUIRE(pool->head() == 0);
}

TEST_CASE("T7.9: Real i8 input creates complex output with Q=0", "[pool][edge][real]") {
    constexpr size_t FRAME_SIZE = 64;
    constexpr size_t OVERLAP = 0;
    constexpr size_t FRAME_COUNT = 4;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Real input: FRAME_SIZE samples with pattern [1, 2, 3, 4, 5, ...]
    vector<uint8_t> real_data(FRAME_SIZE);
    for (size_t i = 0; i < FRAME_SIZE; ++i) {
        real_data[i] = static_cast<uint8_t>((i % 10) + 1);  // Pattern 1-10 repeating
    }

    // Write as REAL input (is_complex=false, stride=1)
    bool result = pool->write_samples(real_data.data(), FRAME_SIZE, &conv, 1, false);
    REQUIRE(result);
    REQUIRE(pool->head() == FRAME_SIZE);

    // Emit frame
    auto frame = pool->try_emit_frame(0);
    REQUIRE(frame.has_value());

    auto data = frame->data();

    // Verify real components are correct and imaginary components are zero
    for (size_t i = 0; i < FRAME_SIZE; ++i) {
        INFO("Checking sample " << i);
        float expected_real = static_cast<float>((i % 10) + 1);
        REQUIRE_THAT(data[i].real(), WithinAbs(expected_real, 0.01f));
        REQUIRE_THAT(data[i].imag(), WithinAbs(0.0f, 0.01f));  // Q=0 for real input
    }
}

TEST_CASE("T7.10: Complex i8 input preserves both I and Q", "[pool][edge][complex]") {
    constexpr size_t FRAME_SIZE = 64;
    constexpr size_t OVERLAP = 0;
    constexpr size_t FRAME_COUNT = 4;

    auto pool = make_shared<framer_pool<complex<float>>>(FRAME_SIZE, OVERLAP, FRAME_COUNT);
    converter_variant<float> conv = converter<int8_t, float>(false);

    // Complex input: FRAME_SIZE samples with pattern [(1+2i), (3+4i), (5+6i), ...]
    vector<uint8_t> complex_data(FRAME_SIZE * 2);  // 2 bytes per complex sample
    for (size_t i = 0; i < FRAME_SIZE; ++i) {
        complex_data[i*2] = static_cast<uint8_t>((i % 10) * 2 + 1);  // I component
        complex_data[i*2 + 1] = static_cast<uint8_t>((i % 10) * 2 + 2);  // Q component
    }

    // Write as COMPLEX input (is_complex=true, stride=2)
    bool result = pool->write_samples(complex_data.data(), FRAME_SIZE, &conv, 2, true);
    REQUIRE(result);
    REQUIRE(pool->head() == FRAME_SIZE);

    // Emit frame
    auto frame = pool->try_emit_frame(0);
    REQUIRE(frame.has_value());

    auto data = frame->data();

    // Verify both I and Q components are correct
    for (size_t i = 0; i < FRAME_SIZE; ++i) {
        INFO("Checking sample " << i);
        float expected_i = static_cast<float>((i % 10) * 2 + 1);
        float expected_q = static_cast<float>((i % 10) * 2 + 2);
        REQUIRE_THAT(data[i].real(), WithinAbs(expected_i, 0.01f));
        REQUIRE_THAT(data[i].imag(), WithinAbs(expected_q, 0.01f));
    }
}

// T7.11: Alignment enforcement - REMOVED
// This test was testing alignment constraints that were removed when we switched
// to unaligned stores in the converters. Frame sizes and overlaps are now used
// exactly as specified by the user, with no rounding.
