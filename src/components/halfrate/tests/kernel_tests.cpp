// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../kernels.hpp"

#include <complex>
#include <vector>
#include <cmath>
#include <random>

using sample_t = std::complex<float>;
using Catch::Matchers::WithinAbs;

// =============================================================================
// ISA Test Configuration
// =============================================================================
#ifndef KERNEL_TEST_ISA
    #define KERNEL_TEST_ISA "unknown"
#endif

inline auto print_isa_info() -> void {
    static bool printed = false;
    if (!printed) {
        printed = true;
        WARN("==> Testing ISA version: " << KERNEL_TEST_ISA);
#ifdef KERNEL_TEST_SCALAR
        WARN("    Expected: Scalar (x86-64 baseline)");
#elif defined(KERNEL_TEST_AVX2)
        WARN("    Expected: AVX2 + FMA");
#elif defined(KERNEL_TEST_AVX512)
        WARN("    Expected: AVX-512F/BW/VL/DQ");
#endif
    }
}

// =============================================================================
// Reference Scalar Implementation (KISS - for validation)
// =============================================================================
// This naive implementation is intentionally simple.
// Used as the ground truth for validating the optimized SIMD kernel.

auto reference_halfband_fused(
    const sample_t* input,
    const sample_t* even_hist,
    const sample_t* odd_hist,
    const float* coeffs,
    std::size_t num_taps,
    float center_tap,
    std::size_t delay_offset,
    sample_t* output,
    std::size_t num_outputs,
    std::size_t history_len
) -> void {
    // Step 1: Build working buffers with [history][new deinterleaved data]
    std::vector<sample_t> even_lane(history_len + num_outputs);
    std::vector<sample_t> odd_lane(history_len + num_outputs);

    // Copy history
    for (std::size_t i = 0; i < history_len; ++i) {
        even_lane[i] = even_hist[i];
        odd_lane[i] = odd_hist[i];
    }

    // Deinterleave: input[2*i] -> even, input[2*i+1] -> odd
    for (std::size_t i = 0; i < num_outputs; ++i) {
        even_lane[history_len + i] = input[2 * i];
        odd_lane[history_len + i] = input[2 * i + 1];
    }

    // Step 2: Apply halfband filter
    // output[i] = even[i + delay_offset] * center_tap + sum(odd[i+k] * coeffs[k])
    for (std::size_t i = 0; i < num_outputs; ++i) {
        sample_t sum = even_lane[i + delay_offset] * center_tap;
        for (std::size_t k = 0; k < num_taps; ++k) {
            sum += odd_lane[i + k] * coeffs[k];
        }
        output[i] = sum;
    }
}

// =============================================================================
// Fused Kernel Tests
// =============================================================================

TEST_CASE("halfband_filter_fused - basic correctness", "[kernels][fused]") {
    print_isa_info();

    constexpr std::size_t num_outputs = 64;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 3;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<sample_t> input(num_outputs * 2);
    for (auto& s : input) {
        s = {dist(gen), dist(gen)};
    }

    // Initialize history with zeros
    std::vector<sample_t> even_hist_ref(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist_ref(history_len, {0.0f, 0.0f});
    std::vector<sample_t> even_hist_fused(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist_fused(history_len, {0.0f, 0.0f});

    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_fused(num_outputs);

    // Reference scalar implementation
    reference_halfband_fused(
        input.data(),
        even_hist_ref.data(), odd_hist_ref.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs,
        history_len
    );

    // SIMD fused kernel
    kernels::halfband_filter_fused(
        input.data(),
        even_hist_fused.data(), odd_hist_fused.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_fused.data(), num_outputs,
        history_len
    );

    // Compare outputs
    for (std::size_t i = 0; i < num_outputs; ++i) {
        INFO("Output index: " << i);
        CHECK_THAT(output_fused[i].real(), WithinAbs(output_ref[i].real(), 1e-4f));
        CHECK_THAT(output_fused[i].imag(), WithinAbs(output_ref[i].imag(), 1e-4f));
    }
}

TEST_CASE("halfband_filter_fused - with non-zero history", "[kernels][fused]") {
    constexpr std::size_t num_outputs = 128;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 3;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<sample_t> input(num_outputs * 2);
    for (auto& s : input) {
        s = {dist(gen), dist(gen)};
    }

    // Initialize history with random data (simulates previous call)
    std::vector<sample_t> even_hist_ref(history_len);
    std::vector<sample_t> odd_hist_ref(history_len);
    for (auto& s : even_hist_ref) s = {dist(gen), dist(gen)};
    for (auto& s : odd_hist_ref) s = {dist(gen), dist(gen)};

    auto even_hist_fused = even_hist_ref;
    auto odd_hist_fused = odd_hist_ref;

    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_fused(num_outputs);

    reference_halfband_fused(
        input.data(),
        even_hist_ref.data(), odd_hist_ref.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs,
        history_len
    );

    kernels::halfband_filter_fused(
        input.data(),
        even_hist_fused.data(), odd_hist_fused.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_fused.data(), num_outputs,
        history_len
    );

    for (std::size_t i = 0; i < num_outputs; ++i) {
        INFO("Output index: " << i);
        CHECK_THAT(output_fused[i].real(), WithinAbs(output_ref[i].real(), 1e-4f));
        CHECK_THAT(output_fused[i].imag(), WithinAbs(output_ref[i].imag(), 1e-4f));
    }
}

TEST_CASE("halfband_filter_fused - tile boundary (512 samples)", "[kernels][fused]") {
    // Test exactly at tile size boundary
    constexpr std::size_t num_outputs = 512;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 3;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<sample_t> input(num_outputs * 2);
    for (auto& s : input) {
        s = {dist(gen), dist(gen)};
    }

    std::vector<sample_t> even_hist_ref(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist_ref(history_len, {0.0f, 0.0f});
    auto even_hist_fused = even_hist_ref;
    auto odd_hist_fused = odd_hist_ref;

    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_fused(num_outputs);

    reference_halfband_fused(
        input.data(),
        even_hist_ref.data(), odd_hist_ref.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs,
        history_len
    );

    kernels::halfband_filter_fused(
        input.data(),
        even_hist_fused.data(), odd_hist_fused.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_fused.data(), num_outputs,
        history_len
    );

    for (std::size_t i = 0; i < num_outputs; ++i) {
        INFO("Output index: " << i);
        CHECK_THAT(output_fused[i].real(), WithinAbs(output_ref[i].real(), 1e-4f));
        CHECK_THAT(output_fused[i].imag(), WithinAbs(output_ref[i].imag(), 1e-4f));
    }
}

TEST_CASE("halfband_filter_fused - multi-tile (1024 samples)", "[kernels][fused]") {
    // Test multiple tiles
    constexpr std::size_t num_outputs = 1024;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 3;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    std::mt19937 gen(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<sample_t> input(num_outputs * 2);
    for (auto& s : input) {
        s = {dist(gen), dist(gen)};
    }

    std::vector<sample_t> even_hist_ref(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist_ref(history_len, {0.0f, 0.0f});
    auto even_hist_fused = even_hist_ref;
    auto odd_hist_fused = odd_hist_ref;

    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_fused(num_outputs);

    reference_halfband_fused(
        input.data(),
        even_hist_ref.data(), odd_hist_ref.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs,
        history_len
    );

    kernels::halfband_filter_fused(
        input.data(),
        even_hist_fused.data(), odd_hist_fused.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_fused.data(), num_outputs,
        history_len
    );

    for (std::size_t i = 0; i < num_outputs; ++i) {
        INFO("Output index: " << i);
        CHECK_THAT(output_fused[i].real(), WithinAbs(output_ref[i].real(), 1e-4f));
        CHECK_THAT(output_fused[i].imag(), WithinAbs(output_ref[i].imag(), 1e-4f));
    }
}

TEST_CASE("halfband_filter_fused - large block (256K)", "[kernels][fused]") {
    constexpr std::size_t num_outputs = 262144;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 3;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    std::mt19937 gen(321);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<sample_t> input(num_outputs * 2);
    for (auto& s : input) {
        s = {dist(gen), dist(gen)};
    }

    std::vector<sample_t> even_hist_ref(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist_ref(history_len, {0.0f, 0.0f});
    auto even_hist_fused = even_hist_ref;
    auto odd_hist_fused = odd_hist_ref;

    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_fused(num_outputs);

    reference_halfband_fused(
        input.data(),
        even_hist_ref.data(), odd_hist_ref.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs,
        history_len
    );

    kernels::halfband_filter_fused(
        input.data(),
        even_hist_fused.data(), odd_hist_fused.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_fused.data(), num_outputs,
        history_len
    );

    // Check samples throughout the block
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < num_outputs; ++i) {
        float real_diff = std::abs(output_fused[i].real() - output_ref[i].real());
        float imag_diff = std::abs(output_fused[i].imag() - output_ref[i].imag());
        if (real_diff > 1e-4f || imag_diff > 1e-4f) {
            if (mismatches < 10) {
                INFO("Mismatch at index " << i);
            }
            mismatches++;
        }
    }

    INFO("Total mismatches: " << mismatches << " out of " << num_outputs);
    CHECK(mismatches == 0);
}

TEST_CASE("halfband_filter_fused - unaligned size", "[kernels][fused]") {
    // Test size that doesn't align to tile or SIMD boundaries
    constexpr std::size_t num_outputs = 537; // Odd number, not tile-aligned
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 3;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    std::mt19937 gen(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<sample_t> input(num_outputs * 2);
    for (auto& s : input) {
        s = {dist(gen), dist(gen)};
    }

    std::vector<sample_t> even_hist_ref(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist_ref(history_len, {0.0f, 0.0f});
    auto even_hist_fused = even_hist_ref;
    auto odd_hist_fused = odd_hist_ref;

    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_fused(num_outputs);

    reference_halfband_fused(
        input.data(),
        even_hist_ref.data(), odd_hist_ref.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs,
        history_len
    );

    kernels::halfband_filter_fused(
        input.data(),
        even_hist_fused.data(), odd_hist_fused.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_fused.data(), num_outputs,
        history_len
    );

    for (std::size_t i = 0; i < num_outputs; ++i) {
        INFO("Output index: " << i);
        CHECK_THAT(output_fused[i].real(), WithinAbs(output_ref[i].real(), 1e-4f));
        CHECK_THAT(output_fused[i].imag(), WithinAbs(output_ref[i].imag(), 1e-4f));
    }
}

TEST_CASE("halfband_filter_fused - different tap counts", "[kernels][fused]") {
    constexpr std::size_t num_outputs = 256;
    constexpr float center_tap = 0.5f;

    std::mt19937 gen(555);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<sample_t> input(num_outputs * 2);
    for (auto& s : input) {
        s = {dist(gen), dist(gen)};
    }

    SECTION("4 taps") {
        constexpr std::size_t num_taps = 4;
        constexpr std::size_t delay_offset = 2;
        constexpr std::size_t history_len = std::max(num_taps, delay_offset);

        std::vector<sample_t> even_hist_ref(history_len, {0.0f, 0.0f});
        std::vector<sample_t> odd_hist_ref(history_len, {0.0f, 0.0f});
        auto even_hist_fused = even_hist_ref;
        auto odd_hist_fused = odd_hist_ref;

        std::vector<float> coeffs = {0.1f, 0.2f, 0.2f, 0.1f};

        std::vector<sample_t> output_ref(num_outputs);
        std::vector<sample_t> output_fused(num_outputs);

        reference_halfband_fused(
            input.data(),
            even_hist_ref.data(), odd_hist_ref.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output_ref.data(), num_outputs,
            history_len
        );

        kernels::halfband_filter_fused(
            input.data(),
            even_hist_fused.data(), odd_hist_fused.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output_fused.data(), num_outputs,
            history_len
        );

        for (std::size_t i = 0; i < num_outputs; ++i) {
            CHECK_THAT(output_fused[i].real(), WithinAbs(output_ref[i].real(), 1e-4f));
            CHECK_THAT(output_fused[i].imag(), WithinAbs(output_ref[i].imag(), 1e-4f));
        }
    }

    SECTION("12 taps") {
        constexpr std::size_t num_taps = 12;
        constexpr std::size_t delay_offset = 6;
        constexpr std::size_t history_len = std::max(num_taps, delay_offset);

        std::vector<sample_t> even_hist_ref(history_len, {0.0f, 0.0f});
        std::vector<sample_t> odd_hist_ref(history_len, {0.0f, 0.0f});
        auto even_hist_fused = even_hist_ref;
        auto odd_hist_fused = odd_hist_ref;

        std::vector<float> coeffs = {0.05f, 0.08f, 0.1f, 0.15f, 0.2f, 0.25f,
                                     0.25f, 0.2f, 0.15f, 0.1f, 0.08f, 0.05f};

        std::vector<sample_t> output_ref(num_outputs);
        std::vector<sample_t> output_fused(num_outputs);

        reference_halfband_fused(
            input.data(),
            even_hist_ref.data(), odd_hist_ref.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output_ref.data(), num_outputs,
            history_len
        );

        kernels::halfband_filter_fused(
            input.data(),
            even_hist_fused.data(), odd_hist_fused.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output_fused.data(), num_outputs,
            history_len
        );

        for (std::size_t i = 0; i < num_outputs; ++i) {
            CHECK_THAT(output_fused[i].real(), WithinAbs(output_ref[i].real(), 1e-4f));
            CHECK_THAT(output_fused[i].imag(), WithinAbs(output_ref[i].imag(), 1e-4f));
        }
    }
}

TEST_CASE("halfband_filter_fused - impulse response", "[kernels][fused]") {
    // Verify filter behavior with known impulse input
    constexpr std::size_t num_outputs = 32;
    constexpr std::size_t num_taps = 4;
    constexpr std::size_t delay_offset = 2;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    // Input: impulse at position 0 (even sample)
    std::vector<sample_t> input(num_outputs * 2, {0.0f, 0.0f});
    input[0] = {1.0f, 0.0f}; // Even sample impulse

    std::vector<sample_t> even_hist(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist(history_len, {0.0f, 0.0f});

    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f};

    std::vector<sample_t> output(num_outputs);

    kernels::halfband_filter_fused(
        input.data(),
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output.data(), num_outputs,
        history_len
    );

    // Even impulse at input[0] goes to even_lane[history_len]
    // Filter output[i] = even_lane[i + delay_offset] * center_tap
    // For i where (i + delay_offset) == history_len, we get the impulse
    // That's i = history_len - delay_offset = 6 - 2 = 4 (with history_len=6)
    // But history_len = max(4,2) = 4, so i = 4 - 2 = 2
    std::size_t impulse_output_idx = history_len - delay_offset;

    // Center tap contribution should appear at the delayed position
    CHECK_THAT(output[impulse_output_idx].real(), WithinAbs(center_tap, 1e-5f));
    CHECK_THAT(output[impulse_output_idx].imag(), WithinAbs(0.0f, 1e-5f));
}

TEST_CASE("halfband_filter_fused - history update", "[kernels][fused]") {
    // Verify that history buffers are correctly updated after processing
    constexpr std::size_t num_outputs = 64;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 3;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    std::mt19937 gen(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<sample_t> input(num_outputs * 2);
    for (auto& s : input) {
        s = {dist(gen), dist(gen)};
    }

    std::vector<sample_t> even_hist(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist(history_len, {0.0f, 0.0f});

    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    kernels::halfband_filter_fused(
        input.data(),
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output.data(), num_outputs,
        history_len
    );

    // After processing, history should contain last history_len deinterleaved samples
    // even_hist[i] should be input[2 * (num_outputs - history_len + i)]
    // odd_hist[i] should be input[2 * (num_outputs - history_len + i) + 1]
    for (std::size_t i = 0; i < history_len; ++i) {
        std::size_t src_idx = num_outputs - history_len + i;
        sample_t expected_even = input[2 * src_idx];
        sample_t expected_odd = input[2 * src_idx + 1];

        CHECK_THAT(even_hist[i].real(), WithinAbs(expected_even.real(), 1e-6f));
        CHECK_THAT(even_hist[i].imag(), WithinAbs(expected_even.imag(), 1e-6f));
        CHECK_THAT(odd_hist[i].real(), WithinAbs(expected_odd.real(), 1e-6f));
        CHECK_THAT(odd_hist[i].imag(), WithinAbs(expected_odd.imag(), 1e-6f));
    }
}
