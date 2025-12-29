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
// These defines are set by CMake to indicate which ISA version is being tested
#ifndef KERNEL_TEST_ISA
    #define KERNEL_TEST_ISA "unknown"
#endif

// Helper to print ISA info (called from within Catch2 context to avoid polluting test discovery)
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
// Reference Scalar Implementations
// =============================================================================

// Reference scalar deinterleave
auto reference_deinterleave(
    const sample_t* src,
    sample_t* dst_even,
    sample_t* dst_odd,
    std::size_t num_pairs
) -> void {
    for (std::size_t i = 0; i < num_pairs; ++i) {
        dst_even[i] = src[2 * i];
        dst_odd[i] = src[2 * i + 1];
    }
}

// Reference scalar halfband filter
auto reference_halfband_filter(
    const sample_t* even_hist,
    const sample_t* odd_hist,
    const float* coeffs,
    std::size_t num_taps,
    float center_tap,
    std::size_t delay_offset,
    sample_t* output,
    std::size_t num_outputs
) -> void {
    for (std::size_t i = 0; i < num_outputs; ++i) {
        auto sum = odd_hist[i + delay_offset] * center_tap;
        for (std::size_t k = 0; k < num_taps; ++k) {
            sum += even_hist[i + k] * coeffs[k];
        }
        output[i] = sum;
    }
}

// =============================================================================
// Deinterleave Kernel Tests
// =============================================================================

TEST_CASE("deinterleave_block - basic operation", "[kernels][deinterleave]") {
    print_isa_info(); // Print ISA version info once

    constexpr std::size_t num_pairs = 8;

    // Create test input: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    std::vector<sample_t> input(num_pairs * 2);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = {static_cast<float>(i), static_cast<float>(i * 2)};
    }

    std::vector<sample_t> even_ref(num_pairs);
    std::vector<sample_t> odd_ref(num_pairs);
    std::vector<sample_t> even_simd(num_pairs);
    std::vector<sample_t> odd_simd(num_pairs);

    // Reference scalar implementation
    reference_deinterleave(input.data(), even_ref.data(), odd_ref.data(), num_pairs);

    // SIMD implementation
    kernels::deinterleave_block(input.data(), even_simd.data(), odd_simd.data(), num_pairs);

    // Verify even lane
    for (std::size_t i = 0; i < num_pairs; ++i) {
        CHECK_THAT(even_simd[i].real(), WithinAbs(even_ref[i].real(), 1e-6f));
        CHECK_THAT(even_simd[i].imag(), WithinAbs(even_ref[i].imag(), 1e-6f));
    }

    // Verify odd lane
    for (std::size_t i = 0; i < num_pairs; ++i) {
        CHECK_THAT(odd_simd[i].real(), WithinAbs(odd_ref[i].real(), 1e-6f));
        CHECK_THAT(odd_simd[i].imag(), WithinAbs(odd_ref[i].imag(), 1e-6f));
    }
}

TEST_CASE("deinterleave_block - aligned 16-sample boundary", "[kernels][deinterleave]") {
    constexpr std::size_t num_pairs = 16; // Exactly one AVX-512 iteration

    std::vector<sample_t> input(num_pairs * 2);
    for (std::size_t i = 0; i < input.size(); ++i) {
        input[i] = {static_cast<float>(i), static_cast<float>(-i)};
    }

    std::vector<sample_t> even_ref(num_pairs);
    std::vector<sample_t> odd_ref(num_pairs);
    std::vector<sample_t> even_simd(num_pairs);
    std::vector<sample_t> odd_simd(num_pairs);

    reference_deinterleave(input.data(), even_ref.data(), odd_ref.data(), num_pairs);
    kernels::deinterleave_block(input.data(), even_simd.data(), odd_simd.data(), num_pairs);

    for (std::size_t i = 0; i < num_pairs; ++i) {
        CHECK_THAT(even_simd[i].real(), WithinAbs(even_ref[i].real(), 1e-6f));
        CHECK_THAT(even_simd[i].imag(), WithinAbs(even_ref[i].imag(), 1e-6f));
        CHECK_THAT(odd_simd[i].real(), WithinAbs(odd_ref[i].real(), 1e-6f));
        CHECK_THAT(odd_simd[i].imag(), WithinAbs(odd_ref[i].imag(), 1e-6f));
    }
}

TEST_CASE("deinterleave_block - unaligned tail", "[kernels][deinterleave]") {
    constexpr std::size_t num_pairs = 21; // 16 SIMD + 5 scalar

    std::vector<sample_t> input(num_pairs * 2);
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (auto& s : input) {
        s = {dist(gen), dist(gen)};
    }

    std::vector<sample_t> even_ref(num_pairs);
    std::vector<sample_t> odd_ref(num_pairs);
    std::vector<sample_t> even_simd(num_pairs);
    std::vector<sample_t> odd_simd(num_pairs);

    reference_deinterleave(input.data(), even_ref.data(), odd_ref.data(), num_pairs);
    kernels::deinterleave_block(input.data(), even_simd.data(), odd_simd.data(), num_pairs);

    for (std::size_t i = 0; i < num_pairs; ++i) {
        CHECK_THAT(even_simd[i].real(), WithinAbs(even_ref[i].real(), 1e-5f));
        CHECK_THAT(even_simd[i].imag(), WithinAbs(even_ref[i].imag(), 1e-5f));
        CHECK_THAT(odd_simd[i].real(), WithinAbs(odd_ref[i].real(), 1e-5f));
        CHECK_THAT(odd_simd[i].imag(), WithinAbs(odd_ref[i].imag(), 1e-5f));
    }
}

TEST_CASE("deinterleave_block - large block", "[kernels][deinterleave]") {
    constexpr std::size_t num_pairs = 1024;

    std::vector<sample_t> input(num_pairs * 2);
    std::mt19937 gen(54321);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    for (auto& s : input) {
        s = {dist(gen), dist(gen)};
    }

    std::vector<sample_t> even_ref(num_pairs);
    std::vector<sample_t> odd_ref(num_pairs);
    std::vector<sample_t> even_simd(num_pairs);
    std::vector<sample_t> odd_simd(num_pairs);

    reference_deinterleave(input.data(), even_ref.data(), odd_ref.data(), num_pairs);
    kernels::deinterleave_block(input.data(), even_simd.data(), odd_simd.data(), num_pairs);

    for (std::size_t i = 0; i < num_pairs; ++i) {
        CHECK_THAT(even_simd[i].real(), WithinAbs(even_ref[i].real(), 1e-4f));
        CHECK_THAT(even_simd[i].imag(), WithinAbs(even_ref[i].imag(), 1e-4f));
        CHECK_THAT(odd_simd[i].real(), WithinAbs(odd_ref[i].real(), 1e-4f));
        CHECK_THAT(odd_simd[i].imag(), WithinAbs(odd_ref[i].imag(), 1e-4f));
    }
}

// =============================================================================
// Halfband Filter Kernel Tests
// =============================================================================

TEST_CASE("halfband_filter_vertical - basic operation", "[kernels][filter]") {
    constexpr std::size_t num_outputs = 8;
    constexpr std::size_t num_taps = 3;
    constexpr std::size_t delay_offset = 2;
    constexpr float center_tap = 1.0f;

    // Create test data
    std::vector<sample_t> even_hist(num_outputs + num_taps, {1.0f, 0.0f});
    std::vector<sample_t> odd_hist(num_outputs + delay_offset, {1.0f, 0.0f});
    std::vector<float> coeffs = {0.5f, 0.3f, 0.2f};

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_simd(num_outputs);

    // Reference scalar implementation
    reference_halfband_filter(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs
    );

    // SIMD implementation
    kernels::halfband_filter_vertical(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_simd.data(), num_outputs
    );

    // Verify results
    for (std::size_t i = 0; i < num_outputs; ++i) {
        CHECK_THAT(output_simd[i].real(), WithinAbs(output_ref[i].real(), 1e-5f));
        CHECK_THAT(output_simd[i].imag(), WithinAbs(output_ref[i].imag(), 1e-5f));
    }
}

TEST_CASE("halfband_filter_vertical - 32-sample boundary", "[kernels][filter]") {
    constexpr std::size_t num_outputs = 32; // Exactly one primary loop iteration
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 2;
    constexpr float center_tap = 1.0f;

    std::vector<sample_t> even_hist(num_outputs + num_taps);
    std::vector<sample_t> odd_hist(num_outputs + delay_offset);

    // Initialize with varying data
    for (std::size_t i = 0; i < even_hist.size(); ++i) {
        even_hist[i] = {static_cast<float>(i) * 0.1f, static_cast<float>(i) * 0.05f};
    }
    for (std::size_t i = 0; i < odd_hist.size(); ++i) {
        odd_hist[i] = {static_cast<float>(i) * 0.2f, static_cast<float>(i) * 0.1f};
    }

    std::vector<float> coeffs = {0.5f, 0.3f, 0.2f, 0.1f, 0.05f, 0.025f};

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_simd(num_outputs);

    reference_halfband_filter(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs
    );

    kernels::halfband_filter_vertical(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_simd.data(), num_outputs
    );

    for (std::size_t i = 0; i < num_outputs; ++i) {
        CHECK_THAT(output_simd[i].real(), WithinAbs(output_ref[i].real(), 1e-4f));
        CHECK_THAT(output_simd[i].imag(), WithinAbs(output_ref[i].imag(), 1e-4f));
    }
}

TEST_CASE("halfband_filter_vertical - 16-sample tail", "[kernels][filter]") {
    constexpr std::size_t num_outputs = 16; // Triggers tail loop
    constexpr std::size_t num_taps = 4;
    constexpr std::size_t delay_offset = 1;
    constexpr float center_tap = 0.5f;

    std::vector<sample_t> even_hist(num_outputs + num_taps);
    std::vector<sample_t> odd_hist(num_outputs + delay_offset);

    std::mt19937 gen(98765);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& s : even_hist) s = {dist(gen), dist(gen)};
    for (auto& s : odd_hist) s = {dist(gen), dist(gen)};

    std::vector<float> coeffs = {0.4f, 0.3f, 0.2f, 0.1f};

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_simd(num_outputs);

    reference_halfband_filter(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs
    );

    kernels::halfband_filter_vertical(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_simd.data(), num_outputs
    );

    for (std::size_t i = 0; i < num_outputs; ++i) {
        CHECK_THAT(output_simd[i].real(), WithinAbs(output_ref[i].real(), 1e-4f));
        CHECK_THAT(output_simd[i].imag(), WithinAbs(output_ref[i].imag(), 1e-4f));
    }
}

TEST_CASE("halfband_filter_vertical - unaligned size", "[kernels][filter]") {
    constexpr std::size_t num_outputs = 37; // 32 primary + 5 scalar tail
    constexpr std::size_t num_taps = 8;
    constexpr std::size_t delay_offset = 3;
    constexpr float center_tap = 1.0f;

    std::vector<sample_t> even_hist(num_outputs + num_taps);
    std::vector<sample_t> odd_hist(num_outputs + delay_offset);

    std::mt19937 gen(11111);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (auto& s : even_hist) s = {dist(gen), dist(gen)};
    for (auto& s : odd_hist) s = {dist(gen), dist(gen)};

    std::vector<float> coeffs(num_taps);
    for (auto& c : coeffs) c = dist(gen) / 100.0f;

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_simd(num_outputs);

    reference_halfband_filter(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs
    );

    kernels::halfband_filter_vertical(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_simd.data(), num_outputs
    );

    for (std::size_t i = 0; i < num_outputs; ++i) {
        CHECK_THAT(output_simd[i].real(), WithinAbs(output_ref[i].real(), 1e-3f));
        CHECK_THAT(output_simd[i].imag(), WithinAbs(output_ref[i].imag(), 1e-3f));
    }
}

TEST_CASE("halfband_filter_vertical - large block", "[kernels][filter]") {
    constexpr std::size_t num_outputs = 1024;
    constexpr std::size_t num_taps = 12;
    constexpr std::size_t delay_offset = 5;
    constexpr float center_tap = 1.0f;

    std::vector<sample_t> even_hist(num_outputs + num_taps);
    std::vector<sample_t> odd_hist(num_outputs + delay_offset);

    std::mt19937 gen(22222);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    for (auto& s : even_hist) s = {dist(gen), dist(gen)};
    for (auto& s : odd_hist) s = {dist(gen), dist(gen)};

    std::vector<float> coeffs(num_taps);
    for (auto& c : coeffs) c = dist(gen) / 1000.0f;

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_simd(num_outputs);

    reference_halfband_filter(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs
    );

    kernels::halfband_filter_vertical(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_simd.data(), num_outputs
    );

    for (std::size_t i = 0; i < num_outputs; ++i) {
        CHECK_THAT(output_simd[i].real(), WithinAbs(output_ref[i].real(), 1e-2f));
        CHECK_THAT(output_simd[i].imag(), WithinAbs(output_ref[i].imag(), 1e-2f));
    }
}

TEST_CASE("halfband_filter_vertical - zero coefficients", "[kernels][filter]") {
    constexpr std::size_t num_outputs = 16;
    constexpr std::size_t num_taps = 4;
    constexpr std::size_t delay_offset = 1;
    constexpr float center_tap = 2.0f;

    std::vector<sample_t> even_hist(num_outputs + num_taps, {1.0f, 1.0f});
    std::vector<sample_t> odd_hist(num_outputs + delay_offset, {3.0f, 2.0f});
    std::vector<float> coeffs(num_taps, 0.0f); // All zero

    std::vector<sample_t> output_ref(num_outputs);
    std::vector<sample_t> output_simd(num_outputs);

    reference_halfband_filter(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_ref.data(), num_outputs
    );

    kernels::halfband_filter_vertical(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output_simd.data(), num_outputs
    );

    // Should only have center tap contribution
    for (std::size_t i = 0; i < num_outputs; ++i) {
        CHECK_THAT(output_simd[i].real(), WithinAbs(output_ref[i].real(), 1e-6f));
        CHECK_THAT(output_simd[i].imag(), WithinAbs(output_ref[i].imag(), 1e-6f));
    }
}
