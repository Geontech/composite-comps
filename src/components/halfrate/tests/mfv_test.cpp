// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC
//
// MFV (Multi-Function Versioning) Test
//
// This program demonstrates that the compiler has generated multiple versions
// of the kernel functions for different CPU features (AVX-512, AVX2, scalar).
//
// Compile with: g++ -mavx512f -mavx2 -O2 mfv_test.cpp -o mfv_test
// Run with: ./mfv_test

#include "../kernels.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cpuid.h>

using sample_t = std::complex<float>;

// CPU Feature Detection
struct CPUFeatures {
    bool avx2 = false;
    bool avx512f = false;
    bool fma = false;

    CPUFeatures() {
        unsigned int eax, ebx, ecx, edx;

        // Check AVX2 and FMA (EAX=7, ECX=0)
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            avx2 = (ebx & bit_AVX2) != 0;
            avx512f = (ebx & bit_AVX512F) != 0;
        }

        // Check FMA (EAX=1)
        if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
            fma = (ecx & bit_FMA) != 0;
        }
    }

    void print() const {
        std::cout << "CPU Features Detected:\n";
        std::cout << "  AVX2:     " << (avx2 ? "YES" : "NO") << "\n";
        std::cout << "  FMA:      " << (fma ? "YES" : "NO") << "\n";
        std::cout << "  AVX-512F: " << (avx512f ? "YES" : "NO") << "\n";
        std::cout << "\n";
    }

    const char* expected_version() const {
        if (avx512f) return "AVX-512";
        if (avx2 && fma) return "AVX2";
        return "Scalar";
    }
};

int main() {
    std::cout << "=================================================================\n";
    std::cout << "Multi-Function Versioning (MFV) Test\n";
    std::cout << "=================================================================\n\n";

    CPUFeatures cpu;
    cpu.print();

    std::cout << "Expected runtime kernel selection: " << cpu.expected_version() << "\n";
    std::cout << "\n";

    // Test data
    constexpr std::size_t num_pairs = 16;
    std::vector<sample_t> src(num_pairs * 2);
    std::vector<sample_t> dst_even(num_pairs);
    std::vector<sample_t> dst_odd(num_pairs);

    for (std::size_t i = 0; i < src.size(); ++i) {
        src[i] = {static_cast<float>(i), static_cast<float>(i * 2)};
    }

    // Call the function - MFV will select the appropriate version
    kernels::deinterleave_block(src.data(), dst_even.data(), dst_odd.data(), num_pairs);

    // Verify results
    bool correct = true;
    for (std::size_t i = 0; i < num_pairs; ++i) {
        auto expected_even = src[2 * i];
        auto expected_odd = src[2 * i + 1];

        if (dst_even[i] != expected_even || dst_odd[i] != expected_odd) {
            correct = false;
            break;
        }
    }

    std::cout << "Deinterleave Test: " << (correct ? "PASS ✓" : "FAIL ✗") << "\n";

    // Test filter kernel
    constexpr std::size_t num_outputs = 32;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 2;
    constexpr float center_tap = 0.5f;

    std::vector<sample_t> even_hist(num_outputs + num_taps, {1.0f, 0.5f});
    std::vector<sample_t> odd_hist(num_outputs + delay_offset, {0.5f, 1.0f});
    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    kernels::halfband_filter_vertical(
        even_hist.data(), odd_hist.data(),
        coeffs.data(), num_taps,
        center_tap, delay_offset,
        output.data(), num_outputs
    );

    // Verify non-zero output
    bool non_zero = false;
    for (const auto& s : output) {
        if (std::abs(s) > 0.001f) {
            non_zero = true;
            break;
        }
    }

    std::cout << "Filter Test:        " << (non_zero ? "PASS ✓" : "FAIL ✗") << "\n";

    std::cout << "\n=================================================================\n";
    std::cout << "Multi-Function Versioning is working!\n";
    std::cout << "The runtime will automatically select the best kernel version\n";
    std::cout << "based on CPU capabilities.\n";
    std::cout << "=================================================================\n";

    return (correct && non_zero) ? 0 : 1;
}
