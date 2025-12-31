// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../coefficients.hpp"
#include "../kernels.hpp"

#include <complex>
#include <vector>
#include <cmath>
#include <random>

using cf32_t = std::complex<float>;
using Catch::Approx;
using Catch::Matchers::WithinAbs;

// =============================================================================
// INTERLEAVED KERNEL TESTS
// =============================================================================

TEST_CASE("prepare_interleaved_coeffs - layout verification", "[pfbc][kernels][interleaved]") {
    // Test coefficient preparation: [phase][tap] -> [tap][phase*2] (doubled)
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 3;

    // Create simple phase-major coefficients
    std::vector<std::vector<float>> phase_coeffs(M);
    for (std::size_t p = 0; p < M; ++p) {
        phase_coeffs[p].resize(K);
        for (std::size_t k = 0; k < K; ++k) {
            phase_coeffs[p][k] = static_cast<float>(p * 10 + k);  // e.g., phase0: 0,1,2; phase1: 10,11,12
        }
    }

    auto interleaved = pfbc::prepare_interleaved_coeffs(phase_coeffs, M, K);

    // Expected layout: [tap][phase*2]
    // tap0: [h0[0],h0[0], h1[0],h1[0], h2[0],h2[0], h3[0],h3[0]]
    // tap1: [h0[1],h0[1], h1[1],h1[1], h2[1],h2[1], h3[1],h3[1]]
    // ...
    REQUIRE(interleaved.size() == K * M * 2);

    for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t p = 0; p < M; ++p) {
            float expected = static_cast<float>(p * 10 + k);
            std::size_t idx = k * M * 2 + p * 2;
            INFO("Tap " << k << ", Phase " << p << ", Index " << idx);
            CHECK(interleaved[idx + 0] == Approx(expected));  // For real part
            CHECK(interleaved[idx + 1] == Approx(expected));  // For imag part
        }
    }
}

TEST_CASE("filter_interleaved_scalar - matches naive reference", "[pfbc][kernels][interleaved]") {
    // Test scalar implementation against manual for-loop computation
    constexpr std::size_t M = 4;
    constexpr std::size_t K = 3;
    constexpr std::size_t num_outputs = 5;

    // Create simple input (interleaved: [t0_p0, t0_p1, t0_p2, t0_p3, t1_p0, ...])
    std::vector<cf32_t> input((K + num_outputs - 1) * M);
    for (std::size_t t = 0; t < K + num_outputs - 1; ++t) {
        for (std::size_t p = 0; p < M; ++p) {
            input[t * M + p] = cf32_t{static_cast<float>(t), static_cast<float>(p)};
        }
    }

    // Create coefficients: phase p, tap k -> p + k*0.1
    std::vector<std::vector<float>> phase_coeffs(M);
    for (std::size_t p = 0; p < M; ++p) {
        phase_coeffs[p].resize(K);
        for (std::size_t k = 0; k < K; ++k) {
            phase_coeffs[p][k] = static_cast<float>(p) + static_cast<float>(k) * 0.1f;
        }
    }
    auto coeffs = pfbc::prepare_interleaved_coeffs(phase_coeffs, M, K);

    std::vector<cf32_t> output(num_outputs * M);

    kernels::filter_interleaved_scalar(
        input.data(), coeffs.data(), output.data(), M, num_outputs, K);

    // Verify against naive for-loop computation
    // y[t,p] = sum_{k=0}^{K-1} h[p,k] * x[t+k,p]
    for (std::size_t t = 0; t < num_outputs; ++t) {
        for (std::size_t p = 0; p < M; ++p) {
            cf32_t expected{0.0f, 0.0f};
            for (std::size_t k = 0; k < K; ++k) {
                cf32_t x = input[(t + k) * M + p];
                float h = phase_coeffs[p][k];
                expected += x * h;
            }
            INFO("Output time " << t << ", phase " << p);
            CHECK(output[t * M + p].real() == Approx(expected.real()).epsilon(0.001));
            CHECK(output[t * M + p].imag() == Approx(expected.imag()).epsilon(0.001));
        }
    }
}

TEST_CASE("filter_interleaved_64ch_avx512 - matches scalar reference", "[pfbc][kernels][interleaved]") {
    // Test AVX-512 64-channel implementation against scalar reference
    constexpr std::size_t M = 64;
    constexpr std::size_t K = 16;
    constexpr std::size_t num_outputs = 128;

    // Random input data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<cf32_t> input((K + num_outputs - 1) * M);
    for (auto& s : input) {
        s = cf32_t{dist(gen), dist(gen)};
    }

    // Random coefficients
    std::vector<std::vector<float>> phase_coeffs(M);
    for (std::size_t p = 0; p < M; ++p) {
        phase_coeffs[p].resize(K);
        for (auto& h : phase_coeffs[p]) {
            h = dist(gen);
        }
    }
    auto coeffs = pfbc::prepare_interleaved_coeffs(phase_coeffs, M, K);

    // Compute with scalar reference
    std::vector<cf32_t> expected(num_outputs * M);
    kernels::filter_interleaved_scalar(
        input.data(), coeffs.data(), expected.data(), M, num_outputs, K);

    // Compute with AVX-512
    std::vector<cf32_t> actual(num_outputs * M);
    kernels::filter_interleaved_64ch_avx512(
        input.data(), coeffs.data(), actual.data(), num_outputs, K);

    // Compare
    for (std::size_t i = 0; i < num_outputs * M; ++i) {
        INFO("Sample " << i << " (time=" << i/M << ", phase=" << i%M << ")");
        CHECK(actual[i].real() == Approx(expected[i].real()).epsilon(0.0001));
        CHECK(actual[i].imag() == Approx(expected[i].imag()).epsilon(0.0001));
    }
}

TEST_CASE("filter_interleaved_64ch_x4_avx512 - matches scalar reference", "[pfbc][kernels][interleaved]") {
    // Test the 4x unrolled AVX-512 implementation
    constexpr std::size_t M = 64;
    constexpr std::size_t K = 16;
    constexpr std::size_t num_outputs = 256;  // Multiple of 4 for main loop + some remainder

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<cf32_t> input((K + num_outputs - 1) * M);
    for (auto& s : input) {
        s = cf32_t{dist(gen), dist(gen)};
    }

    std::vector<std::vector<float>> phase_coeffs(M);
    for (std::size_t p = 0; p < M; ++p) {
        phase_coeffs[p].resize(K);
        for (auto& h : phase_coeffs[p]) {
            h = dist(gen);
        }
    }
    auto coeffs = pfbc::prepare_interleaved_coeffs(phase_coeffs, M, K);

    // Scalar reference
    std::vector<cf32_t> expected(num_outputs * M);
    kernels::filter_interleaved_scalar(
        input.data(), coeffs.data(), expected.data(), M, num_outputs, K);

    // AVX-512 4x unrolled
    std::vector<cf32_t> actual(num_outputs * M);
    kernels::filter_interleaved_64ch_x4_avx512(
        input.data(), coeffs.data(), actual.data(), num_outputs, K);

    // Compare
    for (std::size_t i = 0; i < num_outputs * M; ++i) {
        INFO("Sample " << i << " (time=" << i/M << ", phase=" << i%M << ")");
        CHECK(actual[i].real() == Approx(expected[i].real()).epsilon(0.0001));
        CHECK(actual[i].imag() == Approx(expected[i].imag()).epsilon(0.0001));
    }
}

TEST_CASE("filter_interleaved_64ch_x4 - tail handling", "[pfbc][kernels][interleaved]") {
    // Test with num_outputs that exercises tail path (not multiple of 4)
    constexpr std::size_t M = 64;
    constexpr std::size_t K = 8;
    constexpr std::size_t num_outputs = 13;  // 3 groups of 4 + 1 remainder

    std::mt19937 gen(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<cf32_t> input((K + num_outputs - 1) * M);
    for (auto& s : input) {
        s = cf32_t{dist(gen), dist(gen)};
    }

    std::vector<std::vector<float>> phase_coeffs(M);
    for (std::size_t p = 0; p < M; ++p) {
        phase_coeffs[p].resize(K);
        for (auto& h : phase_coeffs[p]) {
            h = dist(gen);
        }
    }
    auto coeffs = pfbc::prepare_interleaved_coeffs(phase_coeffs, M, K);

    std::vector<cf32_t> expected(num_outputs * M);
    kernels::filter_interleaved_scalar(
        input.data(), coeffs.data(), expected.data(), M, num_outputs, K);

    std::vector<cf32_t> actual(num_outputs * M);
    kernels::filter_interleaved_64ch_x4_avx512(
        input.data(), coeffs.data(), actual.data(), num_outputs, K);

    for (std::size_t i = 0; i < num_outputs * M; ++i) {
        INFO("Sample " << i << " (time=" << i/M << ", phase=" << i%M << ")");
        CHECK(actual[i].real() == Approx(expected[i].real()).epsilon(0.0001));
        CHECK(actual[i].imag() == Approx(expected[i].imag()).epsilon(0.0001));
    }
}

TEST_CASE("filter_interleaved_generic_avx512 - various channel counts", "[pfbc][kernels][interleaved]") {
    // Test the generic implementation with different M values
    std::mt19937 gen(555);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t M : {8, 16, 32, 128}) {
        SECTION("M = " + std::to_string(M)) {
            constexpr std::size_t K = 8;
            constexpr std::size_t num_outputs = 64;

            std::vector<cf32_t> input((K + num_outputs - 1) * M);
            for (auto& s : input) {
                s = cf32_t{dist(gen), dist(gen)};
            }

            std::vector<std::vector<float>> phase_coeffs(M);
            for (std::size_t p = 0; p < M; ++p) {
                phase_coeffs[p].resize(K);
                for (auto& h : phase_coeffs[p]) {
                    h = dist(gen);
                }
            }
            auto coeffs = pfbc::prepare_interleaved_coeffs(phase_coeffs, M, K);

            std::vector<cf32_t> expected(num_outputs * M);
            kernels::filter_interleaved_scalar(
                input.data(), coeffs.data(), expected.data(), M, num_outputs, K);

            std::vector<cf32_t> actual(num_outputs * M);
            kernels::filter_interleaved_generic_avx512(
                input.data(), coeffs.data(), actual.data(), M, num_outputs, K);

            for (std::size_t i = 0; i < num_outputs * M; ++i) {
                INFO("M=" << M << ", Sample " << i);
                CHECK(actual[i].real() == Approx(expected[i].real()).epsilon(0.001));
                CHECK(actual[i].imag() == Approx(expected[i].imag()).epsilon(0.001));
            }
        }
    }
}

TEST_CASE("filter_interleaved - matches naive polyphase filter", "[pfbc][kernels][interleaved]") {
    // Verify that interleaved approach produces same results as naive for-loop implementation
    // This is the key correctness test - validates the mathematical equivalence
    std::mt19937 gen(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t M : {8, 16, 32, 64}) {
        SECTION("M = " + std::to_string(M)) {
            constexpr std::size_t K = 4;
            constexpr std::size_t num_outputs = 16;

            // Input data (time-major interleaved)
            std::vector<cf32_t> input((K + num_outputs - 1) * M);
            for (auto& s : input) {
                s = cf32_t{dist(gen), dist(gen)};
            }

            // Per-phase coefficients
            std::vector<std::vector<float>> phase_coeffs(M);
            for (std::size_t p = 0; p < M; ++p) {
                phase_coeffs[p].resize(K);
                for (auto& h : phase_coeffs[p]) {
                    h = dist(gen);
                }
            }

            // === NAIVE REFERENCE IMPLEMENTATION ===
            // Simple triple-nested loop - easy to verify correctness
            // y[t,p] = sum_{k=0}^{K-1} h[p,k] * x[t+k,p]
            std::vector<cf32_t> naive_output(num_outputs * M);
            for (std::size_t t = 0; t < num_outputs; ++t) {
                for (std::size_t p = 0; p < M; ++p) {
                    cf32_t acc{0.0f, 0.0f};
                    for (std::size_t k = 0; k < K; ++k) {
                        cf32_t x = input[(t + k) * M + p];
                        float h = phase_coeffs[p][k];
                        acc += x * h;
                    }
                    naive_output[t * M + p] = acc;
                }
            }

            // === INTERLEAVED IMPLEMENTATION ===
            auto coeffs = pfbc::prepare_interleaved_coeffs(phase_coeffs, M, K);
            std::vector<cf32_t> interleaved_output(num_outputs * M);
            kernels::filter_interleaved(
                input.data(), coeffs.data(), interleaved_output.data(), M, num_outputs, K);

            // === COMPARE ===
            for (std::size_t i = 0; i < num_outputs * M; ++i) {
                INFO("M=" << M << ", Sample " << i << " (time=" << i/M << ", phase=" << i%M << ")");
                CHECK(interleaved_output[i].real() == Approx(naive_output[i].real()).epsilon(0.0001));
                CHECK(interleaved_output[i].imag() == Approx(naive_output[i].imag()).epsilon(0.0001));
            }
        }
    }
}

// =============================================================================
// TRANSPOSE KERNEL TESTS
// =============================================================================

TEST_CASE("transpose_8x8_cf32_avx512 - basic 8x8 transpose", "[pfbc][kernels][transpose]") {
    // Test basic 8x8 transpose with known values
    constexpr std::size_t M = 8;
    constexpr std::size_t N = 8;

    // Create time-major input: src[t][ch] = t * 100 + ch
    std::vector<cf32_t> src(M * N);
    for (std::size_t t = 0; t < N; ++t) {
        for (std::size_t ch = 0; ch < M; ++ch) {
            float val = static_cast<float>(t * 100 + ch);
            src[t * M + ch] = cf32_t{val, val + 0.5f};
        }
    }

    // Transpose to channel-major: dst[ch][t]
    std::vector<cf32_t> dst(M * N);
    kernels::transpose_8x8_cf32_avx512(src.data(), dst.data(), M, N);

    // Verify: dst[ch * N + t] should equal src[t * M + ch]
    for (std::size_t ch = 0; ch < M; ++ch) {
        for (std::size_t t = 0; t < N; ++t) {
            float expected_re = static_cast<float>(t * 100 + ch);
            float expected_im = expected_re + 0.5f;
            INFO("ch=" << ch << ", t=" << t);
            CHECK(dst[ch * N + t].real() == Approx(expected_re));
            CHECK(dst[ch * N + t].imag() == Approx(expected_im));
        }
    }
}

TEST_CASE("transpose_tm_to_cm - full matrix transpose", "[pfbc][kernels][transpose]") {
    // Test full matrix transpose with various sizes
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto [M, N] : std::vector<std::pair<std::size_t, std::size_t>>{
        {8, 8}, {16, 16}, {64, 64}, {64, 1024}, {32, 100}, {17, 33}
    }) {
        SECTION("M=" + std::to_string(M) + ", N=" + std::to_string(N)) {
            // Create random time-major input
            std::vector<cf32_t> src(M * N);
            for (auto& s : src) {
                s = cf32_t{dist(gen), dist(gen)};
            }

            // Transpose using AVX-512 kernel
            std::vector<cf32_t> dst(M * N);
            kernels::transpose_tm_to_cm(src.data(), dst.data(), M, N);

            // Verify against naive transpose
            for (std::size_t ch = 0; ch < M; ++ch) {
                for (std::size_t t = 0; t < N; ++t) {
                    cf32_t expected = src[t * M + ch];
                    cf32_t actual = dst[ch * N + t];
                    INFO("M=" << M << ", N=" << N << ", ch=" << ch << ", t=" << t);
                    CHECK(actual.real() == Approx(expected.real()));
                    CHECK(actual.imag() == Approx(expected.imag()));
                }
            }
        }
    }
}

TEST_CASE("transpose_to_channel_buffers - scatter to separate buffers", "[pfbc][kernels][transpose]") {
    // Test transpose to separate channel buffers (as used in emit_frames)
    std::mt19937 gen(54321);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto [M, N] : std::vector<std::pair<std::size_t, std::size_t>>{
        {8, 8}, {16, 32}, {64, 1024}, {32, 100}, {17, 33}, {9, 7}
    }) {
        SECTION("M=" + std::to_string(M) + ", N=" + std::to_string(N)) {
            // Create random time-major input
            std::vector<cf32_t> src(M * N);
            for (auto& s : src) {
                s = cf32_t{dist(gen), dist(gen)};
            }

            // Allocate separate output buffers for each channel
            std::vector<std::vector<cf32_t>> channel_buffers(M);
            std::vector<cf32_t*> ptrs(M);
            for (std::size_t ch = 0; ch < M; ++ch) {
                channel_buffers[ch].resize(N);
                ptrs[ch] = channel_buffers[ch].data();
            }

            // Transpose using AVX-512 kernel
            kernels::transpose_to_channel_buffers(src.data(), ptrs.data(), M, N);

            // Verify against naive transpose
            for (std::size_t ch = 0; ch < M; ++ch) {
                for (std::size_t t = 0; t < N; ++t) {
                    cf32_t expected = src[t * M + ch];
                    cf32_t actual = channel_buffers[ch][t];
                    INFO("M=" << M << ", N=" << N << ", ch=" << ch << ", t=" << t);
                    CHECK(actual.real() == Approx(expected.real()));
                    CHECK(actual.imag() == Approx(expected.imag()));
                }
            }
        }
    }
}

TEST_CASE("transpose_to_channel_buffers - handles null pointers", "[pfbc][kernels][transpose]") {
    // Test that null pointers are skipped gracefully
    constexpr std::size_t M = 17;
    constexpr std::size_t N = 33;

    std::mt19937 gen(99999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create random time-major input
    std::vector<cf32_t> src(M * N);
    for (auto& s : src) {
        s = cf32_t{dist(gen), dist(gen)};
    }

    // Allocate buffers for only some channels (simulate pool exhaustion)
    std::vector<std::vector<cf32_t>> channel_buffers(M);
    std::vector<cf32_t*> ptrs(M, nullptr);
    for (std::size_t ch = 0; ch < M; ch += 2) {  // Only even channels
        channel_buffers[ch].resize(N);
        ptrs[ch] = channel_buffers[ch].data();
    }

    // Transpose - should not crash on null pointers
    kernels::transpose_to_channel_buffers(src.data(), ptrs.data(), M, N);

    // Verify only the non-null channels
    for (std::size_t ch = 0; ch < M; ch += 2) {
        for (std::size_t t = 0; t < N; ++t) {
            cf32_t expected = src[t * M + ch];
            cf32_t actual = channel_buffers[ch][t];
            INFO("ch=" << ch << ", t=" << t);
            CHECK(actual.real() == Approx(expected.real()));
            CHECK(actual.imag() == Approx(expected.imag()));
        }
    }
}
