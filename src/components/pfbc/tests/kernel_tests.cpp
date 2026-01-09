// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../coefficients.hpp"
#include "../kernels.hpp"

#include <composite/buffers/aligned_mem.hpp>

#include <complex>
#include <vector>
#include <cmath>
#include <random>

using cf32_t = std::complex<float>;
using Catch::Approx;
using Catch::Matchers::WithinAbs;

// =============================================================================
// FFT-8 KERNEL TESTS
// =============================================================================

#include <fftw3.h>
#include <immintrin.h>

// Naive transpose: time-major to channel buffers (reference for testing)
static void naive_transpose_to_channels(
    const cf32_t* src,
    cf32_t* const* channel_ptrs,
    std::size_t M,
    std::size_t N
) {
    for (std::size_t ch = 0; ch < M; ++ch) {
        for (std::size_t t = 0; t < N; ++t) {
            channel_ptrs[ch][t] = src[t * M + ch];
        }
    }
}

// Reference FFT-8 using FFTW
static void fft8_reference(const cf32_t* input, cf32_t* output) {
    // Create FFTW plan (out-of-place, measure once)
    static fftwf_plan plan = nullptr;
    alignas(64) static cf32_t in_buf[8], out_buf[8];

    if (!plan) {
        plan = fftwf_plan_dft_1d(
            8,
            reinterpret_cast<fftwf_complex*>(in_buf),
            reinterpret_cast<fftwf_complex*>(out_buf),
            FFTW_FORWARD,
            FFTW_MEASURE
        );
    }

    std::copy(input, input + 8, in_buf);
    fftwf_execute(plan);
    std::copy(out_buf, out_buf + 8, output);
}

// Helper to run FFT-8 test with matrix DFT on interleaved data
[[gnu::target("avx512f,avx512dq")]]
static void run_fft8_test(const cf32_t* input, cf32_t* output) {
    __m512 interleaved = _mm512_loadu_ps(reinterpret_cast<const float*>(input));
    __m512 result = kernels::fft8::fft8_interleaved_reg(interleaved);
    _mm512_storeu_ps(reinterpret_cast<float*>(output), result);
}

TEST_CASE("fft8_interleaved_reg - matches FFTW reference", "[pfbc][kernels][fft8]") {
    // Test the matrix DFT FFT-8 against FFTW reference
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int trial = 0; trial < 100; ++trial) {
        SECTION("Trial " + std::to_string(trial)) {
            // Per-trial seed because Catch2 re-runs from the start for each SECTION.
            std::mt19937 gen(42 + trial);
            // Generate random input
            alignas(64) cf32_t input[8];
            for (int i = 0; i < 8; ++i) {
                input[i] = cf32_t{dist(gen), dist(gen)};
            }

            // FFTW reference
            alignas(64) cf32_t expected[8];
            fft8_reference(input, expected);

            // Our FFT-8 implementation
            alignas(64) cf32_t actual[8];
            run_fft8_test(input, actual);

            // Compare
            for (int i = 0; i < 8; ++i) {
                INFO("Trial " << trial << ", bin " << i);
                CHECK(actual[i].real() == Approx(expected[i].real()).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(expected[i].imag()).epsilon(1e-4).margin(3e-5));
            }
        }
    }
}

TEST_CASE("fft8_interleaved_reg - known DFT values", "[pfbc][kernels][fft8]") {
    // Test with known inputs where DFT result is predictable

    SECTION("DC input - all ones") {
        alignas(64) cf32_t input[8];
        std::fill(input, input + 8, cf32_t{1.0f, 0.0f});

        alignas(64) cf32_t actual[8];
        run_fft8_test(input, actual);

        // DC bin should be 8, all others should be 0
        CHECK(actual[0].real() == Approx(8.0f).epsilon(1e-4).margin(3e-5));
        CHECK(actual[0].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        for (int i = 1; i < 8; ++i) {
            INFO("bin " << i);
            CHECK(actual[i].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Single tone at bin 1") {
        // Input: exp(j * 2*pi * 1 * n / 8) for n = 0..7
        alignas(64) cf32_t input[8];
        for (int n = 0; n < 8; ++n) {
            float phase = 2.0f * static_cast<float>(M_PI) * 1 * n / 8.0f;
            input[n] = cf32_t{std::cos(phase), std::sin(phase)};
        }

        alignas(64) cf32_t actual[8];
        run_fft8_test(input, actual);

        // Bin 1 should be 8, all others should be 0
        for (int i = 0; i < 8; ++i) {
            INFO("bin " << i);
            if (i == 1) {
                CHECK(actual[i].real() == Approx(8.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            } else {
                CHECK(actual[i].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Single tone at bin 3") {
        alignas(64) cf32_t input[8];
        for (int n = 0; n < 8; ++n) {
            float phase = 2.0f * static_cast<float>(M_PI) * 3 * n / 8.0f;
            input[n] = cf32_t{std::cos(phase), std::sin(phase)};
        }

        alignas(64) cf32_t actual[8];
        run_fft8_test(input, actual);

        // Bin 3 should be 8, all others should be 0
        for (int i = 0; i < 8; ++i) {
            INFO("bin " << i);
            if (i == 3) {
                CHECK(actual[i].real() == Approx(8.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            } else {
                CHECK(actual[i].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Impulse at n=0 -> all bins = 1") {
        alignas(64) cf32_t input[8] = {};
        input[0] = cf32_t{1.0f, 0.0f};

        alignas(64) cf32_t actual[8];
        run_fft8_test(input, actual);

        // DFT of impulse at n=0 is all 1s
        for (int k = 0; k < 8; ++k) {
            INFO("bin " << k);
            CHECK(actual[k].real() == Approx(1.0f).epsilon(1e-4).margin(3e-5));
            CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Impulse at n=1 -> phase ramp") {
        alignas(64) cf32_t input[8] = {};
        input[1] = cf32_t{1.0f, 0.0f};

        alignas(64) cf32_t actual[8];
        run_fft8_test(input, actual);

        // DFT of delta[n-1] is exp(-j * 2*pi*k / N)
        for (int k = 0; k < 8; ++k) {
            float angle = -2.0f * static_cast<float>(M_PI) * k / 8.0f;
            float expected_re = std::cos(angle);
            float expected_im = std::sin(angle);
            INFO("bin " << k);
            CHECK(actual[k].real() == Approx(expected_re).epsilon(1e-4).margin(3e-5));
            CHECK(actual[k].imag() == Approx(expected_im).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Impulse at n=4 -> alternating ±1") {
        alignas(64) cf32_t input[8] = {};
        input[4] = cf32_t{1.0f, 0.0f};

        alignas(64) cf32_t actual[8];
        run_fft8_test(input, actual);

        // DFT of delta[n-4] is exp(-j * 2*pi*4*k / 8) = exp(-j * pi * k) = (-1)^k
        for (int k = 0; k < 8; ++k) {
            float expected_re = (k % 2 == 0) ? 1.0f : -1.0f;
            INFO("bin " << k);
            CHECK(actual[k].real() == Approx(expected_re).epsilon(1e-4).margin(3e-5));
            CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Nyquist input -> spike at k=4") {
        alignas(64) cf32_t input[8];
        for (int n = 0; n < 8; ++n) {
            input[n] = cf32_t{(n % 2 == 0) ? 1.0f : -1.0f, 0.0f};
        }

        alignas(64) cf32_t actual[8];
        run_fft8_test(input, actual);

        // Nyquist frequency appears at bin N/2 = 4
        for (int k = 0; k < 8; ++k) {
            INFO("bin " << k);
            if (k == 4) {
                CHECK(actual[k].real() == Approx(8.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            } else {
                CHECK(actual[k].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Pure imaginary impulse -> all j") {
        alignas(64) cf32_t input[8] = {};
        input[0] = cf32_t{0.0f, 1.0f};

        alignas(64) cf32_t actual[8];
        run_fft8_test(input, actual);

        // DFT of imaginary impulse at n=0 is all j
        for (int k = 0; k < 8; ++k) {
            INFO("bin " << k);
            CHECK(actual[k].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            CHECK(actual[k].imag() == Approx(1.0f).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Linearity: FFT(a+b) = FFT(a) + FFT(b)") {
        alignas(64) cf32_t a[8], b[8], sum[8];
        for (int n = 0; n < 8; ++n) {
            a[n] = cf32_t{std::sin(n * 0.5f), std::cos(n * 0.3f)};
            b[n] = cf32_t{std::cos(n * 0.7f), std::sin(n * 0.9f)};
            sum[n] = a[n] + b[n];
        }

        alignas(64) cf32_t fft_a[8], fft_b[8], fft_sum[8];
        run_fft8_test(a, fft_a);
        run_fft8_test(b, fft_b);
        run_fft8_test(sum, fft_sum);

        for (int k = 0; k < 8; ++k) {
            cf32_t expected = fft_a[k] + fft_b[k];
            INFO("bin " << k);
            CHECK(fft_sum[k].real() == Approx(expected.real()).epsilon(1e-4).margin(3e-5));
            CHECK(fft_sum[k].imag() == Approx(expected.imag()).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Parseval's theorem: sum|x|^2 = sum|X|^2 / N") {
        alignas(64) cf32_t input[8];
        for (int n = 0; n < 8; ++n) {
            input[n] = cf32_t{std::sin(n * 1.1f + 0.3f), std::cos(n * 0.7f + 0.5f)};
        }

        float time_energy = 0.0f;
        for (int n = 0; n < 8; ++n) {
            time_energy += std::norm(input[n]);  // |x[n]|^2
        }

        alignas(64) cf32_t output[8];
        run_fft8_test(input, output);

        float freq_energy = 0.0f;
        for (int k = 0; k < 8; ++k) {
            freq_energy += std::norm(output[k]);  // |X[k]|^2
        }
        freq_energy /= 8.0f;

        CHECK(time_energy == Approx(freq_energy).epsilon(1e-4).margin(3e-5));
    }
}

// =============================================================================
// VERTICAL FFT-8 KERNEL TESTS
// =============================================================================
// The vertical FFT-8 processes 8 parallel FFT-8s with column-major data layout.
// Each register holds one frequency index across all 8 FFTs.

// Helper to run vertical FFT-8 test
// Input:  8x8 matrix, column-major: data[ch][t] = input for FFT #t, index ch
// Output: 8x8 matrix, column-major: data[k][t] = output of FFT #t, bin k
[[gnu::target("avx512f,avx512dq")]]
static void run_fft8_vertical_test(const cf32_t* input, cf32_t* output) {
    // Load column-major: r[i] holds input[i] from all 8 FFTs
    __m512 r0 = _mm512_loadu_ps(reinterpret_cast<const float*>(input) + 0 * 16);
    __m512 r1 = _mm512_loadu_ps(reinterpret_cast<const float*>(input) + 1 * 16);
    __m512 r2 = _mm512_loadu_ps(reinterpret_cast<const float*>(input) + 2 * 16);
    __m512 r3 = _mm512_loadu_ps(reinterpret_cast<const float*>(input) + 3 * 16);
    __m512 r4 = _mm512_loadu_ps(reinterpret_cast<const float*>(input) + 4 * 16);
    __m512 r5 = _mm512_loadu_ps(reinterpret_cast<const float*>(input) + 5 * 16);
    __m512 r6 = _mm512_loadu_ps(reinterpret_cast<const float*>(input) + 6 * 16);
    __m512 r7 = _mm512_loadu_ps(reinterpret_cast<const float*>(input) + 7 * 16);

    kernels::fft8_vertical::fft8_vertical(r0, r1, r2, r3, r4, r5, r6, r7);

    // Store column-major: r[k] holds X[k] from all 8 FFTs
    _mm512_storeu_ps(reinterpret_cast<float*>(output) + 0 * 16, r0);
    _mm512_storeu_ps(reinterpret_cast<float*>(output) + 1 * 16, r1);
    _mm512_storeu_ps(reinterpret_cast<float*>(output) + 2 * 16, r2);
    _mm512_storeu_ps(reinterpret_cast<float*>(output) + 3 * 16, r3);
    _mm512_storeu_ps(reinterpret_cast<float*>(output) + 4 * 16, r4);
    _mm512_storeu_ps(reinterpret_cast<float*>(output) + 5 * 16, r5);
    _mm512_storeu_ps(reinterpret_cast<float*>(output) + 6 * 16, r6);
    _mm512_storeu_ps(reinterpret_cast<float*>(output) + 7 * 16, r7);
}

TEST_CASE("fft8_vertical - matches FFTW reference", "[pfbc][kernels][fft8_vertical]") {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int trial = 0; trial < 100; ++trial) {
        SECTION("Trial " + std::to_string(trial)) {
            // Per-trial seed because Catch2 re-runs from the start for each SECTION.
            std::mt19937 gen(12345 + trial);
            // Generate 8 random FFT inputs (column-major layout)
            // input[ch * 8 + t] = FFT #t, index ch
            alignas(64) cf32_t input[64];  // 8 FFTs × 8 samples
            for (int i = 0; i < 64; ++i) {
                input[i] = cf32_t{dist(gen), dist(gen)};
            }

            // Compute reference using FFTW for each of the 8 FFTs
            alignas(64) cf32_t expected[64];
            for (int t = 0; t < 8; ++t) {
                // Extract FFT #t input (column t from column-major matrix)
                alignas(64) cf32_t fft_input[8];
                for (int ch = 0; ch < 8; ++ch) {
                    fft_input[ch] = input[ch * 8 + t];
                }
                // Compute reference FFT
                alignas(64) cf32_t fft_output[8];
                fft8_reference(fft_input, fft_output);
                // Store in column-major output
                for (int k = 0; k < 8; ++k) {
                    expected[k * 8 + t] = fft_output[k];
                }
            }

            // Run vertical FFT
            alignas(64) cf32_t actual[64];
            run_fft8_vertical_test(input, actual);

            // Compare all 8 FFTs
            for (int t = 0; t < 8; ++t) {
                for (int k = 0; k < 8; ++k) {
                    INFO("FFT " << t << ", bin " << k);
                    int idx = k * 8 + t;
                    CHECK(actual[idx].real() == Approx(expected[idx].real()).epsilon(1e-4).margin(3e-5));
                    CHECK(actual[idx].imag() == Approx(expected[idx].imag()).epsilon(1e-4).margin(3e-5));
                }
            }
        }
    }
}

TEST_CASE("fft8_vertical - known DFT values", "[pfbc][kernels][fft8_vertical]") {
    SECTION("DC input - all ones for all 8 FFTs") {
        alignas(64) cf32_t input[64];
        std::fill(input, input + 64, cf32_t{1.0f, 0.0f});

        alignas(64) cf32_t actual[64];
        run_fft8_vertical_test(input, actual);

        // For each FFT, DC bin should be 8, all others should be 0
        for (int t = 0; t < 8; ++t) {
            INFO("FFT " << t);
            CHECK(actual[0 * 8 + t].real() == Approx(8.0f).epsilon(1e-4).margin(3e-5));
            CHECK(actual[0 * 8 + t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            for (int k = 1; k < 8; ++k) {
                INFO("bin " << k);
                CHECK(actual[k * 8 + t].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[k * 8 + t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Impulse at n=0 for all 8 FFTs") {
        alignas(64) cf32_t input[64] = {};
        // Set x[0] = 1 for all 8 FFTs (first row in column-major)
        for (int t = 0; t < 8; ++t) {
            input[0 * 8 + t] = cf32_t{1.0f, 0.0f};
        }

        alignas(64) cf32_t actual[64];
        run_fft8_vertical_test(input, actual);

        // DFT of impulse at n=0 is all 1s
        for (int t = 0; t < 8; ++t) {
            INFO("FFT " << t);
            for (int k = 0; k < 8; ++k) {
                INFO("bin " << k);
                CHECK(actual[k * 8 + t].real() == Approx(1.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[k * 8 + t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Different tones for different FFTs") {
        alignas(64) cf32_t input[64];
        // FFT #t gets a tone at bin t
        for (int t = 0; t < 8; ++t) {
            for (int n = 0; n < 8; ++n) {
                float phase = 2.0f * static_cast<float>(M_PI) * t * n / 8.0f;
                input[n * 8 + t] = cf32_t{std::cos(phase), std::sin(phase)};
            }
        }

        alignas(64) cf32_t actual[64];
        run_fft8_vertical_test(input, actual);

        // Each FFT #t should have peak at bin t
        for (int t = 0; t < 8; ++t) {
            INFO("FFT " << t);
            for (int k = 0; k < 8; ++k) {
                INFO("bin " << k);
                if (k == t) {
                    CHECK(actual[k * 8 + t].real() == Approx(8.0f).epsilon(1e-4).margin(3e-5));
                    CHECK(actual[k * 8 + t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                } else {
                    CHECK(actual[k * 8 + t].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                    CHECK(actual[k * 8 + t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                }
            }
        }
    }

    SECTION("Parseval's theorem for all 8 FFTs") {
        alignas(64) cf32_t input[64];
        for (int i = 0; i < 64; ++i) {
            input[i] = cf32_t{std::sin(i * 1.1f + 0.3f), std::cos(i * 0.7f + 0.5f)};
        }

        alignas(64) cf32_t output[64];
        run_fft8_vertical_test(input, output);

        // Check Parseval for each FFT
        for (int t = 0; t < 8; ++t) {
            float time_energy = 0.0f;
            float freq_energy = 0.0f;
            for (int n = 0; n < 8; ++n) {
                time_energy += std::norm(input[n * 8 + t]);
                freq_energy += std::norm(output[n * 8 + t]);
            }
            freq_energy /= 8.0f;
            INFO("FFT " << t);
            CHECK(time_energy == Approx(freq_energy).epsilon(1e-4).margin(3e-5));
        }
    }
}

// =============================================================================
// FFT-16 KERNEL TESTS
// =============================================================================

// Reference FFT-16 using FFTW
static void fft16_reference(const cf32_t* input, cf32_t* output) {
    static fftwf_plan plan = nullptr;
    alignas(64) static cf32_t in_buf[16], out_buf[16];

    if (!plan) {
        plan = fftwf_plan_dft_1d(
            16,
            reinterpret_cast<fftwf_complex*>(in_buf),
            reinterpret_cast<fftwf_complex*>(out_buf),
            FFTW_FORWARD,
            FFTW_MEASURE
        );
    }

    std::copy(input, input + 16, in_buf);
    fftwf_execute(plan);
    std::copy(out_buf, out_buf + 16, output);
}

// Helper to run FFT-16 test
[[gnu::target("avx512f,avx512vl,fma")]]
static void run_fft16_test(const cf32_t* input, cf32_t* output) {
    __m512 lo = _mm512_load_ps(reinterpret_cast<const float*>(input));
    __m512 hi = _mm512_load_ps(reinterpret_cast<const float*>(input) + 16);
    kernels::fft16::fft16_interleaved_reg(lo, hi);
    _mm512_store_ps(reinterpret_cast<float*>(output), lo);
    _mm512_store_ps(reinterpret_cast<float*>(output) + 16, hi);
}

TEST_CASE("fft16_interleaved_reg - matches FFTW reference", "[pfbc][kernels][fft16]") {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int trial = 0; trial < 100; ++trial) {
        SECTION("Trial " + std::to_string(trial)) {
            // Per-trial seed because Catch2 re-runs from the start for each SECTION.
            std::mt19937 gen(4242 + trial);
            alignas(64) cf32_t input[16];
            for (int i = 0; i < 16; ++i) {
                input[i] = cf32_t{dist(gen), dist(gen)};
            }

            alignas(64) cf32_t expected[16];
            fft16_reference(input, expected);

            alignas(64) cf32_t actual[16];
            run_fft16_test(input, actual);

            for (int i = 0; i < 16; ++i) {
                INFO("Trial " << trial << ", bin " << i);
                CHECK(actual[i].real() == Approx(expected[i].real()).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(expected[i].imag()).epsilon(1e-4).margin(3e-5));
            }
        }
    }
}

TEST_CASE("fft16_interleaved_reg - known DFT values", "[pfbc][kernels][fft16]") {
    SECTION("DC input - all ones") {
        alignas(64) cf32_t input[16];
        std::fill(input, input + 16, cf32_t{1.0f, 0.0f});

        alignas(64) cf32_t actual[16];
        run_fft16_test(input, actual);

        CHECK(actual[0].real() == Approx(16.0f).epsilon(1e-4).margin(3e-5));
        CHECK(actual[0].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        for (int i = 1; i < 16; ++i) {
            INFO("bin " << i);
            CHECK(actual[i].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Single tone at bin 1") {
        alignas(64) cf32_t input[16];
        for (int n = 0; n < 16; ++n) {
            float phase = 2.0f * static_cast<float>(M_PI) * 1 * n / 16.0f;
            input[n] = cf32_t{std::cos(phase), std::sin(phase)};
        }

        alignas(64) cf32_t actual[16];
        run_fft16_test(input, actual);

        for (int i = 0; i < 16; ++i) {
            INFO("bin " << i);
            if (i == 1) {
                CHECK(actual[i].real() == Approx(16.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            } else {
                CHECK(actual[i].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Single tone at bin 4") {
        alignas(64) cf32_t input[16];
        for (int n = 0; n < 16; ++n) {
            float phase = 2.0f * static_cast<float>(M_PI) * 4 * n / 16.0f;
            input[n] = cf32_t{std::cos(phase), std::sin(phase)};
        }

        alignas(64) cf32_t actual[16];
        run_fft16_test(input, actual);

        for (int i = 0; i < 16; ++i) {
            INFO("bin " << i);
            if (i == 4) {
                CHECK(actual[i].real() == Approx(16.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            } else {
                CHECK(actual[i].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Impulse at n=0 -> all bins = 1") {
        alignas(64) cf32_t input[16] = {};
        input[0] = cf32_t{1.0f, 0.0f};

        alignas(64) cf32_t actual[16];
        run_fft16_test(input, actual);

        for (int k = 0; k < 16; ++k) {
            INFO("bin " << k);
            CHECK(actual[k].real() == Approx(1.0f).epsilon(1e-4).margin(3e-5));
            CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Impulse at n=1 -> phase ramp") {
        alignas(64) cf32_t input[16] = {};
        input[1] = cf32_t{1.0f, 0.0f};

        alignas(64) cf32_t actual[16];
        run_fft16_test(input, actual);

        for (int k = 0; k < 16; ++k) {
            float angle = -2.0f * static_cast<float>(M_PI) * k / 16.0f;
            float expected_re = std::cos(angle);
            float expected_im = std::sin(angle);
            INFO("bin " << k);
            CHECK(actual[k].real() == Approx(expected_re).epsilon(1e-4).margin(3e-5));
            CHECK(actual[k].imag() == Approx(expected_im).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Impulse at n=8 -> alternating ±1") {
        alignas(64) cf32_t input[16] = {};
        input[8] = cf32_t{1.0f, 0.0f};

        alignas(64) cf32_t actual[16];
        run_fft16_test(input, actual);

        for (int k = 0; k < 16; ++k) {
            float expected_re = (k % 2 == 0) ? 1.0f : -1.0f;
            INFO("bin " << k);
            CHECK(actual[k].real() == Approx(expected_re).epsilon(1e-4).margin(3e-5));
            CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Nyquist input -> spike at k=8") {
        alignas(64) cf32_t input[16];
        for (int n = 0; n < 16; ++n) {
            input[n] = cf32_t{(n % 2 == 0) ? 1.0f : -1.0f, 0.0f};
        }

        alignas(64) cf32_t actual[16];
        run_fft16_test(input, actual);

        for (int k = 0; k < 16; ++k) {
            INFO("bin " << k);
            if (k == 8) {
                CHECK(actual[k].real() == Approx(16.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            } else {
                CHECK(actual[k].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Parseval's theorem") {
        alignas(64) cf32_t input[16];
        for (int n = 0; n < 16; ++n) {
            input[n] = cf32_t{std::sin(n * 1.1f + 0.3f), std::cos(n * 0.7f + 0.5f)};
        }

        float time_energy = 0.0f;
        for (int n = 0; n < 16; ++n) {
            time_energy += std::norm(input[n]);
        }

        alignas(64) cf32_t output[16];
        run_fft16_test(input, output);

        float freq_energy = 0.0f;
        for (int k = 0; k < 16; ++k) {
            freq_energy += std::norm(output[k]);
        }
        freq_energy /= 16.0f;

        CHECK(time_energy == Approx(freq_energy).epsilon(1e-4).margin(3e-5));
    }
}

// =============================================================================
// VERTICAL FFT-16 KERNEL TESTS (8 parallel FFT-16s)
// =============================================================================

// Helper to run vertical FFT-16 test
// Input/output layout: column-major, input[ch * 8 + t] = FFT #t, index ch
[[gnu::target("avx512f,avx512dq,fma")]]
static void run_fft16_vertical_test(const cf32_t* input, cf32_t* output) {
    // Load 16 registers (one per frequency index, each holding 8 FFTs)
    __m512 r0  = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 0 * 8));
    __m512 r1  = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 1 * 8));
    __m512 r2  = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 2 * 8));
    __m512 r3  = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 3 * 8));
    __m512 r4  = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 4 * 8));
    __m512 r5  = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 5 * 8));
    __m512 r6  = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 6 * 8));
    __m512 r7  = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 7 * 8));
    __m512 r8  = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 8 * 8));
    __m512 r9  = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 9 * 8));
    __m512 r10 = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 10 * 8));
    __m512 r11 = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 11 * 8));
    __m512 r12 = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 12 * 8));
    __m512 r13 = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 13 * 8));
    __m512 r14 = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 14 * 8));
    __m512 r15 = _mm512_loadu_ps(reinterpret_cast<const float*>(input + 15 * 8));

    kernels::fft16_vertical::fft16_vertical(
        r0, r1, r2, r3, r4, r5, r6, r7,
        r8, r9, r10, r11, r12, r13, r14, r15
    );

    _mm512_storeu_ps(reinterpret_cast<float*>(output + 0 * 8), r0);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 1 * 8), r1);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 2 * 8), r2);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 3 * 8), r3);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 4 * 8), r4);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 5 * 8), r5);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 6 * 8), r6);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 7 * 8), r7);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 8 * 8), r8);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 9 * 8), r9);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 10 * 8), r10);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 11 * 8), r11);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 12 * 8), r12);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 13 * 8), r13);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 14 * 8), r14);
    _mm512_storeu_ps(reinterpret_cast<float*>(output + 15 * 8), r15);
}

TEST_CASE("fft16_vertical - matches FFTW reference", "[pfbc][kernels][fft16_vertical]") {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int trial = 0; trial < 100; ++trial) {
        SECTION("Trial " + std::to_string(trial)) {
            // Use per-trial seed (Catch2 re-runs from beginning for each SECTION)
            std::mt19937 gen(54321 + trial);

            // Generate 8 random FFT-16 inputs (column-major layout)
            // input[ch * 8 + t] = FFT #t, index ch
            alignas(64) cf32_t input[128];  // 8 FFTs × 16 samples
            for (int i = 0; i < 128; ++i) {
                input[i] = cf32_t{dist(gen), dist(gen)};
            }

            // Compute reference using FFTW for each of the 8 FFTs
            alignas(64) cf32_t expected[128];
            for (int t = 0; t < 8; ++t) {
                // Extract FFT #t input (column t from column-major matrix)
                alignas(64) cf32_t fft_input[16];
                for (int ch = 0; ch < 16; ++ch) {
                    fft_input[ch] = input[ch * 8 + t];
                }
                // Compute reference FFT
                alignas(64) cf32_t fft_output[16];
                fft16_reference(fft_input, fft_output);
                // Store in column-major output
                for (int k = 0; k < 16; ++k) {
                    expected[k * 8 + t] = fft_output[k];
                }
            }

            // Run vertical FFT-16
            alignas(64) cf32_t actual[128];
            run_fft16_vertical_test(input, actual);

            // Compare all 8 FFTs
            // Use 2e-4 relative tolerance (accumulated FP error through 4 stages)
            // plus margin for values near zero where relative error is meaningless
            for (int t = 0; t < 8; ++t) {
                for (int k = 0; k < 16; ++k) {
                    INFO("FFT " << t << ", bin " << k);
                    int idx = k * 8 + t;
                    CHECK(actual[idx].real() == Approx(expected[idx].real()).epsilon(2e-4).margin(3e-5));
                    CHECK(actual[idx].imag() == Approx(expected[idx].imag()).epsilon(2e-4).margin(3e-5));
                }
            }
        }
    }
}

TEST_CASE("fft16_vertical - known DFT values", "[pfbc][kernels][fft16_vertical]") {
    SECTION("DC input - all ones for all 8 FFTs") {
        alignas(64) cf32_t input[128];
        std::fill(input, input + 128, cf32_t{1.0f, 0.0f});

        alignas(64) cf32_t actual[128];
        run_fft16_vertical_test(input, actual);

        // For each FFT, DC bin should be 16, all others should be 0
        for (int t = 0; t < 8; ++t) {
            INFO("FFT " << t);
            CHECK(actual[0 * 8 + t].real() == Approx(16.0f).epsilon(1e-4).margin(3e-5));
            CHECK(actual[0 * 8 + t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            for (int k = 1; k < 16; ++k) {
                INFO("bin " << k);
                CHECK(actual[k * 8 + t].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[k * 8 + t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Impulse at n=0 for all 8 FFTs") {
        alignas(64) cf32_t input[128] = {};
        // Set x[0] = 1 for all 8 FFTs (first row in column-major)
        for (int t = 0; t < 8; ++t) {
            input[0 * 8 + t] = cf32_t{1.0f, 0.0f};
        }

        alignas(64) cf32_t actual[128];
        run_fft16_vertical_test(input, actual);

        // DFT of impulse at n=0 is all 1s
        for (int t = 0; t < 8; ++t) {
            INFO("FFT " << t);
            for (int k = 0; k < 16; ++k) {
                INFO("bin " << k);
                CHECK(actual[k * 8 + t].real() == Approx(1.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[k * 8 + t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Different tones for different FFTs") {
        alignas(64) cf32_t input[128];
        // FFT #t gets a tone at bin t
        for (int t = 0; t < 8; ++t) {
            for (int n = 0; n < 16; ++n) {
                float phase = 2.0f * static_cast<float>(M_PI) * t * n / 16.0f;
                input[n * 8 + t] = cf32_t{std::cos(phase), std::sin(phase)};
            }
        }

        alignas(64) cf32_t actual[128];
        run_fft16_vertical_test(input, actual);

        // FFT #t should have spike at bin t
        for (int t = 0; t < 8; ++t) {
            INFO("FFT " << t);
            for (int k = 0; k < 16; ++k) {
                INFO("bin " << k);
                if (k == t) {
                    CHECK(actual[k * 8 + t].real() == Approx(16.0f).epsilon(1e-4).margin(3e-5));
                    CHECK(actual[k * 8 + t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                } else {
                    CHECK(actual[k * 8 + t].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                    CHECK(actual[k * 8 + t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                }
            }
        }
    }
}

// =============================================================================
// FFT-32 KERNEL TESTS
// =============================================================================

// Reference FFT-32 using FFTW
static void fft32_reference(const cf32_t* input, cf32_t* output) {
    static fftwf_plan plan = nullptr;
    alignas(64) static cf32_t in_buf[32], out_buf[32];

    if (!plan) {
        plan = fftwf_plan_dft_1d(
            32,
            reinterpret_cast<fftwf_complex*>(in_buf),
            reinterpret_cast<fftwf_complex*>(out_buf),
            FFTW_FORWARD,
            FFTW_MEASURE
        );
    }

    std::copy(input, input + 32, in_buf);
    fftwf_execute(plan);
    std::copy(out_buf, out_buf + 32, output);
}

// Helper to run FFT-32 test
[[gnu::target("avx512f,avx512vl,avx512dq,fma")]]
static void run_fft32_test(const cf32_t* input, cf32_t* output) {
    __m512 r0 = _mm512_load_ps(reinterpret_cast<const float*>(input));
    __m512 r1 = _mm512_load_ps(reinterpret_cast<const float*>(input) + 16);
    __m512 r2 = _mm512_load_ps(reinterpret_cast<const float*>(input) + 32);
    __m512 r3 = _mm512_load_ps(reinterpret_cast<const float*>(input) + 48);
    kernels::fft32::fft32_interleaved_reg(r0, r1, r2, r3);
    _mm512_store_ps(reinterpret_cast<float*>(output), r0);
    _mm512_store_ps(reinterpret_cast<float*>(output) + 16, r1);
    _mm512_store_ps(reinterpret_cast<float*>(output) + 32, r2);
    _mm512_store_ps(reinterpret_cast<float*>(output) + 48, r3);
}

TEST_CASE("fft32_interleaved_reg - matches FFTW reference", "[pfbc][kernels][fft32]") {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int trial = 0; trial < 100; ++trial) {
        SECTION("Trial " + std::to_string(trial)) {
            // Per-trial seed because Catch2 re-runs from the start for each SECTION.
            std::mt19937 gen(324232 + trial);
            alignas(64) cf32_t input[32];
            for (int i = 0; i < 32; ++i) {
                input[i] = cf32_t{dist(gen), dist(gen)};
            }

            alignas(64) cf32_t expected[32];
            fft32_reference(input, expected);

            alignas(64) cf32_t actual[32];
            run_fft32_test(input, actual);

            for (int i = 0; i < 32; ++i) {
                INFO("Trial " << trial << ", bin " << i);
                CHECK(actual[i].real() == Approx(expected[i].real()).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(expected[i].imag()).epsilon(1e-4).margin(3e-5));
            }
        }
    }
}

TEST_CASE("fft32_interleaved_reg - known DFT values", "[pfbc][kernels][fft32]") {
    SECTION("DC input - all ones") {
        alignas(64) cf32_t input[32];
        std::fill(input, input + 32, cf32_t{1.0f, 0.0f});

        alignas(64) cf32_t actual[32];
        run_fft32_test(input, actual);

        CHECK(actual[0].real() == Approx(32.0f).epsilon(1e-4).margin(3e-5));
        CHECK(actual[0].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        for (int i = 1; i < 32; ++i) {
            INFO("bin " << i);
            CHECK(actual[i].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Single tone at bin 1") {
        alignas(64) cf32_t input[32];
        for (int n = 0; n < 32; ++n) {
            float phase = 2.0f * static_cast<float>(M_PI) * 1 * n / 32.0f;
            input[n] = cf32_t{std::cos(phase), std::sin(phase)};
        }

        alignas(64) cf32_t actual[32];
        run_fft32_test(input, actual);

        for (int i = 0; i < 32; ++i) {
            INFO("bin " << i);
            if (i == 1) {
                CHECK(actual[i].real() == Approx(32.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            } else {
                CHECK(actual[i].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Single tone at bin 8") {
        alignas(64) cf32_t input[32];
        for (int n = 0; n < 32; ++n) {
            float phase = 2.0f * static_cast<float>(M_PI) * 8 * n / 32.0f;
            input[n] = cf32_t{std::cos(phase), std::sin(phase)};
        }

        alignas(64) cf32_t actual[32];
        run_fft32_test(input, actual);

        for (int i = 0; i < 32; ++i) {
            INFO("bin " << i);
            if (i == 8) {
                CHECK(actual[i].real() == Approx(32.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            } else {
                CHECK(actual[i].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[i].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Impulse at n=0 -> all bins = 1") {
        alignas(64) cf32_t input[32] = {};
        input[0] = cf32_t{1.0f, 0.0f};

        alignas(64) cf32_t actual[32];
        run_fft32_test(input, actual);

        for (int k = 0; k < 32; ++k) {
            INFO("bin " << k);
            CHECK(actual[k].real() == Approx(1.0f).epsilon(1e-4).margin(3e-5));
            CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Impulse at n=16 -> alternating ±1") {
        alignas(64) cf32_t input[32] = {};
        input[16] = cf32_t{1.0f, 0.0f};

        alignas(64) cf32_t actual[32];
        run_fft32_test(input, actual);

        for (int k = 0; k < 32; ++k) {
            float expected_re = (k % 2 == 0) ? 1.0f : -1.0f;
            INFO("bin " << k);
            CHECK(actual[k].real() == Approx(expected_re).epsilon(1e-4).margin(3e-5));
            CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
        }
    }

    SECTION("Nyquist input -> spike at k=16") {
        alignas(64) cf32_t input[32];
        for (int n = 0; n < 32; ++n) {
            input[n] = cf32_t{(n % 2 == 0) ? 1.0f : -1.0f, 0.0f};
        }

        alignas(64) cf32_t actual[32];
        run_fft32_test(input, actual);

        for (int k = 0; k < 32; ++k) {
            INFO("bin " << k);
            if (k == 16) {
                CHECK(actual[k].real() == Approx(32.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            } else {
                CHECK(actual[k].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(actual[k].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Parseval's theorem") {
        alignas(64) cf32_t input[32];
        for (int n = 0; n < 32; ++n) {
            input[n] = cf32_t{std::sin(n * 1.1f + 0.3f), std::cos(n * 0.7f + 0.5f)};
        }

        float time_energy = 0.0f;
        for (int n = 0; n < 32; ++n) {
            time_energy += std::norm(input[n]);
        }

        alignas(64) cf32_t output[32];
        run_fft32_test(input, output);

        float freq_energy = 0.0f;
        for (int k = 0; k < 32; ++k) {
            freq_energy += std::norm(output[k]);
        }
        freq_energy /= 32.0f;

        CHECK(time_energy == Approx(freq_energy).epsilon(1e-4).margin(3e-5));
    }
}

// =============================================================================
// M=32 HYBRID KERNEL TESTS
// =============================================================================

// Reference: staged pipeline (filter_blocked + FFTW + scalar transpose)
[[gnu::target("avx512f,avx512dq,fma")]]
static void m32_staged_reference(
    const cf32_t* input,
    const float* coeffs,
    cf32_t* const* channel_ptrs,
    std::size_t num_outputs,
    std::size_t num_taps
) {
    constexpr std::size_t M = 32;

    // Allocate scratch for one row
    alignas(64) cf32_t scratch[M];

    // Create FFTW plan
    static fftwf_plan plan = nullptr;
    alignas(64) static cf32_t fftw_in[M], fftw_out[M];
    if (!plan) {
        plan = fftwf_plan_dft_1d(static_cast<int>(M),
            reinterpret_cast<fftwf_complex*>(fftw_in),
            reinterpret_cast<fftwf_complex*>(fftw_out),
            FFTW_FORWARD, FFTW_MEASURE);
    }

    for (std::size_t t = 0; t < num_outputs; ++t) {
        // Stage 1: Filter
        kernels::filter_blocked(input + t * M, coeffs, scratch, M, 1, num_taps);

        // Stage 2: FFT
        std::copy(scratch, scratch + M, fftw_in);
        fftwf_execute(plan);

        // Stage 3: Scatter to channels
        for (std::size_t ch = 0; ch < M; ++ch) {
            if (channel_ptrs[ch]) {
                channel_ptrs[ch][t] = fftw_out[ch];
            }
        }
    }
}

TEST_CASE("filter_fft_transpose_32ch_hybrid - correctness", "[pfbc][kernels][m32_hybrid]") {
    constexpr std::size_t M = 32;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t K : {4, 8, 16}) {
        SECTION("K=" + std::to_string(K)) {
            for (std::size_t num_outputs : {4, 8, 16, 100}) {
                SECTION("num_outputs=" + std::to_string(num_outputs)) {
                    std::mt19937 gen(12345 + K + num_outputs);

                    // Generate input
                    std::size_t input_rows = K + num_outputs - 1;
                    std::vector<cf32_t> input(input_rows * M);
                    for (auto& s : input) {
                        s = cf32_t{dist(gen), dist(gen)};
                    }

                    // Generate coefficients
                    std::vector<std::vector<float>> phase_coeffs(M);
                    for (std::size_t p = 0; p < M; ++p) {
                        phase_coeffs[p].resize(K);
                        for (auto& h : phase_coeffs[p]) {
                            h = dist(gen);
                        }
                    }
                    auto blocked_coeffs = pfbc::prepare_blocked_coeffs(phase_coeffs, M, K);
                    auto coeffs = composite::make_aligned<float>(64, blocked_coeffs.size());
                    std::copy(blocked_coeffs.begin(), blocked_coeffs.end(), coeffs->begin());

                    // Allocate outputs for reference
                    std::vector<std::vector<cf32_t>> expected_channels(M);
                    std::vector<cf32_t*> expected_ptrs(M);
                    for (std::size_t ch = 0; ch < M; ++ch) {
                        expected_channels[ch].resize(num_outputs);
                        expected_ptrs[ch] = expected_channels[ch].data();
                    }

                    // Allocate outputs for hybrid
                    std::vector<std::vector<cf32_t>> actual_channels(M);
                    std::vector<cf32_t*> actual_ptrs(M);
                    for (std::size_t ch = 0; ch < M; ++ch) {
                        actual_channels[ch].resize(num_outputs);
                        actual_ptrs[ch] = actual_channels[ch].data();
                    }

                    // Scratch buffer for hybrid kernel
                    alignas(64) cf32_t scratch[4 * M];

                    // Run reference
                    m32_staged_reference(
                        input.data(), coeffs->data(),
                        expected_ptrs.data(), num_outputs, K);

                    // Run hybrid
                    kernels::filter_fft_transpose_32ch_hybrid(
                        input.data(), coeffs->data(),
                        actual_ptrs.data(), scratch, num_outputs, K);

                    // Compare outputs
                    for (std::size_t ch = 0; ch < M; ++ch) {
                        for (std::size_t t = 0; t < num_outputs; ++t) {
                            INFO("K=" << K << ", num_outputs=" << num_outputs
                                 << ", ch=" << ch << ", t=" << t);
                            CHECK(actual_channels[ch][t].real() ==
                                  Approx(expected_channels[ch][t].real()).epsilon(1e-4).margin(3e-5));
                            CHECK(actual_channels[ch][t].imag() ==
                                  Approx(expected_channels[ch][t].imag()).epsilon(1e-4).margin(3e-5));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("filter_fft_transpose_32ch_hybrid - known DFT values", "[pfbc][kernels][m32_hybrid]") {
    constexpr std::size_t M = 32;
    constexpr std::size_t K = 1;  // Trivial filter (pass-through)
    constexpr std::size_t num_outputs = 8;

    // Create pass-through filter (all 1s)
    std::vector<std::vector<float>> phase_coeffs(M);
    for (std::size_t p = 0; p < M; ++p) {
        phase_coeffs[p] = {1.0f};
    }
    auto blocked_coeffs = pfbc::prepare_blocked_coeffs(phase_coeffs, M, K);
    auto coeffs = composite::make_aligned<float>(64, blocked_coeffs.size());
    std::copy(blocked_coeffs.begin(), blocked_coeffs.end(), coeffs->begin());

    // Allocate outputs
    std::vector<std::vector<cf32_t>> channels(M);
    std::vector<cf32_t*> channel_ptrs(M);
    for (std::size_t ch = 0; ch < M; ++ch) {
        channels[ch].resize(num_outputs);
        channel_ptrs[ch] = channels[ch].data();
    }

    alignas(64) cf32_t scratch[4 * M];

    SECTION("DC input - all ones") {
        std::vector<cf32_t> input(num_outputs * M, cf32_t{1.0f, 0.0f});

        kernels::filter_fft_transpose_32ch_hybrid(
            input.data(), coeffs->data(),
            channel_ptrs.data(), scratch, num_outputs, K);

        // DFT of all-ones: X[0] = N, X[k] = 0 for k > 0
        for (std::size_t t = 0; t < num_outputs; ++t) {
            INFO("t=" << t);
            CHECK(channels[0][t].real() == Approx(32.0f).epsilon(1e-4).margin(3e-5));
            CHECK(channels[0][t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            for (std::size_t ch = 1; ch < M; ++ch) {
                INFO("ch=" << ch);
                CHECK(channels[ch][t].real() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
                CHECK(channels[ch][t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }

    SECTION("Impulse at channel 0") {
        std::vector<cf32_t> input(num_outputs * M, cf32_t{0.0f, 0.0f});
        // Set channel 0 to 1 for each time sample
        for (std::size_t t = 0; t < num_outputs; ++t) {
            input[t * M + 0] = cf32_t{1.0f, 0.0f};
        }

        kernels::filter_fft_transpose_32ch_hybrid(
            input.data(), coeffs->data(),
            channel_ptrs.data(), scratch, num_outputs, K);

        // DFT of impulse at n=0: X[k] = 1 for all k
        for (std::size_t t = 0; t < num_outputs; ++t) {
            for (std::size_t ch = 0; ch < M; ++ch) {
                INFO("t=" << t << ", ch=" << ch);
                CHECK(channels[ch][t].real() == Approx(1.0f).epsilon(1e-4).margin(3e-5));
                CHECK(channels[ch][t].imag() == Approx(0.0f).epsilon(1e-4).margin(3e-5));
            }
        }
    }
}

// =============================================================================
// COEFFICIENT LAYOUT TESTS
// =============================================================================

TEST_CASE("M=8 coefficient layouts are equivalent", "[pfbc][kernels][coefficients]") {
    // For M=8, interleaved and blocked layouts produce identical memory layout.
    // This is because:
    //   - Interleaved: [tap0: h0,h0,h1,h1,...,h7,h7][tap1: ...] stride = M*2 = 16
    //   - Blocked (1 block): [tap0: h0,h0,...,h7,h7][tap1: ...] stride = 16
    // This test explicitly verifies they're identical.
    constexpr std::size_t M = 8;

    for (std::size_t K : {1, 4, 8, 16}) {
        SECTION("K=" + std::to_string(K)) {
            std::mt19937 gen(42 + K);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

            std::vector<std::vector<float>> phase_coeffs(M);
            for (std::size_t p = 0; p < M; ++p) {
                phase_coeffs[p].resize(K);
                for (auto& h : phase_coeffs[p]) {
                    h = dist(gen);
                }
            }

            auto interleaved = pfbc::prepare_interleaved_coeffs(phase_coeffs, M, K);
            auto blocked = pfbc::prepare_blocked_coeffs(phase_coeffs, M, K);

            REQUIRE(interleaved.size() == blocked.size());
            for (std::size_t i = 0; i < interleaved.size(); ++i) {
                INFO("index=" << i << ", K=" << K);
                CHECK(interleaved[i] == blocked[i]);
            }
        }
    }
}

TEST_CASE("filter_fft_transpose_8ch_vertical - correctness", "[pfbc][kernels][fft8_vertical]") {
    // Test fused vertical kernel against staged reference: filter_blocked + FFTW + naive_transpose
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr std::size_t M = 8;

    for (std::size_t K : {1, 4, 8, 12}) {
        for (std::size_t num_outputs : {8, 16, 64, 100}) {
            SECTION("K=" + std::to_string(K) + ", outputs=" + std::to_string(num_outputs)) {
                const auto seed = static_cast<unsigned>(12321 + K * 1000 + num_outputs);
                std::mt19937 gen(seed);

                // Generate random input
                std::vector<cf32_t> input((K + num_outputs - 1) * M);
                for (auto& s : input) {
                    s = cf32_t{dist(gen), dist(gen)};
                }

                // Generate random coefficients
                std::vector<std::vector<float>> phase_coeffs(M);
                for (std::size_t p = 0; p < M; ++p) {
                    phase_coeffs[p].resize(K);
                    for (auto& h : phase_coeffs[p]) {
                        h = dist(gen);
                    }
                }
                auto blocked = pfbc::prepare_blocked_coeffs(phase_coeffs, M, K);
                auto coeffs = composite::make_aligned<float>(64, blocked.size());
                std::copy(blocked.begin(), blocked.end(), coeffs->begin());

                // === REFERENCE: filter_blocked + FFTW + naive_transpose ===
                std::vector<cf32_t> filter_output(num_outputs * M);
                kernels::filter_blocked(
                    input.data(), coeffs->data(), filter_output.data(), M, num_outputs, K);

                std::vector<cf32_t> fft_output(num_outputs * M);
                for (std::size_t t = 0; t < num_outputs; ++t) {
                    fft8_reference(filter_output.data() + t * M, fft_output.data() + t * M);
                }

                std::vector<std::vector<cf32_t>> expected_channels(M);
                std::vector<cf32_t*> expected_ptrs(M);
                for (std::size_t ch = 0; ch < M; ++ch) {
                    expected_channels[ch].resize(num_outputs);
                    expected_ptrs[ch] = expected_channels[ch].data();
                }
                naive_transpose_to_channels(fft_output.data(), expected_ptrs.data(), M, num_outputs);

                // === FUSED VERTICAL KERNEL (under test) ===
                std::vector<std::vector<cf32_t>> actual_channels(M);
                std::vector<cf32_t*> actual_ptrs(M);
                for (std::size_t ch = 0; ch < M; ++ch) {
                    actual_channels[ch].resize(num_outputs);
                    actual_ptrs[ch] = actual_channels[ch].data();
                }
                kernels::filter_fft_transpose_8ch_vertical(
                    input.data(), coeffs->data(), actual_ptrs.data(), num_outputs, K);

                // === COMPARE against scalar/FFTW reference ===
                for (std::size_t ch = 0; ch < M; ++ch) {
                    for (std::size_t t = 0; t < num_outputs; ++t) {
                        INFO("K=" << K << ", outputs=" << num_outputs << ", ch=" << ch << ", t=" << t);
                        CHECK(actual_channels[ch][t].real() == Approx(expected_channels[ch][t].real()).epsilon(2e-4).margin(3e-5));
                        CHECK(actual_channels[ch][t].imag() == Approx(expected_channels[ch][t].imag()).epsilon(2e-4).margin(3e-5));
                    }
                }
            }
        }
    }
}

// =============================================================================
// FUSED FILTER+FFT+TRANSPOSE (M=16, VERTICAL) TESTS
// =============================================================================

TEST_CASE("filter_fft_transpose_16ch_vertical - correctness", "[pfbc][kernels][fft16_vertical]") {
    // Test fused vertical kernel against staged reference
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr std::size_t M = 16;
    constexpr std::size_t K = 4;

    for (std::size_t num_outputs : {8, 16, 24, 32, 64}) {
        SECTION("num_outputs=" + std::to_string(num_outputs)) {
            std::mt19937 gen(99999 + static_cast<unsigned>(num_outputs));
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
            auto blocked = pfbc::prepare_blocked_coeffs(phase_coeffs, M, K);
            auto coeffs = composite::make_aligned<float>(64, blocked.size());
            std::copy(blocked.begin(), blocked.end(), coeffs->begin());

            // Reference (using blocked filter to match component)
            std::vector<cf32_t> filter_output(num_outputs * M);
            kernels::filter_blocked(
                input.data(), coeffs->data(), filter_output.data(), M, num_outputs, K);

            std::vector<cf32_t> fft_output(num_outputs * M);
            for (std::size_t t = 0; t < num_outputs; ++t) {
                fft16_reference(filter_output.data() + t * M, fft_output.data() + t * M);
            }

            std::vector<std::vector<cf32_t>> expected_channels(M);
            std::vector<cf32_t*> expected_ptrs(M);
            for (std::size_t ch = 0; ch < M; ++ch) {
                expected_channels[ch].resize(num_outputs);
                expected_ptrs[ch] = expected_channels[ch].data();
            }
            naive_transpose_to_channels(fft_output.data(), expected_ptrs.data(), M, num_outputs);

            // Fused vertical kernel
            std::vector<std::vector<cf32_t>> actual_channels(M);
            std::vector<cf32_t*> actual_ptrs(M);
            for (std::size_t ch = 0; ch < M; ++ch) {
                actual_channels[ch].resize(num_outputs);
                actual_ptrs[ch] = actual_channels[ch].data();
            }
            kernels::filter_fft_transpose_16ch_vertical(
                input.data(), coeffs->data(), actual_ptrs.data(), num_outputs, K);

            // Compare
            for (std::size_t ch = 0; ch < M; ++ch) {
                for (std::size_t t = 0; t < num_outputs; ++t) {
                    INFO("ch=" << ch << ", t=" << t);
                    CHECK(actual_channels[ch][t].real() == Approx(expected_channels[ch][t].real()).epsilon(2e-4).margin(3e-5));
                    CHECK(actual_channels[ch][t].imag() == Approx(expected_channels[ch][t].imag()).epsilon(2e-4).margin(3e-5));
                }
            }
        }
    }
}

TEST_CASE("filter_fft_transpose_16ch_vertical - tail handling", "[pfbc][kernels][fft16_vertical]") {
    // Test fused vertical kernel with non-multiple-of-8 output counts
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr std::size_t M = 16;
    constexpr std::size_t K = 4;

    for (std::size_t num_outputs : {1, 3, 5, 7, 9, 13, 17, 31, 33}) {
        SECTION("num_outputs=" + std::to_string(num_outputs)) {
            std::mt19937 gen(88888 + static_cast<unsigned>(num_outputs));
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
            auto blocked = pfbc::prepare_blocked_coeffs(phase_coeffs, M, K);
            auto coeffs = composite::make_aligned<float>(64, blocked.size());
            std::copy(blocked.begin(), blocked.end(), coeffs->begin());

            // Reference
            std::vector<cf32_t> filter_output(num_outputs * M);
            kernels::filter_blocked(
                input.data(), coeffs->data(), filter_output.data(), M, num_outputs, K);

            std::vector<cf32_t> fft_output(num_outputs * M);
            for (std::size_t t = 0; t < num_outputs; ++t) {
                fft16_reference(filter_output.data() + t * M, fft_output.data() + t * M);
            }

            std::vector<std::vector<cf32_t>> expected_channels(M);
            std::vector<cf32_t*> expected_ptrs(M);
            for (std::size_t ch = 0; ch < M; ++ch) {
                expected_channels[ch].resize(num_outputs);
                expected_ptrs[ch] = expected_channels[ch].data();
            }
            naive_transpose_to_channels(fft_output.data(), expected_ptrs.data(), M, num_outputs);

            // Fused vertical kernel
            std::vector<std::vector<cf32_t>> actual_channels(M);
            std::vector<cf32_t*> actual_ptrs(M);
            for (std::size_t ch = 0; ch < M; ++ch) {
                actual_channels[ch].resize(num_outputs);
                actual_ptrs[ch] = actual_channels[ch].data();
            }
            kernels::filter_fft_transpose_16ch_vertical(
                input.data(), coeffs->data(), actual_ptrs.data(), num_outputs, K);

            // Compare
            for (std::size_t ch = 0; ch < M; ++ch) {
                for (std::size_t t = 0; t < num_outputs; ++t) {
                    INFO("num_outputs=" << num_outputs << ", ch=" << ch << ", t=" << t);
                    CHECK(actual_channels[ch][t].real() == Approx(expected_channels[ch][t].real()).epsilon(2e-4).margin(3e-5));
                    CHECK(actual_channels[ch][t].imag() == Approx(expected_channels[ch][t].imag()).epsilon(2e-4).margin(3e-5));
                }
            }
        }
    }
}
