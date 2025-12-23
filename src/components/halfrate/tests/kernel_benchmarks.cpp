// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include <benchmark/benchmark.h>
#include "../kernels.hpp"

#include <complex>
#include <vector>
#include <random>

using sample_t = std::complex<float>;

// =============================================================================
// Benchmark Utilities
// =============================================================================

auto generate_random_data(std::size_t count) -> std::vector<sample_t> {
    std::vector<sample_t> data(count);
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& s : data) {
        s = {dist(gen), dist(gen)};
    }
    return data;
}

// =============================================================================
// Deinterleave Kernel Benchmarks
// =============================================================================

static void BM_Deinterleave_1K(benchmark::State& state) {
    constexpr std::size_t num_pairs = 512;  // 1024 input samples
    auto src = generate_random_data(num_pairs * 2);
    std::vector<sample_t> dst_even(num_pairs);
    std::vector<sample_t> dst_odd(num_pairs);

    for (auto _ : state) {
        kernels::deinterleave_block(src.data(), dst_even.data(), dst_odd.data(), num_pairs);
        benchmark::DoNotOptimize(dst_even.data());
        benchmark::DoNotOptimize(dst_odd.data());
        benchmark::ClobberMemory();
    }

    // Report throughput
    state.SetBytesProcessed(state.iterations() * num_pairs * 2 * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_pairs * 2); // input samples processed
}
BENCHMARK(BM_Deinterleave_1K);

static void BM_Deinterleave_16K(benchmark::State& state) {
    constexpr std::size_t num_pairs = 8192;  // 16K input samples
    auto src = generate_random_data(num_pairs * 2);
    std::vector<sample_t> dst_even(num_pairs);
    std::vector<sample_t> dst_odd(num_pairs);

    for (auto _ : state) {
        kernels::deinterleave_block(src.data(), dst_even.data(), dst_odd.data(), num_pairs);
        benchmark::DoNotOptimize(dst_even.data());
        benchmark::DoNotOptimize(dst_odd.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_pairs * 2 * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_pairs * 2);
}
BENCHMARK(BM_Deinterleave_16K);

static void BM_Deinterleave_256K(benchmark::State& state) {
    constexpr std::size_t num_pairs = 131072;  // 256K input samples
    auto src = generate_random_data(num_pairs * 2);
    std::vector<sample_t> dst_even(num_pairs);
    std::vector<sample_t> dst_odd(num_pairs);

    for (auto _ : state) {
        kernels::deinterleave_block(src.data(), dst_even.data(), dst_odd.data(), num_pairs);
        benchmark::DoNotOptimize(dst_even.data());
        benchmark::DoNotOptimize(dst_odd.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_pairs * 2 * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_pairs * 2);
}
BENCHMARK(BM_Deinterleave_256K);

// Parameterized benchmark for different sizes
static void BM_Deinterleave_Parameterized(benchmark::State& state) {
    const std::size_t num_pairs = state.range(0);
    auto src = generate_random_data(num_pairs * 2);
    std::vector<sample_t> dst_even(num_pairs);
    std::vector<sample_t> dst_odd(num_pairs);

    for (auto _ : state) {
        kernels::deinterleave_block(src.data(), dst_even.data(), dst_odd.data(), num_pairs);
        benchmark::DoNotOptimize(dst_even.data());
        benchmark::DoNotOptimize(dst_odd.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_pairs * 2 * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_pairs * 2);
    state.SetLabel("pairs=" + std::to_string(num_pairs));
}
// Test powers of 2 from 64 to 256K pairs
BENCHMARK(BM_Deinterleave_Parameterized)
    ->RangeMultiplier(4)
    ->Range(64, 262144)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// Halfband Filter Kernel Benchmarks
// =============================================================================

static void BM_HalfbandFilter_1K_6Taps(benchmark::State& state) {
    constexpr std::size_t num_outputs = 1024;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 2;
    constexpr float center_tap = 0.5f;

    auto even_hist = generate_random_data(num_outputs + num_taps);
    auto odd_hist = generate_random_data(num_outputs + delay_offset);
    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    for (auto _ : state) {
        kernels::halfband_filter_vertical(
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_outputs * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_outputs);
}
BENCHMARK(BM_HalfbandFilter_1K_6Taps);

static void BM_HalfbandFilter_16K_6Taps(benchmark::State& state) {
    constexpr std::size_t num_outputs = 16384;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 2;
    constexpr float center_tap = 0.5f;

    auto even_hist = generate_random_data(num_outputs + num_taps);
    auto odd_hist = generate_random_data(num_outputs + delay_offset);
    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    for (auto _ : state) {
        kernels::halfband_filter_vertical(
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_outputs * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_outputs);
}
BENCHMARK(BM_HalfbandFilter_16K_6Taps);

static void BM_HalfbandFilter_256K_6Taps(benchmark::State& state) {
    constexpr std::size_t num_outputs = 262144;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 2;
    constexpr float center_tap = 0.5f;

    auto even_hist = generate_random_data(num_outputs + num_taps);
    auto odd_hist = generate_random_data(num_outputs + delay_offset);
    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    for (auto _ : state) {
        kernels::halfband_filter_vertical(
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_outputs * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_outputs);
}
BENCHMARK(BM_HalfbandFilter_256K_6Taps);

static void BM_HalfbandFilter_16K_12Taps(benchmark::State& state) {
    constexpr std::size_t num_outputs = 16384;
    constexpr std::size_t num_taps = 12;
    constexpr std::size_t delay_offset = 5;
    constexpr float center_tap = 0.5f;

    auto even_hist = generate_random_data(num_outputs + num_taps);
    auto odd_hist = generate_random_data(num_outputs + delay_offset);
    std::vector<float> coeffs(num_taps, 0.08f);
    std::vector<sample_t> output(num_outputs);

    for (auto _ : state) {
        kernels::halfband_filter_vertical(
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_outputs * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_outputs);
}
BENCHMARK(BM_HalfbandFilter_16K_12Taps);

// Parameterized benchmark for different sizes and tap counts
static void BM_HalfbandFilter_Parameterized(benchmark::State& state) {
    const std::size_t num_outputs = state.range(0);
    const std::size_t num_taps = state.range(1);
    constexpr std::size_t delay_offset = 2;
    constexpr float center_tap = 0.5f;

    auto even_hist = generate_random_data(num_outputs + num_taps);
    auto odd_hist = generate_random_data(num_outputs + delay_offset + num_outputs);
    std::vector<float> coeffs(num_taps, 1.0f / num_taps);
    std::vector<sample_t> output(num_outputs);

    for (auto _ : state) {
        kernels::halfband_filter_vertical(
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_outputs * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_outputs);
    state.SetLabel("outputs=" + std::to_string(num_outputs) + " taps=" + std::to_string(num_taps));
}
// Test different combinations of output sizes and tap counts
BENCHMARK(BM_HalfbandFilter_Parameterized)
    ->Args({1024, 6})
    ->Args({1024, 12})
    ->Args({16384, 6})
    ->Args({16384, 12})
    ->Args({16384, 24})
    ->Args({262144, 6})
    ->Args({262144, 12})
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// Cache Behavior Benchmarks
// =============================================================================

static void BM_Deinterleave_CacheBehavior(benchmark::State& state) {
    const std::size_t num_pairs = state.range(0);
    auto src = generate_random_data(num_pairs * 2);
    std::vector<sample_t> dst_even(num_pairs);
    std::vector<sample_t> dst_odd(num_pairs);

    for (auto _ : state) {
        kernels::deinterleave_block(src.data(), dst_even.data(), dst_odd.data(), num_pairs);
        benchmark::DoNotOptimize(dst_even.data());
        benchmark::DoNotOptimize(dst_odd.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_pairs * 2 * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_pairs * 2);

    // Calculate approximate cache level
    std::size_t bytes = num_pairs * 2 * sizeof(sample_t);
    std::string cache_level;
    if (bytes < 32 * 1024) cache_level = "L1";
    else if (bytes < 256 * 1024) cache_level = "L2";
    else if (bytes < 8 * 1024 * 1024) cache_level = "L3";
    else cache_level = "DRAM";

    state.SetLabel(cache_level + " (" + std::to_string(bytes / 1024) + " KB)");
}
BENCHMARK(BM_Deinterleave_CacheBehavior)
    ->Arg(64)        // L1: ~1 KB
    ->Arg(2048)      // L2: ~32 KB
    ->Arg(32768)     // L3: ~512 KB
    ->Arg(524288)    // DRAM: ~8 MB
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
