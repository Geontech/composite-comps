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
// Fused Halfband Filter Benchmarks
// =============================================================================

static void BM_HalfbandFused_1K_6Taps(benchmark::State& state) {
    constexpr std::size_t num_outputs = 1024;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 3;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    auto input = generate_random_data(num_outputs * 2);
    std::vector<sample_t> even_hist(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist(history_len, {0.0f, 0.0f});
    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    for (auto _ : state) {
        kernels::halfband_filter_fused(
            input.data(),
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs,
            history_len
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_outputs * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_outputs);
}
BENCHMARK(BM_HalfbandFused_1K_6Taps);

static void BM_HalfbandFused_16K_6Taps(benchmark::State& state) {
    constexpr std::size_t num_outputs = 16384;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 3;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    auto input = generate_random_data(num_outputs * 2);
    std::vector<sample_t> even_hist(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist(history_len, {0.0f, 0.0f});
    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    for (auto _ : state) {
        kernels::halfband_filter_fused(
            input.data(),
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs,
            history_len
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_outputs * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_outputs);
}
BENCHMARK(BM_HalfbandFused_16K_6Taps);

static void BM_HalfbandFused_256K_6Taps(benchmark::State& state) {
    constexpr std::size_t num_outputs = 262144;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 3;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    auto input = generate_random_data(num_outputs * 2);
    std::vector<sample_t> even_hist(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist(history_len, {0.0f, 0.0f});
    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    for (auto _ : state) {
        kernels::halfband_filter_fused(
            input.data(),
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs,
            history_len
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_outputs * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_outputs);
}
BENCHMARK(BM_HalfbandFused_256K_6Taps);

static void BM_HalfbandFused_16K_12Taps(benchmark::State& state) {
    constexpr std::size_t num_outputs = 16384;
    constexpr std::size_t num_taps = 12;
    constexpr std::size_t delay_offset = 6;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    auto input = generate_random_data(num_outputs * 2);
    std::vector<sample_t> even_hist(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist(history_len, {0.0f, 0.0f});
    std::vector<float> coeffs(num_taps, 0.08f);
    std::vector<sample_t> output(num_outputs);

    for (auto _ : state) {
        kernels::halfband_filter_fused(
            input.data(),
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs,
            history_len
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_outputs * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_outputs);
}
BENCHMARK(BM_HalfbandFused_16K_12Taps);

// =============================================================================
// Parameterized Benchmark
// =============================================================================

static void BM_HalfbandFused_Parameterized(benchmark::State& state) {
    const std::size_t num_outputs = state.range(0);
    const std::size_t num_taps = state.range(1);
    const std::size_t delay_offset = num_taps / 2;
    const std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    auto input = generate_random_data(num_outputs * 2);
    std::vector<sample_t> even_hist(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist(history_len, {0.0f, 0.0f});
    std::vector<float> coeffs(num_taps, 1.0f / num_taps);
    std::vector<sample_t> output(num_outputs);

    for (auto _ : state) {
        kernels::halfband_filter_fused(
            input.data(),
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs,
            history_len
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_outputs * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_outputs);
    state.SetLabel("outputs=" + std::to_string(num_outputs) + " taps=" + std::to_string(num_taps));
}
BENCHMARK(BM_HalfbandFused_Parameterized)
    ->Args({1024, 6})
    ->Args({1024, 12})
    ->Args({16384, 6})
    ->Args({16384, 12})
    ->Args({262144, 6})
    ->Args({262144, 12})
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// Cache Behavior Benchmark
// =============================================================================

static void BM_HalfbandFused_CacheBehavior(benchmark::State& state) {
    const std::size_t num_outputs = state.range(0);
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 3;
    constexpr std::size_t history_len = std::max(num_taps, delay_offset);
    constexpr float center_tap = 0.5f;

    auto input = generate_random_data(num_outputs * 2);
    std::vector<sample_t> even_hist(history_len, {0.0f, 0.0f});
    std::vector<sample_t> odd_hist(history_len, {0.0f, 0.0f});
    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    for (auto _ : state) {
        kernels::halfband_filter_fused(
            input.data(),
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs,
            history_len
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetBytesProcessed(state.iterations() * num_outputs * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * num_outputs);

    // Calculate approximate cache level (input size dominates)
    std::size_t bytes = num_outputs * 2 * sizeof(sample_t);
    std::string cache_level;
    if (bytes < 32 * 1024) cache_level = "L1";
    else if (bytes < 256 * 1024) cache_level = "L2";
    else if (bytes < 8 * 1024 * 1024) cache_level = "L3";
    else cache_level = "DRAM";

    state.SetLabel(cache_level + " (" + std::to_string(bytes / 1024) + " KB)");
}
BENCHMARK(BM_HalfbandFused_CacheBehavior)
    ->Arg(512)       // L1: ~8 KB
    ->Arg(2048)      // L1: ~32 KB
    ->Arg(16384)     // L2: ~256 KB
    ->Arg(131072)    // L3: ~2 MB
    ->Arg(524288)    // DRAM: ~8 MB
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
