// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include <benchmark/benchmark.h>
#include "../coefficients.hpp"
#include "../kernels.hpp"

#include <complex>
#include <vector>
#include <random>
#include <memory>
#include <cstring>

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

// Helper to create 64-byte aligned buffer
template<typename T>
struct aligned_buffer {
    T* ptr = nullptr;
    std::size_t size = 0;

    aligned_buffer(std::size_t n) : size(n) {
        // aligned_alloc requires size to be a multiple of alignment
        std::size_t alloc_size = n * sizeof(T);
        alloc_size = ((alloc_size + 63) / 64) * 64;  // Round up to multiple of 64
        ptr = static_cast<T*>(std::aligned_alloc(64, alloc_size));
        if (!ptr) throw std::bad_alloc();
        // Default construct elements
        for (std::size_t i = 0; i < n; ++i) {
            new (&ptr[i]) T();
        }
    }

    ~aligned_buffer() {
        if (ptr) {
            for (std::size_t i = 0; i < size; ++i) {
                ptr[i].~T();
            }
            std::free(ptr);
        }
    }

    aligned_buffer(const aligned_buffer&) = delete;
    aligned_buffer& operator=(const aligned_buffer&) = delete;

    aligned_buffer(aligned_buffer&& other) noexcept : ptr(other.ptr), size(other.size) {
        other.ptr = nullptr;
        other.size = 0;
    }

    T* data() { return ptr; }
    const T* data() const { return ptr; }
    T* begin() { return ptr; }
    T* end() { return ptr + size; }
    const T* begin() const { return ptr; }
    const T* end() const { return ptr + size; }
};

template<typename T>
auto make_aligned_vector(std::size_t size) -> aligned_buffer<T> {
    return aligned_buffer<T>(size);
}

// Helper to generate interleaved coefficients
auto generate_interleaved_coeffs(std::size_t M, std::size_t K) -> std::vector<float> {
    // Generate phase-major coefficients first
    std::vector<std::vector<float>> phase_coeffs(M);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t p = 0; p < M; ++p) {
        phase_coeffs[p].resize(K);
        for (std::size_t k = 0; k < K; ++k) {
            phase_coeffs[p][k] = dist(gen);
        }
    }

    return pfbc::prepare_interleaved_coeffs(phase_coeffs, M, K);
}

// =============================================================================
// INTERLEAVED KERNEL BENCHMARKS
// =============================================================================

// -----------------------------------------------------------------------------
// Fixed Configuration Benchmarks
// -----------------------------------------------------------------------------

static void BM_Interleaved_Filter_M64_K16(benchmark::State& state) {
    constexpr std::size_t M = 64;
    constexpr std::size_t K = 16;
    constexpr std::size_t num_outputs = 8192;

    // Input: (K + num_outputs - 1) rows of M interleaved samples
    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    // Coefficients: [K × M × 2] doubled
    auto coeffs = generate_interleaved_coeffs(M, K);

    // Output: [num_outputs × M]
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_interleaved(
            input.data(),
            coeffs.data(),
            output.data(),
            M,
            num_outputs,
            K
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    // Report: input samples processed per second
    const auto total_input_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_input_samples * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * total_input_samples);
    state.counters["Msps"] = benchmark::Counter(
        total_input_samples,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_Interleaved_Filter_M64_K16);

static void BM_Interleaved_Filter_M64_K32(benchmark::State& state) {
    constexpr std::size_t M = 64;
    constexpr std::size_t K = 32;
    constexpr std::size_t num_outputs = 8192;

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_interleaved_coeffs(M, K);
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_interleaved(
            input.data(),
            coeffs.data(),
            output.data(),
            M,
            num_outputs,
            K
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const auto total_input_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_input_samples * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * total_input_samples);
    state.counters["Msps"] = benchmark::Counter(
        total_input_samples,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_Interleaved_Filter_M64_K32);

// -----------------------------------------------------------------------------
// Parameterized Output Count (cache behavior analysis)
// -----------------------------------------------------------------------------

static void BM_Interleaved_Filter_M64_K16_Outputs(benchmark::State& state) {
    constexpr std::size_t M = 64;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_interleaved_coeffs(M, K);
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_interleaved(
            input.data(),
            coeffs.data(),
            output.data(),
            M,
            num_outputs,
            K
        );
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const auto total_input_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_input_samples * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * total_input_samples);
    state.counters["Msps"] = benchmark::Counter(
        total_input_samples,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_Interleaved_Filter_M64_K16_Outputs)
    ->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192)->Arg(16384);

// -----------------------------------------------------------------------------
// Parameterized Channel Count (M = 8, 16, 32, 64, 128, 256)
// -----------------------------------------------------------------------------

static void BM_Interleaved_Parameterized_M(benchmark::State& state) {
    const std::size_t M = state.range(0);
    constexpr std::size_t K = 16;
    constexpr std::size_t num_outputs = 4096;

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_interleaved_coeffs(M, K);
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_interleaved(
            input.data(), coeffs.data(), output.data(), M, num_outputs, K);

        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const auto total_input_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_input_samples * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * total_input_samples);
    state.counters["Msps"] = benchmark::Counter(
        total_input_samples,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_Interleaved_Parameterized_M)
    ->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256);

// -----------------------------------------------------------------------------
// Parameterized Tap Count (K = 4, 8, 16, 32, 64)
// -----------------------------------------------------------------------------

static void BM_Interleaved_Parameterized_K(benchmark::State& state) {
    constexpr std::size_t M = 64;
    const std::size_t K = state.range(0);
    constexpr std::size_t num_outputs = 4096;

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_interleaved_coeffs(M, K);
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_interleaved(
            input.data(), coeffs.data(), output.data(), M, num_outputs, K);

        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const auto total_input_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_input_samples * sizeof(sample_t));
    state.SetItemsProcessed(state.iterations() * total_input_samples);
    state.counters["Msps"] = benchmark::Counter(
        total_input_samples,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_Interleaved_Parameterized_K)
    ->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

BENCHMARK_MAIN();
