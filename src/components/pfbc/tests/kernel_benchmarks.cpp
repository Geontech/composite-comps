// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include <benchmark/benchmark.h>
#include "../coefficients.hpp"
#include "../kernels.hpp"
#include "../pfbc_engine.hpp"

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

// Helper to generate blocked coefficients (64-byte aligned)
auto generate_blocked_coeffs(std::size_t M, std::size_t K) -> aligned_buffer<float> {
    std::vector<std::vector<float>> phase_coeffs(M);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t p = 0; p < M; ++p) {
        phase_coeffs[p].resize(K);
        for (std::size_t k = 0; k < K; ++k) {
            phase_coeffs[p][k] = dist(gen);
        }
    }

    auto blocked = pfbc::prepare_blocked_coeffs(phase_coeffs, M, K);
    aligned_buffer<float> result(blocked.size());
    std::copy(blocked.begin(), blocked.end(), result.data());
    return result;
}

// =============================================================================
// M=8 FUSED KERNEL BENCHMARKS (Filter + FFT + Transpose)
// =============================================================================

#include "../fft_plan.hpp"  // For FFTW comparison
#include <fftw3.h>

// M=8 Fused kernel with VERTICAL FFT-8: filter + 8x8 transpose + vertical FFT-8 + direct store
static void BM_Fused_M8_Vertical_Filter_FFT_Transpose(benchmark::State& state) {
    constexpr std::size_t M = 8;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    // Input: (K + num_outputs - 1) rows of M interleaved samples
    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    // Coefficients: blocked (for M=8, identical to interleaved layout)
    auto coeffs = generate_blocked_coeffs(M, K);

    // Output: 8 channel buffers
    std::array<aligned_buffer<sample_t>, 8> channels = {
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs)
    };

    std::array<sample_t*, 8> channel_ptrs;
    for (std::size_t i = 0; i < 8; ++i) {
        channel_ptrs[i] = channels[i].data();
    }

    for (auto _ : state) {
        kernels::filter_fft_transpose_8ch_vertical(
            input.data(),
            coeffs.data(),
            channel_ptrs.data(),
            num_outputs,
            K
        );
        benchmark::DoNotOptimize(channel_ptrs[0]);
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
BENCHMARK(BM_Fused_M8_Vertical_Filter_FFT_Transpose)
    ->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192);

// M=16 Fused kernel with VERTICAL FFT-16: filter + 2x 8x8 transpose + vertical FFT-16 + direct store
static void BM_Fused_M16_Vertical_Filter_FFT_Transpose(benchmark::State& state) {
    constexpr std::size_t M = 16;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    // Input: (K + num_outputs - 1) rows of M interleaved samples
    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    // Coefficients: blocked layout (same as general pipeline uses)
    auto coeffs = generate_blocked_coeffs(M, K);

    // Output: 16 channel buffers
    std::array<aligned_buffer<sample_t>, 16> channels = {
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs)
    };

    std::array<sample_t*, 16> channel_ptrs;
    for (std::size_t i = 0; i < 16; ++i) {
        channel_ptrs[i] = channels[i].data();
    }

    for (auto _ : state) {
        kernels::filter_fft_transpose_16ch_vertical(
            input.data(),
            coeffs.data(),
            channel_ptrs.data(),
            num_outputs,
            K
        );
        benchmark::DoNotOptimize(channel_ptrs[0]);
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
BENCHMARK(BM_Fused_M16_Vertical_Filter_FFT_Transpose)
    ->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192);

// M=16 Separate stages: filter -> FFTW -> transpose (for comparison)
static void BM_Separate_M16_Filter_FFTW_Transpose(benchmark::State& state) {
    constexpr std::size_t M = 16;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto filter_output = make_aligned_vector<sample_t>(num_outputs * M);
    auto fft = fft_plan<sample_t>(M, 1);

    std::array<aligned_buffer<sample_t>, 16> channels = {
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs)
    };

    alignas(64) sample_t fft_out[16];

    for (auto _ : state) {
        // Stage 1: Filter
        kernels::filter_blocked(
            input.data(), coeffs.data(), filter_output.data(), M, num_outputs, K);

        // Stage 2: FFT each row + transpose to channel buffers
        for (std::size_t t = 0; t < num_outputs; ++t) {
            fft.execute(filter_output.data() + t * M, fft_out);
            for (std::size_t ch = 0; ch < M; ++ch) {
                channels[ch].data()[t] = fft_out[ch];
            }
        }

        benchmark::DoNotOptimize(channels[0].data());
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
BENCHMARK(BM_Separate_M16_Filter_FFTW_Transpose)
    ->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192);

// M=8 Separate stages: filter -> FFTW -> transpose (for comparison)
static void BM_Separate_M8_Filter_FFTW_Transpose(benchmark::State& state) {
    constexpr std::size_t M = 8;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    // Input
    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    // Coefficients (blocked layout)
    auto coeffs = generate_blocked_coeffs(M, K);

    // Filter output (time-major: [num_outputs × M])
    auto filter_output = make_aligned_vector<sample_t>(num_outputs * M);

    // FFTW setup
    auto fft = fft_plan<sample_t>(M, 1);

    // Channel output buffers
    std::array<aligned_buffer<sample_t>, 8> channels = {
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs),
        make_aligned_vector<sample_t>(num_outputs)
    };

    // Temp buffer for FFT output
    alignas(64) sample_t fft_out[8];

    for (auto _ : state) {
        // Stage 1: Filter
        kernels::filter_blocked(
            input.data(), coeffs.data(), filter_output.data(), M, num_outputs, K);

        // Stage 2: FFT each row + transpose to channel buffers
        for (std::size_t t = 0; t < num_outputs; ++t) {
            fft.execute(filter_output.data() + t * M, fft_out);
            for (std::size_t ch = 0; ch < M; ++ch) {
                channels[ch].data()[t] = fft_out[ch];
            }
        }

        benchmark::DoNotOptimize(channels[0].data());
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
BENCHMARK(BM_Separate_M8_Filter_FFTW_Transpose)
    ->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192);

// =============================================================================
// M=32 HYBRID KERNEL BENCHMARKS
// =============================================================================

// M=32 Hybrid kernel: filter + FFT-32 in registers, transpose via L1 scratch
static void BM_Hybrid_M32_Filter_FFT_Transpose(benchmark::State& state) {
    constexpr std::size_t M = 32;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    // Input: (K + num_outputs - 1) rows of M interleaved samples
    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    // Coefficients: blocked layout
    auto coeffs = generate_blocked_coeffs(M, K);

    // Output: 32 channel buffers
    std::vector<aligned_buffer<sample_t>> channels;
    channels.reserve(32);
    for (std::size_t i = 0; i < 32; ++i) {
        channels.push_back(make_aligned_vector<sample_t>(num_outputs));
    }

    std::array<sample_t*, 32> channel_ptrs;
    for (std::size_t i = 0; i < 32; ++i) {
        channel_ptrs[i] = channels[i].data();
    }

    // Scratch buffer for hybrid kernel (4 rows × 32 channels)
    auto scratch = make_aligned_vector<sample_t>(4 * M);

    for (auto _ : state) {
        kernels::filter_fft_transpose_32ch_hybrid(
            input.data(),
            coeffs.data(),
            channel_ptrs.data(),
            scratch.data(),
            num_outputs,
            K
        );
        benchmark::DoNotOptimize(channel_ptrs[0]);
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
BENCHMARK(BM_Hybrid_M32_Filter_FFT_Transpose)
    ->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192);

// M=32 Separate stages: filter -> FFTW -> transpose (for comparison)
static void BM_Separate_M32_Filter_FFTW_Transpose(benchmark::State& state) {
    constexpr std::size_t M = 32;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto filter_output = make_aligned_vector<sample_t>(num_outputs * M);
    auto fft = fft_plan<sample_t>(M, 1);

    std::vector<aligned_buffer<sample_t>> channels;
    channels.reserve(32);
    for (std::size_t i = 0; i < 32; ++i) {
        channels.push_back(make_aligned_vector<sample_t>(num_outputs));
    }

    alignas(64) sample_t fft_out[32];

    for (auto _ : state) {
        // Stage 1: Filter
        kernels::filter_blocked(
            input.data(), coeffs.data(), filter_output.data(), M, num_outputs, K);

        // Stage 2: FFT each row + transpose to channel buffers
        for (std::size_t t = 0; t < num_outputs; ++t) {
            fft.execute(filter_output.data() + t * M, fft_out);
            for (std::size_t ch = 0; ch < M; ++ch) {
                channels[ch].data()[t] = fft_out[ch];
            }
        }

        benchmark::DoNotOptimize(channels[0].data());
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
BENCHMARK(BM_Separate_M32_Filter_FFTW_Transpose)
    ->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)->Arg(8192);

// Pure FFT-8: Matrix DFT on interleaved data
static void BM_FFT8_Interleaved(benchmark::State& state) {
    const std::size_t count = state.range(0);

    auto input = make_aligned_vector<sample_t>(count * 8);
    auto input_data = generate_random_data(count * 8);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto output = make_aligned_vector<sample_t>(count * 8);

    for (auto _ : state) {
        for (std::size_t i = 0; i < count; ++i) {
            // Direct FFT on interleaved data - no format conversion needed
            __m512 interleaved = _mm512_loadu_ps(reinterpret_cast<const float*>(input.data() + i * 8));
            __m512 result = kernels::fft8::fft8_interleaved_reg(interleaved);
            _mm512_storeu_ps(reinterpret_cast<float*>(output.data() + i * 8), result);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.counters["FFT8/sec"] = benchmark::Counter(
        count,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_FFT8_Interleaved)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_FFT8_FFTW(benchmark::State& state) {
    const std::size_t count = state.range(0);

    auto input = make_aligned_vector<sample_t>(count * 8);
    auto input_data = generate_random_data(count * 8);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto output = make_aligned_vector<sample_t>(count * 8);

    auto fft = fft_plan<sample_t>(8, 1);

    for (auto _ : state) {
        for (std::size_t i = 0; i < count; ++i) {
            fft.execute(input.data() + i * 8, output.data() + i * 8);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.counters["FFT8/sec"] = benchmark::Counter(
        count,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_FFT8_FFTW)->Arg(1024)->Arg(4096)->Arg(16384);

// Vertical FFT-8: 8 parallel FFT-8s per iteration (throughput-optimized)
// Processes 8 FFTs simultaneously using column-major data layout
[[gnu::target("avx512f,avx512dq")]]
static void BM_FFT8_Vertical(benchmark::State& state) {
    const std::size_t num_batches = state.range(0);  // Number of 8-FFT batches
    const std::size_t total_ffts = num_batches * 8;

    // Column-major layout: input[ch * 8 + t] = FFT #t, index ch
    auto input = make_aligned_vector<sample_t>(num_batches * 64);  // 64 samples per batch
    auto input_data = generate_random_data(num_batches * 64);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto output = make_aligned_vector<sample_t>(num_batches * 64);

    for (auto _ : state) {
        for (std::size_t b = 0; b < num_batches; ++b) {
            const float* in_ptr = reinterpret_cast<const float*>(input.data() + b * 64);
            float* out_ptr = reinterpret_cast<float*>(output.data() + b * 64);

            // Load 8 registers (one per frequency index)
            __m512 r0 = _mm512_loadu_ps(in_ptr + 0 * 16);
            __m512 r1 = _mm512_loadu_ps(in_ptr + 1 * 16);
            __m512 r2 = _mm512_loadu_ps(in_ptr + 2 * 16);
            __m512 r3 = _mm512_loadu_ps(in_ptr + 3 * 16);
            __m512 r4 = _mm512_loadu_ps(in_ptr + 4 * 16);
            __m512 r5 = _mm512_loadu_ps(in_ptr + 5 * 16);
            __m512 r6 = _mm512_loadu_ps(in_ptr + 6 * 16);
            __m512 r7 = _mm512_loadu_ps(in_ptr + 7 * 16);

            kernels::fft8_vertical::fft8_vertical(r0, r1, r2, r3, r4, r5, r6, r7);

            _mm512_storeu_ps(out_ptr + 0 * 16, r0);
            _mm512_storeu_ps(out_ptr + 1 * 16, r1);
            _mm512_storeu_ps(out_ptr + 2 * 16, r2);
            _mm512_storeu_ps(out_ptr + 3 * 16, r3);
            _mm512_storeu_ps(out_ptr + 4 * 16, r4);
            _mm512_storeu_ps(out_ptr + 5 * 16, r5);
            _mm512_storeu_ps(out_ptr + 6 * 16, r6);
            _mm512_storeu_ps(out_ptr + 7 * 16, r7);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * total_ffts);
    state.counters["FFT8/sec"] = benchmark::Counter(
        total_ffts,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
// Use same total FFT count as horizontal for fair comparison:
// Horizontal does 1024/4096/16384 FFTs, so vertical does 128/512/2048 batches of 8
BENCHMARK(BM_FFT8_Vertical)->Arg(128)->Arg(512)->Arg(2048);

// Pure FFT-16: Custom AVX-512 implementation on interleaved data
static void BM_FFT16_Interleaved(benchmark::State& state) {
    const std::size_t count = state.range(0);

    auto input = make_aligned_vector<sample_t>(count * 16);
    auto input_data = generate_random_data(count * 16);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto output = make_aligned_vector<sample_t>(count * 16);

    for (auto _ : state) {
        for (std::size_t i = 0; i < count; ++i) {
            // FFT-16 uses 2 ZMM registers (lo/hi for 16 complex samples)
            __m512 lo = _mm512_loadu_ps(reinterpret_cast<const float*>(input.data() + i * 16));
            __m512 hi = _mm512_loadu_ps(reinterpret_cast<const float*>(input.data() + i * 16 + 8));
            kernels::fft16::fft16_interleaved_reg(lo, hi);
            _mm512_storeu_ps(reinterpret_cast<float*>(output.data() + i * 16), lo);
            _mm512_storeu_ps(reinterpret_cast<float*>(output.data() + i * 16 + 8), hi);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.counters["FFT16/sec"] = benchmark::Counter(
        count,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_FFT16_Interleaved)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_FFT16_FFTW(benchmark::State& state) {
    const std::size_t count = state.range(0);

    auto input = make_aligned_vector<sample_t>(count * 16);
    auto input_data = generate_random_data(count * 16);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto output = make_aligned_vector<sample_t>(count * 16);

    auto fft = fft_plan<sample_t>(16, 1);

    for (auto _ : state) {
        for (std::size_t i = 0; i < count; ++i) {
            fft.execute(input.data() + i * 16, output.data() + i * 16);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.counters["FFT16/sec"] = benchmark::Counter(
        count,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_FFT16_FFTW)->Arg(1024)->Arg(4096)->Arg(16384);

// Pure FFT-32: Custom AVX-512 implementation on interleaved data
[[gnu::target("avx512f,avx512vl,avx512dq,fma")]]
static void BM_FFT32_Interleaved(benchmark::State& state) {
    const std::size_t count = state.range(0);

    auto input = make_aligned_vector<sample_t>(count * 32);
    auto input_data = generate_random_data(count * 32);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto output = make_aligned_vector<sample_t>(count * 32);

    for (auto _ : state) {
        for (std::size_t i = 0; i < count; ++i) {
            // FFT-32 uses 4 ZMM registers for 32 complex samples
            __m512 r0 = _mm512_loadu_ps(reinterpret_cast<const float*>(input.data() + i * 32));
            __m512 r1 = _mm512_loadu_ps(reinterpret_cast<const float*>(input.data() + i * 32 + 8));
            __m512 r2 = _mm512_loadu_ps(reinterpret_cast<const float*>(input.data() + i * 32 + 16));
            __m512 r3 = _mm512_loadu_ps(reinterpret_cast<const float*>(input.data() + i * 32 + 24));
            kernels::fft32::fft32_interleaved_reg(r0, r1, r2, r3);
            _mm512_storeu_ps(reinterpret_cast<float*>(output.data() + i * 32), r0);
            _mm512_storeu_ps(reinterpret_cast<float*>(output.data() + i * 32 + 8), r1);
            _mm512_storeu_ps(reinterpret_cast<float*>(output.data() + i * 32 + 16), r2);
            _mm512_storeu_ps(reinterpret_cast<float*>(output.data() + i * 32 + 24), r3);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.counters["FFT32/sec"] = benchmark::Counter(
        count,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_FFT32_Interleaved)->Arg(1024)->Arg(4096)->Arg(16384);

static void BM_FFT32_FFTW(benchmark::State& state) {
    const std::size_t count = state.range(0);

    auto input = make_aligned_vector<sample_t>(count * 32);
    auto input_data = generate_random_data(count * 32);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto output = make_aligned_vector<sample_t>(count * 32);

    auto fft = fft_plan<sample_t>(32, 1);

    for (auto _ : state) {
        for (std::size_t i = 0; i < count; ++i) {
            fft.execute(input.data() + i * 32, output.data() + i * 32);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.counters["FFT32/sec"] = benchmark::Counter(
        count,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_FFT32_FFTW)->Arg(1024)->Arg(4096)->Arg(16384);

// Pure FFT-32 2D: Custom AVX-512 implementation using 2D decomposition (4×8)
[[gnu::target("avx512f,avx512dq,avx,avx2,fma")]]
static void BM_FFT32_2D(benchmark::State& state) {
    const std::size_t count = state.range(0);

    auto input = make_aligned_vector<sample_t>(count * 32);
    auto input_data = generate_random_data(count * 32);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto output = make_aligned_vector<sample_t>(count * 32);

    for (auto _ : state) {
        for (std::size_t i = 0; i < count; ++i) {
            __m512 r0 = _mm512_loadu_ps(reinterpret_cast<const float*>(input.data() + i * 32));
            __m512 r1 = _mm512_loadu_ps(reinterpret_cast<const float*>(input.data() + i * 32 + 8));
            __m512 r2 = _mm512_loadu_ps(reinterpret_cast<const float*>(input.data() + i * 32 + 16));
            __m512 r3 = _mm512_loadu_ps(reinterpret_cast<const float*>(input.data() + i * 32 + 24));
            kernels::fft32_2d::fft32_2d_reg(r0, r1, r2, r3);
            _mm512_storeu_ps(reinterpret_cast<float*>(output.data() + i * 32), r0);
            _mm512_storeu_ps(reinterpret_cast<float*>(output.data() + i * 32 + 8), r1);
            _mm512_storeu_ps(reinterpret_cast<float*>(output.data() + i * 32 + 16), r2);
            _mm512_storeu_ps(reinterpret_cast<float*>(output.data() + i * 32 + 24), r3);
        }
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * count);
    state.counters["FFT32/sec"] = benchmark::Counter(
        count,
        benchmark::Counter::kIsIterationInvariantRate,
        benchmark::Counter::kIs1000
    );
}
BENCHMARK(BM_FFT32_2D)->Arg(1024)->Arg(4096)->Arg(16384);

// =============================================================================
// BLOCKED COEFFICIENT KERNEL BENCHMARKS
// =============================================================================

// -----------------------------------------------------------------------------
// Blocked Filter: M=16, K=16
// -----------------------------------------------------------------------------

static void BM_Blocked_M16_K16(benchmark::State& state) {
    constexpr std::size_t M = 16;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_blocked(
            input.data(), coeffs.data(), output.data(), M, num_outputs, K);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_Blocked_M16_K16)->Arg(256)->Arg(1024)->Arg(4096)->Arg(8192);

// -----------------------------------------------------------------------------
// Blocked Filter: M=32, K=16
// -----------------------------------------------------------------------------

static void BM_Blocked_M32_K16(benchmark::State& state) {
    constexpr std::size_t M = 32;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_blocked(
            input.data(), coeffs.data(), output.data(), M, num_outputs, K);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_Blocked_M32_K16)->Arg(256)->Arg(1024)->Arg(4096)->Arg(8192);

// -----------------------------------------------------------------------------
// Blocked Filter: M=64, K=16
// -----------------------------------------------------------------------------

static void BM_Blocked_M64_K16(benchmark::State& state) {
    constexpr std::size_t M = 64;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_blocked(
            input.data(), coeffs.data(), output.data(), M, num_outputs, K);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_Blocked_M64_K16)->Arg(256)->Arg(1024)->Arg(4096)->Arg(8192);

// -----------------------------------------------------------------------------
// Blocked Filter: M=128, K=16
// -----------------------------------------------------------------------------

static void BM_Blocked_M128_K16(benchmark::State& state) {
    constexpr std::size_t M = 128;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_blocked(
            input.data(), coeffs.data(), output.data(), M, num_outputs, K);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_Blocked_M128_K16)->Arg(256)->Arg(1024)->Arg(4096);

// -----------------------------------------------------------------------------
// Blocked Filter: M=256, K=16
// -----------------------------------------------------------------------------

static void BM_Blocked_M256_K16(benchmark::State& state) {
    constexpr std::size_t M = 256;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_blocked(
            input.data(), coeffs.data(), output.data(), M, num_outputs, K);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_Blocked_M256_K16)->Arg(256)->Arg(1024)->Arg(4096);

// -----------------------------------------------------------------------------
// Blocked Filter: M=512, K=12 (typical SDR channelizer config)
// -----------------------------------------------------------------------------

static void BM_Blocked_M512_K12(benchmark::State& state) {
    constexpr std::size_t M = 512;
    constexpr std::size_t K = 12;
    const std::size_t num_outputs = state.range(0);

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_blocked(
            input.data(), coeffs.data(), output.data(), M, num_outputs, K);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_Blocked_M512_K12)->Arg(8)->Arg(32)->Arg(128)->Arg(512)->Arg(1024);

// -----------------------------------------------------------------------------
// Parameterized M (Blocked filter across channel counts)
// -----------------------------------------------------------------------------

static void BM_Blocked_VaryM(benchmark::State& state) {
    const std::size_t M = state.range(0);
    constexpr std::size_t K = 16;
    constexpr std::size_t num_outputs = 1024;

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto output = make_aligned_vector<sample_t>(num_outputs * M);

    for (auto _ : state) {
        kernels::filter_blocked(
            input.data(), coeffs.data(), output.data(), M, num_outputs, K);
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_Blocked_VaryM)->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512);

// =============================================================================
// FULL PIPELINE INTEGRATION BENCHMARKS (Filter → FFT → Transpose)
// =============================================================================
// These benchmarks measure the complete PFBC pipeline, not just the filter kernel.
// L1-tiled: filter_blocked → FFT in-place (tiled) → transpose_tile_to_channel_buffers

// -----------------------------------------------------------------------------
// L1-Tiled Full Pipeline (filter → FFT → transpose in tiles)
// -----------------------------------------------------------------------------
static void BM_FullPipeline_Tiled(benchmark::State& state) {
    const std::size_t M = state.range(0);
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(1);
    constexpr std::size_t TILE_SIZE = 8;

    // Input buffer
    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    // Coefficients (blocked for V2)
    auto coeffs = generate_blocked_coeffs(M, K);

    // L1 scratch buffer (TILE_SIZE * M) - stays in L1!
    auto l1_scratch = make_aligned_vector<sample_t>(TILE_SIZE * M);

    // FFT plan (in-place capable)
    auto fft = fft_plan<sample_t>(static_cast<uint32_t>(M));

    // Channel output buffers
    std::vector<aligned_buffer<sample_t>> channel_storage;
    channel_storage.reserve(M);
    for (std::size_t ch = 0; ch < M; ++ch) {
        channel_storage.push_back(make_aligned_vector<sample_t>(num_outputs));
    }
    std::vector<sample_t*> channel_ptrs(M);
    for (std::size_t ch = 0; ch < M; ++ch) {
        channel_ptrs[ch] = channel_storage[ch].data();
    }

    for (auto _ : state) {
        std::size_t frame_idx = 0;

        // Process in TILE_SIZE (8) row tiles
        for (std::size_t tile_start = 0; tile_start < num_outputs; tile_start += TILE_SIZE) {
            const std::size_t tile_count = std::min(TILE_SIZE, num_outputs - tile_start);
            const sample_t* tile_input = input.data() + tile_start * M;

            // Stage A: Filter tile → L1 scratch
            kernels::filter_blocked(
                tile_input, coeffs.data(), l1_scratch.data(),
                M, tile_count, K);

            // Stage B: FFT in-place on L1 scratch (data still hot!)
            for (std::size_t t = 0; t < tile_count; ++t) {
                sample_t* row = l1_scratch.data() + t * M;
                fft.execute(row, row);
            }

            // Stage C: Transpose tile to channel buffers
            if (tile_count == TILE_SIZE) {
                // Full tile - use optimized 8x8 transpose
                kernels::transpose_tile_to_channel_buffers(
                    l1_scratch.data(), channel_ptrs.data(), M, frame_idx);
            } else {
                // Partial tile - scalar transpose
                for (std::size_t t = 0; t < tile_count; ++t) {
                    for (std::size_t ch = 0; ch < M; ++ch) {
                        channel_ptrs[ch][frame_idx + t] = l1_scratch.data()[t * M + ch];
                    }
                }
            }

            frame_idx += tile_count;
        }

        benchmark::DoNotOptimize(channel_ptrs[0]);
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_FullPipeline_Tiled)
    ->Args({16, 256})->Args({16, 1024})->Args({16, 4096})
    ->Args({32, 256})->Args({32, 1024})->Args({32, 4096})
    ->Args({64, 256})->Args({64, 1024})->Args({64, 4096})
    ->Args({128, 256})->Args({128, 1024})->Args({128, 4096})
    ->Args({256, 256})->Args({256, 1024})->Args({256, 4096})
    ->Args({512, 256})->Args({512, 1024});

// -----------------------------------------------------------------------------
// Pipeline benchmark: Fixed M=64 K=16 varying outputs
// -----------------------------------------------------------------------------
static void BM_Pipeline_M64(benchmark::State& state) {
    constexpr std::size_t M = 64;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);
    constexpr std::size_t TILE_SIZE = 8;

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto l1_scratch = make_aligned_vector<sample_t>(TILE_SIZE * M);
    auto fft = fft_plan<sample_t>(M);

    std::vector<aligned_buffer<sample_t>> channel_storage;
    for (std::size_t ch = 0; ch < M; ++ch) {
        channel_storage.push_back(make_aligned_vector<sample_t>(num_outputs));
    }
    std::vector<sample_t*> channel_ptrs(M);
    for (std::size_t ch = 0; ch < M; ++ch) {
        channel_ptrs[ch] = channel_storage[ch].data();
    }

    for (auto _ : state) {
        std::size_t frame_idx = 0;

        for (std::size_t tile_start = 0; tile_start < num_outputs; tile_start += TILE_SIZE) {
            const std::size_t tile_count = std::min(TILE_SIZE, num_outputs - tile_start);
            const sample_t* tile_input = input.data() + tile_start * M;

            kernels::filter_blocked(
                tile_input, coeffs.data(), l1_scratch.data(), M, tile_count, K);

            for (std::size_t t = 0; t < tile_count; ++t) {
                sample_t* row = l1_scratch.data() + t * M;
                fft.execute(row, row);
            }

            if (tile_count == TILE_SIZE) {
                kernels::transpose_tile_to_channel_buffers(
                    l1_scratch.data(), channel_ptrs.data(), M, frame_idx);
            } else {
                for (std::size_t t = 0; t < tile_count; ++t) {
                    for (std::size_t ch = 0; ch < M; ++ch) {
                        channel_ptrs[ch][frame_idx + t] = l1_scratch.data()[t * M + ch];
                    }
                }
            }

            frame_idx += tile_count;
        }

        benchmark::DoNotOptimize(channel_ptrs[0]);
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_Pipeline_M64)->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096);

// =============================================================================
// V3: Channel-Tiled Full Pipeline (fixes L1 thrashing for large M)
// =============================================================================
// Uses filter_blocked_smart_dispatch which tiles channels for M >= 256

static void BM_V3_FullPipeline_ChannelTiled(benchmark::State& state) {
    const std::size_t M = state.range(0);
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(1);
    constexpr std::size_t TILE_SIZE = 8;

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto l1_scratch = make_aligned_vector<sample_t>(TILE_SIZE * M);
    auto fft = fft_plan<sample_t>(static_cast<uint32_t>(M));

    std::vector<aligned_buffer<sample_t>> channel_storage;
    channel_storage.reserve(M);
    for (std::size_t ch = 0; ch < M; ++ch) {
        channel_storage.push_back(make_aligned_vector<sample_t>(num_outputs));
    }
    std::vector<sample_t*> channel_ptrs(M);
    for (std::size_t ch = 0; ch < M; ++ch) {
        channel_ptrs[ch] = channel_storage[ch].data();
    }

    for (auto _ : state) {
        std::size_t frame_idx = 0;

        for (std::size_t tile_start = 0; tile_start < num_outputs; tile_start += TILE_SIZE) {
            const std::size_t tile_count = std::min(TILE_SIZE, num_outputs - tile_start);
            const sample_t* tile_input = input.data() + tile_start * M;

            // Stage A: Channel-tiled filter (keeps history in L1!)
            kernels::filter_blocked(
                tile_input, coeffs.data(), l1_scratch.data(),
                M, tile_count, K);

            // Stage B: FFT (sequential L2 access is fine)
            for (std::size_t t = 0; t < tile_count; ++t) {
                sample_t* row = l1_scratch.data() + t * M;
                fft.execute(row, row);
            }

            // Stage C: Transpose
            if (tile_count == TILE_SIZE) {
                kernels::transpose_tile_to_channel_buffers(
                    l1_scratch.data(), channel_ptrs.data(), M, frame_idx);
            } else {
                for (std::size_t t = 0; t < tile_count; ++t) {
                    for (std::size_t ch = 0; ch < M; ++ch) {
                        channel_ptrs[ch][frame_idx + t] = l1_scratch.data()[t * M + ch];
                    }
                }
            }

            frame_idx += tile_count;
        }

        benchmark::DoNotOptimize(channel_ptrs[0]);
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_V3_FullPipeline_ChannelTiled)
    ->Args({256, 256})->Args({256, 1024})->Args({256, 4096})
    ->Args({512, 256})->Args({512, 1024})->Args({512, 4096});

// Pipeline benchmark: M=512 (large channel count)
static void BM_Pipeline_M512(benchmark::State& state) {
    constexpr std::size_t M = 512;
    constexpr std::size_t K = 16;
    const std::size_t num_outputs = state.range(0);
    constexpr std::size_t TILE_SIZE = 8;

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto l1_scratch = make_aligned_vector<sample_t>(TILE_SIZE * M);
    auto fft = fft_plan<sample_t>(M);

    std::vector<aligned_buffer<sample_t>> channel_storage;
    for (std::size_t ch = 0; ch < M; ++ch) {
        channel_storage.push_back(make_aligned_vector<sample_t>(num_outputs));
    }
    std::vector<sample_t*> channel_ptrs(M);
    for (std::size_t ch = 0; ch < M; ++ch) {
        channel_ptrs[ch] = channel_storage[ch].data();
    }

    for (auto _ : state) {
        std::size_t frame_idx = 0;

        for (std::size_t tile_start = 0; tile_start < num_outputs; tile_start += TILE_SIZE) {
            const std::size_t tile_count = std::min(TILE_SIZE, num_outputs - tile_start);
            const sample_t* tile_input = input.data() + tile_start * M;

            // V3: With channel tiling (keeps history in L1!)
            kernels::filter_blocked(
                tile_input, coeffs.data(), l1_scratch.data(), M, tile_count, K);

            for (std::size_t t = 0; t < tile_count; ++t) {
                sample_t* row = l1_scratch.data() + t * M;
                fft.execute(row, row);
            }

            if (tile_count == TILE_SIZE) {
                kernels::transpose_tile_to_channel_buffers(
                    l1_scratch.data(), channel_ptrs.data(), M, frame_idx);
            } else {
                for (std::size_t t = 0; t < tile_count; ++t) {
                    for (std::size_t ch = 0; ch < M; ++ch) {
                        channel_ptrs[ch][frame_idx + t] = l1_scratch.data()[t * M + ch];
                    }
                }
            }

            frame_idx += tile_count;
        }

        benchmark::DoNotOptimize(channel_ptrs[0]);
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}
BENCHMARK(BM_Pipeline_M512)->Arg(256)->Arg(512)->Arg(1024);

// =============================================================================
// Comprehensive Pipeline benchmark across all M values
// =============================================================================
// Fixed: K=16, num_outputs=1024

static void BM_Pipeline_Comprehensive(benchmark::State& state) {
    const std::size_t M = state.range(0);
    constexpr std::size_t K = 16;
    constexpr std::size_t num_outputs = 1024;

    // Use engine dispatch (will select fused for M=8,16, hybrid for M=32, staged otherwise)
    auto engine = pfbc::make_engine(M, K);
    const std::size_t TILE_SIZE = engine->tile_size();

    const std::size_t input_rows = K + num_outputs - 1;
    auto input = make_aligned_vector<sample_t>(input_rows * M);
    auto input_data = generate_random_data(input_rows * M);
    std::copy(input_data.begin(), input_data.end(), input.begin());

    auto coeffs = generate_blocked_coeffs(M, K);
    auto l1_scratch = make_aligned_vector<sample_t>(TILE_SIZE * M);
    auto fft = fft_plan<sample_t>(static_cast<uint32_t>(M));

    std::vector<aligned_buffer<sample_t>> channel_storage;
    for (std::size_t ch = 0; ch < M; ++ch) {
        channel_storage.push_back(make_aligned_vector<sample_t>(num_outputs));
    }
    std::vector<sample_t*> channel_ptrs(M);
    for (std::size_t ch = 0; ch < M; ++ch) {
        channel_ptrs[ch] = channel_storage[ch].data();
    }

    // Pre-allocate tile pointer array outside the benchmark loop
    std::vector<sample_t*> tile_ptrs(M);

    for (auto _ : state) {
        std::size_t frame_idx = 0;

        for (std::size_t tile_start = 0; tile_start < num_outputs; tile_start += TILE_SIZE) {
            const std::size_t tile_count = std::min(TILE_SIZE, num_outputs - tile_start);
            const sample_t* tile_input = input.data() + tile_start * M;

            // Offset channel pointers to current frame position
            for (std::size_t ch = 0; ch < M; ++ch) {
                tile_ptrs[ch] = channel_ptrs[ch] + frame_idx;
            }

            if (tile_count == TILE_SIZE) {
                engine->process_tile(
                    tile_input, coeffs.data(), tile_ptrs.data(),
                    l1_scratch.data(), &fft);
            } else {
                engine->process_partial(
                    tile_input, coeffs.data(), tile_ptrs.data(),
                    l1_scratch.data(), &fft, tile_count);
            }

            frame_idx += tile_count;
        }

        benchmark::DoNotOptimize(channel_ptrs[0]);
        benchmark::ClobberMemory();
    }

    const auto total_samples = num_outputs * M;
    state.SetBytesProcessed(state.iterations() * total_samples * sizeof(sample_t));
    state.counters["Msps"] = benchmark::Counter(
        total_samples, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
    state.counters["policy"] = benchmark::Counter(0);  // Can't easily show string, but engine->name() would be useful
}
BENCHMARK(BM_Pipeline_Comprehensive)
    ->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024);

BENCHMARK_MAIN();
