// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "../kernels.hpp"
#include "../component.hpp"

#include <composite/composite.hpp>
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>

#include <complex>
#include <vector>
#include <random>
#include <iomanip>

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

template<typename T>
auto bytes_to_mb(T bytes) -> double {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

template<typename T>
auto samples_to_msps(T samples, std::chrono::nanoseconds duration) -> double {
    auto seconds = std::chrono::duration<double>(duration).count();
    return (static_cast<double>(samples) / 1e6) / seconds;
}

// =============================================================================
// Kernel Benchmarks
// =============================================================================

TEST_CASE("Benchmark: deinterleave_block - 1K samples", "[.][benchmark][kernel][deinterleave]") {
    constexpr std::size_t num_pairs = 512;  // 1024 input samples
    auto src = generate_random_data(num_pairs * 2);
    std::vector<sample_t> dst_even(num_pairs);
    std::vector<sample_t> dst_odd(num_pairs);

    BENCHMARK("deinterleave 1K samples") {
        kernels::deinterleave_block(src.data(), dst_even.data(), dst_odd.data(), num_pairs);
        return dst_even[0];
    };

    // Report throughput
    auto duration = std::chrono::nanoseconds(1); // Placeholder, actual timing from Catch2
    INFO("Input: " << (num_pairs * 2) << " samples = "
         << bytes_to_mb(num_pairs * 2 * sizeof(sample_t)) << " MB");
}

TEST_CASE("Benchmark: deinterleave_block - 16K samples", "[.][benchmark][kernel][deinterleave]") {
    constexpr std::size_t num_pairs = 8192;  // 16K input samples
    auto src = generate_random_data(num_pairs * 2);
    std::vector<sample_t> dst_even(num_pairs);
    std::vector<sample_t> dst_odd(num_pairs);

    BENCHMARK("deinterleave 16K samples") {
        kernels::deinterleave_block(src.data(), dst_even.data(), dst_odd.data(), num_pairs);
        return dst_even[0];
    };
}

TEST_CASE("Benchmark: deinterleave_block - 256K samples", "[.][benchmark][kernel][deinterleave]") {
    constexpr std::size_t num_pairs = 131072;  // 256K input samples
    auto src = generate_random_data(num_pairs * 2);
    std::vector<sample_t> dst_even(num_pairs);
    std::vector<sample_t> dst_odd(num_pairs);

    BENCHMARK("deinterleave 256K samples") {
        kernels::deinterleave_block(src.data(), dst_even.data(), dst_odd.data(), num_pairs);
        return dst_even[0];
    };

    INFO("Input: " << (num_pairs * 2) << " samples = "
         << bytes_to_mb(num_pairs * 2 * sizeof(sample_t)) << " MB");
}

TEST_CASE("Benchmark: halfband_filter_vertical - 1K outputs, 6 taps", "[.][benchmark][kernel][filter]") {
    constexpr std::size_t num_outputs = 1024;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 2;
    constexpr float center_tap = 0.5f;

    auto even_hist = generate_random_data(num_outputs + num_taps);
    auto odd_hist = generate_random_data(num_outputs + delay_offset);
    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    BENCHMARK("filter 1K outputs, 6 taps") {
        kernels::halfband_filter_vertical(
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs
        );
        return output[0];
    };
}

TEST_CASE("Benchmark: halfband_filter_vertical - 16K outputs, 6 taps", "[.][benchmark][kernel][filter]") {
    constexpr std::size_t num_outputs = 16384;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 2;
    constexpr float center_tap = 0.5f;

    auto even_hist = generate_random_data(num_outputs + num_taps);
    auto odd_hist = generate_random_data(num_outputs + delay_offset);
    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    BENCHMARK("filter 16K outputs, 6 taps") {
        kernels::halfband_filter_vertical(
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs
        );
        return output[0];
    };
}

TEST_CASE("Benchmark: halfband_filter_vertical - 256K outputs, 6 taps", "[.][benchmark][kernel][filter]") {
    constexpr std::size_t num_outputs = 262144;
    constexpr std::size_t num_taps = 6;
    constexpr std::size_t delay_offset = 2;
    constexpr float center_tap = 0.5f;

    auto even_hist = generate_random_data(num_outputs + num_taps);
    auto odd_hist = generate_random_data(num_outputs + delay_offset);
    std::vector<float> coeffs = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f};
    std::vector<sample_t> output(num_outputs);

    BENCHMARK("filter 256K outputs, 6 taps") {
        kernels::halfband_filter_vertical(
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs
        );
        return output[0];
    };

    INFO("Output: " << num_outputs << " samples = "
         << bytes_to_mb(num_outputs * sizeof(sample_t)) << " MB");
}

TEST_CASE("Benchmark: halfband_filter_vertical - 16K outputs, 12 taps", "[.][benchmark][kernel][filter]") {
    constexpr std::size_t num_outputs = 16384;
    constexpr std::size_t num_taps = 12;
    constexpr std::size_t delay_offset = 5;
    constexpr float center_tap = 0.5f;

    auto even_hist = generate_random_data(num_outputs + num_taps);
    auto odd_hist = generate_random_data(num_outputs + delay_offset);
    std::vector<float> coeffs(num_taps, 0.08f);
    std::vector<sample_t> output(num_outputs);

    BENCHMARK("filter 16K outputs, 12 taps") {
        kernels::halfband_filter_vertical(
            even_hist.data(), odd_hist.data(),
            coeffs.data(), num_taps,
            center_tap, delay_offset,
            output.data(), num_outputs
        );
        return output[0];
    };
}

// =============================================================================
// Component End-to-End Benchmarks
// =============================================================================

struct HalfrateBenchFixture {
    std::shared_ptr<halfrate> uut;
    std::shared_ptr<composite::output_port<composite::immutable_buffer<sample_t>>> source_port;
    std::shared_ptr<composite::input_port<composite::immutable_buffer<sample_t>>> sink_port;

    HalfrateBenchFixture() {
        uut = std::make_shared<halfrate>("bench_halfrate");
        source_port = std::make_shared<composite::output_port<composite::immutable_buffer<sample_t>>>("source");
        sink_port = std::make_shared<composite::input_port<composite::immutable_buffer<sample_t>>>("sink");

        auto* in_port = uut->get_port<composite::input_port<composite::mutable_buffer<sample_t>>>("data_in");
        auto* out_port = uut->get_port<composite::output_port<composite::immutable_buffer<sample_t>>>("data_out");

        source_port->connect(in_port);
        out_port->connect(sink_port.get());

        uut->property_change_handler();
    }

    void send_data(const std::vector<sample_t>& data) {
        auto data_vec = std::make_shared<std::vector<sample_t>>(data);
        composite::immutable_buffer<sample_t> buf(data_vec);
        source_port->send_data(buf, composite::timestamp{});
    }

    auto get_output() -> std::vector<sample_t> {
        if (sink_port->size() == 0) return {};
        auto [data, ts, meta] = sink_port->get_data();
        return std::vector<sample_t>(data.begin(), data.end());
    }
};

TEST_CASE("Benchmark: halfrate component - 1K input", "[.][benchmark][component]") {
    HalfrateBenchFixture fixture;
    constexpr std::size_t block_size = 1024;
    auto input = generate_random_data(block_size);

    BENCHMARK("process 1K samples") {
        fixture.send_data(input);
        fixture.uut->process();
        return fixture.get_output();
    };

    INFO("Input: " << block_size << " samples ("
         << bytes_to_mb(block_size * sizeof(sample_t)) << " MB), "
         << "Output: " << (block_size / 2) << " samples");
}

TEST_CASE("Benchmark: halfrate component - 16K input", "[.][benchmark][component]") {
    HalfrateBenchFixture fixture;
    constexpr std::size_t block_size = 16384;
    auto input = generate_random_data(block_size);

    BENCHMARK("process 16K samples") {
        fixture.send_data(input);
        fixture.uut->process();
        return fixture.get_output();
    };

    INFO("Input: " << block_size << " samples ("
         << bytes_to_mb(block_size * sizeof(sample_t)) << " MB), "
         << "Output: " << (block_size / 2) << " samples");
}

TEST_CASE("Benchmark: halfrate component - 256K input", "[.][benchmark][component]") {
    HalfrateBenchFixture fixture;
    constexpr std::size_t block_size = 262144;
    auto input = generate_random_data(block_size);

    BENCHMARK("process 256K samples") {
        fixture.send_data(input);
        fixture.uut->process();
        return fixture.get_output();
    };

    INFO("Input: " << block_size << " samples ("
         << bytes_to_mb(block_size * sizeof(sample_t)) << " MB), "
         << "Output: " << (block_size / 2) << " samples");
}

TEST_CASE("Benchmark: halfrate component - 1M input", "[.][benchmark][component]") {
    HalfrateBenchFixture fixture;
    constexpr std::size_t block_size = 1048576;
    auto input = generate_random_data(block_size);

    BENCHMARK("process 1M samples") {
        fixture.send_data(input);
        fixture.uut->process();
        return fixture.get_output();
    };

    INFO("Input: " << block_size << " samples ("
         << bytes_to_mb(block_size * sizeof(sample_t)) << " MB), "
         << "Output: " << (block_size / 2) << " samples");
}

TEST_CASE("Benchmark: halfrate streaming - sustained throughput", "[.][benchmark][component][streaming]") {
    HalfrateBenchFixture fixture;
    constexpr std::size_t block_size = 65536;  // 64K samples per call
    constexpr std::size_t num_iterations = 16; // Process 1M total samples
    auto input = generate_random_data(block_size);

    BENCHMARK("sustained: 16 × 64K samples") {
        for (std::size_t i = 0; i < num_iterations; ++i) {
            fixture.send_data(input);
            fixture.uut->process();
            auto output = fixture.get_output();
        }
        return true;
    };

    INFO("Total processed: " << (block_size * num_iterations) << " samples = "
         << bytes_to_mb(block_size * num_iterations * sizeof(sample_t)) << " MB");
}

// =============================================================================
// Cache Effects Benchmark
// =============================================================================

TEST_CASE("Benchmark: cache behavior - varying block sizes", "[.][benchmark][cache]") {
    HalfrateBenchFixture fixture;

    SECTION("128 samples (L1 cache)") {
        auto input = generate_random_data(128);
        BENCHMARK("128 samples") {
            fixture.send_data(input);
            fixture.uut->process();
            return fixture.get_output();
        };
    }

    SECTION("4K samples (L2 cache)") {
        auto input = generate_random_data(4096);
        BENCHMARK("4K samples") {
            fixture.send_data(input);
            fixture.uut->process();
            return fixture.get_output();
        };
    }

    SECTION("64K samples (L3 cache)") {
        auto input = generate_random_data(65536);
        BENCHMARK("64K samples") {
            fixture.send_data(input);
            fixture.uut->process();
            return fixture.get_output();
        };
    }

    SECTION("1M samples (main memory)") {
        auto input = generate_random_data(1048576);
        BENCHMARK("1M samples") {
            fixture.send_data(input);
            fixture.uut->process();
            return fixture.get_output();
        };
    }
}

// =============================================================================
// Throughput Calculation Helper
// =============================================================================

TEST_CASE("Throughput Report", "[.][benchmark][report]") {
    // This is just a documentation test that shows how to interpret results

    INFO("=================================================================");
    INFO("Throughput Calculation Guide:");
    INFO("=================================================================");
    INFO("");
    INFO("For benchmark results, calculate throughput as:");
    INFO("  - Input samples/sec = (num_input_samples / mean_time_ns) * 1e9");
    INFO("  - Output samples/sec = (num_output_samples / mean_time_ns) * 1e9");
    INFO("  - MSPS (Mega-samples/sec) = samples/sec / 1e6");
    INFO("  - MB/s = (samples * 8 bytes/sample / mean_time_ns) * 1e9 / 1e6");
    INFO("");
    INFO("Example: If 256K samples process in 500μs:");
    INFO("  - 262144 samples / 500e-6 sec = 524.3 Msamples/sec");
    INFO("  - 524.3 Msamples/sec * 8 bytes = 4194 MB/s");
    INFO("");
    INFO("Target: Modern CPUs should achieve 500+ MSPS single-threaded");
    INFO("=================================================================");
}
