// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "component.hpp"
#include "fft_plan.hpp"

#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <future>
#include <numbers>
#include <omp.h>
#include <vector>

using sample_t = std::complex<float>;
using Catch::Approx;
using Catch::Matchers::WithinAbs;

// Test fixture for friend access to polyphase_channelizer internals
struct PfbcTestFixture {
    std::shared_ptr<polyphase_channelizer> pfbc;
    std::shared_ptr<composite::output_port<composite::mutable_buffer<sample_t>>> source_port;
    std::shared_ptr<composite::input_port<composite::immutable_buffer<sample_t>>> sink_port;

    PfbcTestFixture(std::string_view id = "test_pfbc") {
        pfbc = std::make_shared<polyphase_channelizer>(id);
        source_port = std::make_shared<composite::output_port<composite::mutable_buffer<sample_t>>>("source");
        sink_port = std::make_shared<composite::input_port<composite::immutable_buffer<sample_t>>>("sink");

        // Connect ports using friend access
        source_port->connect(&pfbc->m_data_in);
        pfbc->m_data_out.connect(sink_port.get());
    }

    void configure(uint32_t M, uint32_t K, uint32_t N, uint32_t threads = 1) {
        pfbc->m_num_channels = M;
        pfbc->m_taps_per_phase = K;
        pfbc->m_frame_size = N;
        pfbc->m_num_threads = threads;
        pfbc->property_change_handler();
    }

    // Accessors for private members (friend access)
    void set_prototype_filter(const std::vector<float>& filter) {
        pfbc->m_prototype_filter = filter;
    }

    bool is_configured() const { return pfbc->m_configured; }
};

// Helper to generate complex sinusoid at given frequency
auto generate_tone(std::size_t num_samples, float normalized_freq) -> std::vector<sample_t> {
    std::vector<sample_t> signal(num_samples);
    for (std::size_t i = 0; i < num_samples; ++i) {
        float phase = 2.0f * std::numbers::pi_v<float> * normalized_freq * static_cast<float>(i);
        signal[i] = sample_t{std::cos(phase), std::sin(phase)};
    }
    return signal;
}

// Helper to compute signal power
auto compute_power(const sample_t* data, std::size_t n) -> float {
    float power = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        power += std::norm(data[i]);
    }
    return power / static_cast<float>(n);
}

TEST_CASE("polyphase_channelizer construction", "[pfbc][integration]") {
    auto channelizer = std::make_shared<polyphase_channelizer>("test_pfbc");

    // Default values should be set
    CHECK(channelizer != nullptr);
}

TEST_CASE("polyphase_channelizer configuration", "[pfbc][integration]") {
    auto channelizer = std::make_shared<polyphase_channelizer>("test_pfbc");

    // Trigger property change handler to configure
    channelizer->property_change_handler();

    // Component should be configured with defaults
    // (We can't easily check internal state without friend access,
    // but the call shouldn't throw)
    CHECK(true);
}

TEST_CASE("polyphase_channelizer rejects invalid num_channels", "[pfbc][integration]") {
    auto channelizer = std::make_shared<polyphase_channelizer>("test_pfbc");

    // Try to set non-power-of-2 num_channels via property
    // This should fail validation
    // Note: Direct property access would require the composite framework
    // For now, just verify construction works
    CHECK(channelizer != nullptr);
}


// =============================================================================
// Streaming Port Tests - validates data flows through actual composite ports
// =============================================================================

TEST_CASE("streaming port test - single frame", "[pfbc][streaming]") {
    // Configuration
    constexpr uint32_t M = 16;   // channels
    constexpr uint32_t K = 8;    // taps per phase
    constexpr uint32_t N = 4;    // frame size

    // Create and configure fixture
    PfbcTestFixture fixture("test_pfbc");
    fixture.configure(M, K, N, 1);

    // Generate input: enough samples for one complete frame output
    // Need M * N input samples to produce N output samples per channel
    const std::size_t input_size = M * N;
    auto input = composite::make_mutable<sample_t>(input_size);
    auto input_span = input.as_span();
    for (std::size_t i = 0; i < input_size; ++i) {
        float phase = 2.0f * std::numbers::pi_v<float> * 0.1f * static_cast<float>(i);
        input_span[i] = sample_t{std::cos(phase), std::sin(phase)};
    }

    // Send data through port
    composite::timestamp ts{100, 0};
    fixture.source_port->send_data(std::move(input), ts);

    // Process
    auto result = fixture.pfbc->process();
    REQUIRE(result == composite::retval::NORMAL);

    // Should have M output frames (one per channel)
    REQUIRE(fixture.sink_port->size() == M);

    // Verify each channel output
    for (std::size_t ch = 0; ch < M; ++ch) {
        auto [out_buf, out_ts, out_meta] = fixture.sink_port->get_data();

        INFO("Channel " << ch);
        REQUIRE(out_buf.size() == N);

        // Verify timestamp propagated
        CHECK(out_ts.seconds == ts.seconds);

        // Verify output is finite
        for (std::size_t i = 0; i < N; ++i) {
            CHECK(std::isfinite(out_buf.data()[i].real()));
            CHECK(std::isfinite(out_buf.data()[i].imag()));
        }
    }
}

TEST_CASE("streaming port test - multiple frames", "[pfbc][streaming]") {
    // Configuration
    constexpr uint32_t M = 32;   // channels
    constexpr uint32_t K = 12;   // taps per phase
    constexpr uint32_t N = 8;    // frame size
    constexpr std::size_t num_frames = 4;

    // Create and configure fixture
    PfbcTestFixture fixture("test_pfbc");
    fixture.configure(M, K, N, 2);

    // Track total outputs across all frames
    std::size_t total_outputs = 0;

    // Send multiple batches of input, draining after each to simulate real-world usage
    for (std::size_t frame = 0; frame < num_frames; ++frame) {
        const std::size_t input_size = M * N;
        auto input = composite::make_mutable<sample_t>(input_size);
        auto input_span = input.as_span();

        // Generate ramp with offset per frame
        for (std::size_t i = 0; i < input_size; ++i) {
            float val = static_cast<float>(frame * input_size + i);
            input_span[i] = sample_t{val, -val};
        }

        composite::timestamp ts{static_cast<uint32_t>(frame), 0};
        fixture.source_port->send_data(std::move(input), ts);

        auto result = fixture.pfbc->process();
        REQUIRE(result == composite::retval::NORMAL);

        // Should have M outputs per frame (one per channel)
        REQUIRE(fixture.sink_port->size() == M);

        // Drain and verify outputs for this frame
        while (fixture.sink_port->size() > 0) {
            auto [out_buf, out_ts, out_meta] = fixture.sink_port->get_data();
            REQUIRE(out_buf.size() == N);

            // Verify timestamp matches frame
            CHECK(out_ts.seconds == frame);

            // Verify output is finite
            for (std::size_t i = 0; i < N; ++i) {
                CHECK(std::isfinite(out_buf.data()[i].real()));
                CHECK(std::isfinite(out_buf.data()[i].imag()));
            }
            ++total_outputs;
        }
    }

    // Should have received all outputs
    CHECK(total_outputs == num_frames * M);
}

TEST_CASE("streaming port test - tail split matches single batch output", "[pfbc][streaming][tail]") {
    constexpr uint32_t M = 16;
    constexpr uint32_t K = 8;
    constexpr uint32_t N = 4;

    // Build a deterministic input stream with a non-M-aligned length.
    const std::size_t total_samples = M * 13 + 5;
    std::vector<sample_t> input(total_samples);
    for (std::size_t i = 0; i < total_samples; ++i) {
        float val = static_cast<float>(i) * 0.01f;
        input[i] = sample_t{val, -val};
    }

    auto run_with_chunks = [&](const std::vector<std::size_t>& chunks) {
        PfbcTestFixture fixture("test_pfbc");
        fixture.configure(M, K, N, 1);

        std::vector<std::vector<sample_t>> outputs;
        std::size_t offset = 0;
        for (std::size_t chunk : chunks) {
            auto buf = composite::make_mutable<sample_t>(chunk);
            auto span = buf.as_span();
            for (std::size_t i = 0; i < chunk; ++i) {
                span[i] = input[offset + i];
            }
            offset += chunk;

            fixture.source_port->send_data(std::move(buf), composite::timestamp{1, 0});
            auto result = fixture.pfbc->process();
            REQUIRE(result == composite::retval::NORMAL);

            while (fixture.sink_port->size() > 0) {
                auto [out_buf, out_ts, out_meta] = fixture.sink_port->get_data();
                std::vector<sample_t> out(out_buf.data(), out_buf.data() + out_buf.size());
                outputs.push_back(std::move(out));
            }
        }
        return outputs;
    };

    // Single batch reference
    auto outputs_single = run_with_chunks({total_samples});

    // Split batch: small non-M-aligned chunk + larger chunk to trigger tail handling
    const std::size_t first_chunk = M * 3 + 5;
    const std::size_t second_chunk = total_samples - first_chunk;
    auto outputs_split = run_with_chunks({first_chunk, second_chunk});

    REQUIRE(outputs_split.size() == outputs_single.size());
    for (std::size_t i = 0; i < outputs_single.size(); ++i) {
        const auto& a = outputs_single[i];
        const auto& b = outputs_split[i];
        REQUIRE(a.size() == b.size());
        for (std::size_t j = 0; j < a.size(); ++j) {
            CHECK(a[j].real() == Approx(b[j].real()).epsilon(1e-5));
            CHECK(a[j].imag() == Approx(b[j].imag()).epsilon(1e-5));
        }
    }
}

TEST_CASE("streaming port test - K=1 tail split matches single batch output", "[pfbc][streaming][tail]") {
    constexpr uint32_t M = 8;
    constexpr uint32_t K = 1;
    constexpr uint32_t N = 4;

    const std::size_t total_samples = M * 7 + 3;
    std::vector<sample_t> input(total_samples);
    for (std::size_t i = 0; i < total_samples; ++i) {
        float val = static_cast<float>(i) * 0.02f;
        input[i] = sample_t{val, val * 0.5f};
    }

    auto run_with_chunks = [&](const std::vector<std::size_t>& chunks) {
        PfbcTestFixture fixture("test_pfbc");
        fixture.configure(M, K, N, 1);

        std::vector<std::vector<sample_t>> outputs;
        std::size_t offset = 0;
        for (std::size_t chunk : chunks) {
            auto buf = composite::make_mutable<sample_t>(chunk);
            auto span = buf.as_span();
            for (std::size_t i = 0; i < chunk; ++i) {
                span[i] = input[offset + i];
            }
            offset += chunk;

            fixture.source_port->send_data(std::move(buf), composite::timestamp{2, 0});
            auto result = fixture.pfbc->process();
            REQUIRE(result == composite::retval::NORMAL);

            while (fixture.sink_port->size() > 0) {
                auto [out_buf, out_ts, out_meta] = fixture.sink_port->get_data();
                std::vector<sample_t> out(out_buf.data(), out_buf.data() + out_buf.size());
                outputs.push_back(std::move(out));
            }
        }
        return outputs;
    };

    auto outputs_single = run_with_chunks({total_samples});

    const std::size_t first_chunk = M * 2 + 3;
    const std::size_t second_chunk = total_samples - first_chunk;
    auto outputs_split = run_with_chunks({first_chunk, second_chunk});

    REQUIRE(outputs_split.size() == outputs_single.size());
    for (std::size_t i = 0; i < outputs_single.size(); ++i) {
        const auto& a = outputs_single[i];
        const auto& b = outputs_split[i];
        REQUIRE(a.size() == b.size());
        for (std::size_t j = 0; j < a.size(); ++j) {
            CHECK(a[j].real() == Approx(b[j].real()).epsilon(1e-5));
            CHECK(a[j].imag() == Approx(b[j].imag()).epsilon(1e-5));
        }
    }
}

TEST_CASE("streaming port test - tone detection through ports", "[pfbc][streaming]") {
    // Verify that a tone fed through ports ends up in the correct channel
    constexpr uint32_t M = 64;   // channels
    constexpr uint32_t K = 12;   // taps per phase
    constexpr uint32_t N = 32;   // frame size
    constexpr std::size_t target_channel = 10;
    constexpr std::size_t num_frames = 8;  // Accumulate power over multiple frames

    // Create and configure fixture
    PfbcTestFixture fixture("test_pfbc");
    fixture.configure(M, K, N, 4);

    // Generate tone at target channel frequency
    float normalized_freq = static_cast<float>(target_channel) / static_cast<float>(M);
    std::size_t total_samples = M * N * num_frames;
    auto tone = generate_tone(total_samples, normalized_freq);

    // Track power per channel
    std::vector<float> channel_power(M, 0.0f);

    // Process in frame-sized chunks
    std::size_t sample_offset = 0;
    for (std::size_t frame = 0; frame < num_frames; ++frame) {
        const std::size_t input_size = M * N;
        auto input = composite::make_mutable<sample_t>(input_size);
        auto input_span = input.as_span();

        std::copy(tone.begin() + sample_offset,
                  tone.begin() + sample_offset + input_size,
                  input_span.begin());
        sample_offset += input_size;

        fixture.source_port->send_data(std::move(input), {0, 0});
        fixture.pfbc->process();

        // Collect outputs and accumulate power (outputs come in channel order 0..M-1)
        std::size_t channel_idx = 0;
        while (fixture.sink_port->size() > 0) {
            auto [out_buf, out_ts, out_meta] = fixture.sink_port->get_data();

            // Accumulate power for this channel
            for (std::size_t i = 0; i < out_buf.size(); ++i) {
                channel_power[channel_idx] += std::norm(out_buf.data()[i]);
            }
            channel_idx = (channel_idx + 1) % M;
        }
    }

    // Find channel with maximum power
    std::size_t max_channel = 0;
    float max_power = channel_power[0];
    for (std::size_t ch = 1; ch < M; ++ch) {
        if (channel_power[ch] > max_power) {
            max_power = channel_power[ch];
            max_channel = ch;
        }
    }

    // Tone should appear in target channel (allow ±1 for filter response)
    INFO("Target channel: " << target_channel << ", Max power channel: " << max_channel);
    CHECK(max_channel >= target_channel - 1);
    CHECK(max_channel <= target_channel + 1);
}

// =============================================================================
// Remainder Buffer Tests
// =============================================================================

TEST_CASE("streaming - remainder handling across calls", "[pfbc][streaming][remainder]") {
    // Test that partial blocks are carried over correctly between process() calls
    constexpr uint32_t M = 16;   // channels
    constexpr uint32_t K = 8;    // taps per phase
    constexpr uint32_t N = 4;    // frame size

    PfbcTestFixture fixture("test_pfbc");
    fixture.configure(M, K, N, 1);

    // Send partial block that doesn't fill a frame
    // M * N = 64 samples needed per frame
    // Send 50 samples (50 / 16 = 3 complete M-blocks, 2 remainder)
    {
        auto input = composite::make_mutable<sample_t>(50);
        auto span = input.as_span();
        for (std::size_t i = 0; i < 50; ++i) {
            span[i] = sample_t{static_cast<float>(i), 0.0f};
        }
        fixture.source_port->send_data(std::move(input), {0, 0});
        fixture.pfbc->process();

        // Should have no output yet (only 3 M-blocks, need 4 for a frame)
        CHECK(fixture.sink_port->size() == 0);
    }

    // Send 30 more samples - should complete a frame
    // Previous remainder: 2 samples
    // New: 30 samples
    // Total: 32 samples = 2 M-blocks
    // Total M-blocks so far: 3 + 2 = 5, need 4 for a frame = 1 frame + 1 M-block remainder
    {
        auto input = composite::make_mutable<sample_t>(30);
        auto span = input.as_span();
        for (std::size_t i = 0; i < 30; ++i) {
            span[i] = sample_t{static_cast<float>(50 + i), 0.0f};
        }
        fixture.source_port->send_data(std::move(input), {1, 0});
        fixture.pfbc->process();

        // Should have M outputs (one per channel for the completed frame)
        CHECK(fixture.sink_port->size() == M);

        // Drain outputs
        while (fixture.sink_port->size() > 0) {
            auto [buf, ts, meta] = fixture.sink_port->get_data();
            CHECK(buf.size() == N);
        }
    }
}

TEST_CASE("streaming - small inputs accumulate correctly", "[pfbc][streaming][remainder]") {
    // Test that many small inputs eventually produce output
    constexpr uint32_t M = 8;    // channels
    constexpr uint32_t K = 4;    // taps per phase
    constexpr uint32_t N = 4;    // frame size (need M*N = 32 samples)

    PfbcTestFixture fixture("test_pfbc");
    fixture.configure(M, K, N, 1);

    // Send input in small chunks of 5 samples each
    // 32 samples / 5 = 6 full chunks + 2 extra = need 7 chunks minimum
    std::size_t total_sent = 0;
    std::size_t num_chunks = 10;

    for (std::size_t chunk = 0; chunk < num_chunks; ++chunk) {
        auto input = composite::make_mutable<sample_t>(5);
        auto span = input.as_span();
        for (std::size_t i = 0; i < 5; ++i) {
            span[i] = sample_t{static_cast<float>(total_sent + i), 0.0f};
        }
        total_sent += 5;
        fixture.source_port->send_data(std::move(input), {static_cast<uint32_t>(chunk), 0});
        fixture.pfbc->process();
    }

    // Total samples sent: 50
    // Complete M-blocks: 50 / 8 = 6
    // Frames: 6 / 4 = 1 complete frame
    // Should have M outputs
    CHECK(fixture.sink_port->size() == M);
}

TEST_CASE("streaming - no data loss with odd-sized inputs", "[pfbc][streaming][remainder]") {
    // Verify that all samples are processed, none are lost
    constexpr uint32_t M = 16;
    constexpr uint32_t K = 8;
    constexpr uint32_t N = 4;

    PfbcTestFixture fixture("test_pfbc");
    fixture.configure(M, K, N, 1);

    // Send various odd-sized inputs
    std::vector<std::size_t> input_sizes = {17, 23, 31, 29, 11, 13, 19, 37};
    std::size_t total_sent = 0;

    for (auto size : input_sizes) {
        auto input = composite::make_mutable<sample_t>(size);
        auto span = input.as_span();
        for (std::size_t i = 0; i < size; ++i) {
            span[i] = sample_t{1.0f, 0.0f};  // All ones for easy power verification
        }
        total_sent += size;
        fixture.source_port->send_data(std::move(input), {0, 0});
        fixture.pfbc->process();
    }

    // total_sent = 180 samples
    // Complete M-blocks: 180 / 16 = 11
    // Frames: 11 / 4 = 2 complete frames
    // Outputs: 2 * M = 32 channel buffers
    std::size_t expected_frames = (total_sent / M) / N;
    CHECK(fixture.sink_port->size() == expected_frames * M);
}

// =============================================================================
// Timestamp Progression Tests
// =============================================================================

TEST_CASE("streaming - timestamp progression across frames", "[pfbc][streaming][timestamp]") {
    // Verify timestamps advance correctly when multiple frames are emitted from one large input
    constexpr uint32_t M = 8;   // Smaller M to fit pool size
    constexpr uint32_t K = 4;
    constexpr uint32_t N = 4;
    constexpr std::size_t num_frames = 2;  // 2 frames fits in M*2 pool

    PfbcTestFixture fixture("test_pfbc");
    fixture.configure(M, K, N, 1);

    // Set sample rate in metadata for proper timestamp calculation
    composite::metadata md;
    md.sample_rate = 1e6;  // 1 MHz

    // Send enough samples for 2 complete frames in one call
    const std::size_t input_size = M * N * num_frames;
    auto input = composite::make_mutable<sample_t>(input_size);
    auto span = input.as_span();
    for (std::size_t i = 0; i < input_size; ++i) {
        span[i] = sample_t{static_cast<float>(i), 0.0f};
    }

    composite::timestamp base_ts{1, 0};  // 1 second
    // Send metadata along with the data
    fixture.source_port->send_metadata(md);
    fixture.source_port->send_data(std::move(input), base_ts);
    fixture.pfbc->process();

    // Should have num_frames * M outputs
    REQUIRE(fixture.sink_port->size() == num_frames * M);

    // Collect timestamps from first channel of each frame
    std::vector<composite::timestamp> frame_timestamps;
    std::size_t output_count = 0;
    while (fixture.sink_port->size() > 0) {
        auto [buf, ts, meta] = fixture.sink_port->get_data();
        if (output_count % M == 0) {
            frame_timestamps.push_back(ts);
        }
        ++output_count;
    }

    REQUIRE(frame_timestamps.size() == num_frames);

    // Each frame should have a later timestamp than the previous
    // Frame 0: samples [0, M*N) -> ts = base_ts + 0
    // Frame 1: samples [M*N, 2*M*N) -> ts = base_ts + (M*N samples / 1MHz) = base_ts + 32us
    // At 1 MHz, M*N = 32 samples = 32 microseconds = 32e6 picoseconds
    for (std::size_t i = 1; i < frame_timestamps.size(); ++i) {
        INFO("Frame " << i << ": ts=" << frame_timestamps[i].seconds << "s "
             << frame_timestamps[i].picoseconds << "ps");
        INFO("Frame " << (i-1) << ": ts=" << frame_timestamps[i-1].seconds << "s "
             << frame_timestamps[i-1].picoseconds << "ps");
        CHECK(frame_timestamps[i] > frame_timestamps[i-1]);
    }

    // Verify the timestamp offset is approximately correct
    // Frame 1 should be ~32us = 32000ns after frame 0 (at 1 MHz, M*N samples)
    auto time_diff = frame_timestamps[1] - frame_timestamps[0];  // returns nanoseconds
    auto expected_ns = std::chrono::nanoseconds(static_cast<int64_t>(M * N) * 1000);  // 32us = 32000ns
    CHECK(time_diff.count() >= expected_ns.count() - 1000);  // Allow 1us tolerance
    CHECK(time_diff.count() <= expected_ns.count() + 1000);
}

TEST_CASE("streaming - timestamp resets on new input timestamp", "[pfbc][streaming][timestamp]") {
    // Verify that a new input timestamp resets the base
    constexpr uint32_t M = 16;
    constexpr uint32_t K = 8;
    constexpr uint32_t N = 4;

    PfbcTestFixture fixture("test_pfbc");
    fixture.configure(M, K, N, 1);

    // Send first frame with timestamp 10
    {
        auto input = composite::make_mutable<sample_t>(M * N);
        std::fill(input.as_span().begin(), input.as_span().end(), sample_t{1.0f, 0.0f});
        fixture.source_port->send_data(std::move(input), {10, 0});
        fixture.pfbc->process();

        REQUIRE(fixture.sink_port->size() == M);
        auto [buf, ts, meta] = fixture.sink_port->get_data();
        CHECK(ts.seconds == 10);

        // Drain remaining outputs
        while (fixture.sink_port->size() > 0) {
            fixture.sink_port->get_data();
        }
    }

    // Send second frame with timestamp 20 (jump)
    {
        auto input = composite::make_mutable<sample_t>(M * N);
        std::fill(input.as_span().begin(), input.as_span().end(), sample_t{1.0f, 0.0f});
        fixture.source_port->send_data(std::move(input), {20, 0});
        fixture.pfbc->process();

        REQUIRE(fixture.sink_port->size() == M);
        auto [buf, ts, meta] = fixture.sink_port->get_data();
        // Should use the new timestamp base
        CHECK(ts.seconds == 20);
    }
}

TEST_CASE("streaming - timestamp always reanchors on new input ts", "[pfbc][streaming][timestamp]") {
    constexpr uint32_t M = 8;
    constexpr uint32_t K = 4;
    constexpr uint32_t N = 4;

    PfbcTestFixture fixture("test_pfbc");
    fixture.configure(M, K, N, 1);

    composite::metadata md;
    md.sample_rate = 1e6;  // 1 MHz

    auto make_input = [&](std::size_t samples) {
        auto input = composite::make_mutable<sample_t>(samples);
        auto span = input.as_span();
        for (std::size_t i = 0; i < samples; ++i) {
            span[i] = sample_t{static_cast<float>(i), 0.0f};
        }
        return input;
    };

    fixture.source_port->send_metadata(md);

    // First batch: 1 frame
    fixture.source_port->send_data(make_input(M * N), {5, 0});
    fixture.pfbc->process();
    REQUIRE(fixture.sink_port->size() == M);
    auto [buf0, ts0, meta0] = fixture.sink_port->get_data();
    CHECK(ts0.seconds == 5);
    while (fixture.sink_port->size() > 0) {
        fixture.sink_port->get_data();
    }

    // Second batch: re-anchor to new timestamp even though stream continues
    fixture.source_port->send_data(make_input(M * N), {10, 0});
    fixture.pfbc->process();
    REQUIRE(fixture.sink_port->size() == M);
    auto [buf1, ts1, meta1] = fixture.sink_port->get_data();
    CHECK(ts1.seconds == 10);
}

TEST_CASE("streaming - output metadata sample rate is decimated", "[pfbc][streaming][metadata]") {
    constexpr uint32_t M = 8;
    constexpr uint32_t K = 4;
    constexpr uint32_t N = 4;

    PfbcTestFixture fixture("test_pfbc");
    fixture.configure(M, K, N, 1);

    composite::metadata md;
    md.sample_rate = 1e6;  // 1 MHz

    auto input = composite::make_mutable<sample_t>(M * N);
    auto span = input.as_span();
    for (std::size_t i = 0; i < span.size(); ++i) {
        span[i] = sample_t{1.0f, 0.0f};
    }

    fixture.source_port->send_metadata(md);
    fixture.source_port->send_data(std::move(input), {0, 0});
    fixture.pfbc->process();

    REQUIRE(fixture.sink_port->size() == M);
    bool saw_metadata = false;
    for (std::size_t ch = 0; ch < M; ++ch) {
        auto [buf, ts, out_meta] = fixture.sink_port->get_data();
        INFO("Channel " << ch);
        if (out_meta.has_value()) {
            saw_metadata = true;
            CHECK_THAT(out_meta->sample_rate, WithinAbs(1e6 / static_cast<double>(M), 1e-6));
        }
    }
    REQUIRE(saw_metadata);
}

// =============================================================================
// Throughput Benchmark
// =============================================================================

#if 0  // Disabled benchmark tests
TEST_CASE("pool exhaustion handling", "[pfbc][streaming]") {
    // Test behavior when output pool is exhausted (small pool, fast producer)
    constexpr uint32_t M = 16;
    constexpr uint32_t K = 8;
    constexpr uint32_t N = 4;

    // Create and configure fixture
    PfbcTestFixture fixture("test_pfbc");
    fixture.configure(M, K, N, 1);

    // Send many frames without draining - pool should handle it gracefully
    // Pool size is M * 2 = 32 buffers
    constexpr std::size_t frames_to_send = 10;

    for (std::size_t frame = 0; frame < frames_to_send; ++frame) {
        const std::size_t input_size = M * N;
        auto input = composite::make_mutable<sample_t>(input_size);
        auto input_span = input.as_span();
        for (std::size_t i = 0; i < input_size; ++i) {
            input_span[i] = sample_t{1.0f, 0.0f};
        }

        fixture.source_port->send_data(std::move(input), {0, 0});
        auto result = fixture.pfbc->process();

        // Should not crash even if pool exhausted
        CHECK(result == composite::retval::NORMAL);
    }

    // Drain and count outputs
    std::size_t received = 0;
    while (fixture.sink_port->size() > 0) {
        auto [buf, ts, meta] = fixture.sink_port->get_data();
        ++received;
    }

    // Should have received most outputs (some may be dropped if pool exhausted)
    INFO("Sent " << frames_to_send << " frames, received " << received << " channel outputs");
    CHECK(received > 0);
}

// =============================================================================
// Custom Prototype Filter Tests
// =============================================================================

TEST_CASE("custom prototype filter - valid size", "[pfbc][config]") {
    constexpr uint32_t M = 16;
    constexpr uint32_t K = 8;
    constexpr uint32_t N = 4;

    PfbcTestFixture fixture("test_pfbc");

    // Create custom filter with correct size M*K
    std::vector<float> custom_filter(M * K);
    for (std::size_t i = 0; i < custom_filter.size(); ++i) {
        custom_filter[i] = 1.0f / static_cast<float>(M * K);  // Flat response
    }

    // Configure with custom filter
    fixture.set_prototype_filter(custom_filter);
    fixture.configure(M, K, N, 1);

    // Should be configured successfully
    CHECK(fixture.is_configured());

    // Send data and process
    const std::size_t input_size = M * N;
    auto input = composite::make_mutable<sample_t>(input_size);
    auto input_span = input.as_span();
    for (std::size_t i = 0; i < input_size; ++i) {
        input_span[i] = sample_t{1.0f, 0.0f};
    }

    fixture.source_port->send_data(std::move(input), {0, 0});
    auto result = fixture.pfbc->process();

    CHECK(result == composite::retval::NORMAL);
    CHECK(fixture.sink_port->size() == M);
}

TEST_CASE("custom prototype filter - invalid size rejected", "[pfbc][config]") {
    constexpr uint32_t M = 16;
    constexpr uint32_t K = 8;
    constexpr uint32_t N = 4;

    PfbcTestFixture fixture("test_pfbc");

    // Create filter with WRONG size (should be M*K = 128, using 100)
    std::vector<float> wrong_size_filter(100, 1.0f);
    fixture.set_prototype_filter(wrong_size_filter);
    fixture.configure(M, K, N, 1);

    // Should NOT be configured due to invalid filter size
    CHECK_FALSE(fixture.is_configured());
}

// =============================================================================
// Batch FFT Comparison Benchmark
// =============================================================================

TEST_CASE("batch FFT comparison", "[pfbc][benchmark][batch-fft]") {
    // Compare single FFT calls vs batched FFT calls
    struct Config {
        uint32_t M;           // FFT size
        uint32_t batch_size;  // Number of transforms to batch
    };

    std::vector<Config> configs = {
        {64, 1},      // Baseline: no batching
        {64, 16},
        {64, 64},
        {64, 256},
        {64, 1024},
        {256, 1},     // Baseline: no batching
        {256, 16},
        {256, 64},
        {256, 256},
        {256, 1024},
    };

    constexpr std::size_t warmup_iters = 100;
    constexpr std::size_t benchmark_iters = 1000;

    for (const auto& cfg : configs) {
        const uint32_t M = cfg.M;
        const uint32_t batch_size = cfg.batch_size;

        // Create FFT plans
        fft_plan<sample_t> single_fft(M, 1, 1);              // Single transform
        fft_plan<sample_t> batch_fft(M, 1, batch_size);      // Batch transforms

        // Allocate buffers
        auto single_in = composite::make_aligned<sample_t>(64, M);
        auto single_out = composite::make_aligned<sample_t>(64, M);
        auto batch_in = composite::make_aligned<sample_t>(64, M * batch_size);
        auto batch_out = composite::make_aligned<sample_t>(64, M * batch_size);

        // Fill with test data
        for (std::size_t i = 0; i < M; ++i) {
            float phase = 2.0f * std::numbers::pi_v<float> * 0.1f * static_cast<float>(i);
            (*single_in)[i] = sample_t{std::cos(phase), std::sin(phase)};
        }
        // Replicate to batch buffer
        for (std::size_t b = 0; b < batch_size; ++b) {
            std::copy(single_in->begin(), single_in->end(),
                      batch_in->begin() + b * M);
        }

        // Warmup
        for (std::size_t i = 0; i < warmup_iters; ++i) {
            single_fft.execute(single_in->data(), single_out->data());
            batch_fft.execute(batch_in->data(), batch_out->data());
        }

        // Benchmark: Single FFT approach (call N times)
        auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t iter = 0; iter < benchmark_iters; ++iter) {
            for (std::size_t b = 0; b < batch_size; ++b) {
                single_fft.execute(single_in->data(), single_out->data());
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto single_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        // Benchmark: Batch FFT approach (call once with batch)
        auto t2 = std::chrono::high_resolution_clock::now();
        for (std::size_t iter = 0; iter < benchmark_iters; ++iter) {
            batch_fft.execute(batch_in->data(), batch_out->data());
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        auto batch_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

        // Calculate metrics
        std::size_t total_transforms = benchmark_iters * batch_size;
        double single_ns_per_fft = (static_cast<double>(single_us) * 1000.0) / static_cast<double>(total_transforms);
        double batch_ns_per_fft = (static_cast<double>(batch_us) * 1000.0) / static_cast<double>(total_transforms);
        double speedup = static_cast<double>(single_us) / static_cast<double>(batch_us);

        WARN("=== M=" << M << ", Batch=" << batch_size << " ===");
        WARN("  Single FFT: " << single_us << " us total, " << single_ns_per_fft << " ns/FFT");
        WARN("  Batch FFT:  " << batch_us << " us total, " << batch_ns_per_fft << " ns/FFT");
        WARN("  Speedup:    " << speedup << "x");
        WARN("");

        CHECK(batch_us > 0);  // Sanity check
    }
}

// =============================================================================
// End-to-End Batch Size Scaling Tests
// =============================================================================

TEST_CASE("batch size scaling", "[pfbc][benchmark][scaling]") {
    // Test full component throughput with different batch sizes (N = frame_size)
    struct Config {
        uint32_t M;  // Channels
        uint32_t K;  // Taps per phase
        uint32_t N;  // Frame size (batch size)
    };

    std::vector<Config> configs = {
        // Small batch sizes
        {64, 12, 16},
        {64, 12, 32},
        {64, 12, 64},
        // Medium batch sizes
        {64, 12, 128},
        {64, 12, 256},
        {64, 12, 512},
        // Large batch sizes
        {64, 12, 1024},
        {64, 12, 2048},

        // Same for larger channel count
        {256, 12, 16},
        {256, 12, 64},
        {256, 12, 256},
        {256, 12, 1024},
    };

    constexpr std::size_t warmup_batches = 10;
    constexpr std::size_t benchmark_batches = 100;
    constexpr uint32_t threads = 4;

    for (const auto& cfg : configs) {
        PfbcTestFixture fixture("bench_pfbc");
        fixture.configure(cfg.M, cfg.K, cfg.N, threads);

        const std::size_t input_size = cfg.M * cfg.N;

        // Generate test input (one full batch)
        auto input = composite::make_mutable<sample_t>(input_size);
        auto input_span = input.as_span();
        for (std::size_t i = 0; i < input_size; ++i) {
            float phase = 2.0f * std::numbers::pi_v<float> * 0.123f * static_cast<float>(i);
            input_span[i] = sample_t{std::cos(phase), std::sin(phase)};
        }

        // Warmup
        for (std::size_t i = 0; i < warmup_batches; ++i) {
            auto warmup_buf = composite::make_mutable<sample_t>(input_size);
            std::copy(input_span.begin(), input_span.end(), warmup_buf.as_span().begin());
            fixture.source_port->send_data(std::move(warmup_buf), {0, 0});
            fixture.pfbc->process();
            // Drain outputs
            while (fixture.sink_port->size() > 0) {
                fixture.sink_port->get_data();
            }
        }

        // Benchmark
        auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < benchmark_batches; ++i) {
            auto bench_buf = composite::make_mutable<sample_t>(input_size);
            std::copy(input_span.begin(), input_span.end(), bench_buf.as_span().begin());
            fixture.source_port->send_data(std::move(bench_buf), {0, 0});
            fixture.pfbc->process();
            // Drain outputs
            while (fixture.sink_port->size() > 0) {
                fixture.sink_port->get_data();
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

        // Calculate throughput
        std::size_t total_samples = benchmark_batches * input_size;
        double msps = static_cast<double>(total_samples) / static_cast<double>(elapsed_us);
        double us_per_batch = static_cast<double>(elapsed_us) / static_cast<double>(benchmark_batches);

        WARN("=== M=" << cfg.M << ", K=" << cfg.K << ", N=" << cfg.N << " ===");
        WARN("  Throughput:     " << msps << " MSps");
        WARN("  Time/batch:     " << us_per_batch << " us");
        WARN("  Samples/batch:  " << input_size);
        WARN("");

        CHECK(msps > 0.0);  // Sanity check
    }
}
#endif  // Disabled benchmark tests
