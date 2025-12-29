// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../component.hpp"

#include <composite/composite.hpp>
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>

#include <complex>
#include <vector>
#include <cmath>
#include <numbers>

#include <windows.hpp>

using sample_t = std::complex<float>;
using Catch::Matchers::WithinAbs;

// =============================================================================
// Test Fixture - Provides source and sink ports for testing
// =============================================================================

struct HalfrateTestFixture {
    std::shared_ptr<halfrate> uut;
    std::shared_ptr<composite::output_port<composite::immutable_buffer<sample_t>>> source_port;
    std::shared_ptr<composite::input_port<composite::immutable_buffer<sample_t>>> sink_port;

    HalfrateTestFixture() {
        uut = std::make_shared<halfrate>("test_halfrate");
        source_port = std::make_shared<composite::output_port<composite::immutable_buffer<sample_t>>>("source");
        sink_port = std::make_shared<composite::input_port<composite::immutable_buffer<sample_t>>>("sink");

        // Connect ports
        auto* in_port = uut->get_port<composite::input_port<composite::mutable_buffer<sample_t>>>("data_in");
        auto* out_port = uut->get_port<composite::output_port<composite::immutable_buffer<sample_t>>>("data_out");

        REQUIRE(in_port != nullptr);
        REQUIRE(out_port != nullptr);

        source_port->connect(in_port);
        out_port->connect(sink_port.get());
    }

    void reset() {
        uut = std::make_shared<halfrate>("test_halfrate");
        source_port = std::make_shared<composite::output_port<composite::immutable_buffer<sample_t>>>("source");
        sink_port = std::make_shared<composite::input_port<composite::immutable_buffer<sample_t>>>("sink");

        auto* in_port = uut->get_port<composite::input_port<composite::mutable_buffer<sample_t>>>("data_in");
        auto* out_port = uut->get_port<composite::output_port<composite::immutable_buffer<sample_t>>>("data_out");

        source_port->connect(in_port);
        out_port->connect(sink_port.get());
    }

    void send_data(const std::vector<sample_t>& data) {
        auto data_vec = std::make_shared<std::vector<sample_t>>(data);
        composite::immutable_buffer<sample_t> buf(data_vec);
        source_port->send_data(buf, composite::timestamp{});
    }

    auto get_output() -> std::vector<sample_t> {
        if (sink_port->size() == 0) {
            return {};
        }
        auto [data, ts, meta] = sink_port->get_data();
        return std::vector<sample_t>(data.begin(), data.end());
    }
};

// =============================================================================
// Helper Functions
// =============================================================================

// Generate a complex sinusoid: cos(2πft) + j*sin(2πft)
auto generate_tone(std::size_t num_samples, double freq_normalized) -> std::vector<sample_t> {
    std::vector<sample_t> signal(num_samples);
    for (std::size_t n = 0; n < num_samples; ++n) {
        auto phase = 2.0 * std::numbers::pi * freq_normalized * n;
        signal[n] = {
            static_cast<float>(std::cos(phase)),
            static_cast<float>(std::sin(phase))
        };
    }
    return signal;
}

// Compute average power in a signal
auto compute_power(const std::vector<sample_t>& signal) -> float {
    float power = 0.0f;
    for (const auto& s : signal) {
        power += std::norm(s);
    }
    return power / signal.size();
}

// =============================================================================
// Component Integration Tests
// =============================================================================

TEST_CASE("halfrate component - initialization", "[halfrate][integration]") {
    HalfrateTestFixture fixture;
    REQUIRE_NOTHROW(fixture.uut->property_change_handler());
}

TEST_CASE("halfrate component - decimation by 2", "[halfrate][integration]") {
    HalfrateTestFixture fixture;
    fixture.uut->property_change_handler();

    // Create input buffer with 1024 samples
    constexpr std::size_t n_in = 1024;
    auto input = std::vector<sample_t>(n_in, {1.0f, 0.5f});

    fixture.send_data(input);

    REQUIRE(fixture.uut->process() == composite::retval::NORMAL);

    auto output = fixture.get_output();
    REQUIRE(output.size() == n_in / 2); // Decimated by 2
}

TEST_CASE("halfrate component - passband tone preservation", "[halfrate][integration]") {
    HalfrateTestFixture fixture;
    fixture.uut->property_change_handler();

    // Generate a low-frequency tone (well within passband: Fs/8)
    constexpr std::size_t n_in = 2048;
    constexpr double freq = 0.125; // Fs/8 (passband for Fs/4 cutoff filter)
    auto input = generate_tone(n_in, freq);

    fixture.send_data(input);
    REQUIRE(fixture.uut->process() == composite::retval::NORMAL);

    auto output = fixture.get_output();
    REQUIRE(output.size() == n_in / 2);

    // Check that average power per sample is preserved (allowing some filter ripple)
    auto input_power = compute_power(input);
    auto output_power = compute_power(output);

    // Measure filter gain in dB (comparing per-sample power, not total energy)
    // For a decimator, output samples have same amplitude as input, so gain should be near 0 dB
    float gain_db = 10.0f * std::log10(output_power / input_power);
    CHECK(gain_db > -1.0f);
    CHECK(gain_db < 1.0f);
}

TEST_CASE("halfrate component - stopband attenuation", "[halfrate][integration]") {
    HalfrateTestFixture fixture;
    fixture.uut->property_change_handler();

    // Generate a high-frequency tone (in stopband: 0.4*Fs)
    constexpr std::size_t n_in = 2048;
    constexpr double freq = 0.4; // 0.4*Fs (well into stopband)
    auto input = generate_tone(n_in, freq);

    fixture.send_data(input);
    REQUIRE(fixture.uut->process() == composite::retval::NORMAL);

    auto output = fixture.get_output();
    REQUIRE(output.size() == n_in / 2);

    // Output power should be significantly attenuated
    auto input_power = compute_power(input);
    auto output_power = compute_power(output);

    // For M=3 halfband, expect ~40 dB stopband attenuation (reasonable for 13-tap filter)
    float attenuation_db = 10.0f * std::log10(output_power / input_power);
    CHECK(attenuation_db < -35.0f);
}

TEST_CASE("halfrate component - multiple process calls (streaming)", "[halfrate][integration]") {
    HalfrateTestFixture fixture_stream;
    fixture_stream.uut->property_change_handler();

    // Reference fixture processes the entire stream in one shot
    HalfrateTestFixture fixture_ref;
    fixture_ref.uut->property_change_handler();

    constexpr std::size_t block_size = 128;
    constexpr std::size_t num_blocks = 8;

    std::vector<sample_t> full_input(block_size * num_blocks);
    for (std::size_t i = 0; i < full_input.size(); ++i) {
        full_input[i] = {static_cast<float>(i), static_cast<float>(i * 2)};
    }

    // Streamed processing
    std::vector<sample_t> streamed_outputs;
    for (std::size_t b = 0; b < num_blocks; ++b) {
        auto begin = full_input.begin() + b * block_size;
        auto end = begin + block_size;
        std::vector<sample_t> block(begin, end);

        fixture_stream.send_data(block);
        auto ret = fixture_stream.uut->process();
        REQUIRE(ret == composite::retval::NORMAL);

        auto output = fixture_stream.get_output();
        REQUIRE(output.size() == block_size / 2);
        streamed_outputs.insert(streamed_outputs.end(), output.begin(), output.end());
    }

    // Reference single-block processing
    fixture_ref.send_data(full_input);
    REQUIRE(fixture_ref.uut->process() == composite::retval::NORMAL);
    auto reference_output = fixture_ref.get_output();

    REQUIRE(streamed_outputs.size() == reference_output.size());
    for (std::size_t i = 0; i < streamed_outputs.size(); ++i) {
        CHECK_THAT(streamed_outputs[i].real(), WithinAbs(reference_output[i].real(), 1e-4f));
        CHECK_THAT(streamed_outputs[i].imag(), WithinAbs(reference_output[i].imag(), 1e-4f));
    }
}

TEST_CASE("halfrate component - repeated initialization", "[halfrate][integration]") {
    HalfrateTestFixture fixture;

    // Verify that property_change_handler can be called multiple times
    REQUIRE_NOTHROW(fixture.uut->property_change_handler());
    REQUIRE_NOTHROW(fixture.uut->property_change_handler());
    REQUIRE_NOTHROW(fixture.uut->property_change_handler());
}

TEST_CASE("halfrate component - impulse response", "[halfrate][integration]") {
    HalfrateTestFixture fixture;
    fixture.uut->property_change_handler();

    // Create impulse: [1, 0, 0, 0, ...]
    constexpr std::size_t n_in = 512;
    auto input = std::vector<sample_t>(n_in, {0.0f, 0.0f});
    input[0] = {1.0f, 0.0f}; // Unit impulse

    fixture.send_data(input);
    REQUIRE(fixture.uut->process() == composite::retval::NORMAL);

    auto output = fixture.get_output();
    REQUIRE(output.size() == n_in / 2);

    // Output should be the impulse response of the filter; peak near group delay (M outputs)
    constexpr std::size_t semi_len = 3; // default property value
    auto peak_it = std::max_element(
        output.begin(), output.end(),
        [](const sample_t& a, const sample_t& b) { return std::norm(a) < std::norm(b); });
    auto peak_idx = static_cast<std::size_t>(std::distance(output.begin(), peak_it));
    CHECK(peak_idx >= semi_len - 1);
    CHECK(peak_idx <= semi_len + 1);
    CHECK_THAT(peak_it->real(), WithinAbs(0.5f, 5e-2f));
    CHECK_THAT(peak_it->imag(), WithinAbs(0.0f, 5e-2f));

    // Most of the energy should be in the first few taps after the peak
    float early_energy = 0.0f;
    for (std::size_t i = 0; i < 20 && i < output.size(); ++i) {
        early_energy += std::norm(output[i]);
    }

    float total_energy = compute_power(output) * output.size();

    // At least 95% of energy should be in first 20 samples
    CHECK(early_energy > total_energy * 0.95f);
}

TEST_CASE("halfrate component - impulse parity", "[halfrate][integration]") {
    HalfrateTestFixture fixture;
    fixture.uut->property_change_handler();

    constexpr std::size_t n_in = 512;
    constexpr std::size_t semi_len = 3; // default property

    // Impulse at index 0
    auto input0 = std::vector<sample_t>(n_in, {0.0f, 0.0f});
    input0[0] = {1.0f, 0.0f};
    fixture.send_data(input0);
    REQUIRE(fixture.uut->process() == composite::retval::NORMAL);
    auto out0 = fixture.get_output();

    // Expect center-only response at m = M
    CHECK_THAT(out0[semi_len].real(), WithinAbs(0.5f, 5e-2f));

    // Impulse at index 1
    auto input1 = std::vector<sample_t>(n_in, {0.0f, 0.0f});
    input1[1] = {1.0f, 0.0f};
    fixture.send_data(input1);
    REQUIRE(fixture.uut->process() == composite::retval::NORMAL);
    auto out1 = fixture.get_output();

    // Expect odd-tap response at k=-1 (index center-1 in the prototype).
    // This matches the odd-lane impulse response at output index M.
    constexpr std::size_t L = 4 * semi_len + 1;
    constexpr std::size_t center = L / 2;
    constexpr int k = -1;
    auto window = windows::hamming<float>(L, false);
    auto sinc_val = std::sin(0.5f * std::numbers::pi_v<float> * k)
        / (std::numbers::pi_v<float> * k);
    float expected_tap = sinc_val * window->at(center - 1);

    CHECK_THAT(out1[semi_len].real(), WithinAbs(expected_tap, 5e-2f));
    CHECK_THAT(out1[semi_len].imag(), WithinAbs(0.0f, 5e-2f));
}


TEST_CASE("halfrate component - DC signal", "[halfrate][integration]") {
    HalfrateTestFixture fixture;
    fixture.uut->property_change_handler();

    // DC signal (constant value)
    constexpr std::size_t n_in = 1024;
    constexpr sample_t dc_value = {5.0f, 3.0f};
    auto input = std::vector<sample_t>(n_in, dc_value);

    fixture.send_data(input);
    REQUIRE(fixture.uut->process() == composite::retval::NORMAL);

    auto output = fixture.get_output();
    REQUIRE(output.size() == n_in / 2);

    // DC should pass through (it's at f=0, well within passband)
    // After filter settling, output should be close to DC value
    // Check last half of output (after transients)
    std::size_t start = output.size() / 2;
    for (std::size_t i = start; i < output.size(); ++i) {
        CHECK_THAT(output[i].real(), WithinAbs(dc_value.real(), 1.5f));
        CHECK_THAT(output[i].imag(), WithinAbs(dc_value.imag(), 1.5f));
    }
}

TEST_CASE("halfrate component - alternating signal", "[halfrate][integration]") {
    HalfrateTestFixture fixture;
    fixture.uut->property_change_handler();

    // Alternating signal: [+1, -1, +1, -1, ...] (Nyquist frequency)
    constexpr std::size_t n_in = 1024;
    auto input = std::vector<sample_t>(n_in);
    for (std::size_t i = 0; i < n_in; ++i) {
        input[i] = (i % 2 == 0) ? sample_t{1.0f, 0.0f} : sample_t{-1.0f, 0.0f};
    }

    fixture.send_data(input);
    REQUIRE(fixture.uut->process() == composite::retval::NORMAL);

    auto output = fixture.get_output();
    REQUIRE(output.size() == n_in / 2);

    // Alternating at Nyquist (f=0.5) is well into stopband
    // Output power should be heavily attenuated
    auto input_power = compute_power(input);
    auto output_power = compute_power(output);

    // For M=3 halfband, expect ~40 dB stopband attenuation
    float attenuation_db = 10.0f * std::log10(output_power / input_power);
    CHECK(attenuation_db < -35.0f);
}

TEST_CASE("halfrate component - zero input", "[halfrate][integration]") {
    HalfrateTestFixture fixture;
    fixture.uut->property_change_handler();

    // All zeros input
    constexpr std::size_t n_in = 1024;
    auto input = std::vector<sample_t>(n_in, {0.0f, 0.0f});

    fixture.send_data(input);
    REQUIRE(fixture.uut->process() == composite::retval::NORMAL);

    auto output = fixture.get_output();
    REQUIRE(output.size() == n_in / 2);

    // Output should be all zeros
    for (const auto& sample : output) {
        CHECK_THAT(sample.real(), WithinAbs(0.0f, 1e-6f));
        CHECK_THAT(sample.imag(), WithinAbs(0.0f, 1e-6f));
    }
}

TEST_CASE("halfrate component - metadata propagation and sample rate update", "[halfrate][integration][metadata]") {
    HalfrateTestFixture fixture;
    fixture.uut->property_change_handler();

    // Create test data
    constexpr std::size_t n_in = 1024;
    auto input_data = std::vector<sample_t>(n_in, {1.0f, 0.0f});
    auto data_vec = std::make_shared<std::vector<sample_t>>(input_data);
    composite::immutable_buffer<sample_t> buf(data_vec);

    // Create metadata with input sample rate
    composite::metadata input_metadata;
    input_metadata.sample_rate = 10e6;  // 10 MHz input sample rate
    input_metadata.format.is_complex = true;
    input_metadata.format.type = composite::data_type::floating_point;
    input_metadata.format.bit_width = 32;

    // Send metadata and data
    fixture.source_port->send_metadata(input_metadata);
    fixture.source_port->send_data(buf, composite::timestamp{});

    // Process
    REQUIRE(fixture.uut->process() == composite::retval::NORMAL);

    // Get output with metadata
    REQUIRE(fixture.sink_port->size() > 0);
    auto [output_data, output_ts, output_meta] = fixture.sink_port->get_data();

    // Verify data size
    REQUIRE(output_data.size() == n_in / 2);

    // Verify metadata was propagated
    REQUIRE(output_meta.has_value());

    // Verify sample rate was halved (decimation by 2)
    CHECK_THAT(output_meta->sample_rate, WithinAbs(5e6, 1.0));  // Should be 5 MHz

    // Verify other metadata fields were preserved
    CHECK(output_meta->format.is_complex == true);
    CHECK(output_meta->format.type == composite::data_type::floating_point);
    CHECK(output_meta->format.bit_width == 32);
}

TEST_CASE("halfrate component - window and tap configurations", "[halfrate][integration][properties]") {
    struct config {
        uint32_t semi_length;
        const char* window;
        float stopband_db_limit;
    };

    std::vector<config> configs = {
        {6, "HAMMING", -40.0f},
        {6, "BLACKMAN_HARRIS", -40.0f}
    };

    for (const auto& cfg : configs) {
        HalfrateTestFixture fixture;
        fixture.uut->set_properties({
            {"filter_semi_length", std::to_string(cfg.semi_length)},
            {"window", cfg.window}
        });
        fixture.uut->property_change_handler();

        constexpr double stopband_tone = 0.4;
        constexpr std::size_t n_in = 4096;
        auto input = generate_tone(n_in, stopband_tone);

        fixture.send_data(input);
        REQUIRE(fixture.uut->process() == composite::retval::NORMAL);

        auto output = fixture.get_output();
        REQUIRE(output.size() == n_in / 2);

        auto input_power = compute_power(input);
        auto output_power = compute_power(output);
        float attenuation_db = 10.0f * std::log10(output_power / input_power);

        CHECK(attenuation_db < cfg.stopband_db_limit);
    }
}
