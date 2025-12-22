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

    // Check that energy is preserved (allowing some filter loss)
    auto input_power = compute_power(input);
    auto output_power = compute_power(output);

    // Power should be roughly preserved (within 40% loss for filter + decimation)
    CHECK(output_power > input_power * 0.4f);
    CHECK(output_power < input_power * 1.5f);
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

    CHECK(output_power < input_power * 0.05f); // Less than 5% of input power
}

TEST_CASE("halfrate component - multiple process calls (streaming)", "[halfrate][integration]") {
    HalfrateTestFixture fixture;
    fixture.uut->property_change_handler();

    // Process multiple small blocks to verify history handling
    constexpr std::size_t block_size = 128;
    constexpr std::size_t num_blocks = 8;

    std::vector<sample_t> all_outputs;

    for (std::size_t b = 0; b < num_blocks; ++b) {
        // Generate ramping signal
        auto input = std::vector<sample_t>(block_size);
        for (std::size_t i = 0; i < block_size; ++i) {
            auto idx = b * block_size + i;
            input[i] = {static_cast<float>(idx), static_cast<float>(idx * 2)};
        }

        fixture.send_data(input);
        auto ret = fixture.uut->process();
        REQUIRE(ret == composite::retval::NORMAL);

        auto output = fixture.get_output();
        REQUIRE(output.size() == block_size / 2);

        // Collect outputs
        all_outputs.insert(all_outputs.end(), output.begin(), output.end());
    }

    // Should have produced num_blocks * (block_size/2) outputs
    REQUIRE(all_outputs.size() == num_blocks * (block_size / 2));

    // Verify continuity (no discontinuities from history errors)
    for (std::size_t i = 1; i < all_outputs.size(); ++i) {
        auto diff_real = std::abs(all_outputs[i].real() - all_outputs[i-1].real());
        auto diff_imag = std::abs(all_outputs[i].imag() - all_outputs[i-1].imag());

        // Differences should be smooth (not huge jumps)
        CHECK(diff_real < 100.0f);
        CHECK(diff_imag < 200.0f);
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

    // Output should be the impulse response of the filter
    // Check that it's non-zero near the beginning and decays
    CHECK(std::abs(output[0]) > 0.005f); // Non-trivial output

    // Most of the energy should be in the first few taps
    float early_energy = 0.0f;
    for (std::size_t i = 0; i < 20 && i < output.size(); ++i) {
        early_energy += std::norm(output[i]);
    }

    float total_energy = compute_power(output) * output.size();

    // At least 90% of energy should be in first 20 samples
    CHECK(early_energy > total_energy * 0.9f);
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

    CHECK(output_power < input_power * 0.05f); // Less than 5% power
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
