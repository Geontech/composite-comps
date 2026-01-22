/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Test harness component that measures observed sample rate.
 * Prints rate statistics to console for verification.
 */

#pragma once

#include <composite/composite.hpp>

#include <chrono>
#include <complex>
#include <cstdint>

template<typename T>
class rate_monitor : public composite::component {
    using input_port_t = composite::input_port<composite::immutable_buffer<T>>;

public:
    explicit rate_monitor(std::string_view id = "rate_monitor");
    ~rate_monitor() override = default;

    auto initialize() -> void override;
    auto start() -> void override;
    auto stop() -> void override;
    auto process() -> composite::retval override;

private:
    void print_stats();

    input_port_t m_in_port{"data_in"};

    // Configuration
    double m_report_interval_sec{1.0};  // How often to print stats

    // Measurement state
    std::chrono::steady_clock::time_point m_start_time;
    std::chrono::steady_clock::time_point m_last_report_time;
    uint64_t m_total_samples{0};
    uint64_t m_samples_since_report{0};
    uint64_t m_chunks_received{0};
    bool m_first_chunk{true};
};

// Type aliases
using rate_monitor_cf32 = rate_monitor<std::complex<float>>;
using rate_monitor_ci16 = rate_monitor<std::complex<int16_t>>;
using rate_monitor_ci8 = rate_monitor<std::complex<int8_t>>;
using rate_monitor_f32 = rate_monitor<float>;
using rate_monitor_i16 = rate_monitor<int16_t>;

#ifndef UNIT_TESTS
extern "C" {
auto create(std::string_view id, std::string_view type = "cf32") -> std::shared_ptr<composite::component>;
}
#endif
