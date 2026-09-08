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
#include <cstddef>
#include <cstdint>

template<typename T>
class rate_monitor : public composite::component {
    using input_port_t = composite::input_port<composite::immutable_buffer<T>>;

public:
    explicit rate_monitor(std::string_view id = "rate_monitor");
    ~rate_monitor() override = default;

    auto initialize() -> void override;
    auto process() -> composite::retval override;

protected:
    // composite 0.5.2 makes start()/stop() final; per-run state belongs in the
    // worker lifecycle hooks instead.
    auto on_worker_start() -> void override;
    auto on_worker_stop() -> void override;
    // A source that signals end-of-stream now reaches the sink here, so a
    // completed run reports its totals without waiting for stop().
    auto on_end_of_stream() -> void override;

private:
    void print_stats();
    void print_final_stats();

    input_port_t m_in_port{"data_in"};

    // Configuration
    double m_report_interval_sec{1.0};  // How often to print stats
    // Divisor from buffer elements to samples. The typed variants receive one
    // element per sample and leave this at 1; the byte variant receives a raw
    // byte stream (sigmf_source emits std::byte), so a cf32 recording needs 8.
    double m_elements_per_sample{1.0};

    // Measurement state
    std::chrono::steady_clock::time_point m_start_time;
    std::chrono::steady_clock::time_point m_last_report_time;
    uint64_t m_total_samples{0};
    uint64_t m_samples_since_report{0};
    uint64_t m_chunks_received{0};
    bool m_first_chunk{true};
    bool m_final_printed{false};

    // MUST be the last data member: stops the worker while this object is still
    // fully alive (~component stops too late). See component.hpp auto_stop.
    composite::component::auto_stop m_auto_stop{*this};
};

// Type aliases
using rate_monitor_cf32 = rate_monitor<std::complex<float>>;
using rate_monitor_ci16 = rate_monitor<std::complex<int16_t>>;
using rate_monitor_ci8 = rate_monitor<std::complex<int8_t>>;
using rate_monitor_f32 = rate_monitor<float>;
using rate_monitor_i16 = rate_monitor<int16_t>;
// Raw byte stream, for sources that emit untyped bytes such as sigmf_source.
using rate_monitor_bytes = rate_monitor<std::byte>;

// The factory ABI is emitted by COMPOSITE_REGISTER_COMPONENT in component.cpp.
