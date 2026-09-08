/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Test harness component that measures observed sample rate.
 */

#include "component.hpp"

#include <composite/core/register.hpp>

#include <spdlog/spdlog.h>

#include <format>
#include <iostream>

template<typename T>
rate_monitor<T>::rate_monitor(std::string_view id)
    : composite::component(id) {
    using enum composite::properties::config_type;
    add_port(&m_in_port);
    add_property("report_interval_sec", m_report_interval_sec, RUNTIME);
    add_property("elements_per_sample", m_elements_per_sample, RUNTIME);
}

template<typename T>
auto rate_monitor<T>::initialize() -> void {
    logger()->info("rate_monitor initialized, reporting every {} seconds", m_report_interval_sec);
}

template<typename T>
auto rate_monitor<T>::on_worker_start() -> void {
    m_total_samples = 0;
    m_samples_since_report = 0;
    m_chunks_received = 0;
    m_first_chunk = true;
    m_final_printed = false;
    // Seeded here as well as on the first chunk: without it a run that never
    // receives data measures from the steady_clock epoch and reports a nonsense
    // duration.
    m_start_time = std::chrono::steady_clock::now();
    m_last_report_time = m_start_time;

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "     RATE MONITOR - Sample Rate Test   \n";
    std::cout << "========================================\n";
    std::cout << std::format("Report interval: {:.1f} seconds\n", m_report_interval_sec);
    std::cout << "----------------------------------------\n";
}

template<typename T>
auto rate_monitor<T>::on_worker_stop() -> void {
    print_final_stats();
}

template<typename T>
auto rate_monitor<T>::on_end_of_stream() -> void {
    // The upstream source ran to completion. Report now, while the run is still
    // distinguishable from an operator-initiated stop.
    std::cout << "End of stream reached.\n";
    print_final_stats();
}

template<typename T>
void rate_monitor<T>::print_final_stats() {
    // Both end-of-stream and worker stop land here; totals print once per run.
    if (m_final_printed) {
        return;
    }
    m_final_printed = true;

    auto now = std::chrono::steady_clock::now();
    auto total_duration = std::chrono::duration<double>(now - m_start_time).count();

    std::cout << "----------------------------------------\n";
    std::cout << "FINAL STATISTICS:\n";
    if (m_first_chunk) {
        std::cout << "  No data received.\n";
    } else if (total_duration > 0) {
        double overall_rate = static_cast<double>(m_total_samples) / total_duration;
        std::cout << std::format("  Total samples:    {}\n", m_total_samples);
        std::cout << std::format("  Total chunks:     {}\n", m_chunks_received);
        std::cout << std::format("  Total duration:   {:.3f} sec\n", total_duration);
        std::cout << std::format("  Overall rate:     {:.0f} Hz ({:.3f} MHz)\n",
                                overall_rate, overall_rate / 1e6);
        if (m_chunks_received > 0) {
            std::cout << std::format("  Avg samples/chunk: {}\n", m_total_samples / m_chunks_received);
        }
    }
    std::cout << "========================================\n\n";
}

template<typename T>
void rate_monitor<T>::print_stats() {
    auto now = std::chrono::steady_clock::now();
    auto interval_duration = std::chrono::duration<double>(now - m_last_report_time).count();
    auto total_duration = std::chrono::duration<double>(now - m_start_time).count();

    if (interval_duration > 0) {
        double interval_rate = static_cast<double>(m_samples_since_report) / interval_duration;
        double overall_rate = static_cast<double>(m_total_samples) / total_duration;

        std::cout << std::format("[{:6.1f}s] Interval: {:12.0f} Hz ({:8.3f} MHz) | "
                                "Overall: {:12.0f} Hz ({:8.3f} MHz) | "
                                "Chunks: {:6} | Samples: {}\n",
                                total_duration,
                                interval_rate, interval_rate / 1e6,
                                overall_rate, overall_rate / 1e6,
                                m_chunks_received,
                                m_total_samples);
    }

    m_samples_since_report = 0;
    m_last_report_time = now;
}

template<typename T>
auto rate_monitor<T>::process() -> composite::retval {
    using enum composite::retval;

    auto [data, timestamp, metadata_opt] = m_in_port.get_data();
    if (!data) {
        return NOOP;
    }

    auto now = std::chrono::steady_clock::now();

    if (m_first_chunk) {
        m_start_time = now;
        m_last_report_time = now;
        m_first_chunk = false;
        std::cout << "First chunk received, starting measurement...\n";
    }

    // Count samples
    const double divisor = m_elements_per_sample > 0.0 ? m_elements_per_sample : 1.0;
    auto num_samples = static_cast<std::size_t>(static_cast<double>(data.size()) / divisor);
    m_total_samples += num_samples;
    m_samples_since_report += num_samples;
    m_chunks_received++;

    // Check if it's time to report
    auto time_since_report = std::chrono::duration<double>(now - m_last_report_time).count();
    if (time_since_report >= m_report_interval_sec) {
        print_stats();
    }

    return NORMAL;
}

// Explicit template instantiations
template class rate_monitor<std::complex<float>>;
template class rate_monitor<std::complex<int16_t>>;
template class rate_monitor<std::complex<int8_t>>;
template class rate_monitor<float>;
template class rate_monitor<int16_t>;
template class rate_monitor<std::byte>;

#ifndef UNIT_TESTS
// create() now takes create_args, and the template discriminator arrives as
// args.type() -- which is why a graph selects the variant with
// "args": {"type": "cf32"} rather than the old top-level "subtype".
COMPOSITE_REGISTER_COMPONENT([](std::string_view id, const composite::create_args& args)
                                 -> std::shared_ptr<composite::component> {
    const auto type = args.type();
    if (type == "cf32" || type.empty()) {
        return composite::make_component<rate_monitor_cf32>(id);
    }
    if (type == "ci16") {
        return composite::make_component<rate_monitor_ci16>(id);
    }
    if (type == "ci8") {
        return composite::make_component<rate_monitor_ci8>(id);
    }
    if (type == "f32") {
        return composite::make_component<rate_monitor_f32>(id);
    }
    if (type == "i16") {
        return composite::make_component<rate_monitor_i16>(id);
    }
    if (type == "bytes" || type == "u8") {
        return composite::make_component<rate_monitor_bytes>(id);
    }
    throw std::runtime_error(std::format("rate_monitor: unknown type '{}'", type));
})
#endif
