/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Debug component for VITA-49 packets from pkt_builder.
 * Parses and logs packet headers, context fields, and payload info.
 */

#pragma once

#include <composite/composite.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

class pkt_debug : public composite::component {
    using input_port_t = composite::input_port<composite::immutable_buffer<uint8_t>>;

public:
    explicit pkt_debug(std::string_view id = "pkt_debug");
    ~pkt_debug() override;

    auto initialize() -> void override;
    auto stop() -> void override;
    auto process() -> composite::retval override;

private:
    // Context information parsed from VITA-49 context packets
    struct context_info {
        double bandwidth_hz{0.0};
        double center_frequency_hz{0.0};
        double sample_rate_hz{0.0};
        bool is_complex{false};
        uint8_t data_format{0};      // VITA-49 format code
        uint8_t bit_width{0};
        bool valid{false};           // Has context been received?
    };

    struct stream_stats {
        // Packet counts
        uint64_t data_packets{0};
        uint64_t context_packets{0};
        uint64_t total_bytes{0};
        uint64_t total_payload_bytes{0};

        // Sequence tracking
        uint16_t last_data_seq{0};
        uint16_t last_context_seq{0};
        uint64_t seq_errors{0};
        bool first_packet{true};

        // Last known context info for this stream
        context_info context;
    };

    void parse_packet(const uint8_t* data, std::size_t size);
    void parse_context_fields(const uint8_t* data, std::size_t size, std::size_t offset,
                              uint32_t cif0, stream_stats& stats);
    void log_stats();

    // Helper to format data type as human-readable string
    static auto format_type_string(const context_info& ctx) -> std::string;
    static auto format_frequency(double hz) -> std::string;

    input_port_t m_in_port{"data_in"};

    // Configuration
    unsigned int m_log_interval_sec{5};
    bool m_verbose{false};

    // Per-stream statistics
    std::map<uint32_t, stream_stats> m_stream_stats;
    std::mutex m_stats_mutex;

    // Metrics thread
    std::jthread m_metrics_thread;
    std::condition_variable m_stop_cv;
    std::mutex m_stop_mutex;
    std::atomic<bool> m_stopping{false};
    uint64_t m_total_packets{0};
}; // class pkt_debug

extern "C" {
auto create(std::string_view id, std::string_view type = "") -> std::shared_ptr<composite::component>;
}
