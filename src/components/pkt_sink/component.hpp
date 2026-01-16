/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Simple VITA-49 packet sink for debugging pkt_builder output.
 * Logs packet headers, stream IDs, timestamps, and payload info.
 */

#pragma once

#include <composite/composite.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

class pkt_sink : public composite::component {
    using input_port_t = composite::input_port<composite::immutable_buffer<uint8_t>>;

public:
    explicit pkt_sink(std::string_view id = "pkt_sink");
    ~pkt_sink() override;

    auto initialize() -> void override;
    auto stop() -> void override;
    auto process() -> composite::retval override;

private:
    struct stream_stats {
        uint64_t data_packets{0};
        uint64_t context_packets{0};
        uint64_t total_bytes{0};
        uint64_t total_payload_bytes{0};
        uint16_t last_data_seq{0};
        uint16_t last_context_seq{0};
        uint64_t seq_errors{0};
        bool first_packet{true};
    };

    void parse_packet(const uint8_t* data, std::size_t size);
    void log_stats();

    input_port_t m_in_port{"data_in"};

    // Configuration
    unsigned int m_log_interval_sec{5};
    bool m_verbose{false};

    // Per-stream statistics
    std::map<uint32_t, stream_stats> m_stream_stats;
    std::mutex m_stats_mutex;

    // Metrics thread
    std::jthread m_metrics_thread;
    uint64_t m_total_packets{0};
};

extern "C" {
auto create(std::string_view id) -> std::shared_ptr<composite::component>;
}