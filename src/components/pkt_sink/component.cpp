/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Simple VITA-49 packet sink for debugging pkt_builder output.
 */

#include "component.hpp"

#include <spdlog/spdlog.h>

#include <bit>
#include <cstring>

namespace {

// VITA-49 packet types
constexpr uint8_t PKT_TYPE_DATA = 0x0;              // IF Data without Stream ID
constexpr uint8_t PKT_TYPE_DATA_WITH_STREAM = 0x1;  // IF Data with Stream ID
constexpr uint8_t PKT_TYPE_CONTEXT = 0x4;           // IF Context (always has Stream ID)

// Read big-endian uint32
uint32_t read_be32(const uint8_t* p) {
    uint32_t val;
    std::memcpy(&val, p, 4);
    if constexpr (std::endian::native == std::endian::little) {
        return std::byteswap(val);
    }
    return val;
}

// Read big-endian uint64
uint64_t read_be64(const uint8_t* p) {
    uint64_t val;
    std::memcpy(&val, p, 8);
    if constexpr (std::endian::native == std::endian::little) {
        return std::byteswap(val);
    }
    return val;
}

const char* packet_type_str(uint8_t type) {
    switch (type) {
        case PKT_TYPE_DATA: return "DATA";
        case PKT_TYPE_DATA_WITH_STREAM: return "DATA+SID";
        case PKT_TYPE_CONTEXT: return "CONTEXT";
        default: return "UNKNOWN";
    }
}

} // namespace

pkt_sink::pkt_sink(std::string_view id)
    : composite::component(id) {
    using enum composite::properties::config_type;
    add_port(&m_in_port);
    add_property("log_interval_sec", m_log_interval_sec, RUNTIME).units("s");
    add_property("verbose", m_verbose, RUNTIME);
}

pkt_sink::~pkt_sink() {
    stop();
}

auto pkt_sink::initialize() -> void {
    {
        std::lock_guard<std::mutex> lock(m_stats_mutex);
        m_stream_stats.clear();
        m_total_packets = 0;
    }

    m_metrics_thread = std::jthread([this](std::stop_token st) {
        using namespace std::chrono_literals;
        while (!st.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::seconds(m_log_interval_sec));
            if (st.stop_requested()) break;
            log_stats();
        }
    });

    logger()->info("pkt_sink initialized - VITA-49 packet debugging sink");
}

auto pkt_sink::stop() -> void {
    if (m_metrics_thread.joinable()) {
        m_metrics_thread.request_stop();
        m_metrics_thread.join();
    }
    log_stats();
    composite::component::stop();
}

auto pkt_sink::process() -> composite::retval {
    using enum composite::retval;

    auto [data, ts, meta] = m_in_port.get_data();
    if (!data) {
        return NOOP;
    }

    parse_packet(data.data(), data.size());
    m_total_packets++;

    return NORMAL;
}

void pkt_sink::parse_packet(const uint8_t* data, std::size_t size) {
    if (size < 4) {
        logger()->warn("Packet too small: {} bytes", size);
        return;
    }

    // Parse VITA-49 header (first 32-bit word)
    uint32_t header = read_be32(data);

    uint8_t pkt_type = (header >> 28) & 0x0F;
    bool has_class_id = (header >> 27) & 0x01;
    bool has_trailer = (header >> 26) & 0x01;
    // bool tsm = (header >> 24) & 0x01;
    uint8_t tsi = (header >> 22) & 0x03;
    uint8_t tsf = (header >> 20) & 0x03;
    uint16_t pkt_count = (header >> 16) & 0x0F;
    uint16_t pkt_size_words = header & 0xFFFF;

    std::size_t offset = 4;

    // Stream ID (if present)
    // Data packets: 0x1 has stream ID, 0x0 does not
    // Context packets (0x4): always have stream ID
    uint32_t stream_id = 0;
    bool has_stream_id = (pkt_type == PKT_TYPE_DATA_WITH_STREAM ||
                          pkt_type == PKT_TYPE_CONTEXT);
    if (has_stream_id && offset + 4 <= size) {
        stream_id = read_be32(data + offset);
        offset += 4;
    }

    // Class ID (if present)
    uint32_t oui = 0;
    uint16_t icc = 0, pcc = 0;
    if (has_class_id && offset + 8 <= size) {
        uint32_t class_id_hi = read_be32(data + offset);
        uint32_t class_id_lo = read_be32(data + offset + 4);
        oui = class_id_hi & 0x00FFFFFF;
        icc = (class_id_lo >> 16) & 0xFFFF;
        pcc = class_id_lo & 0xFFFF;
        offset += 8;
    }

    // Timestamp
    uint32_t ts_int = 0;
    uint64_t ts_frac = 0;
    if (tsi != 0 && offset + 4 <= size) {
        ts_int = read_be32(data + offset);
        offset += 4;
    }
    if (tsf != 0 && offset + 8 <= size) {
        ts_frac = read_be64(data + offset);
        offset += 8;
    }

    // Calculate payload size
    std::size_t header_size = offset;
    std::size_t trailer_size = has_trailer ? 4 : 0;
    std::size_t payload_size = (pkt_size_words * 4) - header_size - trailer_size;
    if (payload_size > size) {
        payload_size = size > header_size ? size - header_size : 0;
    }

    // Update statistics
    bool is_context = (pkt_type == PKT_TYPE_CONTEXT);
    {
        std::lock_guard<std::mutex> lock(m_stats_mutex);
        auto& stats = m_stream_stats[stream_id];

        if (is_context) {
            // Check context sequence
            if (!stats.first_packet) {
                uint16_t expected = (stats.last_context_seq + 1) & 0x0F;
                if (pkt_count != expected) {
                    stats.seq_errors++;
                }
            }
            stats.last_context_seq = pkt_count;
            stats.context_packets++;
        } else {
            // Check data sequence
            if (!stats.first_packet) {
                uint16_t expected = (stats.last_data_seq + 1) & 0x0F;
                if (pkt_count != expected) {
                    stats.seq_errors++;
                }
            }
            stats.last_data_seq = pkt_count;
            stats.data_packets++;
        }

        stats.total_bytes += size;
        stats.total_payload_bytes += payload_size;
        stats.first_packet = false;
    }

    // Verbose logging
    if (m_verbose) {
        logger()->info(
            "PKT type={} stream_id={} seq={} size={} words payload={} bytes "
            "class_id={:06x}:{:04x}:{:04x} ts={}s+{}ps",
            packet_type_str(pkt_type),
            stream_id,
            pkt_count,
            pkt_size_words,
            payload_size,
            oui, icc, pcc,
            ts_int,
            ts_frac
        );

        // For context packets, try to parse some fields
        if (is_context && offset + 4 <= size) {
            uint32_t cif0 = read_be32(data + offset);
            logger()->debug("  CIF0={:#010x}", cif0);
        }
    }
}

void pkt_sink::log_stats() {
    std::lock_guard<std::mutex> lock(m_stats_mutex);

    if (m_stream_stats.empty()) {
        logger()->trace("No packets received yet");
        return;
    }

    logger()->info("=== pkt_sink stats: {} total packets, {} streams ===",
                   m_total_packets, m_stream_stats.size());

    for (const auto& [stream_id, stats] : m_stream_stats) {
        double avg_payload = stats.data_packets > 0
            ? static_cast<double>(stats.total_payload_bytes) / stats.data_packets
            : 0.0;

        logger()->info(
            "  stream_id={}: data_pkts={} ctx_pkts={} total_bytes={} "
            "avg_payload={:.1f} seq_errors={}",
            stream_id,
            stats.data_packets,
            stats.context_packets,
            stats.total_bytes,
            avg_payload,
            stats.seq_errors
        );
    }
}

extern "C" {
auto create(std::string_view id) -> std::shared_ptr<composite::component> {
    return std::make_shared<pkt_sink>(id);
}
}