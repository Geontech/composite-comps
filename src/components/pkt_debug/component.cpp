/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Debug component for VITA-49 packets from pkt_builder.
 */

#include "component.hpp"

#include <spdlog/spdlog.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <format>

namespace {

// VITA-49 packet types
constexpr uint8_t PKT_TYPE_DATA = 0x0;              // IF Data without Stream ID
constexpr uint8_t PKT_TYPE_DATA_WITH_STREAM = 0x1;  // IF Data with Stream ID
constexpr uint8_t PKT_TYPE_CONTEXT = 0x4;           // IF Context (always has Stream ID)

// CIF0 bit positions for fields we care about
constexpr uint32_t CIF0_BANDWIDTH = (1U << 31);
constexpr uint32_t CIF0_RF_REF_FREQ = (1U << 27);
constexpr uint32_t CIF0_SAMPLE_RATE = (1U << 23);
constexpr uint32_t CIF0_DATA_FORMAT = (1U << 7);

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

// Convert VITA-49 fixed-point (radix at bit 20) to double
double fixed_to_double(uint64_t fixed) {
    return static_cast<double>(fixed) / static_cast<double>(1ULL << 20);
}

const char* packet_type_str(uint8_t type) {
    switch (type) {
        case PKT_TYPE_DATA: return "DATA";
        case PKT_TYPE_DATA_WITH_STREAM: return "DATA+SID";
        case PKT_TYPE_CONTEXT: return "CONTEXT";
        default: return "UNKNOWN";
    }
}

// Convert VITA-49 format code to string
const char* format_code_str(uint8_t code) {
    switch (code) {
        case 0x01: return "signed_int";
        case 0x11: return "unsigned_int";
        case 0x08: return "float";
        default: return "unknown";
    }
}

} // namespace

pkt_debug::pkt_debug(std::string_view id)
    : composite::component(id) {
    using enum composite::properties::config_type;
    add_port(&m_in_port);
    add_property("log_interval_sec", m_log_interval_sec, RUNTIME).units("s");
    add_property("verbose", m_verbose, RUNTIME);
}

pkt_debug::~pkt_debug() {
    stop();
}

auto pkt_debug::initialize() -> void {
    {
        std::lock_guard<std::mutex> lock(m_stats_mutex);
        m_stream_stats.clear();
        m_total_packets = 0;
    }
    m_stopping = false;

    m_metrics_thread = std::jthread([this](std::stop_token st) {
        while (!st.stop_requested()) {
            std::unique_lock<std::mutex> lock(m_stop_mutex);
            // Wait for timeout or stop signal
            if (m_stop_cv.wait_for(lock, std::chrono::seconds(m_log_interval_sec),
                                   [&] { return m_stopping.load(); })) {
                break;  // Stop was requested
            }
            if (st.stop_requested()) break;
            log_stats();
        }
    });

    logger()->info("pkt_debug initialized - VITA-49 packet debugging component");
}

auto pkt_debug::stop() -> void {
    if (m_metrics_thread.joinable()) {
        m_stopping = true;
        m_stop_cv.notify_all();
        m_metrics_thread.request_stop();
        m_metrics_thread.join();
    }
    log_stats();
    composite::component::stop();
}

auto pkt_debug::process() -> composite::retval {
    using enum composite::retval;

    auto [data, ts, meta] = m_in_port.get_data();
    if (!data) {
        return NOOP;
    }

    parse_packet(data.data(), data.size());
    m_total_packets++;

    return NORMAL;
}

void pkt_debug::parse_packet(const uint8_t* data, std::size_t size) {
    if (size < 4) {
        logger()->warn("Packet too small: {} bytes", size);
        return;
    }

    // Parse VITA-49 header (first 32-bit word)
    uint32_t header = read_be32(data);

    uint8_t pkt_type = (header >> 28) & 0x0F;
    bool has_class_id = (header >> 27) & 0x01;
    bool has_trailer = (header >> 26) & 0x01;
    uint8_t tsi = (header >> 22) & 0x03;
    uint8_t tsf = (header >> 20) & 0x03;
    uint16_t pkt_count = (header >> 16) & 0x0F;
    uint16_t pkt_size_words = header & 0xFFFF;

    std::size_t offset = 4;

    // Stream ID (if present)
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
        oui = (class_id_hi >> 8) & 0x00FFFFFF;
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

            // Parse context fields if we have CIF0
            if (offset + 4 <= size) {
                uint32_t cif0 = read_be32(data + offset);
                parse_context_fields(data, size, offset + 4, cif0, stats);

                // Log context info in verbose mode
                if (m_verbose && stats.context.valid) {
                    logger()->info(
                        "CONTEXT stream_id={} seq={} format={} sample_rate={} bandwidth={} center_freq={}",
                        stream_id,
                        pkt_count,
                        format_type_string(stats.context),
                        format_frequency(stats.context.sample_rate_hz),
                        format_frequency(stats.context.bandwidth_hz),
                        format_frequency(stats.context.center_frequency_hz)
                    );
                }
            }
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

    // Verbose logging for data packets
    if (m_verbose && !is_context) {
        logger()->debug(
            "DATA stream_id={} seq={} size={} words payload={} bytes "
            "class_id={:06x}:{:04x}:{:04x} ts={}s+{}ps",
            stream_id,
            pkt_count,
            pkt_size_words,
            payload_size,
            oui, icc, pcc,
            ts_int,
            ts_frac
        );
    }
}

void pkt_debug::parse_context_fields(const uint8_t* data, std::size_t size, std::size_t offset,
                                      uint32_t cif0, stream_stats& stats) {
    auto& ctx = stats.context;

    // Parse fields in order they appear (based on CIF0 bits, MSB to LSB)
    // The order matters because fields are packed sequentially

    // Bandwidth (CIF0 bit 31) - 64-bit fixed point
    if (cif0 & CIF0_BANDWIDTH) {
        if (offset + 8 <= size) {
            uint64_t bw_fixed = read_be64(data + offset);
            ctx.bandwidth_hz = fixed_to_double(bw_fixed);
            offset += 8;
        }
    }

    // RF Reference Frequency (CIF0 bit 27) - 64-bit fixed point
    if (cif0 & CIF0_RF_REF_FREQ) {
        if (offset + 8 <= size) {
            uint64_t rf_fixed = read_be64(data + offset);
            ctx.center_frequency_hz = fixed_to_double(rf_fixed);
            offset += 8;
        }
    }

    // Sample Rate (CIF0 bit 23) - 64-bit fixed point
    if (cif0 & CIF0_SAMPLE_RATE) {
        if (offset + 8 <= size) {
            uint64_t sr_fixed = read_be64(data + offset);
            ctx.sample_rate_hz = fixed_to_double(sr_fixed);
            offset += 8;
        }
    }

    // Data Packet Payload Format (CIF0 bit 7) - 64-bit
    if (cif0 & CIF0_DATA_FORMAT) {
        if (offset + 8 <= size) {
            uint64_t format = read_be64(data + offset);

            // Real/Complex (bit 63)
            ctx.is_complex = (format >> 63) & 0x01;

            // Data Item Format (bits 62-56)
            ctx.data_format = (format >> 56) & 0x7F;

            // Data Item Size (bits 20-16) - size in bits minus 1
            ctx.bit_width = ((format >> 16) & 0x1F) + 1;

            offset += 8;
        }
    }

    ctx.valid = true;
}

auto pkt_debug::format_type_string(const context_info& ctx) -> std::string {
    if (!ctx.valid) {
        return "unknown";
    }

    std::string result;
    result += ctx.is_complex ? "complex_" : "real_";
    result += format_code_str(ctx.data_format);
    result += "_";
    result += std::to_string(ctx.bit_width);
    return result;
}

auto pkt_debug::format_frequency(double hz) -> std::string {
    if (hz == 0.0) {
        return "0Hz";
    } else if (hz >= 1e9) {
        return std::format("{:.3f}GHz", hz / 1e9);
    } else if (hz >= 1e6) {
        return std::format("{:.3f}MHz", hz / 1e6);
    } else if (hz >= 1e3) {
        return std::format("{:.3f}kHz", hz / 1e3);
    } else {
        return std::format("{:.1f}Hz", hz);
    }
}

void pkt_debug::log_stats() {
    std::lock_guard<std::mutex> lock(m_stats_mutex);

    if (m_stream_stats.empty()) {
        logger()->trace("No packets received yet");
        return;
    }

    logger()->info("=== pkt_debug stats: {} total packets, {} streams ===",
                   m_total_packets, m_stream_stats.size());

    for (const auto& [stream_id, stats] : m_stream_stats) {
        double avg_payload = stats.data_packets > 0
            ? static_cast<double>(stats.total_payload_bytes) / stats.data_packets
            : 0.0;

        // Build format string if context is available
        std::string format_str = stats.context.valid
            ? format_type_string(stats.context)
            : "no_context";

        std::string freq_info;
        if (stats.context.valid) {
            freq_info = std::format(" sr={} bw={} cf={}",
                format_frequency(stats.context.sample_rate_hz),
                format_frequency(stats.context.bandwidth_hz),
                format_frequency(stats.context.center_frequency_hz));
        }

        logger()->info(
            "  stream_id={}: format={} data_pkts={} ctx_pkts={} bytes={} "
            "avg_payload={:.1f} seq_errs={}{}",
            stream_id,
            format_str,
            stats.data_packets,
            stats.context_packets,
            stats.total_bytes,
            avg_payload,
            stats.seq_errors,
            freq_info
        );
    }
}

#ifndef UNIT_TESTS
extern "C" {
auto create(std::string_view id, [[maybe_unused]] std::string_view type) -> std::shared_ptr<composite::component> {
    return std::make_shared<pkt_debug>(id);
}
}
#endif
