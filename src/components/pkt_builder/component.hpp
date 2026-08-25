/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * This file is part of composite-comps.
 *
 * composite-comps is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * composite-comps is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include <composite/composite.hpp>
#include <composite/metrics/metrics.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

/**
 * @brief VITA-49 packet builder component with dynamic datatype support
 *
 * Converts signal data into VITA-49 formatted packets with proper headers,
 * timestamps, and context packets — the egress counterpart of pkt_parser.
 * The datatype is determined at runtime from incoming metadata.format, so
 * format changes flow through without reconfiguration.
 *
 * Packets are built with the same vrtgen packing classes pkt_parser parses
 * with, so the wire layout cannot drift between the two: what pkt_builder
 * emits, pkt_parser (and any spec-conforming VITA 49.2 receiver) reads back.
 *
 * Context packets are sent on the first packet of a stream, whenever the
 * signal metadata actually changes, and periodically (context_interval_ms).
 * The incoming shared metadata instance is used for change detection, so
 * steady-state packets do no metadata work.
 *
 * Input:  immutable_buffer<std::byte> with metadata.format describing the data
 * Output: immutable_buffer<uint8_t> containing VITA-49 packets (the input
 *         metadata rides along for downstream routing, e.g. udp_sink)
 */
class pkt_builder : public composite::component {
    friend struct PktBuilderTestFixture;
    using input_port_t = composite::input_port<composite::immutable_buffer<std::byte>>;
    using output_port_t = composite::output_port<composite::immutable_buffer<uint8_t>>;

public:
    explicit pkt_builder(std::string_view id);
    ~pkt_builder() override = default;
    auto property_change_handler(const composite::properties::json& diff) -> void override;
    auto process() -> composite::retval override;
    // Best-effort flush of an in-flight packetization; per-stream residues (< 4 bytes each)
    // cannot be emitted without fabricating pad samples and are counted as dropped.
    auto on_end_of_stream() -> void override;

private:
    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties - packet configuration
    uint32_t m_context_interval_ms{1000};     // Send context every N milliseconds
    uint32_t m_max_payload_size{1024};        // Max samples per packet
    std::string m_stream_id_key{"stream_id"}; // Metadata key for stream ID
    uint32_t m_default_stream_id{0};          // Default if not in metadata
    uint32_t m_oui{0x00123456};               // Organizationally Unique Identifier
    uint16_t m_packet_class_code{0};          // Packet class code

    // Properties - default metadata (used when not provided by upstream)
    double m_default_center_frequency{0.0};
    double m_default_bandwidth{0.0};
    double m_default_sample_rate{0.0};

    // Properties - optional features
    bool m_include_class_id{true};
    bool m_include_timestamp{true};
    bool m_warn_on_missing_metadata{true};
    bool m_payload_clamp_warned{false};  // one-time warning that max_payload_size was clamped to the VITA word limit

    // Per-stream state tracking
    struct stream_state {
        uint32_t stream_id{0};
        composite::metadata_ptr last_in_meta;  // incoming instance the state was built from
        composite::metadata_ptr out_meta;      // effective metadata (defaults applied); rides the output
        uint16_t data_packet_count{0};         // Sequence 0-15 for data packets
        uint16_t context_packet_count{0};      // Sequence 0-15 for context packets
        std::chrono::steady_clock::time_point last_context_time{};
        bool first_packet{true};
        bool warned_missing_metadata{false};
        bool warned_bad_format{false};   ///< one-shot: unsupported format on this stream
        bool warned_invalid_rf{false};   ///< one-shot: non-finite/out-of-range RF values reset
        bool warned_unaligned{false};    ///< one-shot: partial-sample bytes dropped
        // Whole-sample bytes (< one 32-bit word) left over from the previous buffer: carried
        // into the next packet rather than discarded or zero-padded (VITA payloads are
        // word-granular). At most 3 bytes; dropped (counted) on a signal change, since the
        // format they were captured under may no longer apply.
        std::vector<std::byte> residue;
        std::chrono::steady_clock::time_point last_seen{};  ///< for LRU eviction at the cap
    };

    /// Distinct-stream state cap: ids come from upstream metadata, so unbounded distinct ids
    /// must not grow memory without limit. At the cap the least-recently-seen stream is
    /// EVICTED (its next packet recreates state, resending context with a fresh sequence) —
    /// a hard reject would let 64 disposable ids permanently lock legitimate new streams out.
    static constexpr std::size_t MAX_STREAM_STATES = 64;

    /// Chunks emitted per process() invocation: with a writable output, one enormous input
    /// and a tiny max_payload_size must not pin the worker in the emit loop (stop/property
    /// handling runs between process() calls). The pending state carries across calls.
    static constexpr std::size_t EMIT_BUDGET_PER_CALL = 64;

    /// One input buffer mid-packetization. Packetizing is RESUMABLE: emit_pending() sends the
    /// (optional) context packet and word-aligned whole-sample chunks only while the output
    /// has room, returning AWAIT_OUTPUT when full so the worker parks on the reverse doorbell
    /// instead of drop-flooding send_data() — no packet of a partially-emitted buffer is lost.
    /// Per-chunk timestamps are derived from the total emitted-sample offset (drift-free and
    /// trivially resumable).
    struct pending_send {
        composite::immutable_buffer<std::byte> data;
        composite::timestamp base_ts{};
        uint32_t stream_id{0};
        std::shared_ptr<std::vector<uint8_t>> context;  // non-null until sent
        std::vector<std::byte> lead;  // residue carried from the previous buffer (< 4 bytes)
        std::size_t offset{0};        // consumed bytes of `data` (lead excluded)
        std::size_t payload_end{0};
        std::size_t sample_size{0};
        std::size_t max_chunk_bytes{0};
        double rate{0.0};  // sanitized; 0 = no per-chunk timestamp advance
        composite::metadata_ptr out_meta;
        composite::data_format fmt{};
    };
    std::optional<pending_send> m_pending;
    auto emit_pending() -> composite::retval;

    std::unordered_map<uint32_t, stream_state> m_stream_states;
    bool m_stream_cap_warned{false};  ///< one-shot: distinct-stream cap reached (see MAX_STREAM_STATES)

    // Stream-id lookup cache keyed on the incoming shared metadata instance: steady state
    // (same instance every packet) skips the annotation find + parse.
    composite::metadata_ptr m_sid_cache_meta;
    uint32_t m_sid_cache_id{0};

    // Observability: input buffers dropped (unsupported format / stream-cap overflow), and
    // trailing bytes dropped to keep packets whole-sample and word-aligned.
    composite::metrics::counter<uint64_t>* m_packets_dropped{nullptr};
    composite::metrics::counter<uint64_t>* m_bytes_dropped{nullptr};

    // Helper methods
    auto get_stream_id(const composite::metadata& metadata) -> uint32_t;
    /// nullptr when the distinct-stream cap is reached (caller drops + counts).
    auto find_or_create_stream_state(uint32_t stream_id) -> stream_state*;
    auto apply_defaults(composite::metadata& metadata) -> void;
    auto validate_metadata(stream_state& state, const composite::metadata& metadata) -> void;
    auto build_context_packet(stream_state& state, const composite::metadata& metadata,
                              const composite::timestamp& ts)
        -> std::shared_ptr<std::vector<uint8_t>>;
    auto build_data_packet(stream_state& state, std::span<const std::byte> lead,
                           std::span<const std::byte> payload, const composite::timestamp& ts,
                           const composite::data_format& fmt)
        -> std::shared_ptr<std::vector<uint8_t>>;

    // MUST be last: stops the framework worker before any member above destructs.
    composite::component::auto_stop m_auto_stop{*this};

}; // class pkt_builder
