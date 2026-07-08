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

#include <chrono>
#include <cstdint>
#include <memory>
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
    };

    std::unordered_map<uint32_t, stream_state> m_stream_states;

    // Helper methods
    auto get_stream_id(const composite::metadata& metadata) -> uint32_t;
    auto get_or_create_stream_state(uint32_t stream_id) -> stream_state&;
    auto apply_defaults(composite::metadata& metadata) -> void;
    auto validate_metadata(stream_state& state, const composite::metadata& metadata) -> void;
    auto build_context_packet(stream_state& state, const composite::metadata& metadata)
        -> std::shared_ptr<std::vector<uint8_t>>;
    auto build_data_packet(stream_state& state, std::span<const std::byte> payload,
                           const composite::timestamp& ts, const composite::data_format& fmt)
        -> std::shared_ptr<std::vector<uint8_t>>;

    // MUST be last: stops the framework worker before any member above destructs.
    composite::component::auto_stop m_auto_stop{*this};

}; // class pkt_builder
