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
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

/**
 * @brief VITA-49 packet builder component with dynamic datatype support
 *
 * Converts signal data into VITA-49 formatted packets with proper headers,
 * timestamps, and context packets. Datatype is determined at runtime from
 * incoming metadata.format, allowing dynamic format changes via REST API.
 *
 * Input: immutable_buffer<std::byte> with metadata.format describing the data
 * Output: immutable_buffer<uint8_t> containing VITA-49 packets
 */
class pkt_builder : public composite::component {
    using input_port_t = composite::input_port<composite::immutable_buffer<std::byte>>;
    using output_port_t = composite::output_port<composite::immutable_buffer<uint8_t>>;

public:
    explicit pkt_builder(std::string_view id);
    ~pkt_builder() override = default;
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
        composite::metadata last_metadata;
        uint32_t stream_id{0};
        uint16_t data_packet_count{0};      // Sequence 0-15 for data packets
        uint16_t context_packet_count{0};   // Sequence 0-15 for context packets
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
    auto metadata_changed(const composite::metadata& current, const composite::metadata& previous) -> bool;
    auto should_send_context(stream_state& state, const composite::metadata& metadata) -> bool;
    auto build_context_packet(const stream_state& state, const composite::metadata& metadata) -> std::shared_ptr<std::vector<uint8_t>>;
    auto build_data_packet(stream_state& state, const composite::immutable_buffer<std::byte>& payload, const composite::timestamp& ts) -> std::shared_ptr<std::vector<uint8_t>>;
    auto write_header(uint8_t* dest, bool is_context, bool has_stream_id, bool has_class_id,
                      bool has_timestamp, uint16_t packet_size, uint16_t packet_count) -> size_t;
    auto write_stream_id(uint8_t* dest, uint32_t stream_id) -> size_t;
    auto write_class_id(uint8_t* dest) -> size_t;
    auto write_timestamp(uint8_t* dest, const composite::timestamp& ts) -> size_t;
    auto write_context_fields(uint8_t* dest, const composite::metadata& metadata) -> size_t;

}; // class pkt_builder

#ifndef UNIT_TESTS
extern "C" {
auto create(std::string_view id) -> std::shared_ptr<composite::component>;
}
#endif