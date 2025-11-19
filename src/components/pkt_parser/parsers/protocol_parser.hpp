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

#include <composite/buffers/buffer.hpp>
#include <composite/core/metadata.hpp>
#include <composite/core/timestamp.hpp>

#include <optional>
#include <string>
#include <string_view>

namespace parsers {

/**
 * @brief Base interface for protocol parsers
 *
 * Each protocol parser implements detection, parsing, and metadata extraction
 * for a specific packet protocol (SDDS, VITA 49, etc.)
 */
class protocol_parser {
public:
    /**
     * @brief Result of parsing a packet
     */
    struct parse_result {
        composite::immutable_buffer<uint8_t> payload;  ///< Header-stripped data (zero-copy slice)
        composite::metadata metadata;                   ///< Extracted/updated metadata
        composite::timestamp timestamp;                 ///< Packet timestamp
        bool should_send{true};                         ///< false for context-only packets
        std::optional<std::string> warning;             ///< Warnings (dropped packets, etc)
    };

    virtual ~protocol_parser() = default;

    /**
     * @brief Attempt to detect if this protocol matches the data
     *
     * @param data Raw packet data
     * @return true if this parser can handle the packet format
     */
    virtual auto can_parse(const composite::immutable_buffer<uint8_t>& data) const -> bool = 0;

    /**
     * @brief Parse packet and extract payload, metadata, timestamp
     *
     * @param data Raw packet data with headers
     * @param current_metadata Current metadata state (will be updated)
     * @return Parsed result with payload, updated metadata, timestamp
     */
    virtual auto parse(
        const composite::immutable_buffer<uint8_t>& data,
        const composite::metadata& current_metadata
    ) -> parse_result = 0;

    /**
     * @brief Get protocol name for logging
     */
    virtual auto name() const -> std::string_view = 0;

}; // class protocol_parser

} // namespace parsers
