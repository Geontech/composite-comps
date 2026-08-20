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

#include "protocol_parser.hpp"
#include "config.hpp"

#include <bit>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace parsers {

/**
 * @brief VITA 49 protocol parser
 *
 * Parses VITA 49 (VRT) packets including:
 * - Data packets (IF/baseband samples)
 * - Context packets (signal metadata)
 * - Extension data packets
 *
 * Handles timestamp conversion for both real-time and sample-count modes.
 */
class vita49_parser : public protocol_parser {
public:
    /// @param transport_annotation the value stamped into annotations["protocol"] on every
    /// metadata (re)build. The V49.1 wrapper delegates here and passes "v49.1", so the
    /// change-detection compare sees the SAME annotation the component actually publishes —
    /// a post-hoc rewrite in the wrapper would make every context packet look changed.
    explicit vita49_parser(const struct_props::signal_overrides& overrides,
                           std::string_view transport_annotation = "v49");
    ~vita49_parser() override = default;

    auto can_parse(const composite::immutable_buffer<uint8_t>& data) const -> bool override;
    auto parse(
        const composite::immutable_buffer<uint8_t>& data,
        const composite::metadata& current_metadata
    ) -> parse_result override;
    auto name() const -> std::string_view override { return "vita49"; }
    auto on_activated() -> void override {
        m_emitted = false;
        m_tsf_warn = false;  // re-arm the sample-count warning for the (likely new) stream
        m_ext_warn = false;  // re-arm the extension-data warning too
        m_pkt_count = 0;     // sequence restarts; the first packet skips the gap check
    }

private:
    struct_props::signal_overrides m_overrides;
    std::string m_transport;  ///< annotations["protocol"] value ("v49", or "v49.1" when wrapped)
    // The string-valued overrides (data type, endianness) resolved to enums ONCE at
    // construction, so parse() applies them with a branch + assign instead of re-comparing
    // strings on every packet. std::nullopt == "not overridden".
    std::optional<composite::data_type> m_ov_type;
    std::optional<std::endian> m_ov_endianness;
    uint16_t m_pkt_count{0};
    bool m_tsf_warn{false};
    bool m_ext_warn{false};  ///< one-shot: an extension-data packet reached parse()
    bool m_emitted{false};  ///< false until this parser has published metadata since (re)activation

}; // class vita49_parser

} // namespace parsers
