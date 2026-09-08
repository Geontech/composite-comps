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

namespace parsers {

/**
 * @brief SDDS protocol parser
 *
 * Parses SDDS (Spectrum Device Data Stream) packets.
 * SDDS packets are fixed size (1080 bytes: 56 byte header + 1024 byte payload).
 */
class sdds_parser : public protocol_parser {
public:
    explicit sdds_parser(const struct_props::signal_overrides& overrides);
    ~sdds_parser() override = default;

    auto can_parse(const composite::immutable_buffer<uint8_t>& data) const -> bool override;
    auto parse(
        const composite::immutable_buffer<uint8_t>& data,
        const composite::metadata& current_metadata
    ) -> parse_result override;
    auto name() const -> std::string_view override { return "sdds"; }
    auto on_activated() -> void override {
        m_emitted = false;
        m_seq_initialized = false;  // sequence restarts; the first packet skips the gap check
        m_gap_warn = false;         // re-arm the one-shot warnings for the (likely new) stream
        m_anomaly_warn = false;
    }

private:
    struct_props::signal_overrides m_overrides;
    std::optional<std::endian> m_ov_endianness;
    uint16_t m_pkt_count{0};
    bool m_seq_initialized{false};  ///< false until a packet has seeded m_pkt_count
    bool m_emitted{false};  ///< false until this parser has published metadata since (re)activation
    bool m_gap_warn{false};      ///< one-shot: a sequence gap was seen (the counter carries the rate)
    bool m_anomaly_warn{false};  ///< one-shot: inconsistent pp_id/seq parity flags seen

}; // class sdds_parser

} // namespace parsers
