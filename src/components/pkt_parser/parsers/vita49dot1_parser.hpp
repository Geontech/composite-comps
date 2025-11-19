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
#include "vita49_parser.hpp"
#include "config.hpp"

#include <cstdint>

namespace parsers {

/**
 * @brief VITA 49.1 protocol parser
 */
class vita49dot1_parser : public protocol_parser {
public:
    explicit vita49dot1_parser(const signal_overrides& overrides);
    ~vita49dot1_parser() override = default;

    auto can_parse(const composite::immutable_buffer<uint8_t>& data) const -> bool override;
    auto parse(
        const composite::immutable_buffer<uint8_t>& data,
        const composite::metadata& current_metadata
    ) -> parse_result override;
    auto name() const -> std::string_view override { return "vita49.1"; }

private:
    signal_overrides m_overrides;
    vita49_parser m_vita49_parser;  // Delegate to standard V49 parser

}; // class vita49dot1_parser

} // namespace parsers
