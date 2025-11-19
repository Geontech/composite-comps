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

#include "config.hpp"
#include "parsers/protocol_parser.hpp"

#include <composite/core/component.hpp>
#include <composite/buffers/buffer.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

class pkt_parser : public composite::component {
    using input_port_t = composite::input_port<composite::immutable_buffer<uint8_t>>;
    using output_port_t = composite::output_port<composite::immutable_buffer<uint8_t>>;
public:
    explicit pkt_parser(std::string_view id);
    ~pkt_parser() override = default;
    auto property_change_handler() -> void override;
    auto process() -> composite::retval override;

private:
    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    ::signal_overrides m_signal_overrides;

    // Members
    std::vector<std::unique_ptr<parsers::protocol_parser>> m_parsers;
    parsers::protocol_parser* m_active_parser{nullptr};
    composite::metadata m_metadata;
    bool m_init_metadata{false};
    bool m_unknown_protocol_warned{false};

}; // class pkt_parser
