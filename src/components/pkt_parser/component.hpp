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
    struct_props::signal_overrides m_signal_overrides;

    // Members
    std::vector<std::unique_ptr<parsers::protocol_parser>> m_parsers;
    parsers::protocol_parser* m_active_parser{nullptr};
    composite::metadata m_metadata;
    bool m_init_metadata{false};
    bool m_unknown_protocol_warned{false};

}; // class pkt_parser

template<>
struct composite::properties::property_traits<struct_props::signal_overrides::format> {
    static void register_fields(composite::properties::property_set& ps, struct_props::signal_overrides::format& f) {
        using enum composite::properties::config_type;
        ps.add("is_complex", f.is_complex, RUNTIME);
        ps.add("type", f.type, RUNTIME).change_listener([&f]() {
            return (f.type == "signed_integer") ||
                   (f.type == "unsigned_integer") ||
                   (f.type == "floating_point");
        });
        ps.add("bit_width", f.bit_width, RUNTIME);
        ps.add("endianness", f.endianness, RUNTIME).change_listener([&f]() {
            return (f.endianness == "big") || (f.endianness == "little");
        });
    }
};

template<>
struct composite::properties::property_traits<struct_props::signal_overrides> {
    static void register_fields(composite::properties::property_set& ps, struct_props::signal_overrides& s) {
        using enum composite::properties::config_type;
        ps.add("center_frequency", s.center_frequency, RUNTIME);
        ps.add("bandwidth", s.bandwidth, RUNTIME);
        ps.add("sample_rate", s.sample_rate, RUNTIME);
        ps.add("data_format", s.data_format, RUNTIME);
        ps.add("transport", s.transport, RUNTIME).change_listener([&s]() {
            return (s.transport == "sdds") ||
                   (s.transport == "vita49") ||
                   (s.transport == "vita49.1");
        });
    }
};
