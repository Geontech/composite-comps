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

#include "component.hpp"
#include "parsers/sdds_parser.hpp"
#include "parsers/vita49_parser.hpp"
#include "parsers/vita49dot1_parser.hpp"

#include <source_location>

pkt_parser::pkt_parser(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_port(&m_out_port);
    using enum composite::properties::config_type;
    add_struct_property("signal_overrides", &m_signal_overrides, [this](auto& set, auto* prop) {
        set.add_property("center_frequency", &prop->center_frequency).configurability(RUNTIME);
        set.add_property("bandwidth", &prop->bandwidth).configurability(RUNTIME);
        set.add_property("sample_rate", &prop->sample_rate).configurability(RUNTIME);
        set.add_struct_property("data_format", &prop->data_format, [this](auto& set, auto* prop) {
            set.add_property("is_complex", &prop->is_complex).configurability(RUNTIME);
            set.add_property("type", &prop->type).configurability(RUNTIME).change_listener([this]() {
                return (m_signal_overrides.data_format.type == "signed_integer") ||
                       (m_signal_overrides.data_format.type == "unsigned_integer") ||
                       (m_signal_overrides.data_format.type == "floating_point");
            });
            set.add_property("bit_width", &prop->bit_width).configurability(RUNTIME);
            set.add_property("endianness", &prop->endianness).configurability(RUNTIME).change_listener([this]() {
                return (m_signal_overrides.data_format.endianness == "big") || (m_signal_overrides.data_format.endianness == "little");
            });
        }).configurability(RUNTIME);
        set.add_property("transport", &prop->transport).configurability(RUNTIME).change_listener([this]() {
            return (m_signal_overrides.transport == "sdds") ||
                   (m_signal_overrides.transport == "vita49") ||
                   (m_signal_overrides.transport == "vita49.1");
        });
    }).configurability(RUNTIME);
}

auto pkt_parser::property_change_handler() -> void {
    logger()->trace(std::source_location::current().function_name());

    // Initialize parser registry
    m_parsers.clear();
    m_active_parser = nullptr;
    m_unknown_protocol_warned = false;

    // Register parsers based on configuration
    if (m_signal_overrides.transport.empty() || m_signal_overrides.transport == "sdds") {
        m_parsers.push_back(std::make_unique<parsers::sdds_parser>(m_signal_overrides));
    }
    // Note: Order matters - try V49.1 before V49 since V49.1 is more specific
    if (m_signal_overrides.transport.empty() || m_signal_overrides.transport == "vita49.1") {
        m_parsers.push_back(std::make_unique<parsers::vita49dot1_parser>(m_signal_overrides));
    }
    if (m_signal_overrides.transport.empty() || m_signal_overrides.transport == "vita49") {
        m_parsers.push_back(std::make_unique<parsers::vita49_parser>(m_signal_overrides));
    }

    logger()->trace("Registered {} protocol parsers", m_parsers.size());
}

auto pkt_parser::process() -> composite::retval {
    using enum composite::retval;

    // Get input data if available
    auto [data, _, __] = m_in_port.get_data();
    if (!data) {
        return NORMAL;
    }

    // Protocol detection: try each registered parser until one matches
    if (!m_active_parser) {
        for (auto& parser : m_parsers) {
            if (parser->can_parse(data)) {
                m_active_parser = parser.get();
                logger()->info("Detected protocol: {}", parser->name());
                break;
            }
        }

        if (!m_active_parser) {
            // Rate limit: only warn once about unknown protocol to avoid log spam
            if (!m_unknown_protocol_warned) {
                logger()->warn("Unknown packet protocol; dumping {} bytes and continuing", data.size());
                m_unknown_protocol_warned = true;
            }
            return NORMAL;
        }
    }

    // Parse packet using active parser
    auto result = m_active_parser->parse(data, m_metadata);

    // Log any warnings from parser
    if (result.warning.has_value()) {
        logger()->warn("{}", result.warning.value());
    }

    // Send metadata if changed
    if (m_metadata != result.metadata) {
        m_metadata = result.metadata;
        logger()->trace("Sending updated metadata:\n{}", m_metadata.to_string());
        m_out_port.send_metadata(m_metadata);
        m_init_metadata = true;
    }

    // Send data if parser says we should and metadata has been initialized
    if (m_init_metadata && result.should_send) [[likely]] {
        m_out_port.send_data(std::move(result.payload), result.timestamp);
    }

    return NORMAL;
}

extern "C" {
    auto create(std::string_view id) -> std::shared_ptr<composite::component> {
        return std::make_shared<pkt_parser>(id);
    }
}
