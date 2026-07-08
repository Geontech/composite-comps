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

#include <composite/core/register.hpp>

#include <source_location>

pkt_parser::pkt_parser(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_port(&m_out_port);
    using enum composite::properties::config_type;
    add_property("signal_overrides", m_signal_overrides, RUNTIME)
        .validate([](const struct_props::signal_overrides& v) {
            // Validate only when the optional override fields are actually set
            // (an empty string means "not overridden" and is always allowed).
            if (!v.data_format.type.empty() &&
                v.data_format.type != "signed_integer" &&
                v.data_format.type != "unsigned_integer" &&
                v.data_format.type != "floating_point") {
                return false;
            }
            if (!v.data_format.endianness.empty() &&
                v.data_format.endianness != "big" &&
                v.data_format.endianness != "little") {
                return false;
            }
            if (!v.transport.empty() &&
                v.transport != "sdds" &&
                v.transport != "vita49" &&
                v.transport != "vita49.1") {
                return false;
            }
            return true;
        });
    // Type-prefixed name, matching udp_source./framer. convention: the component_id label
    // identifies the instance, the prefix scopes the series to the component type so
    // cross-instance aggregation ("all pkt_parser drops") stays a name match.
    m_packets_dropped = &create_counter(
        "pkt_parser.packets_dropped", "Packets dropped (unknown protocol / malformed / unparseable)");
}

auto pkt_parser::property_change_handler(const composite::properties::json& diff) -> void {
    (void)diff;
    logger()->trace(std::source_location::current().function_name());

    // Initialize parser registry
    m_parsers.clear();
    m_active_parser = nullptr;
    m_drop_warned = false;
    m_consecutive_parse_failures = 0;

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
    auto pkt = m_in_port.try_get();
    if (!pkt) {
        // No input: NOOP so the worker arms the read-doorbell and parks until upstream
        // delivers, instead of busy-spinning process() and burning a core while idle. At
        // end-of-stream the base promotes this NOOP to FINISH (no buffered state to flush).
        return NOOP;
    }
    auto& [data, _, __] = *pkt;

    // The packet bytes are UNTRUSTED (raw UDP). A malformed/short datagram must
    // never propagate an exception out of process() — that would FINISH the
    // component (a one-packet remote DoS) — nor read out of bounds. Drop + count
    // instead. Parsers bounds-check their reads and throw std::out_of_range on a
    // packet that doesn't fit its claimed geometry; we catch it here.
    auto drop = [&](std::string_view why) {
        if (m_packets_dropped != nullptr) { m_packets_dropped->inc(); }
        if (!m_drop_warned) {  // rate-limited; the counter carries the real signal
            logger()->warn("pkt_parser: dropping packet ({} bytes): {}", data.size(), why);
            m_drop_warned = true;
        }
    };

    // Protocol detection: try each registered parser until one matches. can_parse
    // parses untrusted bytes, so guard it too (a malformed candidate -> not a match).
    if (!m_active_parser) {
        for (auto& parser : m_parsers) {
            try {
                if (parser->can_parse(data)) {
                    m_active_parser = parser.get();
                    m_active_parser->on_activated();  // force a fresh metadata publish on the next packet
                    logger()->info("Detected protocol: {}", parser->name());
                    break;
                }
            } catch (const std::exception& e) {
                logger()->debug("{} can_parse rejected packet: {}", parser->name(), e.what());
            }
        }

        if (!m_active_parser) {
            drop("unknown packet protocol");
            return NORMAL;
        }
    }

    // Parse packet using active parser. Lock-in does NOT trust subsequent packets:
    // each is re-validated and a bad one is dropped, not allowed to read OOB/throw.
    parsers::protocol_parser::parse_result result;
    try {
        result = m_active_parser->parse(data, m_metadata);
        m_consecutive_parse_failures = 0;  // this packet matches the locked-in protocol
    } catch (const std::exception& e) {
        drop(std::string{"parse error: "} + e.what());
        // A sustained run of failures means the stream's framing likely changed (e.g. a
        // warm-pool re-steer). Un-lock so the next packet re-runs detection and we self-heal.
        // A single good packet above resets the counter, so isolated corruption never trips it.
        if (++m_consecutive_parse_failures >= REDETECT_AFTER_FAILURES) {
            logger()->info("{} consecutive parse failures for protocol '{}' — re-detecting",
                           m_consecutive_parse_failures, m_active_parser->name());
            m_active_parser = nullptr;
            m_consecutive_parse_failures = 0;
            m_drop_warned = false;  // re-arm the drop warning for the (likely new) stream
        }
        return NORMAL;
    }

    // Log any warnings from parser
    if (result.warning.has_value()) {
        logger()->warn("{}", result.warning.value());
    }

    // Metadata travels WITH the packet as a shared immutable instance. The parser tells us
    // when the parsed metadata actually changed; we rebuild the shared instance only then, so
    // every packet in between attaches the same pointer (refcount bump, no map copy/compare)
    // and downstream consumers detect "unchanged" by pointer identity.
    if (result.metadata_changed) {
        m_metadata = std::move(result.metadata);
        m_metadata_shared = composite::make_metadata(m_metadata);
        logger()->trace("Updated metadata:\n{}", m_metadata.to_string());
        m_init_metadata = true;
    }

    // Send data (carrying the current metadata) if parser says we should and
    // metadata has been initialized.
    if (m_init_metadata && result.should_send) [[likely]] {
        m_out_port.send_data(std::move(result.payload), result.timestamp, m_metadata_shared);
    }

    return NORMAL;
}

COMPOSITE_REGISTER_SIMPLE(pkt_parser)
