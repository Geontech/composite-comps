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
#include "parsers/parser_table.hpp"

#include <composite/core/register.hpp>

#include <cmath>
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
            // A set sample-rate override must be a positive finite number: it feeds
            // timestamp arithmetic (SAMPLE_COUNT fractional timestamps here, anchor
            // extrapolation downstream), where a NaN/negative value is UB-adjacent.
            if (v.sample_rate.has_value() &&
                !(std::isfinite(*v.sample_rate) && *v.sample_rate > 0.0)) {
                return false;
            }
            // Annotation overrides are "key=value" with a non-empty key.
            for (const auto& entry : v.annotations) {
                const auto eq = entry.find('=');
                if (eq == std::string::npos || eq == 0) {
                    return false;
                }
            }
            return true;
        });
    // Type-prefixed name, matching udp_source./framer. convention: the component_id label
    // identifies the instance, the prefix scopes the series to the component type so
    // cross-instance aggregation ("all pkt_parser drops") stays a name match.
    m_packets_dropped = &create_counter(
        "pkt_parser.packets_dropped", "Packets dropped (unknown protocol / malformed / unparseable)");
    m_sequence_gaps = &create_counter(
        "pkt_parser.sequence_gaps",
        "Upstream packet loss/reorder events detected by sequence-number tracking");
}

auto pkt_parser::property_change_handler(const composite::properties::json& diff) -> void {
    (void)diff;
    logger()->trace(std::source_location::current().function_name());

    // Initialize parser registry
    m_parsers.clear();
    m_active_parser = nullptr;
    m_drop_warned = false;
    m_consecutive_parse_failures = 0;

    // Register parsers from the table, which is already ordered most-specific-first (V49.1 before
    // V49, and so on). The set of parsers lives in parsers/parser_table.hpp -- adding one does not
    // touch this file. An empty `transport` override selects every parser and lets can_parse()
    // detection choose; a non-empty one pins a single protocol.
    for (const auto& entry : parsers::parser_table()) {
        if (m_signal_overrides.transport.empty() || m_signal_overrides.transport == entry.transport) {
            m_parsers.push_back(entry.make(m_signal_overrides));
        }
    }

    // Resolve the "key=value" annotation overrides once, off the per-packet path (they are
    // merged into the published metadata only when it is rebuilt; see process_packet).
    m_annotation_overrides.clear();
    for (const auto& entry : m_signal_overrides.annotations) {
        const auto eq = entry.find('=');
        if (eq != std::string::npos && eq != 0) {  // validator-enforced; defensive re-check
            m_annotation_overrides.emplace_back(entry.substr(0, eq), entry.substr(eq + 1));
        }
    }

    logger()->trace("Registered {} protocol parsers", m_parsers.size());
}

auto pkt_parser::process() -> composite::retval {
    using enum composite::retval;

    const auto count = m_in_port.get_batch(std::span{m_input_batch});
    if (count == 0) {
        // No input: NOOP so the worker arms the read-doorbell and parks until upstream
        // delivers, instead of busy-spinning process() and burning a core while idle. At
        // end-of-stream the base promotes this NOOP to FINISH (no buffered state to flush).
        return NOOP;
    }

    // Protocol detection and parser metadata are ordered stream state. Drain a
    // bounded input batch with one ring-head publication, but process each
    // datagram sequentially to preserve exactly the scalar semantics.
    for (std::size_t i = 0; i < count; ++i) {
        process_packet(std::move(m_input_batch[i]));
    }
    return NORMAL;
}

auto pkt_parser::process_packet(input_port_t::queue_type packet) -> void {
    auto& [data, _, __] = packet;

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
            return;
        }
    }

    // Parse packet using active parser. Lock-in does NOT trust subsequent packets:
    // each is re-validated and a bad one is dropped, not allowed to read OOB/throw.
    parsers::protocol_parser::parse_result result;
    try {
        result = m_active_parser->parse(data, m_metadata);
        m_consecutive_parse_failures = 0;  // this packet matches the locked-in protocol
    } catch (const std::exception& e) {
        // Counted always; the message is FORMATTED only when it will actually be logged —
        // a malformed-packet flood otherwise pays a string allocation per packet for a
        // warning that the one-shot latch already muted.
        if (m_packets_dropped != nullptr) { m_packets_dropped->inc(); }
        if (!m_drop_warned) {
            m_drop_warned = true;
            logger()->warn("pkt_parser: dropping packet ({} bytes): parse error: {}", data.size(), e.what());
        }
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
        return;
    }

    // Upstream loss/reorder: the parsers detect it per packet (seq_gap) but warn one-shot;
    // this counter carries the ongoing rate for operators.
    if (result.seq_gap) [[unlikely]] {
        m_sequence_gaps->inc();
    }

    // Log any warnings from parser (each is one-shot on the parser side; see seq_gap above)
    if (result.warning.has_value()) {
        logger()->warn("{}", result.warning.value());
    }

    // Metadata travels WITH the packet as a shared immutable instance. The parser tells us
    // when the parsed metadata actually changed; we rebuild the shared instance only then, so
    // every packet in between attaches the same pointer (refcount bump, no map copy/compare)
    // and downstream consumers detect "unchanged" by pointer identity.
    if (result.metadata_changed) {
        m_metadata = std::move(result.metadata);
        // Operator-declared annotations win over parser-set keys. Applied only on rebuild,
        // and m_metadata (the parsers' change-detection baseline) keeps them, so they do not
        // retrigger a republish per packet.
        for (const auto& [key, value] : m_annotation_overrides) {
            m_metadata.annotations[key] = value;
        }
        m_metadata_shared = composite::make_metadata(m_metadata);
        logger()->trace("Updated metadata:\n{}", m_metadata.to_string());
        m_init_metadata = true;
    }

    // Send data (carrying the current metadata) if parser says we should and
    // metadata has been initialized. Keep this scalar: parsed packets have distinct timestamps,
    // while output_port::send_batch intentionally applies one timestamp to the complete batch.
    if (m_init_metadata && result.should_send) [[likely]] {
        m_out_port.send_data(std::move(result.payload), result.timestamp, m_metadata_shared);
    }
}

COMPOSITE_REGISTER_SIMPLE(pkt_parser)
