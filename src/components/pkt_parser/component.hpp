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
#include <composite/metrics/metrics.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

class pkt_parser : public composite::component {
    using input_port_t = composite::input_port<composite::immutable_buffer<uint8_t>>;
    using output_port_t = composite::output_port<composite::immutable_buffer<uint8_t>>;

    // Grants the re-detection integration test access to the private detection state
    // (m_active_parser / m_consecutive_parse_failures) and the ports, so it can drive
    // process() directly. Mirrors framer's FramerTestFixture friendship.
    friend struct redetect_harness;
public:
    explicit pkt_parser(std::string_view id);
    ~pkt_parser() override = default;
    auto property_change_handler(const composite::properties::json& diff) -> void override;
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
    // m_metadata is the working value the parsers update per packet; m_metadata_shared is the
    // immutable instance that rides the output packets — rebuilt ONLY when the parsed metadata
    // actually changes, so steady-state sends are a refcount bump, not a map copy.
    composite::metadata m_metadata;
    composite::metadata_ptr m_metadata_shared;
    bool m_init_metadata{false};
    bool m_drop_warned{false};   ///< rate-limit drop warnings (the counter carries the real signal)
    uint32_t m_consecutive_parse_failures{0};  ///< a run of these triggers protocol re-detection

    /// A sustained run of parse failures means the stream's framing type has likely changed
    /// (e.g. a warm-pool re-steer to a different digitizer): the locked-in parser can no longer
    /// make sense of the wire layout. After this many *consecutive* failures we un-lock the
    /// active parser so the next packet re-runs detection and the pipeline self-heals — no
    /// operator reconfiguration required. A single successful parse resets the counter, so
    /// isolated corruption on an otherwise-valid stream never trips it. Tunable; promote to a
    /// property if a deployment needs per-instance control.
    static constexpr uint32_t REDETECT_AFTER_FAILURES{32};

    // Observability: malformed/unparseable packets are dropped (not fatal), counted here.
    composite::metrics::counter<uint64_t>* m_packets_dropped{nullptr};

    // MUST be last: stops the framework worker before any member above destructs (the base
    // ~component stops too late). See component.hpp auto_stop.
    composite::component::auto_stop m_auto_stop{*this};

}; // class pkt_parser
