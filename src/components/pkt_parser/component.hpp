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
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
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
    static constexpr std::size_t INPUT_BATCH_SIZE{128};

    auto process_packet(input_port_t::queue_type packet) -> void;

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};
    std::array<input_port_t::queue_type, INPUT_BATCH_SIZE> m_input_batch;

    // Properties
    struct_props::signal_overrides m_signal_overrides;
    // "key=value" annotation overrides, resolved once per property change (see config.hpp:
    // the ingest-boundary hook for stream facts the wire protocol cannot carry).
    std::vector<std::pair<std::string, std::string>> m_annotation_overrides;

    // Members
    std::vector<std::unique_ptr<parsers::protocol_parser>> m_parsers;
    parsers::protocol_parser* m_active_parser{nullptr};
    // m_metadata is the working value the parsers update per packet; m_metadata_shared is the
    // immutable instance that rides the output packets — rebuilt ONLY when the parsed metadata
    // actually changes, so steady-state sends are a refcount bump, not a map copy.
    composite::metadata m_metadata;
    composite::metadata_ptr m_metadata_shared;
    bool m_init_metadata{false};
    // In-band stream-boundary tracking (udp_source's `stream_session` annotation): a session
    // change resets detection + carried metadata. Pointer-compared per packet; parsed only
    // when the upstream instance changes.
    composite::metadata_ptr m_last_in_md;
    composite::annotation_value m_last_session;  // typed: preserved as emitted upstream
    bool m_seen_session{false};
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
    // Upstream loss/reorder events (parser sequence tracking); the parsers' log warnings are
    // one-shot per stream, so this counter carries the ongoing rate.
    composite::metrics::counter<uint64_t>* m_sequence_gaps{nullptr};

    // MUST be last: stops the framework worker before any member above destructs (the base
    // ~component stops too late). See component.hpp auto_stop.
    composite::component::auto_stop m_auto_stop{*this};

}; // class pkt_parser
