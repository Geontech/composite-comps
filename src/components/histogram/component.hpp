/*
 * Copyright (C) 2024 Geon Technologies, LLC
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
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */

#include <composite/core/component.hpp>
#include <composite/properties/config.hpp>
#include <complex>
#include <cstdint>
#include <memory_resource>
#include <string>
#include <vector>

// The histogram's configuration as ONE reflected struct. All fields are
// INITIALIZE (set before start, read on the worker in initialize()/process()), so there is no
// reaction. add_config flattens the fields back to the same top-level property names, so the
// wire contract is unchanged.
struct histogram_config {
    std::string transport;
    uint32_t msg_size{};
    bool byteswap{true};
    uint32_t adc_bits{};
    float sample_rate{};
    float percent_sampled{1.0F};
    bool display_as_bits{};
    COMPOSITE_FIELDS(histogram_config,
        transport,
        (msg_size, unit("bytes")),
        byteswap,
        (adc_bits, unit("bits")),
        (sample_rate, unit("sps")),
        percent_sampled,
        display_as_bits);
};

class histogram : public composite::component {
    // Standard buffer types (the redesigned port requires Buf::value_type). Input is raw
    // bytes (read-only -> immutable); output is the histogram bins (uint64 counts -> mutable).
    using input_port_t = composite::input_port<composite::immutable_buffer<uint8_t>>;
    using output_port_t = composite::output_port<composite::mutable_buffer<uint64_t>>;
public:
    explicit histogram(std::string_view id);
    ~histogram() override = default;
    auto initialize() -> void override;
    auto process() -> composite::retval override;
    // Emit the final below-threshold accumulator at end-of-stream (else it is dropped).
    auto on_end_of_stream() -> void override;

private:
    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties (grouped as one reflected config<T>)
    composite::config<histogram_config> m_cfg{};

    // Members
    composite::mutable_buffer<uint64_t> m_histogram;  // accumulator bins; moved out on send, re-allocated
    std::vector<int8_t> m_sample_bits;
    uint32_t m_histogram_samples{};
    uint32_t m_skip_counter{};
    composite::timestamp m_last_ts{};  // ts of the most recent input frame, for the end-of-stream flush

    // MUST be last: stops the framework worker before any member above destructs (the base
    // ~component stops too late). See component.hpp auto_stop.
    composite::component::auto_stop m_auto_stop{*this};

}; // class histogram
