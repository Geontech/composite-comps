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
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Histogram configuration. Sample width, complex-ness and rate are taken from the incoming
// metadata; these are the display/decimation knobs only. sample_rate is a fallback until metadata
// supplies the real rate. byteswap defaults to metadata-driven (endianness vs host) but can be
// forced on/off when the stream's endianness metadata is unreliable.
struct histogram_config {
    float sample_rate{};
    float percent_sampled{1.0F};      // fraction of frames binned, (0, 1]
    bool display_as_bits{};           // bin ADC-bit magnitudes instead of raw sample values
    std::optional<bool> byteswap{};   // unset = auto (metadata endianness vs host)
    COMPOSITE_FIELDS(histogram_config,
        (sample_rate, unit("sps")),
        (percent_sampled, runtime),
        (display_as_bits, runtime),
        (byteswap, runtime));
};

class histogram : public composite::component {
    // Input is raw bytes (read-only -> immutable); output is the histogram bins (uint64 counts).
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
    auto build_lookup() -> void;
    auto allocate_histogram() -> void;
    template <typename T>
    auto process_samples(const uint8_t* bytes, std::size_t nbytes) -> void;

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties (grouped as one reflected config<T>)
    composite::config<histogram_config> m_cfg{};

    // Members
    composite::mutable_buffer<uint64_t> m_histogram;
    std::vector<int8_t> m_sample_bits;
    uint32_t m_histogram_samples{};
    uint32_t m_skip_counter{};
    uint32_t m_skip_threshold{1};  // = 1 / percent_sampled (whole-frame decimation)
    uint32_t m_send_threshold{};   // = sample_rate * percent_sampled (samples per emitted histogram)

    // Stream characteristics, tracked from metadata:
    float m_sample_rate{};
    uint32_t m_source_bits{16};
    bool m_is_complex{true};
    bool m_byteswap{};        // effective swap used this frame (config override, else auto)
    bool m_auto_byteswap{};   // metadata-derived swap (endianness vs host)
    composite::timestamp m_last_ts{};  // ts of the most recent input frame, for the end-of-stream flush

    // MUST be last: stops the framework worker before any member above destructs (the base
    // ~component stops too late). See component.hpp auto_stop.
    composite::component::auto_stop m_auto_stop{*this};

}; // class histogram
