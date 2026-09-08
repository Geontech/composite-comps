/*
 * Copyright (C) 2024-2025 Geon Technologies, LLC
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

#include "work.hpp"

#include <composite/core/component.hpp>
#include <composite/metrics/metrics.hpp>

#include <array>
#include <cstdint>
#include <optional>

template <typename T>
class exp_smooth : public composite::component {
    using input_port_t = composite::input_port<composite::mutable_buffer<T>>;
    using output_port_t = composite::output_port<composite::mutable_buffer<T>>;
public:
    explicit exp_smooth(std::string_view id);
    ~exp_smooth() override = default;

    auto process() -> composite::retval override;
    // Emit the held 1-frame-delayed PSD at end-of-stream (else the last frame is dropped).
    auto on_end_of_stream() -> void override;

private:
    auto process_packet(typename input_port_t::queue_type pkt) -> void;

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    uint32_t m_num_averages{};

    // Members
    std::optional<T> m_alpha;
    std::optional<work<T>> m_work;  // in-place: work<T> holds only a scalar, no heap indirection
    typename input_port_t::buffer_type m_prev_psd;
    composite::timestamp m_prev_psd_ts;
    composite::metadata_ptr m_prev_meta;  // metadata of m_prev_psd's source frame (rides its send;
                                          // nullptr when that frame arrived bare — per the port
                                          // contract, metadata is per-packet, never latched)

    // Observability: frame-size changes mid-stream (e.g. an upstream fft_size change) that force the
    // EWMA to re-baseline rather than smooth. Shared registry, labeled by component id; auto-removed
    // by ~component.
    composite::metrics::counter<uint64_t>* m_size_mismatch{nullptr};
    // Re-baselines forced by a METADATA change at an unchanged size (retune, window/shift/
    // normalization change): smoothing across one would ship old-configuration energy under
    // new-configuration metadata.
    composite::metrics::counter<uint64_t>* m_meta_rebaselines{nullptr};

    // MUST be last: stops the framework worker before any member above destructs (the base
    // ~component stops too late). See component.hpp auto_stop.
    composite::component::auto_stop m_auto_stop{*this};

}; // class exp_smooth
