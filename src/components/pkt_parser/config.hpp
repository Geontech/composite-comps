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

#pragma once

#include <composite/properties/reflect.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace struct_props {

/**
 * @brief Signal override configuration
 *
 * Allows manual override of protocol-extracted metadata.
 *
 * Reflection is declared in-body via COMPOSITE_FIELDS (the hidden-friend ADL hook), which
 * works inside this namespace and on the nested struct directly — retiring the prior
 * COMPOSITE_STRUCT-at-namespace-scope workaround. The wire contract is unchanged: this
 * whole struct is still registered as the single "signal_overrides" property.
 */
struct signal_overrides {
    std::optional<double> center_frequency;
    std::optional<double> bandwidth;
    std::optional<double> sample_rate;

    struct format {
        std::optional<bool> is_complex;
        std::string type;
        uint32_t bit_width{};
        std::string endianness;
        COMPOSITE_FIELDS(format, is_complex, type, bit_width, endianness);
    };

    format data_format;
    std::string transport;
    // Operator-declared metadata annotations, "key=value" per entry, merged into every
    // published metadata (operator wins over parser-set keys). This is the ingest-boundary
    // hook for stream facts the wire protocol cannot carry — e.g. a stream that is ALREADY
    // FFT data from a remote producer: declaring fft_size / fft_window_sum_sq here lets a
    // downstream psd normalize exactly as it would behind a local fft component.
    std::vector<std::string> annotations;
    COMPOSITE_FIELDS(signal_overrides,
                     center_frequency, bandwidth, sample_rate, data_format, transport,
                     annotations);
};

} // namespace struct_props
