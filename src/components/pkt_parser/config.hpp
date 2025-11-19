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

#include <cstdint>
#include <optional>
#include <string>

/**
 * @brief Signal override configuration
 *
 * Allows manual override of protocol-extracted metadata
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
    };

    format data_format;
    std::string transport;
};
