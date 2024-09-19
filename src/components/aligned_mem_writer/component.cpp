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

#include "component.hpp"

#include <complex>
#include <string_view>

extern "C" {
    auto create(std::string_view type) -> std::shared_ptr<composite::component> {
        if (type == "f32") {
            return std::make_shared<aligned_mem_writer<float>>();
        } else if (type == "f64") {
            return std::make_shared<aligned_mem_writer<double>>();
        } else if (type == "cf32") {
            return std::shared_ptr<composite::component>(new aligned_mem_writer<std::complex<float>>);
        } else if (type == "cf64") {
            return std::shared_ptr<composite::component>(new aligned_mem_writer<std::complex<double>>);
        }
        return std::make_shared<aligned_mem_writer<float>>();
    }
}
