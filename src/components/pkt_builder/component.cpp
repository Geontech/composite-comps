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

#include "component_impl.hpp"

#include <format>
#include <stdexcept>

// Explicit template instantiations for common types
template class pkt_builder<std::complex<float>>;
template class pkt_builder<std::complex<int16_t>>;
template class pkt_builder<std::complex<int8_t>>;
template class pkt_builder<float>;
template class pkt_builder<int16_t>;

// Factory function with type selector
#ifndef UNIT_TESTS
extern "C" {
    auto create(std::string_view id, std::string_view type = "cf32") -> std::shared_ptr<composite::component> {
        if (type == "cf32" || type.empty()) {
            return std::make_shared<pkt_builder_cf32>(id);
        } else if (type == "ci16") {
            return std::make_shared<pkt_builder_ci16>(id);
        } else if (type == "ci8") {
            return std::make_shared<pkt_builder_ci8>(id);
        } else if (type == "f32") {
            return std::make_shared<pkt_builder_f32>(id);
        } else if (type == "i16") {
            return std::make_shared<pkt_builder_i16>(id);
        }
        throw std::runtime_error(std::format("unknown type '{}' for pkt_builder component", type));
    }
}
#endif
