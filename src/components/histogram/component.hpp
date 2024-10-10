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

#include <overlay.hpp>

#include <byteswap.h>
#include <composite/component.hpp>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

class histogram : public composite::component {
    using input_t = std::vector<uint8_t>;
    using input_port_t = composite::input_port<std::shared_ptr<input_t>>;
    using histogram_t = std::vector<uint64_t>;
    using output_port_t = composite::output_port<std::unique_ptr<histogram_t>>;
public:
    histogram();
    ~histogram() override = default;
    auto initialize() -> void override;
    auto process() -> composite::retval override;

private:
    // Ports
    std::unique_ptr<input_port_t> m_in_port{std::make_unique<input_port_t>("data_in")};
    std::unique_ptr<output_port_t> m_out_port{std::make_unique<output_port_t>("data_out")};

    // Properties
    std::string m_transport;
    uint32_t m_msg_size{};
    bool m_byteswap{true};
    uint32_t m_adc_bits{};
    float m_sample_rate{};

    // Members
    std::unique_ptr<histogram_t> m_histogram;
    uint32_t m_histogram_samples{};

}; // class histogram
