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
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */

#include "aligned_mem.hpp"
#include "window_buffer.hpp"

#include <algorithm>
#include <barrier>
#include <composite/component.hpp>
#include <complex>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <semaphore>
#include <thread>
#include <variant>
#include <vector>

class halfrate : public composite::component {
    using input_t = aligned::aligned_mem<std::complex<float>>;
    using input_ptr_t = std::unique_ptr<input_t>;
    using input_port_t = composite::input_port<input_ptr_t>;
    using output_port_t = composite::output_port<input_ptr_t>;
    using enum composite::properties::config_type;
public:
    halfrate();
    ~halfrate() override = default;

    auto property_change_handler() -> void override;
    // auto start() -> void override;
    // auto stop() -> void override;
    auto process() -> composite::retval override;

private:
    // auto process_block(std::stop_token token);
    auto generate_coeffs() -> void;

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    uint32_t m_filter_semi_length{3};
    std::string m_window_type{};

    // Members
    std::vector<float> m_coeffs;
    float m_center_tap{};
    window_buffer<std::complex<float>> m_w0, m_w2;
    window_buffer<std::complex<float>> m_w1, m_w3;

}; // class halfrate
