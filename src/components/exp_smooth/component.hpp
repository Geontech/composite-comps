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

#include "work.hpp"
#include <aligned_mem.hpp>

#include <composite/component.hpp>
#include <tuple>

template <typename T>
class exp_smooth : public composite::component {
    using input_t = aligned::aligned_mem<T>;
    using input_port_t = composite::input_port<input_t>;
    using output_port_t = composite::output_port<input_t>;
public:
    exp_smooth() : composite::component("exp_smooth") {
        add_port(m_in_port.get());
        add_port(m_out_port.get());
        add_property("alpha", &m_alpha);
    }

    ~exp_smooth() override = default;

    auto initialize() -> void override {
        m_work = std::make_unique<work<T>>(m_alpha);
    }

    auto process() -> composite::retval override {
        using enum composite::retval;
        auto [data, ts] = m_in_port->get_data();
        if (data == nullptr) {
            return NOOP;
        }
        if (m_alpha == T{1}) {
            // No smoothing, return as is
            m_out_port->send_data(std::move(data), ts);
            return NORMAL;
        }
        // Run algorithm
        m_work->process(data.get());
        // Send updated data
        m_out_port->send_data(std::move(data), ts);
        return NORMAL;
    }

private:
    // Ports
    std::unique_ptr<input_port_t> m_in_port{std::make_unique<input_port_t>("data_in")};
    std::unique_ptr<output_port_t> m_out_port{std::make_unique<output_port_t>("data_out")};

    // Properties
    T m_alpha{1.0};

    // Members
    std::unique_ptr<work<T>> m_work;

}; // class exp_smooth
