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

#include "component.hpp"

#include <cmath>
#include <format>
#include <string_view>

template <typename T>
exp_smooth<T>::exp_smooth() : composite::component("exp_smooth") {
    using enum composite::properties::config_type;
    add_port(&m_in_port);
    add_port(&m_out_port);
    add_property("num_averages", &m_num_averages).configurability(RUNTIME).change_listener([this]() {
        m_alpha.reset();
        if (m_num_averages > 0) {
            m_alpha = T{1} - std::pow(T{10}, (std::log10(T{1} - T{0.98}) / m_num_averages));
        }
        m_work = std::make_unique<work<T>>(m_alpha.value_or(T{1}));
        m_prev_psd = composite::mutable_buffer<T>{}; // Reset to empty buffer
        return true;
    });
}

template <typename T>
auto exp_smooth<T>::process() -> composite::retval {
    using enum composite::retval;
    auto [data, ts, meta] = m_in_port.get_data();
    if (!data) {
        return NOOP;
    }
    if (meta.has_value()) {
        logger()->trace("pass-through metadata:\n{}", meta->to_string());
        m_out_port.send_metadata(meta.value());
    }
    if (!m_alpha.has_value()) {
        // No smoothing, return as is
        m_out_port.send_data(std::move(data), ts);
        return NORMAL;
    }
    // Handle first PSD
    if (m_prev_psd.empty()) {
        m_prev_psd = std::move(data);
        m_prev_psd_ts = ts;
        return NORMAL;
    }
    // Run algorithm
    m_work->process(data, m_prev_psd);
    // Send previous PSD data and timestamp
    m_out_port.send_data(std::move(m_prev_psd), m_prev_psd_ts);
    // Save current PSD for next pass
    m_prev_psd = std::move(data);
    m_prev_psd_ts = ts;
    return NORMAL;
}

// --- Factory Function ---
extern "C" {
    auto create(std::string_view type) -> std::shared_ptr<composite::component> {
        if (type == "f32") {
            return std::make_shared<exp_smooth<float>>();
        } else if (type == "f64") {
            return std::make_shared<exp_smooth<double>>();
        }
        throw std::runtime_error(std::format("unknown type {} for exp_smooth create", type));
    }
}
