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

#include "component.hpp"
#include "dot_prod.hpp"
#include <windows.hpp>

#include <bit>
#include <immintrin.h>
#include <memory>
#include <numbers>

halfrate::halfrate() : composite::component("halfrate") {
    add_port(&m_in_port);
    add_port(&m_out_port);
    add_property("filter_semi_length", &m_filter_semi_length);
    add_property("window", &m_window_type).change_listener([this]() {
        return (m_window_type == "BLACKMAN_HARRIS") || (m_window_type == "HAMMING");
    });
}

auto halfrate::property_change_handler() -> void {
    m_w0 = window_buffer<std::complex<float>>(2 * m_filter_semi_length);
    m_w1 = window_buffer<std::complex<float>>(2 * m_filter_semi_length);
    generate_coeffs();
}

auto halfrate::process() -> composite::retval {
    using enum composite::retval;
    auto [data, ts] = m_in_port.get_data();
    if (data == nullptr) {
        return NORMAL;
    }

    // Iterate over samples to produce output
    for (auto i = std::size_t{}; i < data->size(); i += 2) {
        // Add samples to window buffers
        m_w0.push(data->at(i));
        m_w1.push(data->at(i + 1));

        // Calculate value with center tap
        auto y0 = m_center_tap * m_w1[m_filter_semi_length - 1];

        // Compute dot-product with filter taps
        auto y1 = dotprod::process(m_coeffs, m_w0);

        // Add result to output vector
        data->at(i / 2) = y0 + y1;
    }

    // Send data
    data->resize(data->size() / 2);
    m_out_port.send_data(std::move(data), ts);

    return NORMAL;
}

auto halfrate::generate_coeffs() -> void {
    auto L = 4 * m_filter_semi_length + 1; // full length
    auto center = L / 2;
    auto h = std::vector<float>(L);

    auto window = windows::hamming<float>(L, false);

    for (int n = 0; n < L; ++n) {
        int k = n - center;
        if (k == 0) {
            h[n] = 0.5;  // half-band filter: center tap is always 0.5
            m_center_tap = h[n] * 2.0f;
        } else if (k % 2 == 0) {
            auto sinc_val = std::sin(0.5 * std::numbers::pi * k) / (std::numbers::pi * k);
            h[n] = sinc_val * window->at(n);
        } else {
            h[n] = 0.0; // zero out odd taps (except center)
        }
    }

    m_coeffs.clear();
    for (int i = h.size() - 1; i >= 0; --i) {
        if (i == h.size() / 2) { continue; } // skip center tap
        if (i % 2 == 1) { // use only even indices
            m_coeffs.push_back(h[i] * 2.0f);
        }
    }

    // m_coeffs = {-0.00288903,0.00745904,-0.0163088,0.0313995,-0.0564697,0.100349,-0.194997,0.630703,0.630703,-0.194997,0.100349,-0.0564697,0.0313995,-0.0163088,0.00745904,-0.00288903};
    // m_coeffs = {-0.00171411,0.00307656,-0.00567513,0.00956402,-0.0151604,0.023042,-0.0340987,0.0499172,-0.0738422,0.114591,-0.204352,0.633965,0.633965,-0.204352,0.114591,-0.0738422,0.0499172,-0.0340987,0.023042,-0.0151604,0.00956402,-0.00567513,0.00307656,-0.00171411};
    // m_coeffs = {-0.000965003,0.000899301,-0.00130554,0.00182206,-0.00246723,0.00326131,-0.00422741,0.00539153,-0.0067835,0.00843858,-0.0103997,0.0127202,-0.0154706,0.0187447,-0.0226746,0.027453,-0.0333755,0.0409215,-0.0509201,0.0649463,-0.0863793,0.124028,-0.210215,0.635954,0.635954,-0.210215,0.124028,-0.0863793,0.0649463,-0.0509201,0.0409215,-0.0333755,0.027453,-0.0226746,0.0187447,-0.0154706,0.0127202,-0.0103997,0.00843858,-0.0067835,0.00539153,-0.00422741,0.00326131,-0.00246723,0.00182206,-0.00130554,0.000899301,-0.000965003};
    // m_center_tap = 1.0f;
}

extern "C" {
    auto create() -> std::shared_ptr<composite::component> {
        return std::make_shared<halfrate>();
    }
}
