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

#include "aligned_mem.hpp"
#include "psd_work.hpp"
#include "windows.hpp"

#include <composite/component.hpp>
#include <complex>
#include <immintrin.h>
#include <memory>

template <typename T>
class psd : public composite::component {
    using fft_t = aligned::aligned_mem<std::complex<T>>;
    using psd_t = aligned::aligned_mem<T>;
    using window_t = aligned::aligned_mem<T>;
    using input_port_t = composite::input_port<fft_t>;
    using output_port_t = composite::output_port<psd_t>;
public:
    psd() : composite::component("psd") {
        add_port(m_in_port.get());
        add_port(m_out_port.get());
        add_property("window", &m_window_type);
        add_property("fft_size", &m_fft_size);
        add_property("sample_rate", &m_sample_rate);
    }

    ~psd() override = default;

    auto initialize() -> void override {
        // Init window
        if (m_window_type == "BLACKMAN_HARRIS") {
            m_window = windows::blackman_harris<T>(m_fft_size, false);
        } else if (m_window_type == "HAMMING") {
            m_window = windows::hamming<T>(m_fft_size, false);
        }
        std::transform(
            m_window->data(),
            m_window->data() + m_window->size(),
            m_window->data(),
            [](const float& val) {
                return val * val;
            }
        );
        auto window_sum = std::accumulate(m_window->data(), m_window->data() + m_window->size(), T{});
        m_psd_work = std::make_unique<psd_work<T>>(window_sum, m_sample_rate);
    }

    auto process() -> composite::retval override {
        using enum composite::retval;
        auto [data, ts] = m_in_port->get_data();
        if (data == nullptr) {
            return NOOP;
        }
        // Perform PSD
        auto psd = m_psd_work->process(data.get());
        std::transform(psd->data(), psd->data() + psd->size(), psd->data(), [](const float& val) {
            return 10 * std::log10(val);
        });
        // Send data
        m_out_port->send_data(std::move(psd), ts);

        return NORMAL;
    }

private:
    // Ports
    std::unique_ptr<input_port_t> m_in_port{std::make_unique<input_port_t>("data_in")};
    std::unique_ptr<output_port_t> m_out_port{std::make_unique<output_port_t>("data_out")};

    // Properties
    std::string m_window_type;
    uint32_t m_fft_size{1024};
    T m_sample_rate{1};

    // Members
    std::unique_ptr<window_t> m_window;
    std::unique_ptr<psd_work<T>> m_psd_work;

}; // class psd
