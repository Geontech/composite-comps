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
#include "apply_window.hpp"
#include "fft_plan.hpp"
#include "overlay.hpp"
#include "windows.hpp"

#include <composite/component.hpp>
#include <complex>
#include <fftw3.h>
#include <memory>
#include <vector>

template <typename T>
class fft : public composite::component {
    using plan_t = fft_plan<T, true>;
    using fft_t = aligned::aligned_mem<std::complex<T>>;
    using window_t = aligned::aligned_mem<T>;
    using input_port_t = composite::input_port<fft_t>;
    using output_port_t = composite::output_port<fft_t>;
public:
    fft() : composite::component("fft") {
        add_port(m_in_port.get());
        add_port(m_out_port.get());
        add_property("window", &m_window_type);
        add_property("fft_size", &m_fft_size);
        add_property("fftw_threads", &m_fftw_threads);
        add_property("shift", &m_shift);
    }

    ~fft() override = default;

    auto initialize() -> void override {
        // Init window
        if (m_window_type == "BLACKMAN_HARRIS") {
            m_window = windows::blackman_harris<T>(m_fft_size);
        } else if (m_window_type == "HAMMING") {
            m_window = windows::hamming<T>(m_fft_size);
        }
        // Init fftw
        m_fft_plan = std::make_unique<plan_t>(m_fft_size, m_fftw_threads, m_shift);
    }

    auto process() -> composite::retval override {
        using enum composite::retval;
        auto [data, ts] = m_in_port->get_data();
        if (data == nullptr) {
            return NORMAL;
        }
        // Apply window
        if (m_window) {
            auto i=0u;
            auto stride = 32u / sizeof(T);
            for (; i < data->size(); i += stride) {
                apply_window(
                    reinterpret_cast<const typename fft_t::value_type::value_type*>(data->data() + i),
                    m_window->data() + i * 2,
                    reinterpret_cast<typename fft_t::value_type::value_type*>(data->data() + i)
                );
            }
        }
        // Execute the fft
        // In-place for complex
        m_fft_plan->execute(data.get(), data.get());
        // Send data
        m_out_port->send_data(std::move(data), ts);
        return NORMAL;
    }

private:
    // Ports
    std::unique_ptr<input_port_t> m_in_port{std::make_unique<input_port_t>("data_in")};
    std::unique_ptr<output_port_t> m_out_port{std::make_unique<output_port_t>("data_out")};

    // Properties
    std::string m_window_type;
    uint32_t m_fft_size{1024};
    uint32_t m_fftw_threads{1};
    bool m_shift{true};

    // Members
    std::unique_ptr<fft_plan<T, true>> m_fft_plan{nullptr};
    std::unique_ptr<window_t> m_window{nullptr};

}; // class fft
