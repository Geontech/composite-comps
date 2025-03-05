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

#include "convert.hpp"

#include <aligned_mem.hpp>
#include <overlay.hpp>

#include <algorithm>
#include <composite/component.hpp>
#include <complex>
#include <cstdint>
#include <spdlog/spdlog.h>
#include <vector>

template <typename T>
class stov : public composite::component {
    using input_t = std::vector<uint8_t>;
    using input_port_t = composite::input_port<std::shared_ptr<input_t>>;
    using output_t = aligned::aligned_mem<T>;
    using output_port_t = composite::output_port<std::unique_ptr<output_t>>;
    using enum composite::properties::config_type;
public:
    stov() : composite::component("stov") {
        add_port(m_in_port.get());
        add_port(m_out_port.get());
        add_property("output_size", &m_output_size).units("bytes").configurability(RUNTIME);
        add_property("byteswap", &m_byteswap).configurability(RUNTIME);
    }

    ~stov() override = default;

    auto property_change_handler() -> void override {
        // flush input port
        m_in_port->clear();
    }

    auto process() -> composite::retval override {
        using enum composite::retval;
        auto [data, ts] = m_in_port->get_data();
        if (data == nullptr) {
            return NORMAL;
        }
        auto stride = size_t{4};
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) {
            stride = 8;
        }
        for (auto i = 0u; i < data->size(); i += stride) {
            if (m_output_buf == nullptr) {
                m_output_buf = aligned::make_aligned<typename output_t::value_type>(64, m_output_size);
                m_output_ts = ts;
            }
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) {
                convert(
                    reinterpret_cast<const int16_t*>(data->data() + i),
                    reinterpret_cast<float*>(m_output_buf->data() + m_output_idx),
                    m_byteswap
                );
            } else { // double or std::complex<double>
                convert(
                    reinterpret_cast<const int16_t*>(data->data() + i),
                    reinterpret_cast<double*>(m_output_buf->data() + m_output_idx),
                    m_byteswap
                );
            }
            m_output_idx += stride;
            if (m_output_idx == m_output_size) {
                m_out_port->send_data(std::move(m_output_buf), m_output_ts);
                m_output_buf.reset();
                m_output_idx = 0;
            }
        }
        return NO_YIELD;
    }

private:
    // Ports
    std::unique_ptr<input_port_t> m_in_port{std::make_unique<input_port_t>("data_in")};
    std::unique_ptr<output_port_t> m_out_port{std::make_unique<output_port_t>("data_out")};

    // Properties
    uint32_t m_output_size{};
    bool m_byteswap{true};

    // Members
    std::unique_ptr<output_t> m_output_buf;
    uint32_t m_output_idx{};
    typename output_port_t::timestamp_type m_output_ts;
    uint8_t m_pkt_count{};

}; // class stov
