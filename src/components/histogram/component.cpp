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

#include <byteswap.h>

histogram::histogram() : composite::component("histogram") {
    add_port(m_in_port.get());
    add_port(m_out_port.get());
    add_property("transport", &m_transport);
    add_property("msg_size", &m_msg_size);
    add_property("byteswap", &m_byteswap);
    add_property("adc_bits", &m_adc_bits);
    add_property("sample_rate", &m_sample_rate);
}

auto histogram::initialize() -> void {
    m_histogram = std::make_unique<histogram_t>(static_cast<size_t>(pow(2, m_adc_bits)), 0);
}

auto histogram::process() -> composite::retval {
    using enum composite::retval;
    auto [data, ts] = m_in_port->get_data();
    if (data == nullptr) {
        return NOOP;
    }
    // Histogram
    for (auto idx = size_t{}; idx < data->size(); idx += m_msg_size) {
        auto payload = std::span<const std::complex<uint16_t>>{};
        if (m_transport == "sdds") {
            auto packet = overlay::sdds::overlay({data->data() + idx, m_msg_size});
            payload = packet.payload<std::complex<uint16_t>>();
        } else if (m_transport == "vita49") {
            auto packet = overlay::v49::overlay({data->data() + idx, m_msg_size});
            if (auto& header = packet.header(); !overlay::v49::is_data(header)) {
                continue;
            }
            payload = packet.payload<std::complex<uint16_t>>();
        }
        // Get sample values
        for (auto& sample : payload) {
            auto sample_val = static_cast<int16_t>(m_byteswap ? bswap_16(sample.real()) : sample.real());
            auto sample_val_32 = static_cast<int32_t>(sample_val) + static_cast<int32_t>(m_histogram->size() / 2);
            if (sample_val_32 >= 0 && sample_val_32 < m_histogram->size()) {
                m_histogram->at(sample_val_32) += 1;
            } else if (sample_val_32 < 0) {
                m_histogram->front() += 1;
            } else {
                m_histogram->back() += 1;
            }
            ++m_histogram_samples;
        }
    }
    // Send histogram data
    if (m_histogram_samples > static_cast<uint32_t>(m_sample_rate)) {
        m_out_port->send_data(std::move(m_histogram), ts);
        m_histogram = std::make_unique<histogram_t>(static_cast<size_t>(pow(2, m_adc_bits)), 0);
        m_histogram_samples = 0;
    }
    return NORMAL;
}

extern "C" {
    auto create() -> std::shared_ptr<composite::component> {
        return std::make_shared<histogram>();
    }
}
