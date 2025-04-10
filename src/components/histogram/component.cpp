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

#include <bit>
#include <climits>
#include <complex>
#include <cmath>
#include <span>

histogram::histogram() : composite::component("histogram") {
    add_port(&m_in_port);
    add_port(&m_out_port);
    add_property("transport", &m_transport);
    add_property("msg_size", &m_msg_size).units("bytes");
    add_property("byteswap", &m_byteswap);
    add_property("adc_bits", &m_adc_bits).units("bits");
    add_property("sample_rate", &m_sample_rate).units("sps");
    add_property("percent_sampled", &m_percent_sampled).change_listener([this]() {
        return (m_percent_sampled > 0.0) && (m_percent_sampled <= 1.0);
    });
    add_property("display_as_bits", &m_display_as_bits);
}

auto histogram::initialize() -> void {
    if (m_display_as_bits) {
        m_histogram = std::make_unique<histogram_t>(m_adc_bits * 2 + 1, 0);
    } else {
        m_histogram = std::make_unique<histogram_t>(static_cast<size_t>(pow(2, m_adc_bits)), 0);
    }

    for (auto i = SHRT_MIN; i <= SHRT_MAX; ++i) {
        if (i < 0) {
            m_sample_bits.push_back(-1 * static_cast<int8_t>(std::min(static_cast<double>(m_adc_bits), std::floor(std::log2(std::abs(i))) + 1)));
        } else {
            m_sample_bits.push_back(static_cast<int8_t>(std::min(static_cast<double>(m_adc_bits), std::floor(std::log2(std::abs(i))) + 1)));
        }
    }
}

auto histogram::process() -> composite::retval {
    using enum composite::retval;
    auto [data, ts] = m_in_port.get_data();
    if (data == nullptr) {
        return NOOP;
    }
    // Process frames based on
    if ((m_skip_counter++ % static_cast<uint32_t>((float{1} / m_percent_sampled))) == 0) {
        // Histogram
        auto payload = std::span<const std::complex<int16_t>>{
            reinterpret_cast<const std::complex<int16_t>*>(data->data()),
            data->size() / sizeof(std::complex<int16_t>)
        };
        for (const auto& sample : payload) {
            auto sample_val = static_cast<int16_t>(m_byteswap ? std::byteswap(sample.real()) : sample.real());
            auto histogram_val = static_cast<int32_t>(sample_val) + static_cast<int32_t>(std::numeric_limits<int16_t>::max() + 1);
            if (m_display_as_bits) {
                histogram_val = m_sample_bits[histogram_val] + m_histogram->size() / 2;
            }
            
            if (histogram_val >= 0 && histogram_val < m_histogram->size()) {
                m_histogram->at(histogram_val) += 1;
            } else if (histogram_val < 0) {
                m_histogram->front() += 1;
            } else {
                m_histogram->back() += 1;
            }
            ++m_histogram_samples;
        }
        // Send histogram data
        if (m_histogram_samples > static_cast<uint32_t>(m_sample_rate)) {
            m_out_port.send_data(std::move(m_histogram), ts);
            if (m_display_as_bits) {
                m_histogram = std::make_unique<histogram_t>(m_adc_bits * 2 + 1, 0);
            } else {
                m_histogram = std::make_unique<histogram_t>(static_cast<size_t>(pow(2, m_adc_bits)), 0);
            }
            m_histogram_samples = 0;
        }
    }
    return NORMAL;
}

extern "C" {
    auto create() -> std::shared_ptr<composite::component> {
        return std::make_shared<histogram>();
    }
}
