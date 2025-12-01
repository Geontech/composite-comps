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

#include <algorithm>
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
    add_property("sample_rate", &m_sample_rate).units("sps").change_listener([this]() {
        if (m_sample_rate > 0.0f) {
            m_send_threshold = static_cast<uint32_t>(m_sample_rate * m_percent_sampled);
            return true;
        }
        return false;
    });
    add_property("percent_sampled", &m_percent_sampled).change_listener([this]() {
        if ((m_percent_sampled > 0.0) && (m_percent_sampled <= 1.0)) {
            m_skip_threshold = static_cast<uint32_t>(1.0f / m_percent_sampled);
            if (m_sample_rate > 0.0f) {
                m_send_threshold = static_cast<uint32_t>(m_sample_rate * m_percent_sampled);
            }
            return true;
        }
        return false;
    });
    add_property("display_as_bits", &m_display_as_bits);
}

auto histogram::initialize() -> void {
    if (m_display_as_bits) {
        m_histogram = std::make_shared<histogram_t>(m_adc_bits * 2 + 1, 0);
    } else {
        m_histogram = std::make_shared<histogram_t>(static_cast<size_t>(pow(2, m_adc_bits)), 0);
    }

    for (auto i = SHRT_MIN; i <= SHRT_MAX; ++i) {
        if (i < 0) {
            m_sample_bits.push_back(-1 * static_cast<int8_t>(std::min(static_cast<double>(m_adc_bits), std::floor(std::log2(std::abs(i))) + 1)));
        } else if (i == 0) [[ unlikely ]] {
            // Zero samples have no magnitude, map to center bin (0 bits)
            m_sample_bits.push_back(0);
        } else {
            m_sample_bits.push_back(static_cast<int8_t>(std::min(static_cast<double>(m_adc_bits), std::floor(std::log2(std::abs(i))) + 1)));
        }
    }

    // Calculate skip threshold for percent_sampled
    m_skip_threshold = static_cast<uint32_t>(1.0f / m_percent_sampled);
}

auto histogram::process() -> composite::retval {
    using enum composite::retval;
    auto [data, ts, meta] = m_in_port.get_data();
    if (data == nullptr) {
        return NOOP;
    }
    // Extract sample rate from metadata if present
    if (meta.has_value() && meta->sample_rate > 0.0) {
        if (m_sample_rate != static_cast<float>(meta->sample_rate)) {
            m_sample_rate = static_cast<float>(meta->sample_rate);
            m_send_threshold = static_cast<uint32_t>(m_sample_rate * m_percent_sampled);
            logger()->info("Updated sample_rate from metadata: {} sps, send_threshold: {} samples",
                          m_sample_rate, m_send_threshold);
        }
    }

    // Don't process data until sample_rate is configured
    if (m_send_threshold == 0) {
        logger()->warn("Skipping data - sample_rate not configured (set property or wait for metadata)");
        return NOOP;
    }

    // Process frames based on skip threshold (optimized from modulo)
    if (++m_skip_counter >= m_skip_threshold) {
        m_skip_counter = 0;

        // Work directly with int16_t array for better performance
        auto samples_ptr = reinterpret_cast<const int16_t*>(data->data());
        auto num_samples = data->size() / sizeof(std::complex<int16_t>);
        auto total_samples = num_samples << 1;

        // Hoist histogram size calculations out of loop
        const auto hist_size = static_cast<int32_t>(m_histogram->size());

        // Histogram loop - only process real components
        for (size_t i = 0; i < total_samples; i+=2) {
            auto sample_val = samples_ptr[i];  // Read real component

            // Byteswap locally if needed (don't modify shared buffer!)
            if (m_byteswap) {
                sample_val = std::byteswap(sample_val);
            }

            auto histogram_val = static_cast<uint16_t>(sample_val) ^ 0x8000;  // Offset to positive range

            if (m_display_as_bits) {
                histogram_val = m_sample_bits[histogram_val] + m_adc_bits;
            }

            // Clamp to valid range and increment
            (*m_histogram)[histogram_val]++;
        }
        m_histogram_samples += num_samples;

        // Send histogram data
        if (m_histogram_samples >= m_send_threshold) {
            m_out_port.send_data(std::move(m_histogram), ts);
            // Reuse buffer instead of reallocating
            if (m_display_as_bits) {
                m_histogram = std::make_shared<histogram_t>(m_adc_bits * 2 + 1, 0);
            } else {
                m_histogram = std::make_shared<histogram_t>(static_cast<size_t>(pow(2, m_adc_bits)), 0);
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
