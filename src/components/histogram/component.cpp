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

#include <composite/core/register.hpp>

#include <bit>
#include <climits>
#include <complex>
#include <cmath>
#include <span>

histogram::histogram(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_port(&m_out_port);
    // The whole struct is the validate/commit unit; percent_sampled's range is a cross-field
    // (here single-field) invariant on the candidate. Set the validator BEFORE add_config.
    m_cfg.validate([](const histogram_config& c) {
        return c.percent_sampled > 0.0F && c.percent_sampled <= 1.0F;
    }, "percent_sampled must be in (0, 1]");
    add_config(m_cfg);
}

auto histogram::initialize() -> void {
    const auto adc_bits = m_cfg->adc_bits;
    const std::size_t bins = m_cfg->display_as_bits
        ? static_cast<std::size_t>(adc_bits * 2 + 1)
        : static_cast<std::size_t>(std::pow(2, adc_bits));
    m_histogram = composite::make_mutable<uint64_t>(bins);  // value-initialized to zeros

    for (auto i = SHRT_MIN; i <= SHRT_MAX; ++i) {
        if (i < 0) {
            m_sample_bits.push_back(-1 * static_cast<int8_t>(std::min(static_cast<double>(adc_bits), std::floor(std::log2(std::abs(i))) + 1)));
        } else {
            m_sample_bits.push_back(static_cast<int8_t>(std::min(static_cast<double>(adc_bits), std::floor(std::log2(std::abs(i))) + 1)));
        }
    }
}

auto histogram::process() -> composite::retval {
    using enum composite::retval;
    auto pkt = m_in_port.try_get();
    if (!pkt) {
        return NOOP;
    }
    auto& [data, ts, meta] = *pkt;
    (void)meta;
    m_last_ts = ts;  // remember for the end-of-stream flush
    // Process frames based on the sampling fraction
    if ((m_skip_counter++ % static_cast<uint32_t>((float{1} / m_cfg->percent_sampled))) == 0) {
        // Histogram. data is an immutable byte buffer; reinterpret as interleaved IQ samples.
        auto payload = std::span<const std::complex<int16_t>>{
            reinterpret_cast<const std::complex<int16_t>*>(data.data()),
            data.size() / sizeof(std::complex<int16_t>)
        };
        const auto bins = m_histogram.size();
        for (const auto& sample : payload) {
            auto sample_val = static_cast<int16_t>(m_cfg->byteswap ? std::byteswap(sample.real()) : sample.real());
            auto histogram_val = static_cast<int32_t>(sample_val) + static_cast<int32_t>(std::numeric_limits<int16_t>::max() + 1);
            if (m_cfg->display_as_bits) {
                histogram_val = m_sample_bits[histogram_val] + static_cast<int32_t>(bins / 2);
            }

            if (histogram_val >= 0 && static_cast<std::size_t>(histogram_val) < bins) {
                m_histogram[static_cast<std::size_t>(histogram_val)] += 1;
            } else if (histogram_val < 0) {
                m_histogram[0] += 1;
            } else {
                m_histogram[bins - 1] += 1;
            }
            ++m_histogram_samples;
        }
        // Send histogram data: move the bins out, then re-allocate a fresh (zeroed) buffer.
        if (m_histogram_samples > static_cast<uint32_t>(m_cfg->sample_rate)) {
            m_out_port.send_data(std::move(m_histogram), ts);
            m_histogram = composite::make_mutable<uint64_t>(bins);
            m_histogram_samples = 0;
        }
    }
    return NORMAL;
}

auto histogram::on_end_of_stream() -> void {
    // The accumulator only ships once it crosses the per-message sample threshold; at end-of-stream
    // any below-threshold remainder would be dropped. Emit it (with the last input's timestamp),
    // then re-allocate a fresh zeroed buffer so a restart accumulates cleanly.
    if (m_histogram_samples == 0 || m_histogram.empty()) { return; }
    const auto bins = m_histogram.size();
    m_out_port.send_data(std::move(m_histogram), m_last_ts);
    m_histogram = composite::make_mutable<uint64_t>(bins);
    m_histogram_samples = 0;
}

COMPOSITE_REGISTER_SIMPLE(histogram)
