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

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace {
    constexpr uint32_t LOOKUP_BITS = 16;
    constexpr std::size_t LOOKUP_SIZE = std::size_t{1} << LOOKUP_BITS;
    constexpr uint16_t LOOKUP_PIVOT = uint16_t{1} << (LOOKUP_BITS - 1);
} // namespace

histogram::histogram(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_port(&m_out_port);
    m_cfg.validate([](const histogram_config& c) {
        return c.percent_sampled > 0.0F && c.percent_sampled <= 1.0F;
    }, "percent_sampled must be in (0, 1]");
    // percent_sampled and display_as_bits are RUNTIME-configurable; react to live changes.
    m_cfg.on_apply([this](const histogram_config&, const composite::changes<histogram_config>& ch) {
        // percent_sampled drives both the frame-skip and per-histogram send thresholds.
        m_skip_threshold = static_cast<uint32_t>(1.0F / m_cfg->percent_sampled);
        if (m_sample_rate > 0.0F) {
            m_send_threshold = static_cast<uint32_t>(m_sample_rate * m_cfg->percent_sampled);
        }
        // display_as_bits changes the bin count, so the accumulator must be re-sized.
        if (ch.changed(&histogram_config::display_as_bits)) {
            allocate_histogram();
        }
    });
    add_config(m_cfg);
}

// Precompute, for every possible 16-bit sample value, the number of significant bits it uses
// (signed: negative magnitudes are negated). Indexed by (uint16 sample ^ LOOKUP_PIVOT).
auto histogram::build_lookup() -> void {
    const auto pivot = static_cast<int64_t>(LOOKUP_SIZE >> 1);
    m_sample_bits.assign(LOOKUP_SIZE, 0);
    for (std::size_t o = 0; o < LOOKUP_SIZE; ++o) {
        const auto val = static_cast<int64_t>(o) - pivot;
        if (val == 0) {
            continue;
        }
        const auto mag = static_cast<int8_t>(
            std::min(static_cast<double>(LOOKUP_BITS), std::floor(std::log2(std::abs(val))) + 1));
        m_sample_bits[o] = (val < 0) ? static_cast<int8_t>(-mag) : mag;
    }
}

auto histogram::allocate_histogram() -> void {
    const std::size_t bins = m_cfg->display_as_bits
        ? static_cast<std::size_t>(m_source_bits) * 2 + 1
        : std::size_t{1} << m_source_bits;
    m_histogram = composite::make_mutable<uint64_t>(bins);  // value-initialized to zeros
    m_histogram_samples = 0;
}

template <typename T>
auto histogram::process_samples(const uint8_t* bytes, std::size_t nbytes) -> void {
    using U = std::make_unsigned_t<T>;
    const auto samples = reinterpret_cast<const T*>(bytes);
    const std::size_t count = nbytes / sizeof(T);
    const std::size_t stride = m_is_complex ? 2U : 1U;  // bin the real component only
    const auto pivot = static_cast<U>(U{1} << (m_source_bits - 1));
    const bool byteswap = m_byteswap;
    const bool as_bits = m_cfg->display_as_bits;
    const std::size_t bins = m_histogram.size();

    std::size_t binned = 0;
    for (std::size_t i = 0; i < count; i += stride) {
        T s = samples[i];
        if (byteswap) {
            s = std::byteswap(s);
        }
        std::size_t bin;
        if (as_bits) {
            // Map the sample to its bit-magnitude bin via the precomputed lookup.
            const auto s16 = static_cast<int16_t>(s);
            const std::size_t off16 = static_cast<uint16_t>(s16) ^ LOOKUP_PIVOT;
            bin = static_cast<std::size_t>(m_sample_bits[off16] + static_cast<int>(m_source_bits));
        } else {
            // Value histogram: shift signed sample into an unsigned bin index.
            bin = static_cast<std::size_t>(static_cast<U>(s) ^ pivot);
        }
        if (bin < bins) {
            ++m_histogram[bin];
        }
        ++binned;
    }
    m_histogram_samples += static_cast<uint32_t>(binned);
}

auto histogram::initialize() -> void {
    m_sample_rate = m_cfg->sample_rate;
    m_skip_threshold = static_cast<uint32_t>(1.0F / m_cfg->percent_sampled);
    if (m_sample_rate > 0.0F) {
        m_send_threshold = static_cast<uint32_t>(m_sample_rate * m_cfg->percent_sampled);
    }
    build_lookup();
    allocate_histogram();
}

auto histogram::process() -> composite::retval {
    using enum composite::retval;
    auto pkt = m_in_port.try_get();
    if (!pkt) {
        return NOOP;
    }
    auto& [data, ts, meta] = *pkt;
    m_last_ts = ts;

    // Stream characteristics (rate, bit width, complex, endianness) come from the metadata.
    if (meta != nullptr) {
        if (meta->sample_rate > 0.0 && m_sample_rate != static_cast<float>(meta->sample_rate)) {
            m_sample_rate = static_cast<float>(meta->sample_rate);
            m_send_threshold = static_cast<uint32_t>(m_sample_rate * m_cfg->percent_sampled);
        }
        if (meta->format.bit_width > 0 &&
            (m_source_bits != meta->format.bit_width || m_is_complex != meta->format.is_complex)) {
            m_source_bits = meta->format.bit_width;
            m_is_complex = meta->format.is_complex;
            allocate_histogram();
        }
        // A single byte never needs swapping; otherwise swap when the wire endianness != host.
        m_auto_byteswap = (m_source_bits != 8) && (std::endian::native != meta->format.endianness);
    }
    // Config override wins when set; otherwise use the metadata-derived value.
    m_byteswap = m_cfg->byteswap.value_or(m_auto_byteswap);

    // Can't size the per-message send threshold until the sample rate is known.
    if (m_send_threshold == 0) {
        return NOOP;
    }

    // Decimate whole frames by percent_sampled, then bin the kept frame.
    if (++m_skip_counter >= m_skip_threshold) {
        m_skip_counter = 0;
        if (m_source_bits == 8) {
            process_samples<int8_t>(data.data(), data.size());
        } else {
            process_samples<int16_t>(data.data(), data.size());
        }
        // Send once enough samples have accrued; move the bins out and re-allocate a zeroed buffer.
        if (m_histogram_samples >= m_send_threshold) {
            const auto bins = m_histogram.size();
            m_out_port.send_data(std::move(m_histogram), ts);
            m_histogram = composite::make_mutable<uint64_t>(bins);
            m_histogram_samples = 0;
        }
    }
    return NORMAL;
}

auto histogram::on_end_of_stream() -> void {
    // The accumulator only ships once it crosses the send threshold; at end-of-stream any
    // below-threshold remainder would be dropped. Emit it (with the last input's timestamp),
    // then re-allocate a fresh zeroed buffer so a restart accumulates cleanly.
    if (m_histogram_samples == 0 || m_histogram.empty()) {
        return;
    }
    const auto bins = m_histogram.size();
    m_out_port.send_data(std::move(m_histogram), m_last_ts);
    m_histogram = composite::make_mutable<uint64_t>(bins);
    m_histogram_samples = 0;
}

COMPOSITE_REGISTER_SIMPLE(histogram)
