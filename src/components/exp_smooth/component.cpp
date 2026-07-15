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

#include <composite/core/register.hpp>

#include <cmath>
#include <format>
#include <string_view>

template <typename T>
exp_smooth<T>::exp_smooth(std::string_view id) : composite::component(id) {
    using enum composite::properties::config_type;
    add_port(&m_in_port);
    add_port(&m_out_port);
    m_size_mismatch = &create_counter("exp_smooth.size_mismatch", "Frames whose size differed from the previous frame");
    add_property("num_averages", m_num_averages, RUNTIME).on_change([this](const composite::properties::json&) {
        m_alpha.reset();
        if (m_num_averages > 0) {
            m_alpha = T{1} - std::pow(T{10}, (std::log10(T{1} - T{0.98}) / m_num_averages));
        }
        m_work.emplace(m_alpha.value_or(T{1}));
        m_prev_psd = composite::mutable_buffer<T>{}; // Reset to empty buffer
    });
}

template <typename T>
auto exp_smooth<T>::process() -> composite::retval {
    using enum composite::retval;
    auto pkt = m_in_port.try_get();
    if (!pkt) {
        return NOOP;
    }
    auto& [data, ts, meta] = *pkt;
    // Metadata now rides WITH the data (send_metadata was removed): it travels with the output
    // frame produced from this input frame. exp_smooth has a 1-frame delay (it emits the
    // previous PSD), so metadata is held in m_prev_meta alongside m_prev_psd and sent with it.
    // Only when the shared instance changes — to_string()'s argument is evaluated eagerly,
    // so logging it per packet would allocate on the hot path even with trace disabled.
    if (meta != nullptr && meta != m_prev_meta) {
        logger()->trace("pass-through metadata:\n{}", meta->to_string());
    }
    if (!m_alpha.has_value()) {
        // No smoothing, return as is (data carries its own metadata)
        m_out_port.send_data(std::move(data), ts, meta);
        return NORMAL;
    }
    // Handle first PSD
    if (m_prev_psd.empty()) {
        m_prev_psd = std::move(data);
        m_prev_psd_ts = ts;
        m_prev_meta = meta;
        return NORMAL;
    }
    // Frame size changed mid-stream (e.g. an upstream fft_size change): the EWMA history no longer
    // applies to the new size. Re-baseline instead of running the kernel — flush the last buffered
    // frame, then adopt this frame as the new starting point (like the first PSD). Throwing here
    // would recur on every subsequent packet (m_prev_psd would keep the old size), so the stream
    // could never recover from a size change; re-baselining lets it continue at the new size.
    if (data.size() != m_prev_psd.size()) {
        m_size_mismatch->inc();
        m_out_port.send_data(std::move(m_prev_psd), m_prev_psd_ts, m_prev_meta);
        m_prev_psd = std::move(data);
        m_prev_psd_ts = ts;
        m_prev_meta = meta;
        return NORMAL;
    }
    // Run algorithm (sizes match).
    m_work->process(data, m_prev_psd);
    // Send previous PSD data + timestamp + the metadata that arrived with it.
    m_out_port.send_data(std::move(m_prev_psd), m_prev_psd_ts, m_prev_meta);
    // Save current PSD/ts/metadata for next pass
    m_prev_psd = std::move(data);
    m_prev_psd_ts = ts;
    m_prev_meta = meta;
    return NORMAL;
}

template <typename T>
auto exp_smooth<T>::on_end_of_stream() -> void {
    // 1-frame delay: the most recent PSD is still buffered in m_prev_psd (the steady-state path emits
    // the PREVIOUS frame each call, so the last one arrives with no successor to trigger its send).
    // Emit it before the outputs close so the final frame isn't silently dropped. The no-smoothing
    // path never buffers, so m_prev_psd is empty there and this is a no-op.
    if (!m_prev_psd.empty()) {
        m_out_port.send_data(std::move(m_prev_psd), m_prev_psd_ts, m_prev_meta);
    }
}

// --- Factory Function ---
COMPOSITE_REGISTER_COMPONENT([](std::string_view id, const composite::create_args& args)
                                 -> std::shared_ptr<composite::component> {
    const auto type = args.type();
    if (type == "f32") {
        return composite::make_component<exp_smooth<float>>(id);
    } else if (type == "f64") {
        return composite::make_component<exp_smooth<double>>(id);
    }
    throw std::runtime_error(std::format("unknown type '{}' for exp_smooth component", type));
})
