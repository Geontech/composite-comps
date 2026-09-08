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
    m_meta_rebaselines = &create_counter(
        "exp_smooth.metadata_rebaselines",
        "Re-baselines forced by a metadata change (retune, normalization/window change)");
    add_property("num_averages", m_num_averages, RUNTIME).on_change([this](const composite::properties::json&) {
        m_alpha.reset();
        if (m_num_averages > 0) {
            // alpha = 1 - 10^(log10(0.02)/N) = -expm1(ln(0.02)/N), computed via expm1 in
            // double: the naive 1 - pow(...) form cancels to EXACTLY zero for large N
            // (float rounds 1 - 4e-9 to 1), which froze the filter instead of smoothing
            // very slowly. expm1 keeps full precision arbitrarily close to zero.
            const auto y = std::log(1.0 - 0.98) / static_cast<double>(m_num_averages);
            m_alpha = static_cast<T>(-std::expm1(y));
        }
        m_work.emplace(m_alpha.value_or(T{1}));
        // Deliberately NO m_prev_psd reset: the accumulated average is still valid data — a
        // runtime alpha change simply re-weights future samples. Resetting here silently
        // dropped the held frame (the 1-frame delay's buffer); the pass-through branch in
        // process_packet flushes it on a smoothing -> pass-through switch instead.
    });
}

template <typename T>
auto exp_smooth<T>::process() -> composite::retval {
    using enum composite::retval;
    // One frame per process() call, deliberately NOT the get_batch drain the parsers use:
    // their per-packet work is small enough that ring-head amortization matters, while this
    // component's per-frame kernel (a full-frame SIMD pass) dwarfs one try_get. Draining a
    // batch here would also fire up to a batch of back-to-back drop-on-full send_data()
    // calls and stretch the framework's per-iteration yield cadence — burstier output and a
    // higher downstream-drop risk for no measurable gain.
    auto pkt = m_in_port.try_get();
    if (!pkt) {
        // No input: NOOP parks the worker on the read-doorbell; at end-of-stream the base
        // promotes it to FINISH (see on_end_of_stream for the held-frame flush).
        return NOOP;
    }
    process_packet(std::move(*pkt));
    return NORMAL;
}

template <typename T>
auto exp_smooth<T>::process_packet(typename input_port_t::queue_type pkt) -> void {
    auto& [data, ts, meta] = pkt;
    // Metadata now rides WITH the data (send_metadata was removed): it travels with the output
    // frame produced from this input frame. exp_smooth has a 1-frame delay (it emits the
    // previous PSD), so metadata is held in m_prev_meta alongside m_prev_psd and sent with it.
    // Only when the shared instance changes — to_string()'s argument is evaluated eagerly,
    // so logging it per packet would allocate on the hot path even with trace disabled.
    if (meta != nullptr && meta != m_prev_meta) {
        logger()->trace("pass-through metadata:\n{}", meta->to_string());
    }
    if (!m_alpha.has_value()) {
        // No smoothing. A runtime smoothing -> pass-through switch can leave the 1-frame
        // delay's buffer holding the final smoothed frame; flush it (on the worker thread,
        // respecting the output port's single-producer contract) instead of silently
        // dropping it or leaking it into a much-later end-of-stream flush.
        if (!m_prev_psd.empty()) {
            m_out_port.send_data(std::move(m_prev_psd), m_prev_psd_ts, m_prev_meta);
            m_prev_meta = nullptr;
        }
        m_out_port.send_data(std::move(data), ts, meta);
        return;
    }
    // Handle first PSD
    if (m_prev_psd.empty()) {
        m_prev_psd = std::move(data);
        m_prev_psd_ts = ts;
        m_prev_meta = meta;
        return;
    }
    // Re-baseline instead of running the kernel when the EWMA history no longer applies to
    // this frame: flush the last buffered frame, then adopt this one as the new starting
    // point (like the first PSD). Two triggers:
    //  - the frame SIZE changed (an upstream fft_size change) — averaging across it would
    //    index mismatched bins (and throwing would recur forever on the stale m_prev_psd);
    //  - the METADATA changed (retune, sample-rate/window/shift/normalization change) — the
    //    bins still line up numerically but are semantically incompatible, and smoothing
    //    across the change would ship old-configuration energy under new-configuration
    //    metadata. Per the port contract (output_port.hpp: "nullptr = none", "no latch"),
    //    metadata is a per-packet fact: a transition between declared and absent — in EITHER
    //    direction — is a semantic discontinuity like any value change, and a bare frame's
    //    output stays bare. Steady state is a pointer compare (conforming producers pass the
    //    same instance for every packet between changes, and null == null covers bare
    //    streams); the deep compare runs only on an instance change, so a producer that
    //    rebuilds equal-valued instances does not force a re-baseline per packet.
    bool rebaseline = false;
    if (data.size() != m_prev_psd.size()) {
        m_size_mismatch->inc();
        rebaseline = true;
    } else if (meta != m_prev_meta &&
               (meta == nullptr || m_prev_meta == nullptr || !(*meta == *m_prev_meta))) {
        m_meta_rebaselines->inc();
        rebaseline = true;
    }
    if (rebaseline) {
        m_out_port.send_data(std::move(m_prev_psd), m_prev_psd_ts, m_prev_meta);
        m_prev_psd = std::move(data);
        m_prev_psd_ts = ts;
        m_prev_meta = meta;
        return;
    }
    // Run algorithm (sizes match).
    m_work->process(data, m_prev_psd);
    // Send previous PSD data + timestamp + the metadata that arrived with it.
    m_out_port.send_data(std::move(m_prev_psd), m_prev_psd_ts, m_prev_meta);
    // Save current PSD/ts/metadata for next pass
    m_prev_psd = std::move(data);
    m_prev_psd_ts = ts;
    m_prev_meta = meta;
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
