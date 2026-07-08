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
#include "windows.hpp"

#include <composite/core/register.hpp>

#include <complex>
#include <format>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>

template <typename T>
psd<T>::psd(std::string_view id)
    : base(id, "data_in", "data_out", /*default_workers=*/1) {
    using enum composite::properties::config_type;
    // num_workers is provided by pipeline_component.
    this->add_property("power_based_normalization", m_power_based_normalization, RUNTIME);
    m_pbn.store(m_power_based_normalization, std::memory_order_relaxed);
}

template <typename T>
auto psd<T>::property_change_handler(const composite::properties::json& diff) -> void {
    (void)diff;
    // Publish the config snapshot read by the pool's work() (pool threads are not parked). A
    // num_workers change is handled separately by pipeline_component.
    m_pbn.store(m_power_based_normalization, std::memory_order_relaxed);
    // prepare() stamps the normalization mode onto the shared metadata; tell the pipeline to
    // rebuild it even though the incoming metadata instance is unchanged.
    this->invalidate_prepared_metadata();
}

template <typename T>
auto psd<T>::prepare(composite::metadata& md) -> void {
    // ARRIVAL order, main thread: record the normalization mode (from the snapshot) on the metadata.
    md.annotations["psd_power_based_normalization"] =
        std::to_string(m_pbn.load(std::memory_order_relaxed));
}

template <typename T>
auto psd<T>::compute_norm_const(const window_t* window, T sample_rate, bool power_based) -> T {
    if (sample_rate == T{}) {
        return T{1};
    }
    T window_norm_const = T{1};
    if (window != nullptr) {
        auto window_sum_sq = std::transform_reduce(
            window->data(), window->data() + window->size(), T{0}, std::plus<>{},
            [](T val) { return val * val; });
        window_norm_const = window_sum_sq;
        if (power_based) {
            window_norm_const = window_norm_const / static_cast<T>(window->size());
        }
    }
    return T{1} / (sample_rate * window_norm_const);
}

template <typename T>
auto psd<T>::work(composite::mutable_buffer<std::complex<T>> in, composite::timestamp ts,
                  const composite::metadata& md) -> composite::mutable_buffer<T> {
    (void)ts;
    // Per-pool-worker state: the window + PSD kernel for this worker, rebuilt only when the FFT
    // params / sample rate / mode actually change (consecutive packets usually share them).
    thread_local std::unique_ptr<window_t> tl_window;
    thread_local std::size_t tl_size{0};
    thread_local std::string tl_wtype;
    thread_local T tl_sample_rate{-1};
    thread_local bool tl_pbn{true};
    thread_local bool tl_have_key{false};
    thread_local ::work<T> tl_work;  // ::work disambiguates the kernel class from this method

    // FFT params + sample rate are stamped on the metadata by the upstream fft.
    std::size_t fft_size{0};
    std::string wtype;
    if (md.annotations.contains("fft_size")) {
        try { fft_size = std::stoul(md.annotations.at("fft_size").to_string()); } catch (...) {}
    }
    if (md.annotations.contains("fft_window")) {
        wtype = md.annotations.at("fft_window").to_string();
    }
    const T sample_rate = static_cast<T>(md.sample_rate);
    const bool pbn = m_pbn.load(std::memory_order_relaxed);

    // Rebuild the window only when (size, type) changes.
    if (!tl_have_key || fft_size != tl_size || wtype != tl_wtype) {
        if (wtype == "BLACKMAN_HARRIS") {
            tl_window = windows::blackman_harris<T>(fft_size, false);
        } else if (wtype == "HAMMING") {
            tl_window = windows::hamming<T>(fft_size, false);
        } else {
            tl_window.reset();
        }
        tl_size = fft_size;
        tl_wtype = wtype;
        tl_have_key = true;
        tl_sample_rate = T{-1};  // force a norm-const recompute below
    }
    // Recompute the norm const only when window / sample_rate / mode changes.
    if (sample_rate != tl_sample_rate || pbn != tl_pbn) {
        tl_work.norm_const(compute_norm_const(tl_window.get(), sample_rate, pbn));
        tl_sample_rate = sample_rate;
        tl_pbn = pbn;
    }
    return tl_work.process(in);
}

// Explicit template instantiations
template class psd<float>;
template class psd<double>;

COMPOSITE_REGISTER_COMPONENT([](std::string_view id, const composite::create_args& args)
                                 -> std::shared_ptr<composite::component> {
    const auto type = args.type();
    if (type == "f32") {
        return std::make_shared<psd<float>>(id);
    } else if (type == "f64") {
        return std::make_shared<psd<double>>(id);
    }
    throw std::runtime_error(std::format("unknown type '{}' for psd component", type));
})
