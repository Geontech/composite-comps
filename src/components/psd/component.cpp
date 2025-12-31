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

#include <algorithm>
#include <format>
#include <numeric>
#include <pthread.h>
#include <string_view>

template <typename T>
psd<T>::psd(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_port(&m_out_port);
    using enum composite::properties::config_type;
    add_property("num_workers", m_num_workers, RUNTIME).change_listener([this]() {
        return m_num_workers <= 8u;
    });
    add_property("power_based_normalization", m_power_based_normalization, RUNTIME);
}

template <typename T>
auto psd<T>::property_change_handler() -> void {
    initialize();
}

template <typename T>
auto psd<T>::initialize() -> void {
    // Setup task queue
    if (m_task_queue.thread_name_prefix() != id()) {
        m_task_queue.thread_name_prefix(id());
    }
    m_task_queue.resize(m_num_workers);
    // Calculate normalization constant
    m_work.norm_const(calculate_norm_const());
}

template <typename T>
auto psd<T>::start() -> void {
    m_input_thread = std::jthread([&](std::stop_token stoken) {
        while (!stoken.stop_requested()) {
            // Get data buffer
            auto [data, ts, meta] = m_in_port.get_data();
            if (!data) {
                std::this_thread::yield();
                continue;
            }
            if (meta.has_value()) {
                logger()->trace("received metadata:\n{}", meta->to_string());
                m_sample_rate = meta->sample_rate;
                if (m_metadata.sample_rate != m_sample_rate) {
                    // Calculate new normalization constant
                    m_work.norm_const(calculate_norm_const());
                }
                // Check if we need to rebuild the window
                auto curr_fft_size = std::size_t{};
                auto curr_fft_window = std::string{};
                if (m_metadata.annotations.contains("fft_size")) {
                    try {
                        curr_fft_size = std::stoul(m_metadata.annotations.at("fft_size"));
                    } catch (...) {}
                }
                if (m_metadata.annotations.contains("fft_window")) {
                    curr_fft_window = m_metadata.annotations.at("fft_window");
                }
                if (meta->annotations.contains("fft_size") && meta->annotations.contains("fft_window")) {
                    auto meta_fft_size = std::size_t{};
                    try {
                        meta_fft_size = std::stoul(meta->annotations.at("fft_size"));
                    } catch (...) {}
                    auto meta_fft_window = meta->annotations.at("fft_window");
                    if (curr_fft_size != meta_fft_size || curr_fft_window != meta_fft_window) {
                        logger()->trace(
                            "metadata fft properties differ from current known fft properties; curr=(size: {}, window={}) vs metadata=(size: {}, window: {})",
                            curr_fft_size, curr_fft_window, meta_fft_size, meta_fft_window
                        );
                        // Create new window
                        if (meta_fft_window == "BLACKMAN_HARRIS") {
                            m_window = windows::blackman_harris<T>(meta_fft_size, false);
                        } else if (meta_fft_window == "HAMMING") {
                            m_window = windows::hamming<T>(meta_fft_size, false);
                        } else {
                            m_window.reset();
                        }
                        // Calculate new normalization constant
                        m_work.norm_const(calculate_norm_const());
                    }
                }
                meta->annotations["psd_power_based_normalization"] = std::to_string(m_power_based_normalization);
                m_metadata = meta.value();
            }
            // Submit PSD task to pool
            auto fut = m_task_queue.submit([data = std::move(data), ts, meta = std::move(meta), this]() mutable -> output_tuple_t {
                // Perform PSD
                auto psd = m_work.process(data);
                // Return modified data, meta, and original ts
                return std::make_tuple(std::move(psd), ts, std::move(meta));
            });
            // Push future onto queue
            {
                auto lock = std::scoped_lock{m_mtx};
                m_futures.push_back(std::move(fut));
            }
            m_cv.notify_one();
        }
    });
    pthread_setname_np(m_input_thread.native_handle(), std::format("{}-in", id()).c_str());
    composite::component::start();
}

template <typename T>
auto psd<T>::stop() -> void {
    m_input_thread.request_stop();
    m_cv.notify_all();
    composite::component::stop();
}

template <typename T>
auto psd<T>::process() -> composite::retval {
    using enum composite::retval;
    // Pop a future from the queue
    using namespace std::chrono_literals;
    auto lock = std::unique_lock{m_mtx};
    m_cv.wait_for(lock, 1s, [this]{ return !m_futures.empty(); });
    if (m_futures.empty()) {
        return NORMAL;
    }
    auto fut = std::move(m_futures.front());
    m_futures.pop_front();
    lock.unlock();

    // Get result data from future
    auto [data, ts, meta] = fut.get();
    if (meta.has_value()) {
        logger()->trace("sending updated metadata:\n{}", meta->to_string());
        m_out_port.send_metadata(meta.value());
    }

    // Send data
    m_out_port.send_data(std::move(data), ts);
    return NORMAL;
}

template <typename T>
auto psd<T>::calculate_norm_const() const -> T {
    // If sample_rate is unknown, return 1
    if (m_sample_rate == T{}) {
        return T{1};
    }
    // Calculate window normalization constant
    auto window_norm_const = T{1};
    if (m_window) {
        // Calculate sum of squared window values without modifying the window
        auto window_sum_sq = std::transform_reduce(
            m_window->data(),
            m_window->data() + m_window->size(),
            T{0},
            std::plus<>{},
            [](T val) { return val * val; }
        );
        // Window normalization constant
        window_norm_const = window_sum_sq;
        if (m_power_based_normalization) {
            window_norm_const = window_norm_const / static_cast<T>(m_window->size());
        }
        logger()->trace("calculated window norm constant of: {}", window_norm_const);
    }
    // Calculate normalization constant
    return T{1} / (m_sample_rate * window_norm_const);
}

// Explicit template instantiations
template class psd<float>;
template class psd<double>;

extern "C" {
    auto create(std::string_view id, std::string_view type) -> std::shared_ptr<composite::component> {
        if (type == "f32") {
            return std::make_shared<psd<float>>(id);
        } else if (type == "f64") {
            return std::make_shared<psd<double>>(id);
        }
        throw std::runtime_error(std::format("unknown type '{}' for psd component", type));
    }
}
