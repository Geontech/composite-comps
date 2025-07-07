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

#include "work.hpp"

#include <aligned_mem.hpp>
#include <windows.hpp>

#include <composite/component.hpp>
#include <complex>
#include <immintrin.h>
#include <memory>

template <typename T>
class psd : public composite::component {
    using fft_t = aligned::aligned_mem<std::complex<T>>;
    using psd_t = aligned::aligned_mem<T>;
    using window_t = aligned::aligned_mem<T>;
    using input_port_t = composite::input_port<std::unique_ptr<fft_t>>;
    using output_port_t = composite::output_port<std::unique_ptr<psd_t>>;
public:
    psd() : composite::component("psd") {
        add_port(&m_in_port);
        add_port(&m_out_port);
        using enum composite::properties::config_type;
        add_property("power_based_normalization", &m_power_based_normalization)
            .configurability(RUNTIME);
    }

    ~psd() override = default;

    auto property_change_handler() -> void override {
        initialize();
    }

    auto initialize() -> void override {
        // Calculate normalization constant
        auto norm_const = calculate_norm_const();
        // Create work class (if needed)
        create_worker(norm_const);
    }

    auto process() -> composite::retval override {
        using enum composite::retval;
        auto [data, ts, meta] = m_in_port.get_data();
        if (data == nullptr) {
            return NOOP;
        }
        if (meta.has_value()) {
            logger()->trace("received metadata:\n{}", meta->to_string());
            m_sample_rate = meta->sample_rate;
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
                    }
                    // Calculate new normalization constant
                    auto norm_const = calculate_norm_const();
                    // Create work class (if needed)
                    create_worker(norm_const);
                    // Update flag
                    m_norm_changed = false;
                }
            }
            // Update and send metadata
            m_metadata = meta.value();
            m_metadata.annotations["psd_power_based_normalization"] = std::to_string(m_power_based_normalization);
            logger()->trace("sending updated metadata:\n{}", m_metadata.to_string());
            m_out_port.send_metadata(m_metadata);
        } else if (m_norm_changed) {
            logger()->trace("detected normalization property changed; checking for worker update requirements");
            // Calculate new normalization constant
            auto norm_const = calculate_norm_const();
            // Create work class (if needed)
            create_worker(norm_const);
            // Update and send metadata
            m_metadata.annotations["psd_power_based_normalization"] = std::to_string(m_power_based_normalization);
            logger()->trace("sending updated metadata:\n{}", m_metadata.to_string());
            m_out_port.send_metadata(m_metadata);
            // Update flag
            m_norm_changed = false;
        }

        // Perform PSD
        auto psd = m_work->process(data.get());
        std::transform(psd->data(), psd->data() + psd->size(), psd->data(), [](T val) {
            if (val > T{0}) [[likely]] {
                if constexpr (std::is_same_v<T, float>) {
                    return std::log2f(val);
                }
                return std::log2(val);
            }
            return val;
        });
        // Apply log multiplier
        m_work->apply_multiplier(psd.get());
        // Send data
        m_out_port.send_data(std::move(psd), ts);

        return NORMAL;
    }

private:
    auto calculate_norm_const() const -> T {
        // Calculate window normalization constant
        auto window_norm_const = T{1};
        if (m_window) {
            // Square all values in the window vector
            std::transform(
                m_window->data(),
                m_window->data() + m_window->size(),
                m_window->data(),
                [](float val) {
                    return val * val;
                }
            );
            // Calculate the sum of the squared window
            auto window_sum = std::accumulate(m_window->data(), m_window->data() + m_window->size(), T{});
            // Window normalization constant
            window_norm_const = window_sum;
            if (m_power_based_normalization) {
                window_norm_const = window_norm_const / m_window->size();
            }
            logger()->trace("calculated window norm constant of: {}", window_norm_const);
        }
        // Calculate normalization constant
        if (m_sample_rate == T{}) {
            return T{1};
        }
        return T{1} / (m_sample_rate * window_norm_const);
    }

    auto create_worker(T norm_const) -> void {
        if (m_work == nullptr) {
            logger()->trace("creating new worker with norm const: {}", norm_const);
            m_work = std::make_unique<work<T>>(norm_const);
        } else if (m_work->norm_const() != norm_const) {
            logger()->trace("creating new worker with updated norm const: {} (prev={})", norm_const, m_work->norm_const());
            m_work = std::make_unique<work<T>>(norm_const);
            m_norm_changed = true;
        }
    }

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    bool m_power_based_normalization{true};

    // Members
    composite::metadata m_metadata;
    std::unique_ptr<window_t> m_window;
    std::unique_ptr<work<T>> m_work;
    bool m_norm_changed{false};
    T m_sample_rate{1};

}; // class psd
