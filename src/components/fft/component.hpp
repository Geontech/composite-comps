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

#include "aligned_mem.hpp"
#include "fft_plan.hpp"
#include "windows.hpp"

#include <bit>
#include <composite/component.hpp>
#include <complex>
#include <fftw3.h>
#include <immintrin.h>
#include <memory>
#include <vector>

template <typename T>
class fft : public composite::component {
    using plan_t = fft_plan<T, true>;
    using fft_t = aligned::aligned_mem<std::complex<T>>;
    using window_t = aligned::aligned_mem<T>;
    using input_port_t = composite::input_port<std::unique_ptr<fft_t>>;
    using output_port_t = composite::output_port<std::unique_ptr<fft_t>>;
    using enum composite::properties::config_type;
public:
    fft() : composite::component("fft") {
        add_port(&m_in_port);
        add_port(&m_out_port);
        add_property("window", &m_window_type).change_listener([this]() {
            return (m_window_type == "BLACKMAN_HARRIS") || (m_window_type == "HAMMING");
        });
        add_property("fft_size", &m_fft_size).configurability(RUNTIME).change_listener([this]() {
            return std::has_single_bit(m_fft_size);
        });
        add_property("fftw_threads", &m_fftw_threads);
        add_property("shift", &m_shift).configurability(RUNTIME);
        // Initialize the function pointer based on CPU features
        if (__builtin_cpu_supports("avx512f")) {
            apply_window_func = &fft::apply_window_avx512;
        } else if (__builtin_cpu_supports("avx2")) {
            apply_window_func = &fft::apply_window_avx2;
        }
    }

    ~fft() override = default;

    auto property_change_handler() -> void override {
        if (m_window_type == "BLACKMAN_HARRIS") {
            m_window = windows::blackman_harris<T>(m_fft_size);
        } else if (m_window_type == "HAMMING") {
            m_window = windows::hamming<T>(m_fft_size);
        }
        m_fft_plan = std::make_unique<plan_t>(m_fft_size, m_fftw_threads, m_shift);
    }

    auto process() -> composite::retval override {
        using enum composite::retval;
        auto [data, ts, meta] = m_in_port.get_data();
        if (data == nullptr) {
            return NORMAL;
        }
        if (meta.has_value()) {
            logger()->trace("received metadata:\n{}", meta->to_string());
            meta->annotations["fft_size"] = std::to_string(m_fft_size);
            meta->annotations["fft_window"] = m_window_type;
            logger()->trace("sending updated metadata:\n{}", meta->to_string());
            m_out_port.send_metadata(meta.value());
        }
        // Apply window
        if (m_window) {
            (this->*apply_window_func)(data.get(), m_window.get());
        }
        // Execute the fft
        // In-place for complex
        m_fft_plan->execute(data.get(), data.get());
        // Send data
        m_out_port.send_data(std::move(data), ts);
        return NORMAL;
    }

private:
    [[gnu::target("avx512f")]]
    auto apply_window_avx512(fft_t* data, const window_t* window) -> void {
        // Logic for both types:
        // - load payload data
        // - load window data
        // - multiply payload by window
        // - store payload data
        auto stride = 512u / 8u / sizeof(double) / 2u/*complex*/;
        if constexpr (std::is_same_v<T, float>) {
            stride = 512u / 8u / sizeof(float) / 2u/*complex*/;
        }
        for (auto i=0u; i < data->size(); i += stride) {
            if constexpr (std::is_same_v<T, float>) {
                auto data_ptr = reinterpret_cast<float*>(data->data() + i);
                auto payload = _mm512_load_ps(data_ptr);
                auto window_ps = _mm512_load_ps(window->data() + i * 2);
                payload = _mm512_mul_ps(payload, window_ps);
                _mm512_store_ps(data_ptr, payload);
            } else {
                auto data_ptr = reinterpret_cast<double*>(data->data() + i);
                auto payload = _mm512_load_pd(data_ptr);
                auto window_pd = _mm512_load_pd(window->data() + i * 2);
                payload = _mm512_mul_pd(payload, window_pd);
                _mm512_store_pd(data_ptr, payload);
            }
        }
    }

    [[gnu::target("avx2")]]
    auto apply_window_avx2(fft_t* data, const window_t* window) -> void {
        // Logic for both types:
        // - load payload data
        // - load window data
        // - multiply payload by window
        // - store payload data
        auto stride = 256u / 8u / sizeof(double) / 2u/*complex*/;
        if constexpr (std::is_same_v<T, float>) {
            stride = 256u / 8u / sizeof(float) / 2u/*complex*/;
        }
        for (auto i=0u; i < data->size(); i += stride) {
            if constexpr (std::is_same_v<T, float>) {
                auto data_ptr = reinterpret_cast<float*>(data->data() + i);
                auto payload = _mm256_load_ps(data_ptr);
                auto window_ps = _mm256_load_ps(window->data() + i * 2);
                payload = _mm256_mul_ps(payload, window_ps);
                _mm256_store_ps(data_ptr, payload);
            } else {
                auto data_ptr = reinterpret_cast<double*>(data->data() + i);
                auto payload = _mm256_load_pd(data_ptr);
                auto window_pd = _mm256_load_pd(window->data() + i * 2);
                payload = _mm256_mul_pd(payload, window_pd);
                _mm256_store_pd(data_ptr, payload);
            }
        }
    }

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    std::string m_window_type;
    uint32_t m_fft_size{1024};
    uint32_t m_fftw_threads{1};
    bool m_shift{true};

    // Members
    auto (fft::*apply_window_func)(fft_t*, const window_t*) -> void;
    std::unique_ptr<fft_plan<T, true>> m_fft_plan{nullptr};
    std::unique_ptr<window_t> m_window{nullptr};

}; // class fft
