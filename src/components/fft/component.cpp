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
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */

#include "component.hpp"
#include "windows.hpp"

#include <bit>
#include <fftw3.h>
#include <format>
#include <immintrin.h>
#include <pthread.h>

template <typename T>
fft<T>::fft(std::string_view id) : composite::component(id) {
    using enum composite::properties::config_type;

    add_port(&m_in_port);
    add_port(&m_out_port);
    add_property("window", m_window_type).change_listener([this]() {
        return (m_window_type == "BLACKMAN_HARRIS") || (m_window_type == "HAMMING") || m_window_type.empty();
    });
    add_property("fft_size", m_fft_size).configurability(RUNTIME).change_listener([this]() {
        return std::has_single_bit(m_fft_size);
    });
    add_property("num_workers", m_num_workers).configurability(RUNTIME).change_listener([this]() {
        return m_num_workers <= 8u;
    });
    add_property("fftw_threads", m_fftw_threads);
    add_property("shift", m_shift).configurability(RUNTIME);

    if constexpr (std::is_same_v<scalar_t, float>) {
        fftwf_init_threads();
    } else {
        fftw_init_threads();
    }
}

template <typename T>
fft<T>::~fft() {
    if constexpr (std::is_same_v<scalar_t, float>) {
        fftwf_cleanup_threads();
    } else {
        fftw_cleanup_threads();
    }
}

template <typename T>
auto fft<T>::property_change_handler() -> void {
    if (m_window_type == "BLACKMAN_HARRIS") {
        m_window = windows::blackman_harris<scalar_t>(m_fft_size);
    } else if (m_window_type == "HAMMING") {
        m_window = windows::hamming<scalar_t>(m_fft_size);
    } else {
        m_window.reset();
    }
    if (m_task_queue.thread_name_prefix() != id()) {
        m_task_queue.thread_name_prefix(id());
    }
    m_task_queue.resize(m_num_workers);
}

template <typename T>
auto fft<T>::start() -> void {
    m_input_thread = std::jthread([&](std::stop_token stoken) {
        while (!stoken.stop_requested()) {
            auto [data, ts, meta] = m_in_port.get_data();
            if (!data) {
                std::this_thread::yield();
                continue;
            }

            auto fut = m_task_queue.submit([data = std::move(data), ts, meta = std::move(meta), this]() mutable -> output_tuple_t {
                thread_local std::unique_ptr<plan_t> fft_plan;
                if (!fft_plan || fft_plan->size() != m_fft_size) {
                    fft_plan = std::make_unique<plan_t>(m_fft_size, m_fftw_threads);
                }

                // Allocate working buffer for windowed input data
                auto working_buf = composite::make_aligned_buffer<T>(ALIGNMENT, data.size());

                // Fused copy + window (or just copy if no window)
                if (m_window) {
                    copy_and_window(data, working_buf, m_window.get());
                } else {
                    std::copy(data.begin(), data.end(), working_buf.begin());
                }

                // Allocate output buffer for FFT result
                auto output_buf = composite::make_aligned_buffer<T>(ALIGNMENT, data.size());

                // Execute out-of-place FFT: working -> output
                fft_plan->execute(working_buf.data(), output_buf.data());

                // Apply fftshift if configured
                if (m_shift) {
                    std::rotate(
                        output_buf.begin(),
                        output_buf.begin() + (output_buf.size() / 2),
                        output_buf.end()
                    );
                }

                // Convert to immutable for output
                return std::make_tuple(std::move(output_buf).to_immutable(), ts, std::move(meta));
            });

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
auto fft<T>::stop() -> void {
    m_input_thread.request_stop();
    m_cv.notify_all();
    composite::component::stop();
}

template <typename T>
auto fft<T>::process() -> composite::retval {
    using enum composite::retval;
    using namespace std::chrono_literals;

    auto lock = std::unique_lock{m_mtx};
    m_cv.wait_for(lock, 1s, [this]{ return !m_futures.empty(); });
    if (m_futures.empty()) {
        return NORMAL;
    }
    auto fut = std::move(m_futures.front());
    m_futures.pop_front();
    lock.unlock();

    auto [data, ts, meta] = fut.get();
    if (meta.has_value()) {
        logger()->trace("received metadata:\n{}", meta->to_string());
        meta->annotations["fft_size"] = std::to_string(m_fft_size);
        meta->annotations["fft_window"] = m_window_type;
        logger()->trace("sending updated metadata:\n{}", meta->to_string());
        m_out_port.send_metadata(meta.value());
    }

    m_out_port.send_data(std::move(data), ts);
    return NORMAL;
}

// Fused copy + window: scalar fallback
template <typename T>
[[gnu::target("default")]]
auto fft<T>::copy_and_window(
  const composite::immutable_buffer<T>& input,
  composite::mutable_buffer<T>& output,
  const window_t* window
) -> void {
    const auto* in = reinterpret_cast<const scalar_t*>(input.data());
    auto* out = reinterpret_cast<scalar_t*>(output.data());
    const auto* w = window->data();
    const auto count = input.size() * 2;  // real + imag components
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = in[i] * w[i];
    }
}

// Fused copy + window: AVX-512
template <typename T>
[[gnu::target("avx512f")]]
auto fft<T>::copy_and_window(
  const composite::immutable_buffer<T>& input,
  composite::mutable_buffer<T>& output,
  const window_t* window
) -> void {
    const auto* in = reinterpret_cast<const scalar_t*>(input.data());
    auto* out = reinterpret_cast<scalar_t*>(output.data());
    const auto* w = window->data();
    const auto count = input.size() * 2;

    std::size_t i = 0;
    if constexpr (std::is_same_v<scalar_t, float>) {
        constexpr std::size_t stride = 16;  // 512 bits / 32 bits
        for (; i + stride <= count; i += stride) {
            auto input_v = _mm512_loadu_ps(in + i);
            auto window_v = _mm512_loadu_ps(w + i);
            auto result = _mm512_mul_ps(input_v, window_v);
            _mm512_storeu_ps(out + i, result);
        }
    } else {
        constexpr std::size_t stride = 8;  // 512 bits / 64 bits
        for (; i + stride <= count; i += stride) {
            auto input_v = _mm512_loadu_pd(in + i);
            auto window_v = _mm512_loadu_pd(w + i);
            auto result = _mm512_mul_pd(input_v, window_v);
            _mm512_storeu_pd(out + i, result);
        }
    }
    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = in[i] * w[i];
    }
}

// Fused copy + window: AVX2
template <typename T>
[[gnu::target("avx2")]]
auto fft<T>::copy_and_window(
  const composite::immutable_buffer<T>& input,
  composite::mutable_buffer<T>& output,
  const window_t* window
) -> void {
    const auto* in = reinterpret_cast<const scalar_t*>(input.data());
    auto* out = reinterpret_cast<scalar_t*>(output.data());
    const auto* w = window->data();
    const auto count = input.size() * 2;

    std::size_t i = 0;
    if constexpr (std::is_same_v<scalar_t, float>) {
        constexpr std::size_t stride = 8;  // 256 bits / 32 bits
        for (; i + stride <= count; i += stride) {
            auto input_v = _mm256_loadu_ps(in + i);
            auto window_v = _mm256_loadu_ps(w + i);
            auto result = _mm256_mul_ps(input_v, window_v);
            _mm256_storeu_ps(out + i, result);
        }
    } else {
        constexpr std::size_t stride = 4;  // 256 bits / 64 bits
        for (; i + stride <= count; i += stride) {
            auto input_v = _mm256_loadu_pd(in + i);
            auto window_v = _mm256_loadu_pd(w + i);
            auto result = _mm256_mul_pd(input_v, window_v);
            _mm256_storeu_pd(out + i, result);
        }
    }
    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = in[i] * w[i];
    }
}

extern "C" {
    auto create(std::string_view id, std::string_view type) -> std::shared_ptr<composite::component> {
        if (type == "cf32") {
            return std::make_shared<fft<std::complex<float>>>(id);
        } else if (type == "cf64") {
            return std::make_shared<fft<std::complex<double>>>(id);
        }
        throw std::runtime_error(std::format("unknown type '{}' for fft component", type));
    }
}
