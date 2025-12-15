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
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include "fft_plan.hpp"
#include "task_queue.hpp"
#include "windows.hpp"

#include <bit>
#include <composite/composite.hpp>
#include <composite/buffers/buffer.hpp>
#include <composite/buffers/aligned_mem.hpp>
#include <complex>
#include <deque>
#include <fftw3.h>
#include <immintrin.h>
#include <memory>
#include <vector>

template <typename T>
class fft : public composite::component {
    using complex_t = std::complex<T>;
    using plan_t = fft_plan<T, true>;
    using window_t = composite::aligned_mem<T>;
    using input_port_t = composite::input_port<composite::immutable_buffer<complex_t>>;
    using output_port_t = composite::output_port<composite::immutable_buffer<complex_t>>;
    using output_tuple_t = std::tuple<composite::immutable_buffer<complex_t>, composite::timestamp, std::optional<composite::metadata>>;
    using enum composite::properties::config_type;

    static constexpr std::size_t ALIGNMENT = 64;

public:
    fft() : composite::component("fft") {
        add_port(&m_in_port);
        add_port(&m_out_port);
        add_property("window", &m_window_type).change_listener([this]() {
            return (m_window_type == "BLACKMAN_HARRIS") || (m_window_type == "HAMMING") || m_window_type.empty();
        });
        add_property("fft_size", &m_fft_size).configurability(RUNTIME).change_listener([this]() {
            return std::has_single_bit(m_fft_size);
        });
        add_property("num_workers", &m_num_workers).configurability(RUNTIME).change_listener([this]() {
            return m_num_workers <= 8u;
        });
        add_property("fftw_threads", &m_fftw_threads);
        add_property("shift", &m_shift).configurability(RUNTIME);
        if constexpr (std::is_same_v<T, float>) {
            fftwf_init_threads();
        } else {
            fftw_init_threads();
        }
    }

    ~fft() override {
        if constexpr (std::is_same_v<T, float>) {
            fftwf_cleanup_threads();
        } else {
            fftw_cleanup_threads();
        }
    }

    auto property_change_handler() -> void override {
        if (m_window_type == "BLACKMAN_HARRIS") {
            m_window = windows::blackman_harris<T>(m_fft_size);
        } else if (m_window_type == "HAMMING") {
            m_window = windows::hamming<T>(m_fft_size);
        } else {
            m_window.reset();
        }
        if (m_task_queue.thread_name_prefix() != id()) {
            m_task_queue.thread_name_prefix(id());
        }
        m_task_queue.resize(m_num_workers);
    }

    auto start() -> void override {
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
                    auto working_buf = composite::make_aligned_buffer<complex_t>(ALIGNMENT, data.size());

                    // Fused copy + window (or just copy if no window)
                    if (m_window) {
                        copy_and_window(data, working_buf, m_window.get());
                    } else {
                        std::copy(data.begin(), data.end(), working_buf.begin());
                    }

                    // Allocate output buffer for FFT result
                    auto output_buf = composite::make_aligned_buffer<complex_t>(ALIGNMENT, data.size());

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

    auto stop() -> void override {
        m_input_thread.request_stop();
        m_cv.notify_all();
        composite::component::stop();
    }

    auto process() -> composite::retval override {
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

private:
    // Fused copy + window: reads from immutable input, writes windowed data to mutable output
    [[gnu::target("default")]]
    auto copy_and_window(
        const composite::immutable_buffer<complex_t>& input,
        composite::mutable_buffer<complex_t>& output,
        const window_t* window
    ) -> void {
        const auto* in = reinterpret_cast<const T*>(input.data());
        auto* out = reinterpret_cast<T*>(output.data());
        const auto* w = window->data();
        const auto count = input.size() * 2;  // real + imag components
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = in[i] * w[i];
        }
    }

    [[gnu::target("avx512f")]]
    auto copy_and_window(
        const composite::immutable_buffer<complex_t>& input,
        composite::mutable_buffer<complex_t>& output,
        const window_t* window
    ) -> void {
        const auto* in = reinterpret_cast<const T*>(input.data());
        auto* out = reinterpret_cast<T*>(output.data());
        const auto* w = window->data();
        const auto count = input.size() * 2;

        std::size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
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

    [[gnu::target("avx2")]]
    auto copy_and_window(
        const composite::immutable_buffer<complex_t>& input,
        composite::mutable_buffer<complex_t>& output,
        const window_t* window
    ) -> void {
        const auto* in = reinterpret_cast<const T*>(input.data());
        auto* out = reinterpret_cast<T*>(output.data());
        const auto* w = window->data();
        const auto count = input.size() * 2;

        std::size_t i = 0;
        if constexpr (std::is_same_v<T, float>) {
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

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    std::string m_window_type;
    uint32_t m_fft_size{1024};
    uint32_t m_fftw_threads{1};
    uint32_t m_num_workers{1};
    bool m_shift{true};

    // Members
    std::unique_ptr<window_t> m_window{nullptr};
    std::deque<std::future<output_tuple_t>> m_futures;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    std::jthread m_input_thread;
    task_queue m_task_queue;

}; // class fft
