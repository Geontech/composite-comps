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

#include "aligned_mem.hpp"
#include "fft_plan.hpp"
#include "task_queue.hpp"
#include "windows.hpp"

#include <bit>
#include <composite/component.hpp>
#include <complex>
#include <deque>
#include <fftw3.h>
#include <immintrin.h>
#include <memory>
#include <vector>

template <typename T>
class fft : public composite::component {
    using plan_t = fft_plan<T, true>;
    using fft_t = aligned::aligned_mem<std::complex<T>>;
    using window_t = aligned::aligned_mem<T>;
    using input_t = std::unique_ptr<fft_t>;
    using input_port_t = composite::input_port<input_t>;
    using output_port_t = composite::output_port<std::unique_ptr<fft_t>>;
    using input_tuple_t = std::tuple<input_t, composite::timestamp, std::optional<composite::metadata>>;
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
        }
        if (m_task_queue.thread_name_prefix() != id()) {
            m_task_queue.thread_name_prefix(id());
        }
        m_task_queue.resize(m_num_workers);
    }

    auto start() -> void override {
        m_input_thread = std::jthread([&](std::stop_token stoken) {
            while (!stoken.stop_requested()) {
                // Get data buffer
                auto [data, ts, meta] = m_in_port.get_data();
                if (data == nullptr) {
                    std::this_thread::yield();
                    continue;
                }
                // Submit FFT task to pool
                auto fut = m_task_queue.submit([data = std::move(data), ts, meta = std::move(meta), this]() mutable -> input_tuple_t {
                    // Create a thread_local plan
                    static thread_local std::unique_ptr<plan_t> fft_plan;
                    if (!fft_plan || fft_plan->size() != m_fft_size) {
                        fft_plan = std::make_unique<plan_t>(m_fft_size, m_fftw_threads);
                    }

                    // Apply window
                    if (m_window) {
                        apply_window(data.get(), m_window.get());
                    }

                    // Execute the fft
                    // In-place for complex
                    fft_plan->execute(data.get(), data.get());

                    // Shift based on property
                    if (m_shift) {
                        std::rotate(
                            data->data(),
                            data->data() + (data->size() / 2),
                            data->data() + data->size()
                        );
                    }

                    // Return modified data and original ts/meta
                    return std::make_tuple(std::move(data), ts, std::move(meta));
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

    auto stop() -> void override {
        m_input_thread.request_stop();
        m_cv.notify_all();
        composite::component::stop();
    }

    auto process() -> composite::retval override {
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
            logger()->trace("received metadata:\n{}", meta->to_string());
            meta->annotations["fft_size"] = std::to_string(m_fft_size);
            meta->annotations["fft_window"] = m_window_type;
            logger()->trace("sending updated metadata:\n{}", meta->to_string());
            m_out_port.send_metadata(meta.value());
        }

        // Send data
        m_out_port.send_data(std::move(data), ts);
        return NORMAL;
    }

private:
    [[gnu::target("default")]]
    auto apply_window(fft_t* data, const window_t* window) -> void {
        for (auto i=0u; i< data->size(); ++i) {
            data->at(i) = data->at(i) * window->at(i);
        }
    }

    [[gnu::target("avx512f")]]
    auto apply_window(fft_t* data, const window_t* window) -> void {
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
    auto apply_window(fft_t* data, const window_t* window) -> void {
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
    uint32_t m_num_workers{1};
    bool m_shift{true};

    // Members
    std::unique_ptr<window_t> m_window{nullptr};
    std::deque<std::future<input_tuple_t>> m_futures;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    std::jthread m_input_thread;
    task_queue m_task_queue;

}; // class fft
