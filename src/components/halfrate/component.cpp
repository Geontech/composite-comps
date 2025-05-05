/*
 * Copyright (C) 2025 Geon Technologies, LLC
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

#include "component.par.hpp"
#include "dot_prod.hpp"
#include <windows.hpp>

#include <bit>
#include <format>
#include <immintrin.h>
#include <memory>
#include <numbers>

halfrate::halfrate() : composite::component("halfrate") {
    add_port(&m_in_port);
    add_port(&m_out_port);
    add_property("num_threads", &m_num_threads).change_listener([this]() {
        return m_num_threads >= 1 && m_num_threads <= 8; 
    });
    add_property("filter_semi_length", &m_filter_semi_length);
    add_property("window", &m_window_type).change_listener([this]() {
        return (m_window_type == "BLACKMAN_HARRIS") || (m_window_type == "HAMMING");
    });
}

halfrate::~halfrate() {
    // Notify all threads to stop working
    for (auto& thread : m_threads) {
        thread.request_stop();
    }
    for (auto& worker : m_workers) {
        worker->cv.notify_one();
    }
}

auto halfrate::property_change_handler() -> void {
    m_workers.reserve(m_num_threads);
    for (auto i = 0; i < m_num_threads; ++i) {
        m_workers.emplace_back(std::make_unique<worker_context>());
        m_workers[i]->w0 = window_buffer<std::complex<float>>(2 * m_filter_semi_length);
        m_workers[i]->w1 = window_buffer<std::complex<float>>(2 * m_filter_semi_length);
        if (m_num_threads > 1) {
            m_threads.emplace_back(&halfrate::worker_thread_func, this, std::ref(m_workers[i]));
            pthread_setname_np(m_threads[i].native_handle(), std::format("{}:w{}", id(), i).c_str());
        }
    }
    generate_coeffs();
}

auto halfrate::process() -> composite::retval {
    using enum composite::retval;
    auto [data, ts] = m_in_port.get_data();
    if (data == nullptr) {
        return NORMAL;
    }

    if (m_num_threads == 1) {
        auto& work_ctx = m_workers.front();
        auto& w0 = work_ctx->w0;
        auto& w1 = work_ctx->w1;

        // Iterate over samples to produce output
        for (auto i = std::size_t{}; i < data->size(); i += 2) {
            // Add samples to window buffers
            w0.push(data->at(i));
            w1.push(data->at(i + 1));

            // Calculate value with center tap
            auto y0 = m_center_tap * w1[m_filter_semi_length - 1];

            // Compute dot-product with filter taps
            auto y1 = dotprod::process(m_coeffs, w0);

            // Add result to output vector
            data->at(i / 2) = y0 + y1;
        }

        // Send data
        data->resize(data->size() / 2);
        m_out_port.send_data(std::move(data), ts);
    } else {
        const auto total_size = data->size();
        const auto chunk_size = total_size / m_num_threads;

        auto done_latch = std::latch{m_num_threads};
        auto worker_idx = std::size_t{};

        auto output = aligned::make_aligned<std::complex<float>>(64, data->size() / 2);

        // Initialize worker thread context
        for (auto& worker : m_workers) {
            auto lock = std::scoped_lock{worker->mtx};

            // Calculate data range for this worker
            const auto start = worker_idx * chunk_size;
            const auto end = (worker_idx == m_num_threads - 1) ? total_size : (worker_idx + 1) * chunk_size;

            // Pre-fill window buffers for threads 1 to N-1
            // worker->w0.reset();
            // worker->w1.reset();
            if (worker_idx != 0) {
                const auto pre_start = (start >= 4 * m_filter_semi_length)
                    ? start - 4 * m_filter_semi_length
                    : 0;

                for (auto i = pre_start; i < start; i += 2) {
                    worker->w0.push(data->at(i));
                    worker->w1.push(data->at(i + 1));
                }
            }

            // Initialize data fields on worker
            worker->input = {data->data() + start, end - start};
            worker->output = {output->data() + (start / 2), ((end - start) / 2)};
            worker->done_latch = &done_latch;

            worker_idx++;
        }

        // Notify all threads to start working
        for (auto& worker : m_workers) {
            worker->data_ready = true;
            worker->cv.notify_one();
        }

        // Wait for all threads to complete their work
        done_latch.wait();

        // Preserve tail samples for thread 0 (first thread) for next call
        {
            auto& worker = m_workers[0];
            auto lock = std::scoped_lock{worker->mtx};

            // Grab the last 2 * filter_length samples from data
            const auto preserved_start = (total_size >= 4 * m_filter_semi_length)
                ? total_size - 4 * m_filter_semi_length
                : 0;

            // worker->w0.reset();
            // worker->w1.reset();
            for (auto i = preserved_start; i < total_size; i += 2) {
                worker->w0.push(data->at(i));
                worker->w1.push(data->at(i + 1));
            }
        }

        // Send data
        m_out_port.send_data(std::move(output), ts);
    }

    return NORMAL;
}

auto halfrate::generate_coeffs() -> void {
    auto L = 4 * m_filter_semi_length + 1; // full length
    auto center = L / 2;
    auto h = std::vector<float>(L);

    auto window = windows::hamming<float>(L, false);

    for (int n = 0; n < L; ++n) {
        int k = n - center;
        if (k == 0) {
            h[n] = 0.5;  // half-band filter: center tap is always 0.5
            m_center_tap = h[n] * 2.0f;
        } else if (k % 2 == 0) {
            auto sinc_val = std::sin(0.5 * std::numbers::pi * k) / (std::numbers::pi * k);
            h[n] = sinc_val * window->at(n);
        } else {
            h[n] = 0.0; // zero out odd taps (except center)
        }
    }

    m_coeffs.clear();
    for (int i = h.size() - 1; i >= 0; --i) {
        if (i == h.size() / 2) { continue; } // skip center tap
        if (i % 2 == 1) { // use only even indices
            m_coeffs.push_back(h[i] * 2.0f);
        }
    }

    // m_coeffs = {-0.00288903,0.00745904,-0.0163088,0.0313995,-0.0564697,0.100349,-0.194997,0.630703,0.630703,-0.194997,0.100349,-0.0564697,0.0313995,-0.0163088,0.00745904,-0.00288903};
    // m_coeffs = {-0.00171411,0.00307656,-0.00567513,0.00956402,-0.0151604,0.023042,-0.0340987,0.0499172,-0.0738422,0.114591,-0.204352,0.633965,0.633965,-0.204352,0.114591,-0.0738422,0.0499172,-0.0340987,0.023042,-0.0151604,0.00956402,-0.00567513,0.00307656,-0.00171411};
    // m_coeffs = {-0.000965003,0.000899301,-0.00130554,0.00182206,-0.00246723,0.00326131,-0.00422741,0.00539153,-0.0067835,0.00843858,-0.0103997,0.0127202,-0.0154706,0.0187447,-0.0226746,0.027453,-0.0333755,0.0409215,-0.0509201,0.0649463,-0.0863793,0.124028,-0.210215,0.635954,0.635954,-0.210215,0.124028,-0.0863793,0.0649463,-0.0509201,0.0409215,-0.0333755,0.027453,-0.0226746,0.0187447,-0.0154706,0.0127202,-0.0103997,0.00843858,-0.0067835,0.00539153,-0.00422741,0.00326131,-0.00246723,0.00182206,-0.00130554,0.000899301,-0.000965003};
    // 16
    m_coeffs = {-0.00129471,0.00177744,-0.00294215,0.00455943,-0.00673839,0.00961138,-0.0133425,0.0181527,-0.0243542,0.0324346,-0.0432212,0.0582726,-0.080946,0.120012,-0.20775,0.635122,0.635122,-0.20775,0.120012,-0.080946,0.0582726,-0.0432212,0.0324346,-0.0243542,0.0181527,-0.0133425,0.00961138,-0.00673839,0.00455943,-0.00294215,0.00177744,-0.00129471};
    
    // 24
    m_coeffs = {-0.000965003,0.000899301,-0.00130554,0.00182206,-0.00246723,0.00326131,-0.00422741,0.00539153,-0.0067835,0.00843858,-0.0103997,0.0127202,-0.0154706,0.0187447,-0.0226746,0.027453,-0.0333755,0.0409215,-0.0509201,0.0649463,-0.0863793,0.124028,-0.210215,0.635954,0.635954,-0.210215,0.124028,-0.0863793,0.0649463,-0.0509201,0.0409215,-0.0333755,0.027453,-0.0226746,0.0187447,-0.0154706,0.0127202,-0.0103997,0.00843858,-0.0067835,0.00539153,-0.00422741,0.00326131,-0.00246723,0.00182206,-0.00130554,0.000899301,-0.000965003};
    m_center_tap = 1.0f;
}

auto halfrate::worker_thread_func(std::stop_token token, std::unique_ptr<halfrate::worker_context>& ctx) -> void {
    while (!token.stop_requested()) {
        auto lock = std::unique_lock{ctx->mtx};
        ctx->cv.wait(lock, [&ctx, &token]{ return ctx->data_ready || token.stop_requested(); });

        if (token.stop_requested()) {
            return;
        }

        for (auto i = 0u; i < ctx->input.size(); i += 2) {
            ctx->w0.push(ctx->input[i]);
            ctx->w1.push(ctx->input[i + 1]);

            auto y0 = m_center_tap * ctx->w1[m_filter_semi_length - 1];
            auto y1 = dotprod::process(m_coeffs, ctx->w0);

            ctx->output[i / 2] = y0 + y1;
        }

        ctx->data_ready = false;
        if (ctx->done_latch) {
            ctx->done_latch->count_down();
        }
    }
}

extern "C" {
    auto create() -> std::shared_ptr<composite::component> {
        return std::make_shared<halfrate>();
    }
}
