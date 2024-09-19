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
#include "apply_window.hpp"
#include "avx_helpers.hpp"
#include "fft_helpers.hpp"
#include "overlay.hpp"
#include "windows.hpp"

#include <composite/component.hpp>
#include <complex>
#include <fftw3.h>
#include <memory>
#include <vector>

template <typename T>
class fft : public composite::component {
    using plan_t = fft_plan<T, true>;
    using work_t = fft_work<T, true>;
    using fft_t = aligned::aligned_mem<std::complex<T>>;
    using window_t = aligned::aligned_mem<T>;
    using input_port_t = composite::input_port<std::vector<std::vector<uint8_t>>>;
    using output_port_t = composite::output_port<fft_t>;
    using timestamp_type = typename output_port_t::timestamp_type;
public:
    fft() : composite::component("fft") {
        add_port(m_in_port.get());
        add_port(m_out_port.get());
        add_property("window", &m_window_type);
        add_property("fft_size", &m_fft_size);
        add_property("fftw_threads", &m_fftw_threads);
        add_property("shift", &m_shift);
    }

    ~fft() override = default;

    auto initialize() -> void override {
        // Init window
        if (m_window_type == "BLACKMAN_HARRIS") {
            m_window = windows::blackman_harris<T>(m_fft_size);
        } else if (m_window_type == "HAMMING") {
            m_window = windows::hamming<T>(m_fft_size);
        }
        // Init fftw
        m_fft_plan = std::make_unique<plan_t>(m_fft_size, m_fftw_threads);
        m_fft_work = std::make_unique<work_t>(m_shift);
    }

    auto start() -> void override {
        composite::component::start();
        m_framer = std::jthread(&fft::frame, this);
    }

    auto stop() -> void override {
        m_framer.request_stop();
        if (m_framer.joinable()) {
            m_framer.join();
        }
        composite::component::stop();
    }

    auto process() -> composite::retval override {
        using enum composite::retval;
        using namespace std::chrono_literals;
        auto lock = std::unique_lock{m_data_mtx};
        m_data_cv.wait_for(lock, 1s, [this]{ return !m_queue.empty(); });
        if (m_queue.empty()) {
            return composite::retval::NORMAL;
        }
        auto [buffer, ts] = std::move(m_queue.front());
        m_queue.pop_front();
        lock.unlock();
        if (buffer == nullptr) {
            return composite::retval::NORMAL;
        }
        // Execute the fft
        // In-place for complex
        m_fft_work->execute(m_fft_plan->plan, buffer.get(), buffer.get());
        // Send data
        m_out_port->send_data(std::move(buffer), std::chrono::system_clock::now());
        return composite::retval::NORMAL;
    }

private:
    // Ports
    std::unique_ptr<input_port_t> m_in_port{std::make_unique<input_port_t>("data_in")};
    std::unique_ptr<output_port_t> m_out_port{std::make_unique<output_port_t>("data_out")};

    // Properties
    std::string m_window_type;
    uint32_t m_fft_size{1024};
    uint32_t m_fftw_threads{1};
    bool m_shift{true};

    // Members
    std::unique_ptr<fft_plan<T, true>> m_fft_plan;
    std::unique_ptr<fft_work<T, true>> m_fft_work;
    std::unique_ptr<window_t> m_window;
    std::jthread m_framer;
    std::deque<std::tuple<std::unique_ptr<fft_t>, timestamp_type>> m_queue;
    std::mutex m_data_mtx;
    std::condition_variable m_data_cv;
    std::size_t m_frame_idx{};
    std::unique_ptr<fft_t> m_frame_buf;
    struct timespec m_frame_ts;

    auto frame(std::stop_token token) -> void {
        while (!token.stop_requested()) {
            auto [data, ts] = m_in_port->get_data();
            if (data == nullptr) {
                std::this_thread::yield();
                continue;
            }
            for (auto& v : *data) {
                auto packet = overlay::v49::overlay(v);
                if (auto& header = packet.header(); !overlay::v49::is_data(header)) {
                    continue;
                }
                auto payload = packet.payload<std::complex<int16_t>>();
                auto i=0u;
                auto stride = 32u / sizeof(T);
                for (; i < payload.size(); i += stride) {
                    if (m_frame_buf == nullptr) {
                        m_frame_buf = aligned::make_aligned<typename fft_t::value_type>(64, m_fft_size);
                        // TODO: timestamps
                    }
                    apply_window(
                        reinterpret_cast<const int16_t*>(payload.data() + i),
                        m_window == nullptr ? nullptr : m_window->data() + m_frame_idx * 2,
                        reinterpret_cast<T*>(m_frame_buf->data() + m_frame_idx)
                    );
                    m_frame_idx += stride;
                    if (m_frame_idx == m_fft_size) {
                        const auto lock = std::scoped_lock{m_data_mtx};
                        // TODO: timestamp
                        m_queue.emplace_back(std::move(m_frame_buf), std::chrono::system_clock::now());
                        m_frame_buf.reset();
                        m_data_cv.notify_one();
                        m_frame_idx = 0;
                        continue;
                    }
                }
            }
        }
    }

}; // class fft
