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

#include <bit>
#include <composite/component.hpp>
#include <complex>
#include <fftw3.h>
#include <immintrin.h>
#include <memory>
#include <vector>


template <typename T>
class mt_fft : public composite::component {
    using plan_t = fft_plan<T, true>;
    using fft_t = aligned::aligned_mem<std::complex<T>>;
    using input_port_t = composite::input_port<std::unique_ptr<fft_t>>;
    using output_port_t = composite::output_port<std::unique_ptr<fft_t>>;
    using enum composite::properties::config_type;

public:
    mt_fft() : composite::component("mt_fft") {
        add_port(&m_in_port);
        add_port(&m_out_port);

        add_property("fft_size", &m_fft_size).configurability(RUNTIME).change_listener([this]() {
            return std::has_single_bit(m_fft_size);
        });
        add_property("fftw_threads", &m_fftw_threads);
        add_property("shift", &m_shift).configurability(RUNTIME);
        add_property("fft_workers", &m_fft_workers);
    }

    ~mt_fft() override {
        running = false;
        for (auto& thread : m_threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    auto property_change_handler() -> void override {

    }

    void start() override {
        running = true;
        for (auto& ts : m_thread_timestamps) {
            ts = make_max_timestamp();
        }
    
        for (int i = 0; i < m_fft_workers; ++i) {
            m_threads.emplace_back([this, i]() {
                std::string name = "fft_worker_" + std::to_string(i);
                process_loop(i, name);
            });
        }
        component::start();
    }

private:
    auto process() -> composite::retval override {
        std::this_thread::sleep_for(std::chrono::seconds(1000));
        return composite::retval::NORMAL;
    }
    
    void process_loop(int thread_id, const std::string& name) {
        pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
        logger()->trace("Thread started: {}", name);
    
        thread_local std::unique_ptr<plan_t> thread_fft_plan;
        if (!thread_fft_plan) {
            thread_fft_plan = std::make_unique<plan_t>(m_fft_size, m_fftw_threads, m_shift);
        }
    
        while (running) {
            process_with_plan(thread_fft_plan.get(), thread_id, m_thread_timestamps);
        }
    }

    auto process_with_plan(plan_t* plan, int thread_id, std::vector<composite::timestamp>& thread_ts) -> composite::retval {
        using enum composite::retval;
        auto [data, ts, meta] = m_in_port.get_data();
        if (!data) {
            return NORMAL;
        }
        if (meta.has_value()) {
            logger()->trace("received metadata:\n{}", meta->to_string());
            meta->annotations["fft_size"] = std::to_string(m_fft_size);
            logger()->trace("sending updated metadata:\n{}", meta->to_string());
            m_out_port.send_metadata(meta.value());
        }

        {
            std::scoped_lock lock(m_ts_mutex);
            thread_ts[thread_id] = ts;
        }
        auto output = aligned::make_aligned<std::complex<T>>(64, m_fft_size);
        plan->execute(data.get(), output.get());

        while (running) {
            bool is_earliest = true;
            {
                std::scoped_lock lock(m_ts_mutex);
                for (int i = 0; i < thread_ts.size(); ++i) {
                    if (i == thread_id) continue;
                    if (timestamp_less_than(thread_ts[i], ts)) {
                        is_earliest = false;
                        break;
                    }
                }
            }
            if (is_earliest) break;
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        m_out_port.send_data(std::move(data), ts);
        {
            std::scoped_lock lock(m_ts_mutex);
            thread_ts[thread_id] = make_max_timestamp();
        }
        return NORMAL;
        }


    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};


    // Properties
    uint32_t m_fft_size{1024};
    uint32_t m_fftw_threads{1};
    uint32_t m_fft_workers{2};
    bool m_shift{true};

    // Threading
    std::vector<std::thread> m_threads;
    std::atomic<bool> running{false};
    static std::mutex fftw_plan_mutex;
    std::mutex m_ts_mutex;

    std::vector<composite::timestamp> m_thread_timestamps{m_fft_workers};
    inline composite::timestamp make_max_timestamp() {
        return composite::timestamp{
            std::numeric_limits<uint32_t>::max(),
            std::numeric_limits<uint64_t>::max()
        };
    }
    inline bool timestamp_less_than(const composite::timestamp& lhs, const composite::timestamp& rhs) {
        return (lhs.seconds < rhs.seconds) ||
               (lhs.seconds == rhs.seconds && lhs.picoseconds < rhs.picoseconds);
    }
    bool is_later(const composite::timestamp& a, const composite::timestamp& b) {
        return (a.seconds > b.seconds) ||
            (a.seconds == b.seconds && a.picoseconds > b.picoseconds);
    }
    struct FFTWfPlanDeleter {
        void operator()(fftwf_plan plan) const {
            if (plan) fftwf_destroy_plan(plan);
        }
    };
};
