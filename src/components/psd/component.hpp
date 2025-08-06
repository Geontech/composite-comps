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

#include "work.hpp"

#include <aligned_mem.hpp>
#include <task_queue.hpp>
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
    using input_t = std::unique_ptr<fft_t>;
    using input_port_t = composite::input_port<input_t>;
    using output_t = std::unique_ptr<psd_t>;
    using output_port_t = composite::output_port<output_t>;
    using output_tuple_t = std::tuple<output_t, composite::timestamp, std::optional<composite::metadata>>;
public:
    psd();
    ~psd() override = default;

    auto property_change_handler() -> void override;
    auto initialize() -> void override;
    auto start() -> void override;
    auto stop() -> void override;
    auto process() -> composite::retval override;

private:
    auto calculate_norm_const() const -> T;
    auto create_worker(T norm_const) -> void;

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    uint32_t m_num_workers{1};
    bool m_power_based_normalization{true};

    // Members
    composite::metadata m_metadata;
    std::unique_ptr<window_t> m_window;
    work<T> m_work;
    T m_sample_rate{};
    std::deque<std::future<output_tuple_t>> m_futures;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    std::jthread m_input_thread;
    task_queue m_task_queue;

}; // class psd
