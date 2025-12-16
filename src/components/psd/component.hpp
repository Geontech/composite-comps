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

#pragma once

#include "work.hpp"
#include "task_queue.hpp"

#include <composite/composite.hpp>
#include <composite/buffers/buffer.hpp>
#include <composite/buffers/aligned_mem.hpp>
#include <complex>
#include <deque>
#include <future>
#include <memory>

template <typename T>
class psd : public composite::component {
    using input_port_t = composite::input_port<composite::mutable_buffer<std::complex<T>>>;
    using output_port_t = composite::output_port<composite::mutable_buffer<T>>;
    using output_tuple_t = std::tuple<composite::mutable_buffer<T>, composite::timestamp, std::optional<composite::metadata>>;
    using window_t = composite::aligned_mem<T>;

    static constexpr std::size_t ALIGNMENT = 64;

public:
    explicit psd(std::string_view id);
    ~psd() override = default;

    auto property_change_handler() -> void override;
    auto initialize() -> void override;
    auto start() -> void override;
    auto stop() -> void override;
    auto process() -> composite::retval override;

private:
    auto calculate_norm_const() const -> T;

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

// Explicit template instantiations (defined in component.cpp)
extern template class psd<float>;
extern template class psd<double>;
