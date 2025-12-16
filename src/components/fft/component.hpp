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

#include "fft_plan.hpp"
#include "task_queue.hpp"

#include <composite/composite.hpp>
#include <composite/buffers/buffer.hpp>
#include <composite/buffers/aligned_mem.hpp>
#include <complex>
#include <deque>
#include <future>
#include <memory>

// T is expected to be std::complex<float> or std::complex<double>
template <typename T>
class fft : public composite::component {
    using scalar_t = typename T::value_type;  // float or double
    using plan_t = fft_plan<T>;
    using window_t = composite::aligned_mem<scalar_t>;
    using input_port_t = composite::input_port<composite::immutable_buffer<T>>;
    using output_port_t = composite::output_port<composite::immutable_buffer<T>>;
    using output_tuple_t = std::tuple<composite::immutable_buffer<T>, composite::timestamp, std::optional<composite::metadata>>;

    static constexpr std::size_t ALIGNMENT = 64;

public:
    explicit fft(std::string_view id);
    ~fft() override;

    auto property_change_handler() -> void override;
    auto start() -> void override;
    auto stop() -> void override;
    auto process() -> composite::retval override;

private:
    // Fused copy + window with SIMD variants
    [[gnu::target("default")]]
    auto copy_and_window(
      const composite::immutable_buffer<T>& input,
      composite::mutable_buffer<T>& output,
      const window_t* window
    ) -> void;

    [[gnu::target("avx512f")]]
    auto copy_and_window(
      const composite::immutable_buffer<T>& input,
      composite::mutable_buffer<T>& output,
      const window_t* window
    ) -> void;

    [[gnu::target("avx2")]]
    auto copy_and_window(
      const composite::immutable_buffer<T>& input,
      composite::mutable_buffer<T>& output,
      const window_t* window
    ) -> void;

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
