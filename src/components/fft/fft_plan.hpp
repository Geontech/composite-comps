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

#pragma once

#include <composite/buffers/aligned_mem.hpp>

#include <complex>
#include <fftw3.h>
#include <format>
#include <mutex>
#include <stdexcept>

template <typename T>
class fft_plan;

// FFTW's planner is NOT thread-safe: per its documentation, fftw_execute (and the new-array
// variants) is the ONLY routine safe to call concurrently — plan creation AND destruction
// must be serialized. Each precision (fftwf_/fftw_) has its own independent planner, so the
// mutex is per specialization, and it must guard ~fft_plan too: pool workers rebuild their
// thread_local plans concurrently on an fft_size change (make_unique constructs the new plan,
// then assignment destroys the old one), and destroy racing create corrupts planner state.

template <>
class fft_plan<std::complex<float>> {
public:
    fft_plan(uint32_t fft_size, uint32_t fftw_threads) : m_size(fft_size) {
        auto lock = std::scoped_lock{planner_mutex()};
        fftwf_plan_with_nthreads(fftw_threads);
        // Create separate buffers for out-of-place planning
        auto in_buf = composite::make_aligned<std::complex<float>>(64, fft_size);
        auto out_buf = composite::make_aligned<std::complex<float>>(64, fft_size);
        m_plan = fftwf_plan_dft_1d(
            fft_size,
            reinterpret_cast<fftwf_complex*>(in_buf->data()),
            reinterpret_cast<fftwf_complex*>(out_buf->data()),
            FFTW_FORWARD,
            FFTW_MEASURE
        );
        // The planner returns null on failure (resource exhaustion at large sizes); executing
        // a null plan is a crash. Throw instead — the caller's work() exception path logs and
        // drops the packet, and retries the plan build on the next one.
        if (m_plan == nullptr) {
            throw std::runtime_error(
                std::format("fftwf_plan_dft_1d failed for size {}", fft_size));
        }
    }

    ~fft_plan() {
        auto lock = std::scoped_lock{planner_mutex()};
        fftwf_destroy_plan(m_plan);
    }

    auto size() const noexcept -> std::size_t {
        return m_size;
    }

    auto execute(const std::complex<float>* in, std::complex<float>* out) -> void {
        fftwf_execute_dft(
            m_plan,
            const_cast<fftwf_complex*>(reinterpret_cast<const fftwf_complex*>(in)),
            reinterpret_cast<fftwf_complex*>(out)
        );
    }

private:
    static auto planner_mutex() -> std::mutex& {
        static std::mutex mtx;
        return mtx;
    }

    fftwf_plan m_plan;
    std::size_t m_size;

}; // class fft_plan<std::complex<float>>

template <>
class fft_plan<std::complex<double>> {
public:
    fft_plan(uint32_t fft_size, uint32_t fftw_threads) : m_size(fft_size) {
        auto lock = std::scoped_lock{planner_mutex()};
        fftw_plan_with_nthreads(fftw_threads);
        // Create separate buffers for out-of-place planning
        auto in_buf = composite::make_aligned<std::complex<double>>(64, fft_size);
        auto out_buf = composite::make_aligned<std::complex<double>>(64, fft_size);
        m_plan = fftw_plan_dft_1d(
            fft_size,
            reinterpret_cast<fftw_complex*>(in_buf->data()),
            reinterpret_cast<fftw_complex*>(out_buf->data()),
            FFTW_FORWARD,
            FFTW_MEASURE
        );
        // See the single-precision specialization: a null plan must throw, not execute.
        if (m_plan == nullptr) {
            throw std::runtime_error(
                std::format("fftw_plan_dft_1d failed for size {}", fft_size));
        }
    }

    ~fft_plan() {
        auto lock = std::scoped_lock{planner_mutex()};
        fftw_destroy_plan(m_plan);
    }

    auto size() const noexcept -> std::size_t {
        return m_size;
    }

    auto execute(const std::complex<double>* in, std::complex<double>* out) -> void {
        fftw_execute_dft(
            m_plan,
            const_cast<fftw_complex*>(reinterpret_cast<const fftw_complex*>(in)),
            reinterpret_cast<fftw_complex*>(out)
        );
    }

private:
    static auto planner_mutex() -> std::mutex& {
        static std::mutex mtx;
        return mtx;
    }

    fftw_plan m_plan;
    std::size_t m_size;

}; // class fft_plan<std::complex<double>>
