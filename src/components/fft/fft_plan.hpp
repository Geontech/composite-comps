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
#include <mutex>

template <typename T>
class fft_plan;

template <>
class fft_plan<std::complex<float>> {
public:
    fft_plan(uint32_t fft_size, uint32_t fftw_threads) : m_size(fft_size) {
        static std::mutex plan_mtx;
        auto lock = std::scoped_lock{plan_mtx};
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
    }

    ~fft_plan() {
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
    fftwf_plan m_plan;
    std::size_t m_size;

}; // class fft_plan<std::complex<float>>

template <>
class fft_plan<std::complex<double>> {
public:
    fft_plan(uint32_t fft_size, uint32_t fftw_threads) : m_size(fft_size) {
        static std::mutex plan_mtx;
        auto lock = std::scoped_lock{plan_mtx};
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
    }

    ~fft_plan() {
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
    fftw_plan m_plan;
    std::size_t m_size;

}; // class fft_plan<std::complex<double>>
