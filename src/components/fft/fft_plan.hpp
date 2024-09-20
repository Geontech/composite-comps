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

#pragma once

#include "aligned_mem.hpp"
#include "windows.hpp"

#include <algorithm>
#include <complex>
#include <fftw3.h>

template <typename T>
auto shift(aligned::aligned_mem<T>* out) {
    std::rotate(
        out->data(),
        out->data() + (out->size() / 2),
        out->data() + out->size()
    );
}

template <typename T, bool complex>
class fft_plan{};

template <>
class fft_plan<float, true> {
public:
    fft_plan(uint32_t fft_size, uint32_t fftw_threads, bool do_shift) : m_shift(do_shift) {
        fftwf_init_threads();
        fftwf_plan_with_nthreads(fftw_threads);
        auto plan_buf = aligned::make_aligned<std::complex<float>>(64, fft_size);
        m_plan = fftwf_plan_dft_1d(
            fft_size,
            reinterpret_cast<fftwf_complex*>(plan_buf->data()),
            reinterpret_cast<fftwf_complex*>(plan_buf->data()),
            FFTW_FORWARD,
            FFTW_MEASURE
        );
    }

    ~fft_plan() {
        fftwf_cleanup_threads();
        fftwf_destroy_plan(m_plan);
    }

    auto plan() -> fftwf_plan {
        return m_plan;
    }

    auto execute(aligned::aligned_mem<std::complex<float>>* in, aligned::aligned_mem<std::complex<float>>* out) -> void {
        fftwf_execute_dft(
            m_plan,
            reinterpret_cast<fftwf_complex*>(in->data()),
            reinterpret_cast<fftwf_complex*>(out->data())
        );
        // Shift
        if (m_shift) {
            shift(out);
        }
    }

private:
    fftwf_plan m_plan;
    bool m_shift{false};

}; // class fft_plan<float, true>

template <>
class fft_plan<float, false> {
public:
    fft_plan(uint32_t fft_size, uint32_t fftw_threads, bool do_shift) : m_shift(do_shift) {
        fftwf_init_threads();
        fftwf_plan_with_nthreads(fftw_threads);
        auto in_buf = aligned::make_aligned<float>(64, fft_size);
        auto out_buf = aligned::make_aligned<std::complex<float>>(64, fft_size);
        m_plan = fftwf_plan_dft_r2c_1d(
            fft_size,
            in_buf->data(),
            reinterpret_cast<fftwf_complex*>(out_buf->data()),
            FFTW_MEASURE
        );
    }

    ~fft_plan() {
        fftwf_cleanup_threads();
        fftwf_destroy_plan(m_plan);
    }

    auto plan() -> fftwf_plan {
        return m_plan;
    }

    auto execute(aligned::aligned_mem<float>* in, aligned::aligned_mem<std::complex<float>>* out) -> void {
        fftwf_execute_dft_r2c(
            m_plan,
            in->data(),
            reinterpret_cast<fftwf_complex*>(out->data())
        );
        // Shift
        if (m_shift) {
            shift(out);
        }
    }


private:
    fftwf_plan m_plan;
    bool m_shift{false};

}; // class fft_plan<float, false>

template <>
class fft_plan<double, true> {
public:
    fft_plan(uint32_t fft_size, uint32_t fftw_threads, bool do_shift) : m_shift(do_shift) {
        fftw_init_threads();
        fftw_plan_with_nthreads(fftw_threads);
        auto plan_buf = aligned::make_aligned<std::complex<double>>(64, fft_size);
        m_plan = fftw_plan_dft_1d(
            fft_size,
            reinterpret_cast<fftw_complex*>(plan_buf->data()),
            reinterpret_cast<fftw_complex*>(plan_buf->data()),
            FFTW_FORWARD,
            FFTW_MEASURE
        );
    }

    ~fft_plan() {
        fftw_cleanup_threads();
        fftw_destroy_plan(m_plan);
    }

    auto plan() -> fftw_plan {
        return m_plan;
    }

    auto execute(aligned::aligned_mem<std::complex<double>>* in, aligned::aligned_mem<std::complex<double>>* out) -> void {
        fftw_execute_dft(
            m_plan,
            reinterpret_cast<fftw_complex*>(in->data()),
            reinterpret_cast<fftw_complex*>(out->data())
        );
        // Shift
        if (m_shift) {
            shift(out);
        }
    }

private:
    fftw_plan m_plan;
    bool m_shift{false};

}; // class fft_plan<double, true>

template <>
class fft_plan<double, false> {
public:
    fft_plan(uint32_t fft_size, uint32_t fftw_threads, bool do_shift) : m_shift(do_shift) {
        fftw_init_threads();
        fftw_plan_with_nthreads(fftw_threads);
        auto in_buf = aligned::make_aligned<double>(64, fft_size);
        auto out_buf = aligned::make_aligned<std::complex<double>>(64, fft_size);
        m_plan = fftw_plan_dft_r2c_1d(
            fft_size,
            in_buf->data(),
            reinterpret_cast<fftw_complex*>(out_buf->data()),
            FFTW_MEASURE
        );
    }

    ~fft_plan() {
        fftw_cleanup_threads();
        fftw_destroy_plan(m_plan);
    }

    auto plan() -> fftw_plan {
        return m_plan;
    }

    auto execute(aligned::aligned_mem<double>* in, aligned::aligned_mem<std::complex<double>>* out) -> void {
        fftw_execute_dft_r2c(
            m_plan,
            in->data(),
            reinterpret_cast<fftw_complex*>(out->data())
        );
        // Shift
        if (m_shift) {
            shift(out);
        }
    }

private:
    fftw_plan m_plan;
    bool m_shift{false};

}; // class fft_plan<double, false>
