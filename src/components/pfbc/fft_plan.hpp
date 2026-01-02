// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2024-2025 Geon Technologies, LLC

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
    fft_plan(uint32_t fft_size, uint32_t fftw_threads = 1) : m_size(fft_size) {
        static std::mutex plan_mtx;
        auto lock = std::scoped_lock{plan_mtx};
        fftwf_plan_with_nthreads(fftw_threads);

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

    auto plan() -> fftwf_plan {
        return m_plan;
    }

    auto size() const noexcept -> std::size_t {
        return m_size;
    }

    auto execute(std::complex<float>* in, std::complex<float>* out) -> void {
        fftwf_execute_dft(
            m_plan,
            reinterpret_cast<fftwf_complex*>(in),
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

    auto plan() -> fftw_plan {
        return m_plan;
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
