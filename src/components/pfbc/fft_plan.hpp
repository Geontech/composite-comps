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
        auto lock = std::scoped_lock{fftw_mutex()};
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
        auto lock = std::scoped_lock{fftw_mutex()};
        fftwf_destroy_plan(m_plan);
    }

public:
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

    auto execute_inplace(std::complex<float>* data) -> void {
        fftwf_execute_dft(
            m_plan,
            reinterpret_cast<fftwf_complex*>(data),
            reinterpret_cast<fftwf_complex*>(data)
        );
    }

private:
    static std::mutex& fftw_mutex() {
        static std::mutex mtx;
        return mtx;
    }

    fftwf_plan m_plan;
    std::size_t m_size;

}; // class fft_plan<std::complex<float>>

// =============================================================================
// Batched FFT Plan
// =============================================================================
//
// Uses fftwf_plan_many_dft for efficient batch execution of multiple FFTs.
// Benchmarking shows speedup for small M.
//
// Memory layout: Contiguous rows, each row is one FFT of size M
//   Input:  [row0: M samples][row1: M samples]...[rowN: M samples]
//   Output: Same layout
//

template <typename T>
class fft_batch_plan;

template <>
class fft_batch_plan<std::complex<float>> {
public:
    fft_batch_plan(std::size_t fft_size, std::size_t batch_size, uint32_t fftw_threads = 1) :
      m_fft_size(fft_size),
      m_batch_size(batch_size) {
        auto lock = std::scoped_lock{fftw_mutex()};
        fftwf_plan_with_nthreads(fftw_threads);

        // Allocate planning buffers
        const auto total_samples = fft_size * batch_size;
        auto in_buf = composite::make_aligned<std::complex<float>>(64, total_samples);
        auto out_buf = composite::make_aligned<std::complex<float>>(64, total_samples);

        // Create batched plan using fftwf_plan_many_dft
        // Parameters:
        //   rank=1 (1D FFTs)
        //   n={fft_size} (each FFT is fft_size points)
        //   howmany=batch_size (number of FFTs to execute)
        //   in/out pointers
        //   inembed/onembed=nullptr (use n for embedded dimensions)
        //   istride/ostride=1 (contiguous samples within each FFT)
        //   idist/odist=fft_size (distance between consecutive FFTs)
        int n[] = {static_cast<int>(fft_size)};
        m_plan = fftwf_plan_many_dft(
            1,                                              // rank
            n,                                              // n (dimensions)
            static_cast<int>(batch_size),                   // howmany
            reinterpret_cast<fftwf_complex*>(in_buf->data()),
            nullptr,                                        // inembed
            1,                                              // istride
            static_cast<int>(fft_size),                     // idist
            reinterpret_cast<fftwf_complex*>(out_buf->data()),
            nullptr,                                        // onembed
            1,                                              // ostride
            static_cast<int>(fft_size),                     // odist
            FFTW_FORWARD,
            FFTW_MEASURE
        );
    }

    ~fft_batch_plan() {
        auto lock = std::scoped_lock{fftw_mutex()};
        fftwf_destroy_plan(m_plan);
    }

    // Non-copyable, non-movable (plan is tied to specific memory layout)
    fft_batch_plan(const fft_batch_plan&) = delete;
    fft_batch_plan& operator=(const fft_batch_plan&) = delete;
    fft_batch_plan(fft_batch_plan&&) = delete;
    fft_batch_plan& operator=(fft_batch_plan&&) = delete;

public:
    auto plan() -> fftwf_plan {
        return m_plan;
    }

    auto fft_size() const noexcept -> std::size_t {
        return m_fft_size;
    }

    auto batch_size() const noexcept -> std::size_t {
        return m_batch_size;
    }

    // Execute all FFTs in the batch
    // Input/output must have at least fft_size * batch_size samples
    auto execute(std::complex<float>* in, std::complex<float>* out) -> void {
        fftwf_execute_dft(
            m_plan,
            reinterpret_cast<fftwf_complex*>(in),
            reinterpret_cast<fftwf_complex*>(out)
        );
    }

    // In-place execution (input is overwritten with output)
    auto execute_inplace(std::complex<float>* data) -> void {
        fftwf_execute_dft(
            m_plan,
            reinterpret_cast<fftwf_complex*>(data),
            reinterpret_cast<fftwf_complex*>(data)
        );
    }

private:
    static std::mutex& fftw_mutex() {
        static std::mutex mtx;
        return mtx;
    }

    fftwf_plan m_plan;
    std::size_t m_fft_size;
    std::size_t m_batch_size;

}; // class fft_batch_plan<std::complex<float>>
