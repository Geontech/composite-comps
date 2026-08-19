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

#include "component.hpp"
#include "windows.hpp"

#include <composite/core/register.hpp>

#include <bit>
#include <cstdint>
#include <fftw3.h>
#include <format>
#include <immintrin.h>
#include <pthread.h>
#include "simd_fmv.hpp"

template <typename T>
fft<T>::fft(std::string_view id)
    : base(id, "data_in", "data_out", /*default_workers=*/1) {
    using enum composite::properties::config_type;

    // Ports + num_workers are provided by pipeline_component; register fft's own config here.
    this->add_property("window", m_window_type).validate([](const std::string& v) {
        return (v == "BLACKMAN_HARRIS") || (v == "HAMMING") || v.empty();
    });
    this->add_property("fft_size", m_fft_size, RUNTIME).validate([](const std::uint32_t& v) {
        return std::has_single_bit(v);
    });
    this->add_property("fftw_threads", m_fftw_threads);
    this->add_property("shift", m_shift, RUNTIME);

    if constexpr (std::is_same_v<scalar_t, float>) {
        fftwf_init_threads();
    } else {
        fftw_init_threads();
    }

    // Drop / plan-build counters in the shared registry so operators can see rejected frames and
    // FFTW plan churn (labeled by component id; auto-removed by ~component).
    m_frames_dropped = &this->create_counter("fft.frames_dropped",
                                             "Input frames dropped for an fft_size mismatch");
    m_plan_builds = &this->create_counter("fft.plan_builds", "FFTW plan (re)creations across the worker pool");

    // Publish an initial snapshot so a pool worker always loads a valid config even if no
    // properties are set before start() (property_change_handler republishes on every change).
    m_task_cfg.publish(make_task_config());
}

template <typename T>
fft<T>::~fft() {
    // Stop the pipeline (main worker + pool) BEFORE releasing FFTW's global thread state: the dtor
    // body runs before members, so a pool worker could otherwise still be in work() using a plan
    // when fftw_cleanup_threads() frees FFTW's threading bookkeeping.
    this->stop();
    if constexpr (std::is_same_v<scalar_t, float>) {
        fftwf_cleanup_threads();
    } else {
        fftw_cleanup_threads();
    }
}

template <typename T>
auto fft<T>::property_change_handler(const composite::properties::json& diff) -> void {
    (void)diff;
    // Publish a fresh immutable snapshot (incl. a newly-allocated window). Pool workers already in
    // flight keep their previously-loaded snapshot alive, so the old window is freed only once the
    // last worker using it finishes — no use-after-free even though the pool is not parked. (A
    // num_workers change is applied separately by pipeline_component.)
    m_task_cfg.publish(make_task_config());
    // prepare() stamps fft_size/fft_window from this snapshot onto the shared metadata; tell
    // the pipeline to rebuild it even though the incoming metadata instance is unchanged.
    this->invalidate_prepared_metadata();
}

template <typename T>
auto fft<T>::make_task_config() const -> std::shared_ptr<const task_config> {
    auto cfg = std::make_shared<task_config>();
    cfg->fft_size = m_fft_size;
    cfg->shift = m_shift;
    cfg->fftw_threads = m_fftw_threads;
    cfg->window_type = m_window_type;
    if (m_window_type == "BLACKMAN_HARRIS") {
        cfg->window = windows::blackman_harris<scalar_t>(m_fft_size);
    } else if (m_window_type == "HAMMING") {
        cfg->window = windows::hamming<scalar_t>(m_fft_size);
    }
    return cfg;
}

template <typename T>
auto fft<T>::prepare(composite::metadata& md) -> void {
    // ARRIVAL order, main thread: stamp the FFT params from the published snapshot so the
    // metadata that travels with this packet matches the config work() will use.
    auto cfg = m_task_cfg.load();
    // Keep the prior annotation wire format (string), so this refactor doesn't incidentally
    // change the fft_size annotation type; psd parses it with stoul either way.
    md.annotations["fft_size"] = std::to_string(cfg->fft_size);
    md.annotations["fft_window"] = cfg->window_type;
}

template <typename T>
auto fft<T>::work(composite::immutable_buffer<T> in, composite::timestamp ts,
                  const composite::metadata& md) -> composite::immutable_buffer<T> {
    (void)ts;
    (void)md;
    // Load a consistent config snapshot for this work item (runs on a pool thread, not parked).
    auto cfg = m_task_cfg.load();
    const auto n = cfg->fft_size;

    // The output buffer and the FFTW plan read AND write exactly cfg->fft_size elements. If
    // in.size() != fft_size the plan runs off the end of those buffers: a smaller frame is an
    // out-of-bounds WRITE (heap corruption); a larger frame also walks the window buffer (sized
    // fft_size) out of bounds. A framer/fft size mismatch must be a loud error, not silent
    // corruption — throw so pipeline_component logs it and drops the packet (work() exceptions are
    // captured per-slot) rather than executing the plan.
    if (in.size() != n) {
        m_frames_dropped->inc();
        throw std::runtime_error("fft: input frame size (" + std::to_string(in.size()) +
                                 ") does not match configured fft_size (" + std::to_string(n) +
                                 "); check the upstream framer's frame_size");
    }

    // Per-pool-worker FFTW plan, rebuilt when the size OR the thread count changes (thread_local =>
    // one per worker). Tracking planned_threads separately means a runtime fftw_threads change is
    // actually honoured — previously the plan was keyed on size alone, so a threads change was
    // silently ignored until the size also changed.
    thread_local std::unique_ptr<plan_t> fft_plan;
    thread_local std::uint32_t planned_threads{0};
    if (!fft_plan || fft_plan->size() != n || planned_threads != cfg->fftw_threads) {
        fft_plan = std::make_unique<plan_t>(n, cfg->fftw_threads);
        planned_threads = cfg->fftw_threads;
        m_plan_builds->inc();
    }

    // Per-pool-worker scratch, 64-aligned to match the plan's array alignment and REUSED across
    // packets (resized only when fft_size changes). This replaces a fresh make_aligned_buffer per
    // packet, which allocated AND zero-initialised a whole frame that was then fully overwritten.
    thread_local composite::mutable_buffer<T> scratch;
    auto working = [&]() -> composite::mutable_buffer<T>& {
        if (scratch.size() != n) {
            // Uninitialized: every element is written (fused window or copy) before the FFT reads it.
            scratch = composite::make_aligned_buffer_uninitialized<T>(ALIGNMENT, n);
        }
        return scratch;
    };

    // Out-of-place FFT into a fresh output buffer (sent downstream, so it cannot be pooled here).
    // Uninitialized: the FFT (and the optional shift) write every element before it is read.
    auto output_buf = composite::make_aligned_buffer_uninitialized<T>(ALIGNMENT, n);
    if (cfg->window) {
        // Fused copy + window into the reused scratch, then FFT scratch -> output.
        auto& w = working();
        copy_and_window(in, w, cfg->window.get());
        fft_plan->execute(w.data(), output_buf.data());
    } else if ((reinterpret_cast<std::uintptr_t>(in.data()) % ALIGNMENT) == 0) {
        // No window and the input is already 64-aligned (the common case: an upstream aligned_mem
        // frame) — fftw's new-array execute accepts it because its alignment matches the plan's, so
        // FFT straight from the input and skip the frame copy entirely.
        fft_plan->execute(in.data(), output_buf.data());
    } else {
        // No window but an under-aligned input — copy into the aligned scratch so fftw always sees
        // an array with the alignment its plan was built for.
        auto& w = working();
        std::copy(in.begin(), in.end(), w.begin());
        fft_plan->execute(w.data(), output_buf.data());
    }
    if (cfg->shift) {
        std::rotate(output_buf.begin(), output_buf.begin() + (n / 2), output_buf.end());
    }
    return std::move(output_buf).to_immutable();
}

// Fused copy + window: scalar fallback
template <typename T>
COMPS_FMV_DEFAULT
auto fft<T>::copy_and_window(
  const composite::immutable_buffer<T>& input,
  composite::mutable_buffer<T>& output,
  const window_t* window
) -> void {
    const auto* in = reinterpret_cast<const scalar_t*>(input.data());
    auto* out = reinterpret_cast<scalar_t*>(output.data());
    const auto* w = window->data();
    const auto count = input.size() * 2;  // real + imag components
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = in[i] * w[i];
    }
}

// Fused copy + window: AVX-512
#if COMPS_FMV_ENABLED
template <typename T>
[[gnu::target("avx512f")]]
auto fft<T>::copy_and_window(
  const composite::immutable_buffer<T>& input,
  composite::mutable_buffer<T>& output,
  const window_t* window
) -> void {
    const auto* in = reinterpret_cast<const scalar_t*>(input.data());
    auto* out = reinterpret_cast<scalar_t*>(output.data());
    const auto* w = window->data();
    const auto count = input.size() * 2;

    std::size_t i = 0;
    if constexpr (std::is_same_v<scalar_t, float>) {
        constexpr std::size_t stride = 16;  // 512 bits / 32 bits
        for (; i + stride <= count; i += stride) {
            auto input_v = _mm512_loadu_ps(in + i);
            auto window_v = _mm512_loadu_ps(w + i);
            auto result = _mm512_mul_ps(input_v, window_v);
            _mm512_storeu_ps(out + i, result);
        }
    } else {
        constexpr std::size_t stride = 8;  // 512 bits / 64 bits
        for (; i + stride <= count; i += stride) {
            auto input_v = _mm512_loadu_pd(in + i);
            auto window_v = _mm512_loadu_pd(w + i);
            auto result = _mm512_mul_pd(input_v, window_v);
            _mm512_storeu_pd(out + i, result);
        }
    }
    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = in[i] * w[i];
    }
}
#endif

// Fused copy + window: AVX2
#if COMPS_FMV_ENABLED
template <typename T>
[[gnu::target("avx2")]]
auto fft<T>::copy_and_window(
  const composite::immutable_buffer<T>& input,
  composite::mutable_buffer<T>& output,
  const window_t* window
) -> void {
    const auto* in = reinterpret_cast<const scalar_t*>(input.data());
    auto* out = reinterpret_cast<scalar_t*>(output.data());
    const auto* w = window->data();
    const auto count = input.size() * 2;

    std::size_t i = 0;
    if constexpr (std::is_same_v<scalar_t, float>) {
        constexpr std::size_t stride = 8;  // 256 bits / 32 bits
        for (; i + stride <= count; i += stride) {
            auto input_v = _mm256_loadu_ps(in + i);
            auto window_v = _mm256_loadu_ps(w + i);
            auto result = _mm256_mul_ps(input_v, window_v);
            _mm256_storeu_ps(out + i, result);
        }
    } else {
        constexpr std::size_t stride = 4;  // 256 bits / 64 bits
        for (; i + stride <= count; i += stride) {
            auto input_v = _mm256_loadu_pd(in + i);
            auto window_v = _mm256_loadu_pd(w + i);
            auto result = _mm256_mul_pd(input_v, window_v);
            _mm256_storeu_pd(out + i, result);
        }
    }
    // Scalar remainder
    for (; i < count; ++i) {
        out[i] = in[i] * w[i];
    }
}
#endif

// Explicit template instantiations
template class fft<std::complex<float>>;
template class fft<std::complex<double>>;

COMPOSITE_REGISTER_COMPONENT([](std::string_view id, const composite::create_args& args)
                                 -> std::shared_ptr<composite::component> {
    const auto type = args.type();
    if (type == "cf32") {
        return composite::make_component<fft<std::complex<float>>>(id);
    } else if (type == "cf64") {
        return composite::make_component<fft<std::complex<double>>>(id);
    }
    throw std::runtime_error(std::format("unknown type '{}' for fft component", type));
})
