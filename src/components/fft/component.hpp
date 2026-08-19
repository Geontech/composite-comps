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

#include <composite/core/pipeline_component.hpp>
#include <composite/buffers/buffer.hpp>
#include <composite/buffers/aligned_mem.hpp>
#include <composite/metrics/metrics.hpp>
#include <composite/properties/snapshot.hpp>

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include "simd_fmv.hpp"

// T is expected to be std::complex<float> or std::complex<double>.
//
// fft is a pipeline_component: the framework's single worker ingests in order and the bounded
// slot ring re-serialises output to submission order, while the parallel FFT runs on a worker
// pool (num_workers). This replaces the hand-rolled input-thread + task_queue + ordered-future
// drain (and integrates with the park coordinator, so config writes / stop quiesce promptly).
template <typename T>
class fft : public composite::pipeline_component<composite::immutable_buffer<T>, composite::immutable_buffer<T>> {
    using base = composite::pipeline_component<composite::immutable_buffer<T>, composite::immutable_buffer<T>>;
    using scalar_t = typename T::value_type;  // float or double
    using plan_t = fft_plan<T>;
    using window_t = composite::aligned_mem<scalar_t>;

    static constexpr std::size_t ALIGNMENT = 64;

public:
    explicit fft(std::string_view id);
    ~fft() override;

    auto property_change_handler(const composite::properties::json& diff) -> void override;

protected:
    // ARRIVAL order, main thread: stamp the FFT params onto the metadata that travels with the
    // packet (read from the published config snapshot, so it is consistent with work()).
    auto prepare(composite::metadata& md) -> void override;

    // The parallel stage (pool worker, concurrent across packets): window + FFT + optional shift.
    auto work(composite::immutable_buffer<T> in, composite::timestamp ts,
              const composite::metadata& md) -> composite::immutable_buffer<T> override;

private:
    // Fused copy + window with SIMD variants (use only their arguments — thread-safe in work()).
    COMPS_FMV_DEFAULT
    auto copy_and_window(const composite::immutable_buffer<T>& input,
                         composite::mutable_buffer<T>& output, const window_t* window) -> void;
#if COMPS_FMV_ENABLED
    [[gnu::target("avx512f")]]
    auto copy_and_window(const composite::immutable_buffer<T>& input,
                         composite::mutable_buffer<T>& output, const window_t* window) -> void;
#endif
#if COMPS_FMV_ENABLED
    [[gnu::target("avx2")]]
    auto copy_and_window(const composite::immutable_buffer<T>& input,
                         composite::mutable_buffer<T>& output, const window_t* window) -> void;
#endif

    // Immutable per-task config snapshot. property_change_handler runs under park (the main
    // ingest/retire worker quiesced), but the POOL workers running work() do NOT park — so they
    // must not read the live config members (m_window free-while-used is a UAF; the scalar reads
    // are torn). PCH builds a fresh value and publishes it through composite::snapshot; work()
    // loads it, getting a consistent {fft_size, window, shift} and a window buffer kept alive by
    // the returned shared_ptr.
    struct task_config {
        std::size_t fft_size{1024};
        bool shift{true};
        uint32_t fftw_threads{1};
        std::string window_type;                 // for the output metadata annotation
        std::shared_ptr<const window_t> window;  // null => no windowing
    };
    auto make_task_config() const -> std::shared_ptr<const task_config>;

    // Properties (num_workers is owned by pipeline_component). Live config, written by the engine
    // under park; read only by property_change_handler/make_task_config (park-synchronized).
    std::string m_window_type;
    uint32_t m_fft_size{1024};
    uint32_t m_fftw_threads{1};
    bool m_shift{true};

    composite::snapshot<task_config> m_task_cfg;

    // Observability (shared metrics registry, labeled by component id; auto-removed by ~component).
    // frames_dropped: packets rejected in work() for a frame-size/fft_size mismatch (else only
    // logged per-slot). plan_builds: FFTW plan (re)creations — an FFTW_MEASURE build is expensive,
    // so a climbing count flags fft_size/fftw_threads thrashing across the pool.
    composite::metrics::counter<uint64_t>* m_frames_dropped{nullptr};
    composite::metrics::counter<uint64_t>* m_plan_builds{nullptr};

}; // class fft
