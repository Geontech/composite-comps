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
#include <composite/buffers/slab_pool.hpp>
#include <composite/metrics/metrics.hpp>
#include <composite/properties/snapshot.hpp>

#include <atomic>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
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
        // True when the fftshift is FOLDED INTO the window (odd-index sign flip): the
        // transform then comes out pre-shifted and work() skips the post-FFT rotate pass.
        // Only possible when a window is in use; the unwindowed paths still rotate.
        bool shift_folded{false};
        uint32_t fftw_threads{1};
        std::string window_type;                 // for the output metadata annotation
        std::string size_annotation;             // std::to_string(fft_size), built once
        // sum(w^2) over the window taps, stamped as the fft_window_sum_sq annotation so a
        // downstream psd normalizes from what THIS component actually applied instead of
        // re-deriving the window from its name (which silently mis-normalizes any window
        // type the consumer does not recognize). Empty when no window is in use.
        std::string window_sum_sq_annotation;
        std::shared_ptr<const window_t> window;  // null => no windowing
        // Pooled output frames: work() sends one freshly-owned buffer downstream per packet,
        // which used to be a heap allocation per packet. The pool lives in the snapshot so
        // pool workers acquire from a consistent, correctly-sized pool with no extra locking.
        // Carried FORWARD unchanged across config generations while fft_size stays the same
        // (see make_task_config), so window/shift churn does not strand one pool per
        // generation behind downstream-held frames. Null when frames are too large to pool.
        std::shared_ptr<composite::slab_pool<T>> out_pool;
    };

    // What work() loads is a HISTORY of recent config generations, newest first. prepare()
    // stamps each packet's metadata from the newest generation at INGEST; work() runs later
    // on a pool thread, where "latest" may already be a NEWER generation than the one this
    // packet was stamped with (a RUNTIME change with packets in flight). work() therefore
    // binds each packet to the generation matching its OWN annotations (fft_size /
    // fft_window / fft_shift), so the transform always agrees with the metadata that
    // travels with the frame — and an old-size frame accepted before an fft_size change is
    // still transformed rather than spuriously dropped.
    //
    // Retention is DEDUPLICATED BY KEY (fft_size, window_type, shift): two generations with
    // the same key are semantically interchangeable, so property churn between a handful of
    // values never consumes history slots — flipping A/B forever retains exactly two
    // entries. Eviction therefore requires more than CFG_HISTORY_DEPTH *distinct*
    // configurations applied while one packet stays queued, not merely that many property
    // writes. A packet that still misses (or foreign metadata that never went through
    // prepare()) falls back to the newest generation — the pre-history behavior — and is
    // COUNTED (fft.config_binding_misses) plus warned once, so the degraded case is
    // observable instead of silent. Exact per-packet binding with no fallback at all needs
    // a framework channel from ingest to work(); see composite-framework-requests.md FR-1.
    using cfg_history_t = std::vector<std::shared_ptr<const task_config>>;
    static constexpr std::size_t CFG_HISTORY_DEPTH = 16;

    auto make_task_config(const cfg_history_t* prev) const -> std::shared_ptr<const task_config>;
    auto publish_config() -> void;
    static auto config_matches(const task_config& cfg, const composite::metadata& md) -> bool;
    static auto same_key(const task_config& a, const task_config& b) -> bool;

    // Actual pool size, recorded by on_workers_resized() (main worker, pool idle) and read by
    // make_task_config() (PCH thread, park-synchronized) to bound num_workers x fftw_threads.
    auto on_workers_resized(int n) -> void override;
    std::atomic<int> m_active_workers{1};

    // RAII holder for the process-global FFTW threads refcount (helpers in component.cpp). A
    // MEMBER rather than acquire/release calls in the ctor/dtor bodies: if the constructor body
    // throws after acquiring (e.g. a metric registration), the destructor never runs, but
    // fully-constructed members ARE destroyed — so the count cannot leak. Destroyed after the
    // dtor body's stop(), so the pool never outlives the threading state it uses.
    struct fftw_threads_lease {
        explicit fftw_threads_lease(bool single_precision);
        ~fftw_threads_lease();
        fftw_threads_lease(const fftw_threads_lease&) = delete;
        fftw_threads_lease& operator=(const fftw_threads_lease&) = delete;
        bool single;
    };
    fftw_threads_lease m_fftw_lease{std::is_same_v<scalar_t, float>};

    // Properties (num_workers is owned by pipeline_component). Live config, written by the engine
    // under park; read only by property_change_handler/make_task_config (park-synchronized).
    std::string m_window_type;
    uint32_t m_fft_size{1024};
    uint32_t m_fftw_threads{1};
    bool m_shift{true};

    composite::snapshot<cfg_history_t> m_task_cfg;

    // Observability (shared metrics registry, labeled by component id; auto-removed by ~component).
    // frames_dropped: packets rejected in work() for a frame-size/fft_size mismatch (else only
    // logged per-slot). plan_builds: FFTW plan (re)creations — an FFTW_MEASURE build is expensive,
    // so a climbing count flags fft_size/fftw_threads thrashing across the pool.
    composite::metrics::counter<uint64_t>* m_frames_dropped{nullptr};
    composite::metrics::counter<uint64_t>* m_plan_builds{nullptr};
    // config_binding_misses: packets whose stamped config generation was no longer retained
    // (transformed with the newest instead). Nonzero means config churn outran the history.
    composite::metrics::counter<uint64_t>* m_binding_misses{nullptr};
    std::atomic<bool> m_binding_miss_warned{false};

}; // class fft
