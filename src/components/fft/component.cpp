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

#include <algorithm>
#include <bit>
#include <cstdint>
#include <fftw3.h>
#include <format>
#include <immintrin.h>
#include <mutex>
#include <optional>
#include <pthread.h>
#include <thread>
#include "simd_fmv.hpp"

namespace {

/// Ceiling for the fftw_threads property. Generous relative to any real host, small enough
/// that a typo (e.g. 100000) cannot ask FFTW for that many threads per plan.
constexpr std::uint32_t k_max_fftw_threads = 128;

/// Ceiling for fft_size, same rationale: the value drives window/scratch/output allocations
/// and an FFTW_MEASURE plan build. 2^24 (16.7M points) is far beyond any deployed transform,
/// while a typo'd 2^30 would ask for multi-GB buffers and a multi-minute measure.
constexpr std::uint32_t k_max_fft_size = 1u << 24;

// fftw(f)_init_threads()/cleanup_threads() manage PROCESS-GLOBAL state per precision: cleanup
// destroys every plan and all threading bookkeeping for that precision, process-wide. Calling
// them per instance meant destroying one fft component tore FFTW down underneath any other
// instance still executing plans — a use-after-free on the survivor's next transform. Reference-
// count per precision instead, under a mutex (FFTW's own init/cleanup are not thread-safe).
std::mutex g_fftw_threads_mtx;
int g_fftwf_thread_users = 0;
int g_fftw_thread_users = 0;

auto acquire_fftw_threads(bool single_precision) -> void {
    const std::scoped_lock lk{g_fftw_threads_mtx};
    // init_threads returns 0 on failure. Increment the refcount only AFTER a successful
    // init: throwing keeps the count consistent (the lease constructor propagates, so the
    // component fails construction loudly instead of executing threaded plans on an
    // uninitialized threading layer).
    if (single_precision) {
        if (g_fftwf_thread_users == 0 && fftwf_init_threads() == 0) {
            throw std::runtime_error("fftwf_init_threads() failed");
        }
        ++g_fftwf_thread_users;
    } else {
        if (g_fftw_thread_users == 0 && fftw_init_threads() == 0) {
            throw std::runtime_error("fftw_init_threads() failed");
        }
        ++g_fftw_thread_users;
    }
}

auto release_fftw_threads(bool single_precision) noexcept -> void {
    const std::scoped_lock lk{g_fftw_threads_mtx};
    if (single_precision) {
        if (--g_fftwf_thread_users == 0) {
            fftwf_cleanup_threads();
        }
    } else {
        if (--g_fftw_thread_users == 0) {
            fftw_cleanup_threads();
        }
    }
}

} // namespace

template <typename T>
fft<T>::fftw_threads_lease::fftw_threads_lease(bool single_precision) : single(single_precision) {
    acquire_fftw_threads(single);
}

template <typename T>
fft<T>::fftw_threads_lease::~fftw_threads_lease() {
    release_fftw_threads(single);
}

template <typename T>
fft<T>::fft(std::string_view id)
    : base(id, "data_in", "data_out", /*default_workers=*/1) {
    using enum composite::properties::config_type;

    // Ports + num_workers are provided by pipeline_component; register fft's own config here.
    // RUNTIME: make_task_config() already rebuilds the window from m_window_type into a fresh
    // immutable snapshot, and publish() keeps the outgoing one alive for workers still holding it,
    // so a window change is applied by exactly the path fft_size already uses.
    this->add_property("window", m_window_type, RUNTIME).validate([](const std::string& v) {
        return (v == "BLACKMAN_HARRIS") || (v == "HAMMING") || v.empty();
    });
    this->add_property("fft_size", m_fft_size, RUNTIME).validate([](const std::uint32_t& v) {
        return std::has_single_bit(v) && v <= k_max_fft_size;
    });
    // Ceiling on fftw_threads: the value goes straight to fftwf_plan_with_nthreads(), and the
    // effective parallelism is num_workers x fftw_threads per component. An absurd value used to
    // be accepted verbatim; reject it instead of letting FFTW spawn it.
    this->add_property("fftw_threads", m_fftw_threads).validate([](const std::uint32_t& v) {
        return v >= 1 && v <= k_max_fftw_threads;
    });
    this->add_property("shift", m_shift, RUNTIME);

    // Drop / plan-build counters in the shared registry so operators can see rejected frames and
    // FFTW plan churn (labeled by component id; auto-removed by ~component).
    m_frames_dropped = &this->create_counter("fft.frames_dropped",
                                             "Input frames dropped for an fft_size mismatch");
    m_plan_builds = &this->create_counter("fft.plan_builds", "FFTW plan (re)creations across the worker pool");
    m_binding_misses = &this->create_counter(
        "fft.config_binding_misses",
        "Packets transformed with the newest config because their stamped generation was no longer retained");

    // Publish an initial snapshot so a pool worker always loads a valid config even if no
    // properties are set before start() (property_change_handler republishes on every change).
    publish_config();
}

template <typename T>
fft<T>::~fft() {
    // Stop the pipeline (main worker + pool) BEFORE releasing FFTW's global thread state: the dtor
    // body runs before members, so a pool worker could otherwise still be in work() using a plan
    // when fftw_cleanup_threads() frees FFTW's threading bookkeeping.
    this->stop();
    // m_fftw_lease releases the process-global FFTW threads refcount during member destruction,
    // i.e. AFTER this stop() — the pool is already down when the last lease of this precision
    // lets fftw(f)_cleanup_threads() run.
}

template <typename T>
auto fft<T>::property_change_handler(const composite::properties::json& diff) -> void {
    (void)diff;
    // Publish a fresh immutable snapshot (incl. a newly-allocated window). Pool workers already in
    // flight keep their previously-loaded snapshot alive, so the old window is freed only once the
    // last worker using it finishes — no use-after-free even though the pool is not parked. (A
    // num_workers change is applied separately by pipeline_component.)
    publish_config();
    // prepare() stamps fft_size/fft_window/fft_shift from this snapshot onto the shared metadata;
    // tell the pipeline to rebuild it even though the incoming metadata instance is unchanged.
    this->invalidate_prepared_metadata();
}

template <typename T>
auto fft<T>::on_workers_resized(int n) -> void {
    // Main worker, pool idle (no in-flight work). Record the ACHIEVED pool size and republish
    // so the fftw_threads clamp in make_task_config reflects it.
    m_active_workers.store(n, std::memory_order_relaxed);
    publish_config();
}

template <typename T>
auto fft<T>::same_key(const task_config& a, const task_config& b) -> bool {
    // The binding key (see config_matches): generations agreeing on these are semantically
    // interchangeable — same window content, same shift, same frame size.
    return a.fft_size == b.fft_size && a.shift == b.shift && a.window_type == b.window_type;
}

template <typename T>
auto fft<T>::publish_config() -> void {
    // Prepend the new generation to the retained history (see cfg_history_t in the header),
    // DEDUPLICATED BY KEY: an older generation with the same key as a retained newer one is
    // dropped (any packet stamped with it binds to the newer, semantically identical entry).
    // Property churn between a handful of values therefore never consumes history slots.
    auto prev = m_task_cfg.load();  // null before the constructor's first publish
    auto next = std::make_shared<cfg_history_t>();
    next->reserve(CFG_HISTORY_DEPTH);
    next->push_back(make_task_config(prev.get()));
    if (prev != nullptr) {
        for (const auto& cfg : *prev) {
            if (next->size() >= CFG_HISTORY_DEPTH) {
                break;
            }
            const bool duplicate = std::any_of(next->begin(), next->end(),
                [&](const auto& kept) { return same_key(*kept, *cfg); });
            if (!duplicate) {
                next->push_back(cfg);
            }
        }
    }
    m_task_cfg.publish(std::move(next));
}

template <typename T>
auto fft<T>::config_matches(const task_config& cfg, const composite::metadata& md) -> bool {
    static const std::string k_size{"fft_size"};
    static const std::string k_window{"fft_window"};
    static const std::string k_shift{"fft_shift"};
    const auto size_it = md.annotations.find(k_size);
    const auto window_it = md.annotations.find(k_window);
    const auto shift_it = md.annotations.find(k_shift);
    if (size_it == md.annotations.end() || window_it == md.annotations.end() ||
        shift_it == md.annotations.end()) {
        return false;  // not stamped by this component's prepare(); nothing to bind to
    }
    return size_it->second == cfg.size_annotation && window_it->second == cfg.window_type &&
           shift_it->second == (cfg.shift ? "1" : "0");
}

namespace {
// Output-frame pool sizing: cap total pooled memory rather than buffer count, so small
// transforms get deep pools and giant ones skip pooling entirely (their per-packet heap
// allocation is dwarfed by the O(n log n) transform).
constexpr std::size_t k_pool_target_bytes = std::size_t{16} << 20;  // 16 MiB per snapshot
constexpr std::size_t k_pool_min_buffers = 8;   // below this, pooling can't cover in-flight frames
constexpr std::size_t k_pool_max_buffers = 64;

/// Ceiling on the component's EFFECTIVE parallelism (pool workers x FFTW threads per plan).
/// Both factors are independently validated, but their product was not: num_workers can be
/// 1024 and fftw_threads 128 — a 131072-way thread ask from one component. Budget the
/// product against the machine instead; fftw_threads is clamped per-plan to fit.
auto max_total_parallelism() -> uint32_t {
    const auto hw = std::thread::hardware_concurrency();
    return std::max(hw != 0 ? 2 * hw : 0u, 16u);
}
} // namespace

template <typename T>
auto fft<T>::make_task_config(const cfg_history_t* prev) const -> std::shared_ptr<const task_config> {
    auto cfg = std::make_shared<task_config>();
    cfg->fft_size = m_fft_size;
    cfg->shift = m_shift;
    cfg->window_type = m_window_type;
    cfg->size_annotation = std::to_string(m_fft_size);

    // Bound num_workers x fftw_threads (each is individually capped, but the product goes to
    // the OS as real threads). Clamp the per-plan thread count to the remaining budget.
    // When num_workers ALONE exceeds the budget, this component cannot shrink the pool —
    // num_workers and its ceiling are framework-owned (pipeline_component; see
    // composite-framework-requests.md FR-4) — so FFTW threading is pinned to 1 and the
    // overrun is called out for the operator to reduce num_workers.
    const auto workers = static_cast<uint32_t>(std::max(1, m_active_workers.load(std::memory_order_relaxed)));
    const auto budget = std::max<uint32_t>(max_total_parallelism() / workers, 1);
    cfg->fftw_threads = std::min(m_fftw_threads, budget);
    if (workers > max_total_parallelism()) {
        this->logger()->warn("fft: num_workers ({}) alone exceeds the total-parallelism budget ({}); "
                             "FFTW threading disabled — reduce num_workers",
                             workers, max_total_parallelism());
    } else if (cfg->fftw_threads != m_fftw_threads) {
        this->logger()->info("fft: clamping fftw_threads {} -> {} ({} pool workers, {} total-parallelism budget)",
                             m_fftw_threads, cfg->fftw_threads, workers, max_total_parallelism());
    }

    std::shared_ptr<window_t> window;
    if (m_window_type == "BLACKMAN_HARRIS") {
        window = windows::blackman_harris<scalar_t>(m_fft_size);
    } else if (m_window_type == "HAMMING") {
        window = windows::hamming<scalar_t>(m_fft_size);
    }
    if (window != nullptr) {
        // Declare sum(w^2) for downstream normalization (see task_config). Taps are stored
        // complex-interleaved (each tap twice); sum the even slots, in double for accuracy.
        // Computed BEFORE any shift fold, though squaring makes the sign flip immaterial.
        double sum_sq = 0.0;
        const auto* w = window->data();
        for (std::size_t k = 0; k < m_fft_size; ++k) {
            sum_sq += static_cast<double>(w[2 * k]) * static_cast<double>(w[2 * k]);
        }
        cfg->window_sum_sq_annotation = std::format("{:.17g}", sum_sq);
    }
    if (window && cfg->shift) {
        // Fold the fftshift into the window: FFT(x[k]·(−1)^k)[m] = X[(m + n/2) mod n]
        // exactly (a half-spectrum shift is multiplication by e^{jπk} = ±1 in time), so
        // flipping the window's sign at odd k makes the transform come out pre-shifted and
        // the post-FFT rotate pass over the whole output disappears. Sign flips are exact
        // in floating point — the result is bit-equivalent mathematics, not an approximation.
        auto* w = window->data();
        for (std::size_t k = 1; k < m_fft_size; k += 2) {
            w[2 * k] = -w[2 * k];
            w[2 * k + 1] = -w[2 * k + 1];
        }
        cfg->shift_folded = true;
    }
    cfg->window = std::move(window);

    // Pooled output frames (see task_config): pool only when the memory cap yields enough
    // buffers to actually cover the in-flight window; work() falls back to the heap when
    // the pool is exhausted or absent. REUSE any retained generation's pool with the same
    // frame size — window/shift/threads churn shares one pool, and ALTERNATING fft_size
    // values (A/B/A/...) cycle between the retained pools instead of creating a fresh one
    // per change and stranding the old behind downstream-held frames.
    if (prev != nullptr) {
        for (const auto& old : *prev) {
            if (old->out_pool != nullptr && old->fft_size == cfg->fft_size) {
                cfg->out_pool = old->out_pool;
                break;
            }
        }
    }
    if (cfg->out_pool == nullptr) {
        const auto frame_bytes = static_cast<std::size_t>(m_fft_size) * sizeof(T);
        if (const auto want = k_pool_target_bytes / frame_bytes; want >= k_pool_min_buffers) {
            cfg->out_pool = composite::slab_pool<T>::create(
                m_fft_size, std::min(want, k_pool_max_buffers));
        }
    }
    return cfg;
}

template <typename T>
auto fft<T>::prepare(composite::metadata& md) -> void {
    // ARRIVAL order, main thread: stamp the FFT params from the newest config generation so
    // the metadata that travels with this packet identifies the config work() must use —
    // work() binds each packet back to these annotations (see config_matches), so a RUNTIME
    // change with packets in flight cannot make the transform disagree with the metadata.
    auto history = m_task_cfg.load();
    const auto& cfg = *history->front();
    // Keep the prior annotation wire format (string), so this refactor doesn't incidentally
    // change the fft_size annotation type; psd parses it with stoul either way. fft_shift
    // also tells downstream whether the spectrum is fftshifted — previously unknowable.
    md.annotations["fft_size"] = cfg.size_annotation;
    md.annotations["fft_window"] = cfg.window_type;
    md.annotations["fft_shift"] = cfg.shift ? "1" : "0";
    // Normalization contract with downstream psd: declare sum(w^2) of the window actually
    // applied, so consumers do not have to re-derive it from the window's name. Stamped only
    // when a window is in use (the established no-window convention downstream is a unit
    // window factor, NOT sum(1^2) = N).
    if (!cfg.window_sum_sq_annotation.empty()) {
        md.annotations["fft_window_sum_sq"] = cfg.window_sum_sq_annotation;
    } else {
        md.annotations.erase("fft_window_sum_sq");
    }
}

template <typename T>
auto fft<T>::work(composite::immutable_buffer<T> in, composite::timestamp ts,
                  const composite::metadata& md) -> composite::immutable_buffer<T> {
    (void)ts;
    // Bind this packet to the config generation its metadata was STAMPED with at ingest.
    // "Newest" is the common case (fast path); after a RUNTIME change with packets in
    // flight, the stamped generation is an older retained entry — using the newest one
    // instead would transform the frame with a window/shift/size its metadata does not
    // describe. A packet older than the retained history (or foreign metadata that never
    // went through prepare()) falls back to the newest generation.
    auto history = m_task_cfg.load();
    const task_config* cfg = history->front().get();
    if (!config_matches(*cfg, md)) [[unlikely]] {
        bool bound = false;
        for (const auto& candidate : *history) {
            if (config_matches(*candidate, md)) {
                cfg = candidate.get();
                bound = true;
                break;
            }
        }
        if (!bound) {
            // Config churn outran the retained history (needs > CFG_HISTORY_DEPTH DISTINCT
            // configs while this packet was queued), or the metadata never went through
            // prepare(). Fall back to the newest generation — pre-history behavior — but
            // make it observable: the counter carries the rate, the warning fires once.
            m_binding_misses->inc();
            if (!m_binding_miss_warned.exchange(true, std::memory_order_relaxed)) {
                this->logger()->warn("fft: a packet's stamped config generation is no longer retained "
                                     "(config churn outran the {}-deep history); transformed with the "
                                     "newest config instead (counted in fft.config_binding_misses)",
                                     CFG_HISTORY_DEPTH);
            }
        }
    }
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

    // Out-of-place FFT into a fresh output buffer (sent downstream). Pooled slab in steady
    // state (zero heap allocations per packet; slab slots are 64-aligned, matching the plan);
    // heap fallback when downstream holds more frames than the pool's capacity or the frames
    // are too large to pool. Uninitialized either way: the FFT (and the optional shift) write
    // every element before it is read.
    std::optional<composite::external_buffer<T>> slab;
    composite::mutable_buffer<T> heap_buf;
    T* out = nullptr;
    if (cfg->out_pool && (slab = cfg->out_pool->acquire())) {
        out = slab->data();
    } else {
        heap_buf = composite::make_aligned_buffer_uninitialized<T>(ALIGNMENT, n);
        out = heap_buf.data();
    }

    if (cfg->window) {
        // Fused copy + window into the reused scratch, then FFT scratch -> output. When shift
        // is on, the window carries the fftshift too (see make_task_config) — no rotate pass.
        auto& w = working();
        copy_and_window(in, w, cfg->window.get());
        fft_plan->execute(w.data(), out);
    } else if ((reinterpret_cast<std::uintptr_t>(in.data()) % ALIGNMENT) == 0) {
        // No window and the input is already 64-aligned (the common case: an upstream aligned_mem
        // frame) — fftw's new-array execute accepts it because its alignment matches the plan's, so
        // FFT straight from the input and skip the frame copy entirely.
        fft_plan->execute(in.data(), out);
    } else {
        // No window but an under-aligned input — copy into the aligned scratch so fftw always sees
        // an array with the alignment its plan was built for.
        auto& w = working();
        std::copy(in.begin(), in.end(), w.begin());
        fft_plan->execute(w.data(), out);
    }
    if (cfg->shift && !cfg->shift_folded) {
        std::rotate(out, out + (n / 2), out + n);
    }
    return slab.has_value() ? composite::immutable_buffer<T>(std::move(*slab))
                            : std::move(heap_buf).to_immutable();
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
