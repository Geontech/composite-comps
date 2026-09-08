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

#include "component.hpp"
#include "windows.hpp"

#include <composite/core/register.hpp>
#include <composite/buffers/external_buffer.hpp>
#include <composite/buffers/slab_pool.hpp>

#include <algorithm>
#include <bit>
#include <charconv>
#include <complex>
#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>

namespace {

/// Ceiling for a metadata-declared fft_size, matching the fft component's own property
/// ceiling: a larger value is malformed upstream metadata, not a real window request (and
/// would ask for a multi-GB window allocation).
constexpr std::size_t k_max_annotated_fft_size = std::size_t{1} << 24;

// Output pool sizing. The budget is COMPONENT-WIDE and divided across the pool workers
// (each worker owns a thread_local pool): a per-worker constant would multiply by
// num_workers, which the framework accepts up to 1024. Frames too large for a worker's
// share to cover the in-flight window are simply not pooled.
constexpr std::size_t k_out_pool_total_bytes = std::size_t{32} << 20;      // whole component
constexpr std::size_t k_out_pool_worker_cap_bytes = std::size_t{8} << 20;  // per worker, few-worker case
constexpr std::size_t k_out_pool_min_buffers = 8;
constexpr std::size_t k_out_pool_max_buffers = 64;

/// Positive, finite, nonzero — classified on the BIT PATTERN. This translation unit builds
/// with -ffast-math (FAST_MATH in CMakeLists), under which NaN comparisons and
/// std::isfinite/std::isnan may be constant-folded away; integer classification is immune.
/// Overloads per width: a double that passes can still overflow to inf (or flush to zero)
/// when narrowed to float, so validate in the precision that will actually be used.
auto positive_finite(double v) -> bool {
    const auto bits = std::bit_cast<std::uint64_t>(v);
    const bool negative_or_zero = (bits >> 63) != 0 || (bits << 1) == 0;  // sign bit, or +/-0
    const bool nan_or_inf = ((bits >> 52) & 0x7FF) == 0x7FF;              // exponent all-ones
    return !negative_or_zero && !nan_or_inf;
}
auto positive_finite(float v) -> bool {
    const auto bits = std::bit_cast<std::uint32_t>(v);
    const bool negative_or_zero = (bits >> 31) != 0 || (bits << 1) == 0;
    const bool nan_or_inf = ((bits >> 23) & 0xFF) == 0xFF;
    return !negative_or_zero && !nan_or_inf;
}

/// Strict full-string numeric parse (stoul/stod accept trailing junk like "1024abc").
template <typename V>
auto parse_full(const std::string& s, V& out) -> bool {
    const auto* end = s.data() + s.size();
    const auto [ptr, ec] = std::from_chars(s.data(), end, out);
    return ec == std::errc{} && ptr == end;
}

/// Container for pooled output frames with genuine VALUE semantics. mutable_buffer's
/// type-erased deep copy invokes the CONTAINER's copy constructor (output-port fan-out to
/// multiple mutable consumers relies on it for isolation) — and external_buffer's copy
/// constructor shares the slab, so wrapping a slab directly would hand every fan-out
/// consumer the SAME memory. This container's copy constructor copies the samples into
/// owned heap storage instead: the single-consumer fast path stays pooled and zero-copy,
/// and only fan-out copies pay an allocation (exactly what fan-out already paid pre-pool).
template <typename T>
struct pooled_frame {
    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    struct aligned_delete {
        auto operator()(T* p) const noexcept -> void { ::operator delete[](p, std::align_val_t{64}); }
    };
    static auto aligned_alloc_n(std::size_t count) -> std::unique_ptr<T[], aligned_delete> {
        return std::unique_ptr<T[], aligned_delete>(
            static_cast<T*>(::operator new[](count * sizeof(T), std::align_val_t{64})));
    }

    composite::external_buffer<T> slab{};       // engaged when pool-backed
    std::unique_ptr<T[], aligned_delete> heap{};  // engaged on deep copies; 64-aligned so
                                                  // downstream SIMD (exp_smooth) keeps its
                                                  // aligned fast path on fan-out copies too
    std::size_t n{0};

    pooled_frame() = default;
    explicit pooled_frame(composite::external_buffer<T> s) : slab(std::move(s)), n(slab.size()) {}
    pooled_frame(const pooled_frame& other) : heap(aligned_alloc_n(other.n)), n(other.n) {
        std::copy_n(other.data(), n, heap.get());
    }
    auto operator=(const pooled_frame& other) -> pooled_frame& {
        if (this != &other) {
            heap = aligned_alloc_n(other.n);
            std::copy_n(other.data(), other.n, heap.get());
            n = other.n;
            slab = {};
        }
        return *this;
    }
    pooled_frame(pooled_frame&&) noexcept = default;
    auto operator=(pooled_frame&&) noexcept -> pooled_frame& = default;

    auto data() noexcept -> T* { return heap ? heap.get() : slab.data(); }
    auto data() const noexcept -> const T* { return heap ? heap.get() : slab.data(); }
    auto size() const noexcept -> std::size_t { return n; }
    auto begin() noexcept -> iterator { return data(); }
    auto begin() const noexcept -> const_iterator { return data(); }
    auto end() noexcept -> iterator { return data() + n; }
    auto end() const noexcept -> const_iterator { return data() + n; }
};

} // namespace

template <typename T>
psd<T>::psd(std::string_view id)
    : base(id, "data_in", "data_out", /*default_workers=*/1) {
    using enum composite::properties::config_type;
    // num_workers is provided by pipeline_component.
    this->add_property("power_based_normalization", m_power_based_normalization, RUNTIME);
    m_pbn.publish(m_power_based_normalization);
    m_bad_metadata = &this->create_counter("psd.bad_metadata", "Packets with an unparseable fft_size annotation");
}

template <typename T>
auto psd<T>::property_change_handler(const composite::properties::json& diff) -> void {
    (void)diff;
    // Publish the config snapshot read by the pool's work() (pool threads are not parked). A
    // num_workers change is handled separately by pipeline_component.
    m_pbn.publish(m_power_based_normalization);
    // prepare() stamps the normalization mode onto the shared metadata; tell the pipeline to
    // rebuild it even though the incoming metadata instance is unchanged.
    this->invalidate_prepared_metadata();
}

template <typename T>
auto psd<T>::prepare(composite::metadata& md) -> void {
    // ARRIVAL order, main thread: record the normalization mode (from the snapshot) on the metadata.
    md.annotations["psd_power_based_normalization"] = std::to_string(*m_pbn.load());
}

template <typename T>
auto psd<T>::compute_norm_const(const window_t* window, T sample_rate, bool power_based,
                                std::size_t spectrum_size) -> T {
    if (sample_rate == T{}) {
        return T{1};
    }
    T window_norm_const = T{1};
    if (window != nullptr && window->size() > 0) {
        auto window_sum_sq = std::transform_reduce(
            window->data(), window->data() + window->size(), T{0}, std::plus<>{},
            [](T val) { return val * val; });
        window_norm_const = window_sum_sq;
        if (power_based) {
            window_norm_const = window_norm_const / static_cast<T>(window->size());
        }
        // A degenerate window (all zeros, or a zero-length one guarded above) must not turn
        // the norm into inf/NaN that then poisons every output sample.
        if (!(window_norm_const > T{0})) {
            window_norm_const = T{1};
        }
    } else if (!power_based && spectrum_size > 0) {
        // No window in energy mode: the rectangular window has sum(w^2) = N, and the windowed
        // branch above uses sum(w^2) — using a unit factor here (as this component
        // historically did) left unwindowed energy-mode PSDs high by a factor of N. Power
        // mode is unchanged: sum(w^2)/N is exactly 1 for the rectangular window.
        window_norm_const = static_cast<T>(spectrum_size);
    }
    return T{1} / (sample_rate * window_norm_const);
}

template <typename T>
auto psd<T>::work(composite::immutable_buffer<std::complex<T>> in, composite::timestamp ts,
                  const composite::metadata& md) -> composite::mutable_buffer<T> {
    (void)ts;
    // Per-pool-worker state: the window + PSD kernel for this worker, rebuilt only when the FFT
    // params / sample rate / mode actually change (consecutive packets usually share them).
    thread_local std::unique_ptr<window_t> tl_window;
    thread_local std::size_t tl_size{0};
    thread_local std::string tl_wtype;
    thread_local T tl_sum_sq{0};        // annotation-declared sum(w^2); 0 => not declared
    thread_local T tl_sample_rate{0};
    thread_local bool tl_pbn{true};
    thread_local bool tl_have_key{false};
    thread_local bool tl_norm_valid{false};  // a dedicated flag, not a sample-rate sentinel: a
                                             // real (if pathological) rate equal to the sentinel
                                             // used to skip the recompute after a window change
    thread_local ::work<T> tl_work;  // ::work disambiguates the kernel class from this method
    // Per-pool-worker output pool (exact frame size, recreated on the rare size change).
    thread_local std::shared_ptr<composite::slab_pool<T>> tl_pool;
    thread_local std::size_t tl_pool_size{0};

    // FFT params + sample rate are stamped on the metadata by the upstream fft.
    std::size_t fft_size{0};
    if (md.annotations.contains("fft_size")) {
        // Present-but-unusable is malformed upstream metadata (an absent annotation is a valid
        // no-window request, so it is NOT counted); surface it rather than silently falling
        // back. Unusable covers unparseable (full-string parse; "1024abc" is malformed), zero,
        // absurd (> the fft component's own ceiling — a multi-GB allocation request), and a
        // size that CONTRADICTS the spectrum actually delivered: a local fft always agrees,
        // but the parser annotation-override route lets an operator declare one length while
        // the wire supplies another, and the power normalization divides by the declaration.
        if (!parse_full(md.annotations.at("fft_size").to_string(), fft_size)) {
            fft_size = 0;
        }
        if (fft_size == 0 || fft_size > k_max_annotated_fft_size || fft_size != in.size()) {
            m_bad_metadata->inc();
            fft_size = 0;
        }
    }
    // A window is only meaningful with a usable size (guards the size-0 window whose
    // sum(w^2) = 0 would poison the norm constant).
    std::string wtype;
    if (fft_size != 0 && md.annotations.contains("fft_window")) {
        wtype = md.annotations.at("fft_window").to_string();
    }
    // Preferred normalization contract: the upstream fft DECLARES sum(w^2) directly
    // (fft_window_sum_sq), so psd needs no window recipe of its own and cannot drift from
    // the window fft actually applied (e.g. a window type this component has never heard
    // of). The name-keyed rebuild below remains as the fallback for older producers.
    // Validated in BOTH precisions: a double like 1e300 is positive-finite yet becomes inf
    // when narrowed to float (and a denormal can flush to zero).
    T sum_sq{0};
    if (fft_size != 0 && md.annotations.contains("fft_window_sum_sq")) {
        double declared{0};
        if (parse_full(md.annotations.at("fft_window_sum_sq").to_string(), declared) &&
            positive_finite(declared) && positive_finite(static_cast<T>(declared))) {
            sum_sq = static_cast<T>(declared);
        } else {
            m_bad_metadata->inc();
        }
    }

    // Sanitize the metadata-derived rate ONCE, bit-level (this TU is -ffast-math; see
    // positive_finite), in both precisions: a NaN/inf/negative/overflowing rate must degrade
    // to "unnormalized" (norm 1), not flow into the arithmetic.
    const T sample_rate =
        positive_finite(md.sample_rate) && positive_finite(static_cast<T>(md.sample_rate))
            ? static_cast<T>(md.sample_rate) : T{0};
    // Normalization mode: bind to the packet's OWN annotation (stamped by prepare() at
    // ingest), so a RUNTIME flip with packets in flight cannot make the computation disagree
    // with the metadata that travels with the frame. Snapshot fallback for metadata that
    // never went through prepare().
    bool pbn = *m_pbn.load();
    if (const auto it = md.annotations.find("psd_power_based_normalization"); it != md.annotations.end()) {
        pbn = (it->second.to_string() == "1");
    }

    // Rebuild the window only when (size, type) changes — and only when no declared sum(w^2)
    // makes the window itself unnecessary.
    if (!tl_have_key || fft_size != tl_size || wtype != tl_wtype || sum_sq != tl_sum_sq) {
        if (sum_sq == T{0} && wtype == "BLACKMAN_HARRIS") {
            tl_window = windows::blackman_harris<T>(fft_size, false);
        } else if (sum_sq == T{0} && wtype == "HAMMING") {
            tl_window = windows::hamming<T>(fft_size, false);
        } else {
            tl_window.reset();
        }
        tl_size = fft_size;
        tl_wtype = wtype;
        tl_sum_sq = sum_sq;
        tl_have_key = true;
        tl_norm_valid = false;
    }
    // Recompute the norm const only when window / sample_rate / mode / spectrum size changes
    // (the size matters for the no-window energy mode, where sum(w^2) of the implicit
    // rectangular window is N).
    thread_local std::size_t tl_norm_size{0};
    if (!tl_norm_valid || sample_rate != tl_sample_rate || pbn != tl_pbn || in.size() != tl_norm_size) {
        if (sum_sq != T{0} && sample_rate != T{0}) {
            // Declared-sum contract: norm = 1 / (fs * sum(w^2) [/ N]), no window needed.
            const auto wnc = pbn ? sum_sq / static_cast<T>(fft_size) : sum_sq;
            tl_work.norm_const(T{1} / (sample_rate * wnc));
        } else {
            tl_work.norm_const(compute_norm_const(tl_window.get(), sample_rate, pbn, in.size()));
        }
        tl_sample_rate = sample_rate;
        tl_pbn = pbn;
        tl_norm_size = in.size();
        tl_norm_valid = true;
    }

    // Output buffer: pooled slab in steady state (the full-frame heap allocation per packet
    // becomes a slot pop; only the small type-erasure container is allocated), heap fallback
    // when the pool is exhausted or frames are too large to pool. The budget is divided by
    // the ACTUAL worker count (each worker owns its pool), so total pooled memory stays
    // bounded whatever num_workers is set to. The pooled_frame wrapper (not a bare
    // external_buffer) keeps mutable fan-out deep copies genuinely independent.
    const auto out_size = in.size();
    const auto workers = static_cast<std::size_t>(std::max(1, m_active_workers.load(std::memory_order_relaxed)));
    const auto budget = std::min(k_out_pool_worker_cap_bytes, k_out_pool_total_bytes / workers);
    thread_local std::size_t tl_pool_budget{0};
    if (tl_pool == nullptr || tl_pool_size != out_size || tl_pool_budget != budget) {
        tl_pool = nullptr;
        tl_pool_size = out_size;
        tl_pool_budget = budget;
        const auto frame_bytes = out_size * sizeof(T);
        if (frame_bytes != 0 && budget / frame_bytes >= k_out_pool_min_buffers) {
            tl_pool = composite::slab_pool<T>::create(
                out_size, std::min(budget / frame_bytes, k_out_pool_max_buffers));
        }
    }
    composite::mutable_buffer<T> out;
    if (tl_pool != nullptr) {
        if (auto slab = tl_pool->acquire()) {
            out = composite::mutable_buffer<T>(
                std::make_unique<pooled_frame<T>>(std::move(*slab)));
        }
    }
    if (out.size() != out_size) {
        out = composite::make_aligned_buffer_uninitialized<T>(64, out_size);
    }
    tl_work.process(in, out.data());
    return out;
}

template <typename T>
auto psd<T>::on_workers_resized(int n) -> void {
    m_active_workers.store(n, std::memory_order_relaxed);
}

// Explicit template instantiations
template class psd<float>;
template class psd<double>;

COMPOSITE_REGISTER_COMPONENT([](std::string_view id, const composite::create_args& args)
                                 -> std::shared_ptr<composite::component> {
    const auto type = args.type();
    if (type == "f32") {
        return composite::make_component<psd<float>>(id);
    } else if (type == "f64") {
        return composite::make_component<psd<double>>(id);
    }
    throw std::runtime_error(std::format("unknown type '{}' for psd component", type));
})
