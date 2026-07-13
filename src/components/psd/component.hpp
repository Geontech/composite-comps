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

#include "work.hpp"

#include <composite/core/pipeline_component.hpp>
#include <composite/buffers/buffer.hpp>
#include <composite/buffers/aligned_mem.hpp>
#include <composite/metrics/metrics.hpp>

#include <atomic>
#include <complex>
#include <cstdint>
#include <memory>
#include <string>

// psd is a pipeline_component: the framework worker ingests in order and the slot ring re-
// serialises output to submission order, while the parallel PSD runs on a worker pool. Its window
// is DATA-DRIVEN (rebuilt from each packet's fft_size/fft_window metadata), so work() builds and
// caches the window PER POOL WORKER (keyed by size+type, like fft's thread_local plan) and derives
// the normalization constant locally from the window + the packet's sample_rate + the
// power_based_normalization config snapshot — no shared mutable window/work state.
template <typename T>
class psd : public composite::pipeline_component<composite::immutable_buffer<std::complex<T>>, composite::mutable_buffer<T>> {
    using base = composite::pipeline_component<composite::immutable_buffer<std::complex<T>>, composite::mutable_buffer<T>>;
    using window_t = composite::aligned_mem<T>;

public:
    explicit psd(std::string_view id);
    ~psd() override = default;

    auto property_change_handler(const composite::properties::json& diff) -> void override;

protected:
    // ARRIVAL order, main thread: record the (snapshot) normalization mode onto the metadata.
    auto prepare(composite::metadata& md) -> void override;

    // The parallel stage (pool worker): build/lookup the window for this packet's fft params,
    // compute its norm const, and run the PSD. Per-worker state lives in work() (thread_local).
    // Input is immutable (psd only reads it, producing a fresh real output) so it connects
    // zero-copy to fft's immutable_buffer output.
    auto work(composite::immutable_buffer<std::complex<T>> in, composite::timestamp ts,
              const composite::metadata& md) -> composite::mutable_buffer<T> override;

private:
    /// Normalization constant for a given window + sample rate + mode. Pure (no member state) so
    /// it is safe to call from any pool worker. norm = 1 / (sample_rate * sum(window^2)[/size]).
    static auto compute_norm_const(const window_t* window, T sample_rate, bool power_based) -> T;

    // Properties (num_workers is owned by pipeline_component). power_based_normalization is written
    // by the engine under park; work() (pool threads, not parked) reads the m_pbn atomic snapshot.
    bool m_power_based_normalization{true};
    std::atomic<bool> m_pbn{true};

    // Observability: packets whose fft_size annotation was present but unparseable (a malformed
    // upstream metadata that would otherwise silently fall back to a no-window PSD). In the shared
    // registry, labeled by component id; auto-removed by ~component.
    composite::metrics::counter<uint64_t>* m_bad_metadata{nullptr};

    // MUST be last: stops the pipeline (main worker + pool) before any member above destructs, so
    // a pool worker in work() (reading m_pbn) can't touch freed state. See component.hpp auto_stop.
    composite::component::auto_stop m_auto_stop{*this};

}; // class psd

// Explicit template instantiations (defined in component.cpp)
extern template class psd<float>;
extern template class psd<double>;
