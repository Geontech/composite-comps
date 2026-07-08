/*
 * Copyright (C) 2025 Geon Technologies, LLC
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

#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>

#include <composite/buffers/buffer.hpp>
#include <composite/buffers/overlap_ring.hpp>
#include <composite/composite.hpp>
#include <composite/metrics/metrics.hpp>
#include <composite/properties/config.hpp>

#include "convert_variant.hpp"

namespace framer_detail {
inline constexpr auto PS_PER_SEC = 1'000'000'000'000ULL;

// The framer's configuration as ONE reflected struct. The fields are
// type-independent, so a single struct serves every framer<T> instantiation. add_config()
// projects the fields back onto the property-set namespace, so the wire contract is
// unchanged (PATCH {"frame_size": 1024} still works); but the whole struct is the
// validate/commit unit — a single-field write re-checks the cross-field invariants
// (overlap < frame_size) against the proposed combined state, and the on_apply reaction
// (rebuild the pool) runs once at the worker loop-top.
struct framer_config {
    uint32_t frame_size{};
    uint32_t overlap{};
    uint32_t frame_count{64};
    COMPOSITE_FIELDS(framer_config,
        (frame_size, unit("samples")),
        (overlap, unit("samples")),
        frame_count);
};
} // namespace framer_detail

/**
 * @brief Reframes a byte stream into fixed-size, optionally overlapping frames of complex
 *        samples (T = std::complex<float> or std::complex<int16_t>), converting the input
 *        wire format into T on the way into the framing ring.
 *
 * The supported (input format, T) combinations — and their strides and converters — live in
 * one table: `supported_input_formats<T::value_type>` (convert_variant.hpp). Conversion is
 * fused into the ring write, so each input sample is converted exactly once even with
 * overlapping frames.
 *
 * Conversion semantics worth knowing for spectral processing:
 *  - Integer inputs are widened numerically, NOT normalized: i8 -> float yields values in
 *    [-128, 127], i16 -> float in [-32768, 32767]. Downstream power/dB levels therefore
 *    carry a constant, input-format-dependent offset.
 *  - Real inputs are expanded to complex with imag = 0, which produces a conjugate-symmetric
 *    spectrum downstream.
 *
 * Frame timestamps are extrapolated from an anchor (first buffer's timestamp) by sample
 * count and sample rate; the anchor is re-established after any sample drop and on any
 * sample-rate change so extrapolation never drifts from wall time.
 */
template <typename T>
class framer : public composite::component {
    friend struct FramerTestFixture;
    friend class framer_integration_tests;
    using input_port_t = composite::input_port<composite::immutable_buffer<uint8_t>>;
    using output_port_t = composite::output_port<composite::immutable_buffer<T>>;

public:
    explicit framer(std::string_view);
    ~framer() override = default;

    auto process() -> composite::retval override;
    // At end-of-stream, zero-pad + emit the final partial frame still buffered in the ring.
    auto on_end_of_stream() -> void override;

private:
    auto reset_state() -> void;
    auto initialize_pool() -> void;
    auto handle_metadata(const composite::metadata_ptr& meta) -> void;
    auto configure_output_metadata() -> void;
    auto process_buffer(const composite::immutable_buffer<uint8_t>& buffer, composite::timestamp ts) -> void;
    auto try_emit_frames() -> void;
    auto compute_frame_timestamp(std::size_t start_sample) const -> composite::timestamp;

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties (grouped as one reflected config<T>)
    composite::config<framer_detail::framer_config> m_cfg{};

    // Drop statistics, exposed through the shared metrics registry (labeled by component id;
    // the series are removed by ~component's label cleanup).
    composite::metrics::counter<uint64_t>* m_samples_dropped{nullptr};
    composite::metrics::counter<uint64_t>* m_drops_batch_too_large{nullptr};
    composite::metrics::counter<uint64_t>* m_drops_backpressure{nullptr};

    // Metadata tracking. Metadata arrives as a shared immutable instance that upstream
    // latches, so handle_metadata() early-outs on pointer identity (the common case) —
    // no copy, no deep compare. m_metadata is the working output value (timestamps read
    // sample_rate from it); m_out_metadata is the shared instance frames are sent with,
    // rebuilt only when the input metadata changes.
    composite::metadata m_metadata{};
    composite::metadata_ptr m_out_metadata{};
    composite::metadata_ptr m_last_input_meta{};
    bool m_metadata_ready{false};
    composite::data_format m_input_format{};
    std::size_t m_input_stride{};

    // Overlapped framing ring (framework-provided): manages the sample ring + frame slots.
    std::shared_ptr<composite::overlap_ring<T>> m_pool;
    std::size_t m_next_frame_start{};  // Next frame start sample

    // Type converter (AVX-optimized) - converts to scalar type (float or int16_t)
    std::optional<converter_variant<typename T::value_type>> m_converter;

    // Timestamp tracking: frame timestamps extrapolate from this anchor. Cleared (forcing a
    // re-anchor on the next buffer) after a drop and on a sample-rate change, since either
    // one breaks the sample-index -> wall-time mapping.
    composite::timestamp m_timestamp_origin{};
    std::size_t m_timestamp_origin_sample{};
    bool m_timestamp_initialized{false};

    // MUST be last: stops the framework worker before any member above destructs. The base
    // ~component stops too late (after derived members are gone), so a still-running worker's
    // process() could touch freed state. (Documented pattern; see component.hpp auto_stop.)
    composite::component::auto_stop m_auto_stop{*this};
};
