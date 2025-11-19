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

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include <composite/buffers/aligned_mem.hpp>
#include <composite/buffers/buffer.hpp>
#include <composite/buffers/external_buffer.hpp>
#include <composite/composite.hpp>

#include "convert.hpp"
#include "framer_pool.hpp"

namespace framer_detail {
inline constexpr auto PS_PER_SEC = 1'000'000'000'000ULL;
} // namespace framer_detail

template <typename T>
class framer : public composite::component {
    using input_port_t = composite::input_port<composite::immutable_buffer<uint8_t>>;
    using output_port_t = composite::output_port<composite::immutable_buffer<T>>;

public:
    framer();
    ~framer() override = default;

    auto property_change_handler() -> void override;
    auto process() -> composite::retval override;

private:
    enum class source_format {
        unknown,
        real_i8,
        complex_i8,
        complex_i16,
        complex_cf32
    };

    auto reset_state() -> void;
    auto initialize_pool() -> void;
    auto handle_metadata(const composite::metadata& meta) -> void;
    auto detect_source_format(const composite::metadata& meta) const -> source_format;
    auto bytes_per_input_sample() const -> std::size_t;
    auto configure_output_metadata() -> void;
    auto create_converter() -> void;
    auto process_buffer(const composite::immutable_buffer<uint8_t>& buffer, composite::timestamp ts) -> void;
    auto try_emit_frames() -> void;
    auto compute_frame_timestamp(std::size_t start_sample) const -> composite::timestamp;

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    uint32_t m_frame_size{};
    uint32_t m_overlap{};
    uint32_t m_frame_count{64};
    bool m_byteswap{true};

    // Metadata tracking
    composite::metadata m_metadata{};
    bool m_metadata_ready{false};
    source_format m_source_format{source_format::unknown};
    std::size_t m_input_stride{};

    // Type converter (AVX-optimized) - converts to scalar type (float or int16_t)
    std::unique_ptr<converter_base<typename T::value_type>> m_converter;

    // Frame pool manages ring buffer and slot allocation
    std::shared_ptr<framer_pool<T>> m_pool;
    std::size_t m_next_frame_start{};  // Next frame start sample

    // Partial sample handling (when incoming buffers break sample boundaries)
    std::vector<uint8_t> m_partial_sample;

    // Timestamp tracking
    composite::timestamp m_timestamp_origin{};
    std::size_t m_timestamp_origin_sample{};
    bool m_timestamp_initialized{false};
};
