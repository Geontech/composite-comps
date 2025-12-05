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

#include "component.hpp"

#include <complex>
#include <cstring>
#include <format>
#include <string_view>

template <typename T>
framer<T>::framer(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_port(&m_out_port);

    add_property("frame_size", &m_frame_size)
        .units("samples")
        .configurability(composite::properties::config_type::INITIALIZE)
        .change_listener([this]() {
            return m_frame_size > 0;
        });

    add_property("overlap", &m_overlap)
        .units("samples")
        .configurability(composite::properties::config_type::INITIALIZE)
        .change_listener([this]() {
            return m_overlap < m_frame_size;
        });

    add_property("frame_count", &m_frame_count)
        .configurability(composite::properties::config_type::INITIALIZE)
        .change_listener([this]() {
            return m_frame_count >= 2;
        });
}

template <typename T>
auto framer<T>::reset_state() -> void {
    m_in_port.clear();
    m_metadata_ready = false;
    m_input_format = {};  // Reset to default
    m_input_stride = 0;
    m_next_frame_start = 0;
    m_timestamp_initialized = false;
}

template <typename T>
auto framer<T>::initialize_pool() -> void {
    if (m_frame_size == 0) {
        logger()->error("framer: frame_size must be > 0");
        return;
    }

    if (m_overlap >= m_frame_size) {
        logger()->error("framer: overlap ({}) must be < frame_size ({})", m_overlap, m_frame_size);
        return;
    }

    auto frames = std::max<uint32_t>(m_frame_count, 2);

    try {
        m_pool = std::make_shared<framer_pool<T>>(m_frame_size, m_overlap, frames);
        logger()->debug("framer: initialized pool with frame_size={}, overlap={}, frame_count={}, "
                       "hop_size={}, ring_size={}",
                       m_frame_size, m_overlap, frames,
                       m_frame_size - m_overlap, frames * (m_frame_size - m_overlap) + m_overlap);
    } catch (const std::exception& e) {
        logger()->error("framer: failed to initialize pool: {}", e.what());
        m_pool.reset();
        return;
    }

    reset_state();
}

template <typename T>
auto framer<T>::property_change_handler() -> void {
    initialize_pool();
}

template <typename T>
auto framer<T>::is_supported_input_format(const composite::data_format& fmt) const -> bool {
    const auto bit_width = fmt.bit_width;
    const auto type = fmt.type;
    const auto is_complex = fmt.is_complex;

    // Real i8
    if (!is_complex && bit_width == 8 && type == composite::data_type::signed_integer) {
        return true;
    }

    // Complex i8
    if (is_complex && bit_width == 8 && type == composite::data_type::signed_integer) {
        return true;
    }

    // Complex i16
    if (is_complex && bit_width == 16 && type == composite::data_type::signed_integer) {
        return true;
    }

    // Complex cf32
    if (is_complex && bit_width == 32 && type == composite::data_type::floating_point) {
        return true;
    }

    return false;
}

template <typename T>
auto framer<T>::bytes_per_input_sample() const -> std::size_t {
    const auto bit_width = m_input_format.bit_width;
    const auto is_complex = m_input_format.is_complex;

    // Real i8: 1 byte
    if (!is_complex && bit_width == 8) {
        return 1;
    }

    // Complex i8: 2 bytes (I + Q)
    if (is_complex && bit_width == 8) {
        return 2;
    }

    // Complex i16: 4 bytes (I + Q)
    if (is_complex && bit_width == 16) {
        return 4;
    }

    // Complex cf32: 8 bytes (I + Q as floats)
    if (is_complex && bit_width == 32) {
        return 8;
    }

    return 0;
}

template <typename T>
auto framer<T>::configure_output_metadata() -> void {
    if constexpr (std::is_same_v<T, std::complex<float>>) {
        m_metadata.format.type = composite::data_type::floating_point;
        m_metadata.format.bit_width = 32; // per component
    } else {
        m_metadata.format.type = composite::data_type::signed_integer;
        m_metadata.format.bit_width = 16;
    }
    m_metadata.format.is_complex = true;
    m_metadata.format.endianness = std::endian::native;
}

template <typename T>
auto framer<T>::create_converter() -> void {
    using scalar_t = typename T::value_type;  // float or int16_t from complex<T>

    // Automatically determine if byte swapping is needed based on INPUT endianness
    bool needs_swap = (m_input_format.endianness != std::endian::native);

    const auto bit_width = m_input_format.bit_width;
    const auto type = m_input_format.type;
    const auto is_complex = m_input_format.is_complex;

    // Create appropriate converter based on input format and output type
    if constexpr (std::is_same_v<scalar_t, float>) {
        // Output type is float
        if (bit_width == 8 && type == composite::data_type::signed_integer) {
            // i8 -> float (handles both real and complex)
            m_converter = converter<int8_t, float>(needs_swap);
        } else if (is_complex && bit_width == 16 && type == composite::data_type::signed_integer) {
            // complex i16 -> float
            m_converter = converter<int16_t, float>(needs_swap);
        } else if (is_complex && bit_width == 32 && type == composite::data_type::floating_point) {
            // complex cf32 -> float (passthrough)
            m_converter = converter<uint32_t, float>(needs_swap);
        } else {
            m_converter = std::nullopt;
        }
    } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
        // Output type is int16_t
        if (bit_width == 8 && type == composite::data_type::signed_integer) {
            // i8 -> int16_t (handles both real and complex)
            m_converter = converter<int8_t, int16_t>(needs_swap);
        } else if (is_complex && bit_width == 16 && type == composite::data_type::signed_integer) {
            // complex i16 -> int16_t (passthrough)
            m_converter = converter<int16_t, int16_t>(needs_swap);
        } else {
            m_converter = std::nullopt;
        }
    }
}

template <typename T>
auto framer<T>::handle_metadata(const composite::metadata& meta) -> void {
    if (!is_supported_input_format(meta.format)) {
        logger()->warn("framer: unsupported input format (bit_width={}, type={}, complex={})",
                       meta.format.bit_width, static_cast<int>(meta.format.type), meta.format.is_complex);
        m_metadata_ready = false;
        return;
    }

    auto changed = (!m_metadata_ready) || (meta.format != m_input_format) || (meta != m_metadata);

    // Save input format before overwriting metadata
    m_input_format = meta.format;
    m_metadata = meta;
    m_metadata_ready = true;

    m_input_stride = bytes_per_input_sample();
    if (m_input_stride == 0) {
        logger()->error("framer: computed input stride is zero");
        m_metadata_ready = false;
        return;
    }

    configure_output_metadata();
    create_converter();

    if (changed) {
        logger()->trace("framer: updated metadata:\n{}", m_metadata.to_string());
        m_out_port.send_metadata(m_metadata);
    }
}

template <typename T>
auto framer<T>::process_buffer(const composite::immutable_buffer<uint8_t>& buffer, composite::timestamp ts) -> void {
    if (!m_metadata_ready || m_input_stride == 0 || !m_converter || !m_pool) {
        return;
    }

    if (!m_timestamp_initialized) {
        m_timestamp_origin = ts;
        m_timestamp_origin_sample = m_pool->head();
        m_timestamp_initialized = true;
    }

    auto span = buffer.as_span();
    const uint8_t* bytes = span.data();
    std::size_t byte_count = span.size();
    const bool is_complex = m_input_format.is_complex;

    // Process complete samples in batch
    std::size_t complete_samples = byte_count / m_input_stride;

    if (complete_samples > 0) {
        if (!m_pool->write_samples(bytes, complete_samples, &m_converter.value(), m_input_stride, is_complex)) {
            auto diag = m_pool->get_diagnostics();

            m_samples_dropped += complete_samples;

            switch (diag.last_drop_reason) {
                case decltype(m_pool)::element_type::drop_reason::BATCH_TOO_LARGE:
                    ++m_drops_batch_too_large;
                    logger()->error("framer: failed to write {} samples (BATCH TOO LARGE) - "
                                   "batch size exceeds ring_size ({}). Dropping samples.",
                                   complete_samples, diag.ring_size);
                    break;
                case decltype(m_pool)::element_type::drop_reason::BACKPRESSURE_TIMEOUT:
                    ++m_drops_backpressure;
                    logger()->warn("framer: failed to write {} samples (BACKPRESSURE TIMEOUT) - "
                                  "downstream holding frames. Dropping samples. "
                                  "Slots in use: {}/{}, available space: {} samples, write_head: {}, "
                                  "oldest protected: {}",
                                  complete_samples, diag.slots_in_use, diag.total_slots,
                                  diag.available_space, diag.write_head, diag.oldest_protected_sample);
                    break;
                default:
                    logger()->warn("framer: failed to write {} samples (UNKNOWN), dropping", complete_samples);
                    break;
            }
        }
    }

    // If there are leftover bytes, it indicates a non-sample-aligned buffer, which is now considered an error.
    if (byte_count % m_input_stride != 0) {
        logger()->warn("framer: received a buffer that is not aligned to sample boundaries ({} bytes, stride {}). "
                       "Partial sample handling has been disabled, so leftover bytes will be discarded.",
                       byte_count, m_input_stride);
    }
}

template <typename T>
auto framer<T>::try_emit_frames() -> void {
    if (!m_pool) {
        return;
    }

    std::size_t head = m_pool->head();
    std::size_t hop = m_pool->hop_size();

    while (head >= m_next_frame_start + m_frame_size) {
        auto buffer_opt = m_pool->try_emit_frame(m_next_frame_start);
        if (!buffer_opt.has_value()) {
            break;  // Slot busy, stop emitting
        }

        auto ts = compute_frame_timestamp(m_next_frame_start);
        m_out_port.send_data(std::move(buffer_opt.value()), ts);

        m_next_frame_start += hop;
    }
}

template <typename T>
auto framer<T>::compute_frame_timestamp(std::size_t start_sample) const -> composite::timestamp {
    if (!m_timestamp_initialized || m_metadata.sample_rate <= 0.0) {
        return m_timestamp_origin;
    }

    // Handle potential underflow by checking if start_sample is less than origin
    if (start_sample < m_timestamp_origin_sample) {
        logger()->warn("framer: start_sample ({}) < origin_sample ({}), using origin timestamp",
                       start_sample, m_timestamp_origin_sample);
        return m_timestamp_origin;
    }

    auto delta_samples = static_cast<long double>(start_sample - m_timestamp_origin_sample);
    auto ps_per_sample = static_cast<long double>(framer_detail::PS_PER_SEC) / m_metadata.sample_rate;
    auto ps_total = static_cast<uint64_t>(delta_samples * ps_per_sample);

    auto ts = m_timestamp_origin;
    ts.seconds += ps_total / framer_detail::PS_PER_SEC;
    auto picoseconds = ts.picoseconds + (ps_total % framer_detail::PS_PER_SEC);
    if (picoseconds >= framer_detail::PS_PER_SEC) {
        ts.seconds += 1;
        picoseconds -= framer_detail::PS_PER_SEC;
    }
    ts.picoseconds = picoseconds;
    return ts;
}


template <typename T>
auto framer<T>::process() -> composite::retval {
    using enum composite::retval;

    auto [buffer, ts, meta] = m_in_port.get_data();
    if (!buffer) {
        return NORMAL;
    }

    if (meta.has_value()) {
        handle_metadata(meta.value());
    }

    if (!m_metadata_ready) {
        return NORMAL;
    }

    process_buffer(buffer, ts);
    try_emit_frames();

    return NORMAL;
}

extern "C" {
    auto create(std::string_view id, std::string_view type) -> std::shared_ptr<composite::component> {
        if (type == "cf32") {
            return std::make_shared<framer<std::complex<float>>>(id);
        } else if (type == "ci16") {
            return std::make_shared<framer<std::complex<int16_t>>>(id);
        }
        throw std::runtime_error(std::format("unknown type '{}' for framer component", type));
    }
}
