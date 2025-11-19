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
#include <string_view>

template <typename T>
framer<T>::framer() : composite::component("framer") {
    using enum composite::properties::config_type;
    add_port(&m_in_port);
    add_port(&m_out_port);

    add_property("frame_size", &m_frame_size)
        .units("samples")
        .configurability(INITIALIZE)
        .change_listener([this]() {
            return m_frame_size > 0;
        });

    add_property("overlap", &m_overlap)
        .units("samples")
        .configurability(INITIALIZE)
        .change_listener([this]() {
            return m_overlap < m_frame_size;
        });

    add_property("frame_count", &m_frame_count)
        .configurability(INITIALIZE)
        .change_listener([this]() {
            return m_frame_count >= 2;
        });
}

template <typename T>
auto framer<T>::reset_state() -> void {
    m_in_port.clear();
    m_metadata_ready = false;
    m_source_format = source_format::unknown;
    m_input_stride = 0;
    m_next_frame_start = 0;
    m_timestamp_initialized = false;
    m_partial_sample.clear();
}

template <typename T>
auto framer<T>::initialize_pool() -> void {
    if (m_frame_size == 0) {
        return;
    }

    auto frames = std::max<uint32_t>(m_frame_count, 2);
    m_pool = std::make_shared<framer_pool<T>>(m_frame_size, m_overlap, frames);

    reset_state();
}

template <typename T>
auto framer<T>::property_change_handler() -> void {
    initialize_pool();
}

template <typename T>
auto framer<T>::detect_source_format(const composite::metadata& meta) const -> source_format {
    const auto bit_width = meta.format.bit_width;
    const auto type = meta.format.type;
    const auto is_complex = meta.format.is_complex;

    if (!is_complex && bit_width == 8 && type == composite::data_type::signed_integer) {
        return source_format::real_i8;
    }

    if (is_complex && bit_width == 8 && type == composite::data_type::signed_integer) {
        return source_format::complex_i8;
    }

    if (is_complex && bit_width == 16 && type == composite::data_type::signed_integer) {
        return source_format::complex_i16;
    }

    if (is_complex && bit_width == 32 && type == composite::data_type::floating_point) {
        return source_format::complex_cf32;
    }

    return source_format::unknown;
}

template <typename T>
auto framer<T>::bytes_per_input_sample() const -> std::size_t {
    switch (m_source_format) {
        case source_format::real_i8: return 1;
        case source_format::complex_i8: return 2;
        case source_format::complex_i16: return 4;
        case source_format::complex_cf32: return 8;
        default: break;
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

    // Automatically determine if byte swapping is needed
    bool needs_swap = (m_metadata.format.endianness != std::endian::native);

    // Create appropriate converter based on source format and output type
    if constexpr (std::is_same_v<scalar_t, float>) {
        switch (m_source_format) {
            case source_format::real_i8:
            case source_format::complex_i8:
                m_converter = std::make_unique<converter<int8_t, float>>(needs_swap);
                break;
            case source_format::complex_i16:
                m_converter = std::make_unique<converter<int16_t, float>>(needs_swap);
                break;
            case source_format::complex_cf32:
                m_converter = std::make_unique<converter<uint32_t, float>>(needs_swap);
                break;
            default:
                break;
        }
    } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
        switch (m_source_format) {
            case source_format::real_i8:
            case source_format::complex_i8:
                m_converter = std::make_unique<converter<int8_t, int16_t>>(needs_swap);
                break;
            case source_format::complex_i16:
                m_converter = std::make_unique<converter<int16_t, int16_t>>(needs_swap);
                break;
            default:
                break;
        }
    }
}

template <typename T>
auto framer<T>::handle_metadata(const composite::metadata& meta) -> void {
    auto fmt = detect_source_format(meta);
    if (fmt == source_format::unknown) {
        logger()->warn("framer: unsupported input format (bit_width={}, type={})",
                       meta.format.bit_width, static_cast<int>(meta.format.type));
        m_metadata_ready = false;
        return;
    }

    auto changed = (!m_metadata_ready) || (fmt != m_source_format) || (meta != m_metadata);
    m_metadata = meta;
    m_source_format = fmt;
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
    m_partial_sample.clear();
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
    std::size_t offset = 0;

    // Handle partial sample from previous buffer
    if (!m_partial_sample.empty()) {
        auto needed = m_input_stride - m_partial_sample.size();
        auto take = std::min<std::size_t>(needed, byte_count);
        m_partial_sample.insert(m_partial_sample.end(), bytes, bytes + take);
        offset += take;

        if (m_partial_sample.size() == m_input_stride) {
            if (!m_pool->write_samples(m_partial_sample.data(), 1, m_converter.get(), m_input_stride)) {
                logger()->warn("framer: failed to write partial sample, dropping");
            }
            m_partial_sample.clear();
        } else {
            return; // Still incomplete, wait for more data
        }
    }

    // Process complete samples in batch
    std::size_t remaining_bytes = byte_count - offset;
    std::size_t complete_samples = remaining_bytes / m_input_stride;

    if (complete_samples > 0) {
        const uint8_t* input_ptr = bytes + offset;
        if (!m_pool->write_samples(input_ptr, complete_samples, m_converter.get(), m_input_stride)) {
            logger()->warn("framer: failed to write {} samples, dropping", complete_samples);
        }
        offset += complete_samples * m_input_stride;
    }

    // Save partial sample for next buffer
    if (offset < byte_count) {
        m_partial_sample.assign(bytes + offset, bytes + byte_count);
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
    auto create(std::string_view type) -> std::shared_ptr<composite::component> {
        if (type == "cf32") {
            return std::make_shared<framer<std::complex<float>>>();
        } else if (type == "ci16") {
            return std::make_shared<framer<std::complex<int16_t>>>();
        }
        return std::make_shared<framer<std::complex<float>>>();
    }
}
