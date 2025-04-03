/*
 * Copyright (C) 2024 Geon Technologies, LLC
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

#include "overlay.hpp"

#include <bit>

namespace overlay {

namespace sdds {

overlay:: overlay(std::span<const uint8_t> data) : m_data(data) {}

auto overlay::pp_id() const -> bool {
    return m_data[0] & 0x20;
}

auto overlay::is_parity() const -> bool {
    return pp_id() && ((seq_num() % 32) == 31);
}

auto overlay::bps() const -> uint8_t {
    return m_data[1] & 0x1F;
}

auto overlay::seq_num() const -> uint16_t {
    return std::byteswap(*reinterpret_cast<const uint16_t*>(m_data.data() + 2));
}

auto overlay::ttv() const -> bool {
    return m_data[4] & 0x40;
}

auto overlay::ttag() const -> uint64_t {
    return std::byteswap(*reinterpret_cast<const uint64_t*>(m_data.data() + 8));
}

auto overlay::ttage() const -> uint32_t {
    return std::byteswap(*reinterpret_cast<const uint32_t*>(m_data.data() + 16));
}

auto overlay::secs() const -> uint32_t {
    return static_cast<uint32_t>(ttag() / PS250_PER_SEC);
}

auto overlay::psecs() const -> uint64_t {
    return (ttag() * uint64_t{250u}) - (static_cast<uint64_t>(secs()) * uint64_t{1'000'000'000'000u});
}

template<typename T>
auto overlay::payload() const -> std::span<const T> {
    auto data = reinterpret_cast<const T*>(m_data.data() + DATA_IDX);
    return std::span<const T>(data, DATA_LEN / sizeof(T));
}

} // namespace sdds

namespace v49 {

overlay::overlay(std::span<const uint8_t> data) : m_data(data) {
    m_header.unpack_from(m_data.data());
    auto curr_idx = m_header.size();
    if (m_header.packet_type() != vrtgen::packing::PacketType::SIGNAL_DATA) {
        auto stream_id = *reinterpret_cast<const uint32_t*>(m_data.data() + curr_idx);
        m_stream_id = std::byteswap(stream_id);
        curr_idx += sizeof(stream_id);
    }
    if (m_header.class_id_enable()) {
        m_class_id = vrtgen::packing::ClassIdentifier{};
        m_class_id->unpack_from(m_data.data() + curr_idx);
        curr_idx += m_class_id->size();
    }
    if (m_header.tsi() != vrtgen::packing::TSI::NONE) {
        auto ts = *reinterpret_cast<const uint32_t*>(m_data.data() + curr_idx);
        m_int_ts = std::byteswap(ts);
        curr_idx += sizeof(ts);
    }
    if (m_header.tsf() != vrtgen::packing::TSF::NONE) {
        auto ts = *reinterpret_cast<const uint64_t*>(m_data.data() + curr_idx);
        m_int_ts = std::byteswap(ts);
        curr_idx += sizeof(ts);
    }
    if (is_data()) {
        m_positions["payload"] = curr_idx;
        auto data_header = vrtgen::packing::DataHeader{};
        data_header.unpack_from(m_data.data());
        if (data_header.trailer_included()) {
            m_trailer = vrtgen::packing::Trailer{};
            auto pos = (m_header.packet_size() - 1) * sizeof(uint32_t)/*word size*/;
            m_trailer->unpack_from(m_data.data() + pos);
        }
    }
}

auto overlay::is_data() const -> bool {
    using enum vrtgen::packing::PacketType;
    return m_header.packet_type() == SIGNAL_DATA || m_header.packet_type() == SIGNAL_DATA_STREAM_ID;
}

auto overlay::is_context() const -> bool {
    using enum vrtgen::packing::PacketType;
    return m_header.packet_type() == CONTEXT;
}

auto overlay::header() const -> const vrtgen::packing::Header& {
    return m_header;
}

auto overlay::stream_id() const -> std::optional<uint32_t> {
    return m_stream_id;
}

auto overlay::class_id() const -> const std::optional<vrtgen::packing::ClassIdentifier>& {
    return m_class_id;
}

auto overlay::integer_timestamp() const -> std::optional<uint32_t> {
    return m_int_ts;
}

auto overlay::fractional_timestamp() const -> std::optional<uint64_t> {
    return m_frac_ts;
}

template<typename T>
auto overlay::payload() const -> std::span<const T> {
    if (!m_positions.contains("payload")) {
        return {};
    }
    auto pos = m_positions.at("payload");
    auto data = reinterpret_cast<const T*>(m_data.data() + pos);
    return std::span<const T>(data, payload_size() / sizeof(T));
}

auto overlay::payload_size() const -> size_t {
    if (!m_positions.contains("payload")) {
        return {};
    }
    auto size = (m_header.packet_size() * sizeof(uint32_t)/*word size*/) - m_positions.at("payload");
    if (m_positions.contains("trailer")) {
        size -= sizeof(uint32_t);
    }
    return size;
}

auto overlay::payload_start() const -> size_t {
    if (!m_positions.contains("payload")) {
        return {};
    }
    return m_positions.at("payload");
}

} // namespace v49
} // namespace overlay
