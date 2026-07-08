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

#include "component.hpp"

#include <composite/core/register.hpp>

#include <fcntl.h>
#include <unistd.h>

file_writer::file_writer(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_property("filename", m_filename);
    add_property("num_bytes", m_num_bytes).units("bytes");
}

file_writer::~file_writer() {
    if (m_file != -1) {
        ::close(m_file);
    }
}

auto file_writer::initialize() -> void {
    m_file = ::open(m_filename.c_str(), O_CREAT|O_TRUNC|O_WRONLY, 0644);
}

auto file_writer::process() -> composite::retval {
    using enum composite::retval;
    auto pkt = m_in_port.try_get();
    if (!pkt) {
        return NOOP;  // base promotes NOOP-at-end-of-stream to FINISH; nothing buffered to flush
    }
    auto& [data, ts, meta] = *pkt;
    (void)ts;
    (void)meta;
    // Append the incoming byte stream, honoring the optional num_bytes cap. The cap can land
    // mid-buffer, so clamp the write to the remaining budget before issuing it.
    std::size_t to_write = data.size();  // immutable_buffer<uint8_t>: 1 byte per element
    if (m_num_bytes > 0) {
        const uint64_t remaining = m_num_bytes - m_total_bytes;
        if (to_write > remaining) {
            to_write = static_cast<std::size_t>(remaining);
        }
    }
    if (auto num_written = ::write(m_file, data.data(), to_write); num_written != -1) {
        m_total_bytes += static_cast<uint64_t>(num_written);
    }
    if (m_num_bytes > 0 && m_total_bytes >= m_num_bytes) {
        return FINISH;
    }
    return NORMAL;
}

COMPOSITE_REGISTER_SIMPLE(file_writer)
