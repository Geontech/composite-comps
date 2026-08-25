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

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <span>
#include <stdexcept>
#include <sys/uio.h>
#include <unistd.h>

file_writer::file_writer(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_property("filename", m_filename);
    add_property("num_bytes", m_num_bytes).units("bytes");
    m_bytes_written = &create_counter("file_writer.bytes_written", "Bytes written to the capture file");
}

file_writer::~file_writer() {
    if (m_file != -1) {
        ::close(m_file);
    }
}

auto file_writer::initialize() -> void {
    // Contract: every initialize() starts a FRESH capture — O_TRUNC, byte counter reset (append
    // is not offered in v0.5). A previous descriptor (re-initialize) must not leak.
    if (m_file != -1) {
        ::close(m_file);
        m_file = -1;
    }
    m_file = ::open(m_filename.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644);
    if (m_file < 0) {
        // Fail the load loudly. The old code stored -1 and every later ::write failed silently:
        // the component ran forever reporting NORMAL while writing nothing.
        throw std::runtime_error(
            std::format("file_writer '{}': cannot open '{}' for writing: {}", id(), m_filename, strerror(errno)));
    }
    m_total_bytes = 0;
}

auto file_writer::process() -> composite::retval {
    using enum composite::retval;
    // Drain a bounded batch with one ring-head publication and write it with ONE writev():
    // a disk sink at capture rates otherwise pays a syscall per packet.
    const auto count = m_in_port.get_batch(std::span{m_input_batch});
    if (count == 0) {
        return NOOP;  // base promotes NOOP-at-end-of-stream to FINISH; nothing buffered to flush
    }
    if (m_file < 0) {
        // initialize() was skipped or failed; running on would silently discard the stream.
        throw std::runtime_error(std::format("file_writer '{}': no open file (initialize() not run?)", id()));
    }

    // Gather the batch into iovecs, honoring the optional num_bytes cap. The cap can land
    // mid-buffer, so the last entry may be clamped; packets past the cap are dropped (the
    // capture is complete — same semantics the single-packet path always had).
    std::array<struct iovec, INPUT_BATCH_SIZE> iov;
    std::size_t niov = 0;
    std::size_t batch_bytes = 0;
    for (std::size_t i = 0; i < count; ++i) {
        auto& [data, ts, meta] = m_input_batch[i];
        (void)ts;
        (void)meta;
        std::size_t take = data.size();  // immutable_buffer<uint8_t>: 1 byte per element
        if (m_num_bytes > 0) {
            const uint64_t remaining = m_num_bytes - m_total_bytes - batch_bytes;
            if (take > remaining) {
                take = static_cast<std::size_t>(remaining);
            }
        }
        if (take > 0) {
            // iov_base is void*; writev never writes through it for O_WRONLY output.
            iov[niov++] = {.iov_base = const_cast<uint8_t*>(data.data()), .iov_len = take};
            batch_bytes += take;
        }
        if (m_num_bytes > 0 && m_total_bytes + batch_bytes >= m_num_bytes) {
            break;  // cap reached: anything still in the batch is past the capture
        }
    }

    // Complete the write: a short write is normal kernel behaviour (signals, quotas, some
    // filesystems), and treating it as done silently TRUNCATED the stream mid-buffer while the
    // byte counter advanced. Retry EINTR and resume a partial writev at the exact byte it
    // stopped; surface any real error as a component error finish (throwing is the framework's
    // post-commit failure contract — the worker records finish_reason::error with this message
    // as finish_error).
    std::size_t idx = 0;
    std::size_t written_total = 0;
    while (idx < niov) {
        const auto n = ::writev(m_file, &iov[idx], static_cast<int>(niov - idx));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error(std::format("file_writer '{}': write to '{}' failed after {} bytes: {}", id(),
                                                 m_filename, m_total_bytes + written_total, strerror(errno)));
        }
        auto advance = static_cast<std::size_t>(n);
        written_total += advance;
        while (idx < niov && advance >= iov[idx].iov_len) {
            advance -= iov[idx].iov_len;
            ++idx;
        }
        if (idx < niov && advance > 0) {
            iov[idx].iov_base = static_cast<uint8_t*>(iov[idx].iov_base) + advance;
            iov[idx].iov_len -= advance;
        }
    }
    m_total_bytes += written_total;
    if (m_bytes_written != nullptr) { m_bytes_written->add(written_total); }

    // Release the drained slots now that the kernel has copied the bytes: entries left in the
    // member array would pin upstream pool buffers until a later batch overwrites them.
    for (std::size_t i = 0; i < count; ++i) {
        m_input_batch[i] = {};
    }

    if (m_num_bytes > 0 && m_total_bytes >= m_num_bytes) {
        return FINISH;
    }
    return NORMAL;
}

COMPOSITE_REGISTER_SIMPLE(file_writer)
