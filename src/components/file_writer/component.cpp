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
#include "overlay.hpp"

#include <fcntl.h>
#include <fstream>
#include <sys/uio.h>

file_writer::file_writer() : composite::component("file_writer") {
    add_port(m_in_port.get());
    add_property("filename", &m_filename);
    add_property("num_bytes", &m_num_bytes);
}

file_writer::~file_writer() {
    close(m_file);
}

auto file_writer::initialize() -> void {
    m_file = open(m_filename.c_str(), O_CREAT|O_TRUNC|O_WRONLY, 0644);
}

auto file_writer::process() -> composite::retval {
    auto [data, ts] = m_in_port->get_data();
    if (data == nullptr) {
        return composite::retval::NOOP;
    }
    auto curr_total = m_total_bytes;
    using iovec_t = struct iovec;
    auto iovecs = std::vector<iovec_t>{};
    for (auto& v : *data) {
        auto packet = overlay::v49::overlay(v);
        auto& header = packet.header();
        if (!overlay::v49::is_data(header)) {
            continue;
        }
        auto payload = packet.payload<uint8_t>();
        iovecs.emplace_back(const_cast<uint8_t*>(payload.data()), payload.size_bytes());
        curr_total += payload.size_bytes();
        if (curr_total >= m_num_bytes) {
            iovecs.back().iov_len -= (curr_total - m_num_bytes);
            break;
        }
    }
    if (auto num_written = writev(m_file, iovecs.data(), iovecs.size()); num_written != -1) {
        m_total_bytes += num_written;
        if (m_total_bytes >= m_num_bytes) {
            m_in_port->clear();
            return composite::retval::FINISH;
        }
    }
    return composite::retval::NORMAL;
}

extern "C" {
    auto create() -> std::shared_ptr<composite::component> {
        return std::make_shared<file_writer>();
    }
}
