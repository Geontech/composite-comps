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

#include "buffer_pool.hpp"

#include <chrono>

namespace udpsrc {

mmsgs::mmsgs(size_t num_msgs, size_t msg_size) :
  msgs(num_msgs),
  iovecs(num_msgs),
  buffer(std::make_unique<std::vector<uint8_t>>(num_msgs * msg_size, 0xFF)) {
    for (auto i=0u; i<num_msgs; ++i) {
        iovecs.at(i).iov_base = buffer->data() + (i * msg_size);
        iovecs.at(i).iov_len = msg_size;
        msgs.at(i).msg_hdr.msg_iov = &iovecs.at(i);
        msgs.at(i).msg_hdr.msg_iovlen = 1;
    }
}

buffer_pool::buffer_pool(size_type init_size, size_type num_msgs, size_type msg_size) :
  m_num_msgs(num_msgs),
  m_msg_size(msg_size),
  m_min_size(init_size / 2) {
    for (auto i=size_type{}; i < init_size; ++i) {
        m_pool.emplace(new buffer_type(m_num_msgs, m_msg_size));
    }
}

buffer_pool::~buffer_pool() {
    auto lock = std::scoped_lock{m_mtx};
    while (!m_pool.empty()) {
        auto buf = m_pool.front();
        m_pool.pop();
        delete buf;
    }
}

auto buffer_pool::acquire() -> value_type {
    auto lock = std::scoped_lock{m_mtx};
    buffer_type* buf = nullptr;
    if (!m_pool.empty()) [[likely]] {
        buf = m_pool.front();
        m_pool.pop();
    } else {
        buf = new buffer_type(m_num_msgs, m_msg_size);
    }
    return value_type(buf, [this](buffer_type* ptr) {
        this->release(ptr);
    });
}

auto buffer_pool::release(buffer_type* buf) -> void {
    auto lock = std::scoped_lock{m_mtx};
    m_pool.emplace(buf);
}

auto buffer_pool::replenish() -> void {
    auto lock = std::scoped_lock{m_mtx};
    while (m_pool.size() < m_min_size) {
        m_pool.emplace(new buffer_type(m_num_msgs, m_msg_size));
    }
}

auto buffer_pool::size() -> size_type {
    return m_pool.size();
}

} // namespace udpsrc