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

buffer_pool::buffer_pool(size_type init_size, size_type buffer_size) :
  m_buffer_size(buffer_size),
  m_min_size(init_size / 2) {
    for (auto i=size_type{}; i < init_size; ++i) {
        m_pool.emplace(new buffer_type(m_buffer_size));
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
        buf = new buffer_type(m_buffer_size);
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
        m_pool.emplace(new buffer_type(m_buffer_size));
    }
}

auto buffer_pool::size() -> size_type {
    return m_pool.size();
}

} // namespace udpsrc