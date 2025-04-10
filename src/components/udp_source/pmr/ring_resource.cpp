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

#include "ring_resource.hpp"

#include <bit>
#include <cstdlib>

namespace {
auto free_func = [](void* p){ std::free(p); };
}

ring_resource::ring_resource(const ring_config& config) :
  m_frame_size(config.frame_size),
  m_frame_count(config.frame_count),
  m_alignment(config.alignment),
  m_read_idx(0),
  m_write_idx(0),
  m_buffer((uint8_t*)std::aligned_alloc(m_alignment, m_frame_size * m_frame_count), free_func) {
    if (!m_buffer.get()) {
        throw std::bad_alloc();
    }
    if (m_frame_size == 0 || m_frame_count == 0) {
        throw std::invalid_argument("frame size and count must be non-zero");
    }
}

auto ring_resource::do_allocate(std::size_t bytes, std::size_t alignment) -> void* {
    if (bytes > m_frame_size || alignment > m_alignment) {
        throw std::bad_alloc();
    }

    auto current_write = m_write_idx.load(std::memory_order_relaxed);
    auto next_write = (current_write + 1) % m_frame_count;
    auto current_read = m_read_idx.load(std::memory_order_acquire);

    if (next_write == current_read) {
        throw std::bad_alloc(); // ring full
    }

    auto ptr = m_buffer.get() + (current_write * m_frame_size);
    m_write_idx.store(next_write, std::memory_order_release);
    return ptr;
}

auto ring_resource::do_deallocate(void* p, std::size_t, std::size_t) -> void {
    auto current_read = m_read_idx.load(std::memory_order_relaxed);
    m_read_idx.store((current_read + 1) % m_frame_count, std::memory_order_release);
}

auto ring_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept -> bool{
    return this == &other;
}
