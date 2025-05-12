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
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include <bit>
#include <cstdlib>
#include <stdexcept>
#include <memory_resource>

class aligned_mem_resource : public std::pmr::memory_resource {
public:
    explicit aligned_mem_resource(std::size_t alignment) : m_alignment(alignment) {
        if (!std::has_single_bit(m_alignment)) {
            throw std::invalid_argument("alignment must be power of two");
        }
    }

protected:
    auto do_allocate(std::size_t bytes, std::size_t alignment) -> void* override {
        alignment = std::max(alignment, m_alignment);
        auto ptr = std::aligned_alloc(alignment, bytes);
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    auto do_deallocate(void* p, std::size_t /*bytes*/, std::size_t /*alignment*/) -> void override {
        std::free(p);
    }

    auto do_is_equal(const std::pmr::memory_resource& other) const noexcept -> bool override {
        auto* o = dynamic_cast<const aligned_mem_resource*>(&other);
        return o && o->m_alignment == m_alignment;
    }

private:
    std::size_t m_alignment;

}; // class aligned_mem_resource
