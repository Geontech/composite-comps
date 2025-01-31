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

#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <memory>
#include <stdexcept>

namespace aligned {

/**
 * @class aligned_mem
 * @brief A template class for managing aligned memory allocations
 * @tparam T The data type stored in the aligned memory buffer
 * 
 * Provides memory allocation with user-specified alignment.
 * Supports deep-copying, moving, iterators, and safe element access.
 * Satisfies `std::ranges::contiguous_range` concept.
 */
template <typename T>
class aligned_mem {
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using iterator = value_type*;
    using const_iterator = const value_type*;

    /**
     * @brief Constructs an aligned memory buffer
     * @param alignment Memory alignment requirement (must be a power of 2)
     * @param count Number of elements of type T to allocate
     * @throws std::invalid_argument if alignment is not valid
     * @throws std::runtime_error if memory allocation fails
     */
    explicit aligned_mem(size_type alignment, size_type count) :
      m_data(),
      m_alignment(alignment),
      m_count(count) {
        if (alignment < alignof(std::max_align_t) || !std::has_single_bit(alignment)) {
            auto err = std::format(
              "invalid alignment: {}: must be a power of 2 and at least alignof(std::max_align_t)",
              alignment
            );
            throw std::invalid_argument(err);
        }
        m_data = static_cast<T*>(std::aligned_alloc(alignment, count * sizeof(T)));
        if (m_data == nullptr) {
            throw std::runtime_error("memory allocation failed");
        }
    }
    
    /**
     * @brief Destructor that frees allocated memory
     */
    ~aligned_mem() {
        std::free(m_data);
    }

    /**
     * @brief Copy constructor (deep copy)
     * @param other The aligned_mem instance to copy
     * @throws std::runtime_error if memory allocation fails
     */
    aligned_mem(const aligned_mem& other) :
      m_alignment(other.m_alignment),
      m_count(other.m_count) {
        m_data = static_cast<T*>(std::aligned_alloc(m_alignment, m_count * sizeof(T)));
        if (m_data == nullptr) {
            throw std::runtime_error("memory allocation failed during copy construction");
        }
        std::copy(other.m_data, other.m_data + m_count, m_data);
    }

    /**
     * @brief Copy assignment operator (deep copy)
     * @param other The aligned_mem instance to copy from
     * @return Reference to this instance
     * @throws std::runtime_error if memory allocation fails
     */
    auto operator=(const aligned_mem& other) -> aligned_mem& {
        if (this != &other) {
            auto new_data = static_cast<T*>(
              std::aligned_alloc(other.m_alignment, other.m_count * sizeof(T))
            );
            if (new_data == nullptr) {
                throw std::runtime_error("memory allocation failed during copy assignment");
            }
            std::copy(other.m_data, other.m_data + other.m_count, new_data);
            std::free(m_data);
            m_data = new_data;
            m_alignment = other.m_alignment;
            m_count = other.m_count;
        }
        return *this;
    }

    /**
     * @brief Move constructor (transfers ownership)
     * @param other The aligned_mem instance to move
     */
    aligned_mem(aligned_mem&& other) noexcept {
        *this = std::move(other);
    }

    /**
     * @brief Move assignment operator (transfers ownership)
     * @param other The aligned_mem instance to move from
     * @return Reference to this instance
     */
    auto operator=(aligned_mem<T>&& other) noexcept -> aligned_mem& {
        if (this != &other) {
            std::free(m_data);
            m_data = other.m_data;
            m_alignment = other.m_alignment;
            m_count = other.m_count;
            other.m_data = nullptr;
            other.m_count = 0;
        }
        return *this;
    }

    /**
     * @brief Returns a pointer to the allocated memory
     * @return A pointer to the memory buffer
     */
    [[nodiscard]]
    auto data() noexcept -> T* {
        return m_data;
    }

    /**
     * @brief Returns a const pointer to the allocated memory
     * @return A const pointer to the memory buffer
     */
    [[nodiscard]]
    auto data() const noexcept -> const T* {
        return m_data;
    }

    /**
     * @brief Provides bounds-checked access to an element
     * @param pos The index of the element
     * @return Reference to the element at specified location @param pos
     * @throws std::out_of_range if @param pos is not within range
     */
    auto at(size_type pos) -> reference {
        if (pos >= m_count) {
            auto err = std::format(
                "aligned_mem range check: pos (which is {}) >= this->size() (which is {})",
                pos, m_count
            );
            throw std::out_of_range(err);
        }
        return m_data[pos];
    }

    /**
     * @brief Provides bounds-checked access to an element
     * @param pos The index of the element
     * @return Const reference to the element at specified location @param pos
     * @throws std::out_of_range if @param pos is not within range
     */
    auto at(size_type pos) const -> const_reference {
        if (pos >= m_count) {
            auto err = std::format(
                "aligned_mem range check: pos (which is {}) >= this->size() (which is {})",
                pos, m_count
            );
            throw std::out_of_range(err);
        }
        return m_data[pos];
    }

    /**
     * @brief Returns the memory alignment of the allocated buffer
     * @return The memory alignment value
     */
    [[nodiscard]]
    auto alignment() const noexcept -> size_type {
        return m_alignment;
    }

    /**
     * @brief Returns the number of elements stored
     * @return The number of elements
     */
    [[nodiscard]]
    auto size() const noexcept -> size_type {
        return m_count;
    }

    /**
     * @brief Returns the number of bytes stored
     * @return The number of allocated bytes
     */
    [[nodiscard]]
    auto size_bytes() const noexcept -> size_type {
        return m_count * sizeof(value_type);
    }

    /**
     * @brief Returns a pointer to the beginning of the buffer
     * @return Pointer to the beginning of the buffer
     */
    [[nodiscard]]
    auto begin() noexcept -> iterator {
        return m_data;
    }

    /**
     * @brief Returns a pointer past the end of the buffer
     * @return Pointer past the end of the buffer
     */
    [[nodiscard]]
    auto end() noexcept -> iterator {
        return m_data + m_count;
    }

    /**
     * @brief Returns a const pointer to the beginning of the buffer
     * @return Const pointer to the beginning of the buffer
     */
    [[nodiscard]]
    auto begin() const noexcept -> const_iterator {
        return m_data;
    }

    /**
     * @brief Returns a const pointer past the end of the buffer
     * @return Const pointer past the end of the buffer
     */
    [[nodiscard]]
    auto end() const noexcept -> const_iterator {
        return m_data + m_count;
    }

    /**
     * @brief Returns a const pointer to the beginning of the buffer
     * @return Const pointer to the beginning of the buffer
     */
    [[nodiscard]]
    auto cbegin() const noexcept -> const_iterator {
        return begin();
    }

    /**
     * @brief Returns a const pointer past the end of the buffer
     * @return Const pointer past the end of the buffer
     */
    [[nodiscard]]
    auto cend() const noexcept -> const_iterator {
        return end();
    };

private:
    value_type* m_data{nullptr}; ///< Pointer allocated memory
    size_type m_alignment{alignof(std::max_align_t)}; ///< Allocated memory alignment
    size_type m_count{}; ///< Number of elements stored

}; // class aligned_mem

/**
 * @brief Creates a `std::unique_ptr` to an aligned_mem instance
 * @tparam T The type stored in the aligned memory
 * @param alignment The memory alignment (must be a power of 2)
 * @param count Number of elements to allocate
 * @return A unique pointer to an aligned_mem<T> instance
 * @throws std::invalid_argument if alignment is invalid
 * @throws std::runtime_error if memory allocation fails
 */
template <typename T>
auto make_aligned(std::size_t alignment, std::size_t count) -> std::unique_ptr<aligned_mem<T>> {
    return std::make_unique<aligned_mem<T>>(alignment, count);
}

} // namespace aligned
