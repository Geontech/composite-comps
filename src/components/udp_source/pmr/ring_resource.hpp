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

#pragma once

#include <atomic>
#include <memory>
#include <memory_resource>

/**
 * @brief A custom polymorphic memory resource backed by a circular ring buffer.
 *
 * This class implements a lock-free single-producer/single-consumer ring allocator
 * using a fixed-size buffer. It is particularly useful for zero-copy pipelines or
 * real-time systems where allocation predictability is important.
 */
class ring_resource : public std::pmr::memory_resource {
public:
    /**
     * @brief Configuration for the ring buffer memory resource.
     */
    struct ring_config {

        /**
         * @brief Size of each frame in bytes.
         */
        std::size_t frame_size{};

        /**
         * @brief Total number of frames in the buffer.
         */
        std::size_t frame_count{};

        /**
         * @brief Alignment for each allocation, in bytes (default is 64).
         */
        std::size_t alignment{64};

    }; // struct ring_config

    /**
     * @brief Construct a ring_resource with the given configuration.
     *
     * @param config Configuration for frame size, count, and alignment.
     * 
     * @throws std::bad_alloc If allocation fails
     * @throws std::invalid_argument If frame size or frame count is zero
     */
    explicit ring_resource(const ring_config& config);

protected:
    /**
     * @brief Allocate a memory block of the given size and alignment.
     *
     * @param bytes Number of bytes to allocate.
     * @param alignment Alignment requirement for the allocation.
     * @return Pointer to the allocated memory.
     */
    auto do_allocate(std::size_t bytes, std::size_t alignment) -> void* override;

    /**
     * @brief Deallocate a memory block.
     *
     * @param p Pointer to the memory block to deallocate.
     * @param bytes Number of bytes that were allocated.
     * @param alignment Alignment that was used for the allocation.
     * 
     * @throws std::bad_alloc If allocation fails
     */
    auto do_deallocate(void* p, std::size_t bytes, std::size_t alignment) -> void override;

    /**
     * @brief Compare this resource to another for equality.
     *
     * @param other Another polymorphic memory resource.
     * @return True if both resources are equal, false otherwise.
     */
    auto do_is_equal(const std::pmr::memory_resource& other) const noexcept -> bool override;

private:
    /**
     * @brief Size of each memory frame.
     */
    std::size_t m_frame_size{};

    /**
     * @brief Total number of frames in the ring buffer.
     */
    std::size_t m_frame_count{};

    /**
     * @brief Byte alignment for memory allocations.
     */
    std::size_t m_alignment{};

    /**
     * @brief Index for reading from the ring buffer.
     */
    std::atomic<std::size_t> m_read_idx{};

    /**
     * @brief Index for writing into the ring buffer.
     */
    std::atomic<std::size_t> m_write_idx{};

    /**
     * @brief Owning pointer to the backing buffer.
     *
     * Uses a custom deleter to properly release memory.
     */
    std::unique_ptr<std::uint8_t, void(*)(void*)> m_buffer;

}; // class ring_resource
