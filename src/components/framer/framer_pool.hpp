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

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>

#include <composite/buffers/aligned_mem.hpp>
#include <composite/buffers/buffer.hpp>
#include <composite/buffers/external_buffer.hpp>

#include "convert.hpp"

template <typename T>
class framer_pool : public std::enable_shared_from_this<framer_pool<T>> {
public:
    struct frame_slot {
        std::atomic<bool> in_use{false};
        std::atomic<std::size_t> start_sample{0};  // Absolute sample index
        std::size_t ring_offset{0};                 // Fixed offset in ring buffer
    };

    framer_pool(std::size_t frame_size, std::size_t overlap, std::size_t frame_count) :
      m_frame_size(frame_size),
      m_overlap(overlap),
      m_hop_size(frame_size - overlap),
      m_frame_count(frame_count),
      m_ring_size(frame_size * frame_count + overlap),
      m_ring(std::make_shared<composite::aligned_mem<T>>(64, m_ring_size)),
      m_slots(frame_count) {
        // Pre-compute ring offsets for all slots
        for (std::size_t i = 0; i < m_frame_count; ++i) {
            m_slots[i].ring_offset = (i * m_hop_size) % m_ring_size;
        }
    }

    // Check if we can write samples at current head position
    auto can_write(std::size_t count) -> bool {
        std::size_t head = m_head.load(std::memory_order_acquire);
        return can_write_range(head, count);
    }

    // Write samples to ring using AVX converter
    auto write_samples(
      const uint8_t* input, std::size_t sample_count,
      converter_base<typename T::value_type>* converter,
      std::size_t input_stride
    ) -> bool {
        if (!converter) {
            return false;
        }

        auto head = m_head.load(std::memory_order_acquire);

        // Check for conflicts with inflight frames
        if (!can_write_range(head, sample_count)) {
            return false;  // Would overwrite inflight frame
        }

        using scalar_t = typename T::value_type;
        auto* ring_base = m_ring->data();
        std::size_t ring_pos = head % m_ring_size;
        std::size_t component_count = sample_count * 2;  // I and Q components

        // Handle wraparound
        if (ring_pos + sample_count <= m_ring_size) {
            // No wrap - direct AVX conversion to ring
            converter->process(input, reinterpret_cast<scalar_t*>(ring_base + ring_pos),
                             component_count);
        } else {
            // Wraps - split into two conversions
            std::size_t first_chunk = m_ring_size - ring_pos;
            std::size_t second_chunk = sample_count - first_chunk;
            std::size_t first_components = first_chunk * 2;
            std::size_t second_components = second_chunk * 2;

            converter->process(input, reinterpret_cast<scalar_t*>(ring_base + ring_pos),
                             first_components);
            converter->process(input + first_chunk * input_stride,
                             reinterpret_cast<scalar_t*>(ring_base),
                             second_components);

            // Copy overlap samples to front for contiguous slicing
            if (m_overlap > 0 && m_ring_size >= m_overlap) {
                std::size_t overlap_src = m_ring_size - m_overlap;
                std::memcpy(ring_base, ring_base + overlap_src, m_overlap * sizeof(T));
            }
        }

        m_head.fetch_add(sample_count, std::memory_order_release);
        return true;
    }

    // Try to emit a frame
    auto try_emit_frame(std::size_t absolute_start) -> std::optional<composite::immutable_buffer<T>> {
        auto frame_num = absolute_start / m_hop_size;
        auto slot_idx = frame_num % m_frame_count;
        auto& slot = m_slots[slot_idx];

        // Try to acquire slot
        bool expected_free = false;
        if (!slot.in_use.compare_exchange_strong(expected_free, true,
                                                   std::memory_order_acq_rel)) {
            return std::nullopt;  // Slot still in use
        }

        // Update slot's absolute position
        slot.start_sample.store(absolute_start, std::memory_order_release);

        // Create buffer pointing to ring at this slot's offset
        auto* data_ptr = m_ring->data() + slot.ring_offset;

        auto self = this->shared_from_this();
        auto deleter = slot_deleter{
            .pool = self,
            .slot_index = slot_idx
        };

        auto buffer = std::make_shared<composite::external_buffer<T, slot_deleter>>(
            data_ptr, m_frame_size, deleter
        );

        return composite::immutable_buffer<T>(buffer);
    }

    auto frame_size() const -> std::size_t { return m_frame_size; }
    auto hop_size() const -> std::size_t { return m_hop_size; }
    auto head() const -> std::size_t { return m_head.load(std::memory_order_acquire); }

private:
    struct slot_deleter {
        std::shared_ptr<framer_pool<T>> pool;
        std::size_t slot_index;

        auto operator()() const -> void {
            // Just clear the in_use flag - that's it!
            pool->m_slots[slot_index].in_use.store(false, std::memory_order_release);
        }
    };

    // Check if writing to [start, start+count) would conflict with any inflight frame
    auto can_write_range(std::size_t start, std::size_t count) -> bool {
        // Calculate ring buffer position for this write
        std::size_t write_ring_pos = start % m_ring_size;

        // Check all in-use slots
        for (const auto& slot : m_slots) {
            if (!slot.in_use.load(std::memory_order_acquire)) {
                continue;  // Slot free, no conflict
            }

            // Slot occupies [ring_offset, ring_offset + frame_size) in ring buffer
            std::size_t slot_start = slot.ring_offset;
            std::size_t slot_end = slot.ring_offset + m_frame_size;

            // Check if write's ring positions overlap with slot's ring positions
            // Write may wrap around, slot positions don't wrap (by design)
            if (write_ring_pos + count <= m_ring_size) {
                // Write doesn't wrap - simple overlap check
                std::size_t write_end = write_ring_pos + count;
                if (write_ring_pos < slot_end && write_end > slot_start) {
                    return false;  // CONFLICT! Would overwrite inflight frame
                }
            } else {
                // Write wraps: covers [write_ring_pos, m_ring_size) and [0, wrap_end)
                std::size_t wrap_end = (write_ring_pos + count) % m_ring_size;

                // Check tail part [write_ring_pos, m_ring_size)
                if (write_ring_pos < slot_end && m_ring_size > slot_start) {
                    return false;  // CONFLICT! Would overwrite inflight frame
                }

                // Check wrapped head part [0, wrap_end)
                if (0 < slot_end && wrap_end > slot_start) {
                    return false;  // CONFLICT! Would overwrite inflight frame
                }
            }
        }

        return true;  // No conflicts
    }

    std::size_t m_frame_size{};
    std::size_t m_overlap{};
    std::size_t m_hop_size{};
    std::size_t m_frame_count{};
    std::size_t m_ring_size{};
    std::shared_ptr<composite::aligned_mem<T>> m_ring{nullptr};
    std::atomic<std::size_t> m_head{0};
    std::vector<frame_slot> m_slots;

}; // class framer_pool
