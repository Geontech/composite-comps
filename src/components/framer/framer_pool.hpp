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
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include <immintrin.h>

#include <composite/buffers/aligned_mem.hpp>
#include <composite/buffers/buffer.hpp>
#include <composite/buffers/external_buffer.hpp>

#include "convert_variant.hpp"

template <typename T>
class framer_pool : public std::enable_shared_from_this<framer_pool<T>> {
public:
    enum class drop_reason {
        NONE,
        BATCH_TOO_LARGE,
        BACKPRESSURE_TIMEOUT
    };

    struct diagnostics {
        std::size_t slots_in_use{0};
        std::size_t total_slots{0};
        std::size_t write_head{0};
        std::size_t ring_size{0};
        std::size_t oldest_protected_sample{0};
        std::size_t available_space{0};
        drop_reason last_drop_reason{drop_reason::NONE};
        std::size_t last_drop_sample_count{0};
    };

    struct frame_slot {
        std::atomic<bool> in_use{false};
        std::atomic<std::size_t> start_sample{0};
    };

    framer_pool(std::size_t frame_size, std::size_t overlap, std::size_t frame_count) :
      m_frame_size(frame_size),
      m_overlap(overlap),
      m_frame_count(frame_count),
      m_slots(frame_count) {
        // Validate parameters
        if (frame_size == 0) {
            throw std::invalid_argument("framer_pool: frame_size must be > 0");
        }
        if (overlap >= frame_size) {
            throw std::invalid_argument("framer_pool: overlap must be < frame_size");
        }
        if (frame_count == 0) {
            throw std::invalid_argument("framer_pool: frame_count must be > 0");
        }

        // Calculate derived values
        m_hop_size = m_frame_size - m_overlap;
        m_ring_size = frame_count * m_hop_size + m_overlap;
        m_ring_capacity = m_ring_size + m_frame_size; // tail for contiguous wrapped frames

        // Allocate ring buffer (still use aligned_mem for better cache performance)
        m_ring = std::make_shared<composite::aligned_mem<T>>(64, m_ring_capacity);

        // Initialize oldest protected to max (no slots in use initially)
        m_oldest_protected.store(std::numeric_limits<std::size_t>::max(), std::memory_order_relaxed);
    }

    // Write samples to ring - lock-free on hot path, spins/yields on backpressure
    auto write_samples(
      const uint8_t* input, std::size_t sample_count,
      converter_variant<typename T::value_type>* converter,
      std::size_t input_stride, bool is_complex
    ) -> bool {
        if (!converter) {
            return false;
        }

        if (sample_count == 0) {
            return true;  // Valid no-op
        }

        if (sample_count > m_ring_size) {
            m_drop_reason.store(drop_reason::BATCH_TOO_LARGE, std::memory_order_relaxed);
            m_last_drop_sample_count.store(sample_count, std::memory_order_relaxed);
            return false;
        }

        // Lock-free backpressure check with spin + yield
        auto write_head = m_write_head.load(std::memory_order_relaxed);

        if (!can_write_fast(write_head, sample_count)) {
            // Spin briefly with pause
            constexpr int SPIN_COUNT = 1000;
            for (int i = 0; i < SPIN_COUNT; ++i) {
                _mm_pause();
                if (can_write_fast(write_head, sample_count)) {
                    goto do_write;
                }
            }

            // Yield and retry
            constexpr int MAX_YIELDS = 100;
            for (int i = 0; i < MAX_YIELDS; ++i) {
                std::this_thread::yield();
                if (can_write_fast(write_head, sample_count)) {
                    goto do_write;
                }
            }

            // Still blocked - timeout
            m_drop_reason.store(drop_reason::BACKPRESSURE_TIMEOUT, std::memory_order_relaxed);
            m_last_drop_sample_count.store(sample_count, std::memory_order_relaxed);
            return false;
        }

    do_write:
        using scalar_t = typename T::value_type;
        auto* ring_base = m_ring->data();
        std::size_t ring_pos = write_head % m_ring_size;

        auto write_complex = [&](std::size_t dst_offset, const uint8_t* src, std::size_t samples) {
            std::size_t component_count = samples * 2;
            convert(*converter, src, reinterpret_cast<scalar_t*>(ring_base + dst_offset),
                   component_count);
        };

        auto write_real = [&](std::size_t dst_offset, const uint8_t* src, std::size_t samples) {
            if (samples == 0) {
                return;
            }

            // Use scratch buffer to convert real samples, then expand to complex
            thread_local std::unique_ptr<composite::aligned_mem<scalar_t>> scratch;
            if (!scratch || scratch->size() < samples) {
                scratch = std::make_unique<composite::aligned_mem<scalar_t>>(64, samples);
            }
            convert(*converter, src, scratch->data(), samples);

            auto* dst = ring_base + dst_offset;
            for (std::size_t i = 0; i < samples; ++i) {
                dst[i] = T{scratch->data()[i], static_cast<scalar_t>(0)};
            }
        };

        auto write_chunk = [&](std::size_t dst_offset, const uint8_t* src, std::size_t samples) {
            if (is_complex) {
                write_complex(dst_offset, src, samples);
            } else {
                write_real(dst_offset, src, samples);
            }
        };

        // Handle wraparound
        if (ring_pos + sample_count <= m_ring_size) {
            // No wrap
            write_chunk(ring_pos, input, sample_count);
        } else {
            // Wraps - split into two conversions
            std::size_t first_chunk = m_ring_size - ring_pos;
            std::size_t second_chunk = sample_count - first_chunk;

            write_chunk(ring_pos, input, first_chunk);
            write_chunk(0, input + first_chunk * input_stride, second_chunk);

            // Mirror wrapped portion into tail for contiguous frame access
            std::memcpy(ring_base + m_ring_size, ring_base, second_chunk * sizeof(T));
        }

        // Update write head - release so readers see the written data
        m_write_head.store(write_head + sample_count, std::memory_order_release);
        return true;
    }

    // Try to emit a frame - lock-free
    auto try_emit_frame(std::size_t absolute_start) -> std::optional<composite::immutable_buffer<T>> {
        // Validate frame alignment - absolute_start MUST be aligned to hop boundaries
        if (absolute_start % m_hop_size != 0) {
            throw std::logic_error(std::format(
                "framer_pool::try_emit_frame: absolute_start ({}) is not aligned to hop_size ({})",
                absolute_start, m_hop_size
            ));
        }

        auto write_head = m_write_head.load(std::memory_order_acquire);

        // Check if frame data is available
        if (absolute_start + m_frame_size > write_head) {
            return std::nullopt;
        }

        auto frame_num = absolute_start / m_hop_size;
        auto slot_idx = frame_num % m_frame_count;
        auto& slot = m_slots[slot_idx];

        // Try to acquire slot atomically
        bool expected = false;
        if (!slot.in_use.compare_exchange_strong(expected, true,
                std::memory_order_acquire, std::memory_order_relaxed)) {
            // Slot busy - downstream still holding previous frame
            return std::nullopt;
        }

        // Slot acquired - store start sample
        slot.start_sample.store(absolute_start, std::memory_order_relaxed);

        // Update oldest protected - this slot might be the new oldest
        update_oldest_protected_on_acquire(absolute_start);

        // Increment slots in use counter
        m_slots_in_use.fetch_add(1, std::memory_order_relaxed);

        // Calculate ring position (contiguous even on wrap due to tail)
        const auto ring_offset = absolute_start % m_ring_size;
        auto* data_ptr = m_ring->data() + ring_offset;

        auto self = this->shared_from_this();
        auto deleter = slot_deleter{
            .pool = self,
            .slot_index = slot_idx
        };

        auto buffer = composite::external_buffer<T>(data_ptr, m_frame_size, deleter);
        return composite::immutable_buffer<T>(std::move(buffer));
    }

    auto frame_size() const noexcept -> std::size_t {
        return m_frame_size;
    }

    auto hop_size() const noexcept -> std::size_t {
        return m_hop_size;
    }

    // Lock-free read of write head
    auto head() const noexcept -> std::size_t {
        return m_write_head.load(std::memory_order_acquire);
    }

    auto get_diagnostics() const -> diagnostics {
        auto write_head = m_write_head.load(std::memory_order_acquire);
        auto slots_in_use = m_slots_in_use.load(std::memory_order_relaxed);
        auto oldest = m_oldest_protected.load(std::memory_order_relaxed);

        // If no slots in use, oldest_protected is meaningless
        if (slots_in_use == 0) {
            oldest = write_head;
        }

        std::size_t available_space = 0;
        if (oldest == std::numeric_limits<std::size_t>::max()) {
            available_space = m_ring_size;
        } else if (write_head >= oldest) {
            available_space = m_ring_size - (write_head - oldest);
        }

        return diagnostics{
            .slots_in_use = slots_in_use,
            .total_slots = m_frame_count,
            .write_head = write_head,
            .ring_size = m_ring_size,
            .oldest_protected_sample = oldest,
            .available_space = available_space,
            .last_drop_reason = m_drop_reason.load(std::memory_order_relaxed),
            .last_drop_sample_count = m_last_drop_sample_count.load(std::memory_order_relaxed)
        };
    }

private:
    struct slot_deleter {
        std::shared_ptr<framer_pool<T>> pool;
        std::size_t slot_index;

        auto operator()() const noexcept -> void {
            // Get the start sample before releasing
            auto released_start = pool->m_slots[slot_index].start_sample.load(std::memory_order_relaxed);

            // Release slot
            pool->m_slots[slot_index].in_use.store(false, std::memory_order_release);

            // Decrement counter
            auto prev_count = pool->m_slots_in_use.fetch_sub(1, std::memory_order_relaxed);

            // Update oldest protected if this was the oldest slot
            if (prev_count == 1) {
                // We were the last slot - no protection needed
                pool->m_oldest_protected.store(std::numeric_limits<std::size_t>::max(), std::memory_order_relaxed);
            } else if (released_start == pool->m_oldest_protected.load(std::memory_order_relaxed)) {
                // We were the oldest - scan forward to find next oldest
                pool->advance_oldest_protected_from_slot(slot_index);
            }
        }
    };

    // Fast O(1) check - just compare against cached oldest
    auto can_write_fast(std::size_t write_head, std::size_t count) const noexcept -> bool {
        if (count == 0) {
            return true;
        }

        auto oldest = m_oldest_protected.load(std::memory_order_relaxed);

        // No slots in use - always can write
        if (oldest == std::numeric_limits<std::size_t>::max()) {
            return true;
        }

        // Simple watermark check: don't overwrite protected region
        return (write_head - oldest + count) <= m_ring_size;
    }

    // Update oldest when acquiring a new slot - only needed when first slot acquired
    auto update_oldest_protected_on_acquire(std::size_t new_start) noexcept -> void {
        // Slots are acquired in strict order, so a new slot is always >= existing slots.
        // Only update if this is the first slot (no protection currently).
        auto current = m_oldest_protected.load(std::memory_order_relaxed);
        if (current == std::numeric_limits<std::size_t>::max()) {
            m_oldest_protected.store(new_start, std::memory_order_relaxed);
        }
    }

    // Advance oldest protected by scanning forward from released slot
    // Slots are acquired/released in circular order, so next in-use slot is the new oldest
    auto advance_oldest_protected_from_slot(std::size_t released_slot_idx) noexcept -> void {
        for (std::size_t i = 1; i < m_frame_count; ++i) {
            auto slot_idx = (released_slot_idx + i) % m_frame_count;
            if (m_slots[slot_idx].in_use.load(std::memory_order_relaxed)) {
                auto start = m_slots[slot_idx].start_sample.load(std::memory_order_relaxed);
                m_oldest_protected.store(start, std::memory_order_relaxed);
                return;
            }
        }
        // No slots in use - shouldn't happen since we checked prev_count > 1
        m_oldest_protected.store(std::numeric_limits<std::size_t>::max(), std::memory_order_relaxed);
    }

    std::size_t m_frame_size{};
    std::size_t m_frame_count{};
    std::size_t m_ring_size{};
    std::size_t m_ring_capacity{};
    std::size_t m_overlap{};
    std::size_t m_hop_size{};
    std::shared_ptr<composite::aligned_mem<T>> m_ring{nullptr};

    // Lock-free synchronization
    std::atomic<std::size_t> m_write_head{0};
    std::atomic<std::size_t> m_oldest_protected{std::numeric_limits<std::size_t>::max()};
    std::atomic<std::size_t> m_slots_in_use{0};
    std::vector<frame_slot> m_slots;

    // Diagnostics (relaxed atomics - not critical path)
    std::atomic<drop_reason> m_drop_reason{drop_reason::NONE};
    std::atomic<std::size_t> m_last_drop_sample_count{0};

}; // class framer_pool
