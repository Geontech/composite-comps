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

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

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
        bool in_use{false};
        std::size_t start_sample{0};
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
    }

    // Write samples to ring - blocks until space available
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
            m_drop_reason = drop_reason::BATCH_TOO_LARGE;
            m_last_drop_sample_count = sample_count;
            return false;
        }

        std::unique_lock lock(m_mutex);

        // Block until space available (backpressure), with timeout
        bool space_available = m_space_available.wait_for(
            lock,
            std::chrono::seconds(1),
            [&]() { return can_write_unlocked(sample_count); }
        );

        if (!space_available) {
            m_drop_reason = drop_reason::BACKPRESSURE_TIMEOUT;
            m_last_drop_sample_count = sample_count;
            return false;  // Timeout - likely deadlock or backpressure issue
        }

        using scalar_t = typename T::value_type;
        auto* ring_base = m_ring->data();
        std::size_t ring_pos = m_write_head % m_ring_size;

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

        m_write_head += sample_count;
        return true;
    }

    // Try to emit a frame
    auto try_emit_frame(std::size_t absolute_start) -> std::optional<composite::immutable_buffer<T>> {
        std::unique_lock lock(m_mutex);

        // Validate frame alignment - absolute_start MUST be aligned to hop boundaries
        if (absolute_start % m_hop_size != 0) {
            // This indicates a logic error in the caller
            throw std::logic_error(std::format(
                "framer_pool::try_emit_frame: absolute_start ({}) is not aligned to hop_size ({})",
                absolute_start, m_hop_size
            ));
        }

        // Check if frame data is available
        if (absolute_start + m_frame_size > m_write_head) {
            return std::nullopt;
        }

        auto frame_num = absolute_start / m_hop_size;
        auto slot_idx = frame_num % m_frame_count;
        auto& slot = m_slots[slot_idx];

        // Check if slot available
        if (slot.in_use) {
            return std::nullopt;
        }

        // Acquire slot
        slot.in_use = true;
        slot.start_sample = absolute_start;

        // Calculate ring position (contiguous even on wrap due to tail)
        const auto ring_offset = absolute_start % m_ring_size;
        auto* data_ptr = m_ring->data() + ring_offset;

        auto self = this->shared_from_this();
        auto deleter = slot_deleter{
            .pool = self,
            .slot_index = slot_idx
        };

        auto buffer = std::make_shared<composite::external_buffer<T>>(
            data_ptr, m_frame_size, deleter
        );

        return composite::immutable_buffer<T>(buffer);
    }

    auto frame_size() const noexcept -> std::size_t {
        return m_frame_size;
    }

    auto hop_size() const noexcept -> std::size_t {
        return m_hop_size;
    }

    auto head() const -> std::size_t {
        std::lock_guard lock(m_mutex);
        return m_write_head;
    }

    auto get_diagnostics() const -> diagnostics {
        std::lock_guard lock(m_mutex);

        // Count slots in use
        std::size_t slots_in_use = 0;
        std::size_t oldest_protected = m_write_head;

        for (const auto& slot : m_slots) {
            if (slot.in_use) {
                ++slots_in_use;
                oldest_protected = std::min(oldest_protected, slot.start_sample);
            }
        }

        std::size_t available_space = 0;
        if (m_write_head >= oldest_protected) {
            available_space = m_ring_size - (m_write_head - oldest_protected);
        }

        return diagnostics{
            .slots_in_use = slots_in_use,
            .total_slots = m_frame_count,
            .write_head = m_write_head,
            .ring_size = m_ring_size,
            .oldest_protected_sample = oldest_protected,
            .available_space = available_space,
            .last_drop_reason = m_drop_reason,
            .last_drop_sample_count = m_last_drop_sample_count
        };
    }

private:
    struct slot_deleter {
        std::shared_ptr<framer_pool<T>> pool;
        std::size_t slot_index;

        auto operator()() const -> void {
            std::lock_guard lock(pool->m_mutex);
            pool->m_slots[slot_index].in_use = false;
            // Wake any blocked writers
            pool->m_space_available.notify_all();
        }
    };

    // Check if writing count samples would overwrite in-use frames
    // Must be called with mutex held
    auto can_write_unlocked(std::size_t count) -> bool {
        if (count == 0) {
            return true;
        }

        // Find the oldest sample we need to protect
        std::size_t min_protected_sample = m_write_head;
        for (const auto& slot : m_slots) {
            if (slot.in_use) {
                min_protected_sample = std::min(min_protected_sample, slot.start_sample);
            }
        }

        // Simple watermark check: don't overwrite protected region
        return (m_write_head - min_protected_sample + count) <= m_ring_size;
    }

    std::size_t m_frame_size{};
    std::size_t m_frame_count{};
    std::size_t m_ring_size{};
    std::size_t m_ring_capacity{};
    std::size_t m_overlap{};
    std::size_t m_hop_size{};
    std::shared_ptr<composite::aligned_mem<T>> m_ring{nullptr};

    // Synchronization (replaces atomics)
    mutable std::mutex m_mutex;
    std::condition_variable m_space_available;
    std::size_t m_write_head{};

    std::vector<frame_slot> m_slots;

    // Diagnostics
    drop_reason m_drop_reason{drop_reason::NONE};
    std::size_t m_last_drop_sample_count{0};

}; // class framer_pool
