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
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sys/mman.h>
#include <utility>

namespace udp {

class frame_pool {
    struct slab_deleter {
        void operator()(uint8_t* ptr) const noexcept { std::free(ptr); }
    };

    struct slot_header {
        std::atomic<uint32_t> status{0};
    };

    static constexpr uint32_t SLOT_FREE = 0;
    static constexpr uint32_t SLOT_IN_USE = 1;
public:
    // Acquire next slot in ring order; returns {index, payload} when free, nullopt otherwise.
    struct slot_view {
        std::size_t index;
        uint8_t* payload;
    };

    frame_pool(std::size_t frame_size, std::size_t frame_count, std::size_t alignment = 64) :
      m_frame_size(frame_size),
      m_payload_offset(((sizeof(slot_header) + alignment - 1) / alignment) * alignment),
      m_stride(m_payload_offset + ((frame_size + alignment - 1) / alignment * alignment)),
      m_frame_count(frame_count) {
        auto total = m_stride * frame_count;
        auto* ptr = reinterpret_cast<uint8_t*>(std::aligned_alloc(alignment, total));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        m_slab = std::unique_ptr<uint8_t, slab_deleter>(ptr);
        madvise(ptr, total, MADV_HUGEPAGE);
    }

    auto frame_size() const -> std::size_t { return m_frame_size; }

    auto try_acquire() -> std::optional<slot_view> {
        auto idx = m_head.load(std::memory_order_relaxed);
        auto* base = slot_base(idx);
        auto* hdr = reinterpret_cast<slot_header*>(base);
        if (hdr->status.load(std::memory_order_acquire) != SLOT_FREE) {
            return std::nullopt;
        }
        if (hdr->status.exchange(SLOT_IN_USE, std::memory_order_acq_rel) != SLOT_FREE) {
            return std::nullopt;
        }
        m_head.store((idx + 1) % m_frame_count, std::memory_order_relaxed);
        return slot_view{idx, base + m_payload_offset};
    }

    auto release(std::size_t index) -> void {
        auto* hdr = reinterpret_cast<slot_header*>(slot_base(index));
        hdr->status.store(SLOT_FREE, std::memory_order_release);
    }

private:
    auto slot_base(std::size_t index) -> uint8_t* { return m_slab.get() + index * m_stride; }

    std::size_t m_frame_size{};
    std::size_t m_payload_offset{};
    std::size_t m_stride{};
    std::size_t m_frame_count{};
    std::unique_ptr<uint8_t, slab_deleter> m_slab{nullptr};
    std::atomic<std::size_t> m_head{0};

}; // class frame_pool

} // namespace udp
