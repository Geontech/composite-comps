// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#pragma once

#include <complex>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <bit>

#include <sys/mman.h>
#include <unistd.h>

namespace pfbc {

// =============================================================================
// Virtual Memory Mirrored Ring Buffer
// =============================================================================
//
// Maps physical pages twice in virtual address space:
//   Virtual:  [Page A][Page B][Page A'][Page B']
//   Physical: [Page A][Page B][Page A ][Page B ]
//
// This allows reading past the "end" without wrap-around logic or boundary
// checks. When a read spans the physical boundary, it automatically continues
// from the beginning of the physical buffer due to the mirrored mapping.
//
// Benefits:
//   - Single memcpy for writes (no split handling)
//   - Contiguous reads across wrap boundary (no assembly needed)
//   - Zero runtime overhead for wrap-around logic
//   - Eliminates history/boundary/tail buffer management
//
// Usage:
//   ring.write(input, count);           // Just memcpy
//   const T* ptr = ring.read_ptr();     // Safe to read up to capacity samples
//   ring.consume(n);                    // Advance read position
//

template<typename T>
class mirrored_ring_buffer {
public:
    mirrored_ring_buffer() = default;

    ~mirrored_ring_buffer() {
        deallocate();
    }

    // Non-copyable, movable
    mirrored_ring_buffer(const mirrored_ring_buffer&) = delete;
    mirrored_ring_buffer& operator=(const mirrored_ring_buffer&) = delete;

    mirrored_ring_buffer(mirrored_ring_buffer&& other) noexcept
        : m_data(other.m_data)
        , m_capacity(other.m_capacity)
        , m_byte_size(other.m_byte_size)
        , m_write_pos(other.m_write_pos)
        , m_read_pos(other.m_read_pos)
    {
        other.m_data = nullptr;
        other.m_capacity = 0;
        other.m_byte_size = 0;
        other.m_write_pos = 0;
        other.m_read_pos = 0;
    }

    mirrored_ring_buffer& operator=(mirrored_ring_buffer&& other) noexcept {
        if (this != &other) {
            deallocate();
            m_data = other.m_data;
            m_capacity = other.m_capacity;
            m_byte_size = other.m_byte_size;
            m_write_pos = other.m_write_pos;
            m_read_pos = other.m_read_pos;
            other.m_data = nullptr;
            other.m_capacity = 0;
            other.m_byte_size = 0;
            other.m_write_pos = 0;
            other.m_read_pos = 0;
        }
        return *this;
    }

    // -------------------------------------------------------------------------
    // Allocation
    // -------------------------------------------------------------------------

    void allocate(std::size_t num_samples) {
        if (m_data) {
            deallocate();
        }

        // Round capacity up to power of 2 for efficient modulo
        m_capacity = std::bit_ceil(num_samples);

        const std::size_t byte_size = m_capacity * sizeof(T);
        const std::size_t page_size = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        m_byte_size = (byte_size + page_size - 1) & ~(page_size - 1);

        // Create anonymous backing file
        int fd = memfd_create("pfbc_ring", 0);
        if (fd < 0) {
            throw std::runtime_error("mirrored_ring_buffer: memfd_create failed");
        }

        if (ftruncate(fd, static_cast<off_t>(m_byte_size)) < 0) {
            close(fd);
            throw std::runtime_error("mirrored_ring_buffer: ftruncate failed");
        }

        // Reserve 2x virtual address space
        void* addr = mmap(nullptr, m_byte_size * 2, PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (addr == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mirrored_ring_buffer: mmap reserve failed");
        }

        // Map first half
        void* first = mmap(addr, m_byte_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_FIXED, fd, 0);
        if (first == MAP_FAILED) {
            munmap(addr, m_byte_size * 2);
            close(fd);
            throw std::runtime_error("mirrored_ring_buffer: mmap first half failed");
        }

        // Map second half to SAME physical pages (the mirror)
        void* second = mmap(static_cast<char*>(addr) + m_byte_size, m_byte_size,
                            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
        if (second == MAP_FAILED) {
            munmap(addr, m_byte_size * 2);
            close(fd);
            throw std::runtime_error("mirrored_ring_buffer: mmap second half failed");
        }

        close(fd);  // fd no longer needed after mmap

        m_data = static_cast<T*>(addr);
        m_write_pos = 0;
        m_read_pos = 0;
    }

    void deallocate() {
        if (m_data) {
            munmap(m_data, m_byte_size * 2);
            m_data = nullptr;
            m_capacity = 0;
            m_byte_size = 0;
            m_write_pos = 0;
            m_read_pos = 0;
        }
    }

    // -------------------------------------------------------------------------
    // Write Interface
    // -------------------------------------------------------------------------

    // Write samples to the buffer. Single memcpy - mirroring handles wrap.
    void write(const T* src, std::size_t count) {
        std::memcpy(m_data + m_write_pos, src, count * sizeof(T));
        m_write_pos = (m_write_pos + count) & (m_capacity - 1);
    }

    // -------------------------------------------------------------------------
    // Read Interface
    // -------------------------------------------------------------------------

    // Get read pointer at current position + offset.
    // Safe to read up to `capacity` samples from this pointer due to mirroring.
    [[nodiscard]] const T* read_ptr(std::size_t offset = 0) const {
        return m_data + ((m_read_pos + offset) & (m_capacity - 1));
    }

    // Mutable read pointer (for in-place operations if needed)
    [[nodiscard]] T* read_ptr_mut(std::size_t offset = 0) {
        return m_data + ((m_read_pos + offset) & (m_capacity - 1));
    }

    // Advance read position after consuming samples
    void consume(std::size_t count) {
        m_read_pos = (m_read_pos + count) & (m_capacity - 1);
    }

    // -------------------------------------------------------------------------
    // State Queries
    // -------------------------------------------------------------------------

    // Number of samples available for reading
    [[nodiscard]] std::size_t available() const {
        return (m_write_pos - m_read_pos + m_capacity) & (m_capacity - 1);
    }

    // Number of samples that can be written before overwriting unread data
    [[nodiscard]] std::size_t free_space() const {
        return m_capacity - available() - 1;
    }

    [[nodiscard]] std::size_t capacity() const { return m_capacity; }
    [[nodiscard]] bool is_allocated() const { return m_data != nullptr; }

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------

    // Pre-fill buffer with zeros (e.g., for filter history)
    void prefill_zeros(std::size_t count) {
        std::memset(m_data + m_write_pos, 0, count * sizeof(T));
        m_write_pos = (m_write_pos + count) & (m_capacity - 1);
    }

    // Reset positions without deallocating
    void reset() {
        m_write_pos = 0;
        m_read_pos = 0;
    }

private:
    T* m_data{nullptr};
    std::size_t m_capacity{0};      // Number of samples (power of 2)
    std::size_t m_byte_size{0};     // Aligned byte size for mmap
    std::size_t m_write_pos{0};     // Next write position
    std::size_t m_read_pos{0};      // Current read position
};

// Type alias for complex float ring buffer
using cf32_ring_buffer = mirrored_ring_buffer<std::complex<float>>;

} // namespace pfbc
