#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <memory_resource>

class aligned_alloc_resource : public std::pmr::memory_resource {
public:
    explicit aligned_alloc_resource(std::size_t alignment = 64) : m_alignment(alignment) {
        assert((alignment & (alignment - 1)) == 0 && "Alignment must be power of two");
    }

private:
    std::size_t m_alignment;

    auto do_allocate(std::size_t bytes, std::size_t alignment) -> void* override {
        alignment = std::max(alignment, m_alignment);
        if ((bytes % alignment) > 0) {
            bytes += (alignment - (bytes % alignment));
        }
        if (auto ptr = std::aligned_alloc(alignment, bytes); ptr != nullptr) {
            return ptr;
        }
        throw std::bad_alloc();
    }

    auto do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) -> void override {
        std::free(ptr);
    }

    auto do_is_equal(const std::pmr::memory_resource& other) const noexcept -> bool override {
        return this == &other;
    }
};
