
#pragma once

#include <cstdlib>
#include <cassert>
#include <deque>
#include <memory>
#include <memory_resource>
#include <vector>

#include <memory_resource>
#include <vector>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <iostream>

template <size_t BufferSize, size_t Alignment>
class PreallocatedMemoryResource : public std::pmr::memory_resource {
public:
    // alignment: desired alignment for all buffers.
    // count: number of buffers to preallocate.
    PreallocatedMemoryResource(size_t count = 100) {
        preallocate(count);
    }

    ~PreallocatedMemoryResource() override {
        freeBuffers();
    }

protected:
    // Attempt to allocate a buffer for requests up to BufferSize.
    void* do_allocate(size_t bytes, size_t alignment) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (bytes <= BufferSize) {
            if (!pool_.empty()) {
                void* ptr = pool_.back();
                pool_.pop_back();
                return ptr;
            }
            // No preallocated buffer available; allocate one.
            return allocateBuffer();
        }
        // For requests larger than BufferSize, fallback to global operator new.
        return ::operator new(bytes);
    }

    // Deallocate memory: return buffers of size <= BufferSize to the pool.
    void do_deallocate(void* p, size_t bytes, size_t alignment) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (bytes <= BufferSize) {
            pool_.push_back(p);
        } else {
            ::operator delete(p);
        }
    }

    // Check for resource equality.
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other;
    }

private:
    // size_t alignment_;
    std::vector<void*> pool_;
    std::mutex mutex_;

    // Preallocate a given number of buffers.
    void preallocate(size_t count) {
        for (size_t i = 0; i < count; ++i) {
            pool_.push_back(allocateBuffer());
        }
    }

    // Allocate an aligned buffer of BufferSize bytes.
    void* allocateBuffer() {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, BufferSize) != 0) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    // Free all buffers in the pool.
    void freeBuffers() {
        for (void* ptr : pool_) {
            std::free(ptr);
        }
        pool_.clear();
    }
};

// class PooledAligned1024Resource : public std::pmr::memory_resource {
// public:
//     explicit PooledAligned1024Resource(size_t prealloc_pages = 256) {
//         for (size_t i = 0; i < prealloc_pages; ++i) {
//             allocate_new_page();
//         }
//     }

// protected:
//     void* do_allocate(size_t bytes, size_t alignment) override {
//         assert(bytes <= 1024 && alignment <= 1024);
//         if (free_blocks_.empty()) allocate_new_page();
//         void* ptr = free_blocks_.front();
//         free_blocks_.pop_front();
//         return ptr;
//     }

//     void do_deallocate(void* ptr, size_t, size_t) override {
//         free_blocks_.push_back(ptr);
//     }

//     bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
//         return this == &other;
//     }

// private:
//     void allocate_new_page() {
//         void* page = nullptr;
//         if (posix_memalign(&page, 4096, 4096) != 0) throw std::bad_alloc();
//         pages_.emplace_back(static_cast<uint8_t*>(page));
//         for (size_t i = 0; i < 4096; i += 1024) {
//             free_blocks_.push_back(static_cast<uint8_t*>(page) + i);
//         }
//     }

//     std::vector<std::unique_ptr<uint8_t[]>> pages_;
//     std::deque<void*> free_blocks_;
// };
