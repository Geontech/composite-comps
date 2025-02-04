#include "buffer_pool.hpp"

#include <chrono>

namespace udpsrc {

buffer_pool::buffer_pool(size_type init_size, size_type buffer_size) :
  m_buffer_size(buffer_size),
  m_min_size(init_size / 2) {
    for (auto i=size_type{}; i < init_size; ++i) {
        m_pool.emplace(new buffer_type(m_buffer_size));
    }
}

buffer_pool::~buffer_pool() {
    auto lock = std::scoped_lock{m_mtx};
    while (!m_pool.empty()) {
        auto buf = m_pool.front();
        m_pool.pop();
        delete buf;
    }
}

auto buffer_pool::acquire() -> value_type {
    auto lock = std::scoped_lock{m_mtx};
    buffer_type* buf = nullptr;
    if (!m_pool.empty()) [[likely]] {
        buf = m_pool.front();
        m_pool.pop();
    } else {
        buf = new buffer_type(m_buffer_size);
    }
    return value_type(buf, [this](buffer_type* ptr) {
        this->release(ptr);
    });
}

auto buffer_pool::release(buffer_type* buf) -> void {
    auto lock = std::scoped_lock{m_mtx};
    m_pool.push(buf);
}

auto buffer_pool::replenish() -> void {
    auto lock = std::scoped_lock{m_mtx};
    while (m_pool.size() < m_min_size) {
        m_pool.emplace(new buffer_type(m_buffer_size));
    }
}

auto buffer_pool::size() -> size_type {
    return m_pool.size();
}

} // namespace udpsrc