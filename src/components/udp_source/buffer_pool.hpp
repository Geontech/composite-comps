#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace udpsrc {

class buffer_pool {
public:
    using size_type = std::size_t;
    using buffer_type = std::vector<uint8_t>;
    using value_type = std::unique_ptr<buffer_type, std::function<void(buffer_type*)>>;

    buffer_pool(size_type init_size, size_type buffer_size);
    ~buffer_pool();

    [[nodiscard]]
    auto acquire() -> value_type;
    auto release(buffer_type* buf) -> void;
    auto replenish() -> void;
    auto size() -> size_type;

private:
    std::queue<buffer_type*> m_pool;
    size_type m_buffer_size{};
    size_type m_min_size{};
    std::mutex m_mtx;

}; // class buffer_pool

} // namespace udpsrc