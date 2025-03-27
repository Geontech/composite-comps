#pragma once

#include "statistics.hpp"

#include <memory>
#include <memory_resource>

namespace udp {

class interface {
public:
    using buffer_t = std::pmr::vector<uint8_t>;
    using buffer_ptr_t = std::shared_ptr<buffer_t>;

    virtual ~interface() = default;
    virtual auto start() -> void = 0;
    virtual auto stop() -> void = 0;
    virtual auto get_data(buffer_ptr_t&) -> bool = 0;
    virtual auto get_stats() -> statistics = 0;

}; // class interface

} // namespace udp