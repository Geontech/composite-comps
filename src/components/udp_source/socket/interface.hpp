#pragma once

#include "pmr/aligned_alloc_resource.hpp"
#include "statistics.hpp"

#include <memory>
#include <memory_resource>
#include <spdlog/spdlog.h>

namespace udp {

struct config {

std::string id;
std::string interface;
std::string ip_addr;
uint16_t port{};
std::size_t recv_buf_size{};

}; // struct config

class interface {
public:
    using buffer_t = std::pmr::vector<uint8_t>;
    using buffer_ptr_t = std::shared_ptr<buffer_t>;

    virtual ~interface() = default;
    virtual auto start_recv() -> void = 0;
    virtual auto stop_recv() -> void = 0;
    virtual auto get_data(buffer_ptr_t&) -> bool = 0;
    virtual auto get_stats() -> statistics = 0;

protected:
    std::string m_id;
    aligned_alloc_resource m_upstream_alloc{64};
    std::pmr::synchronized_pool_resource m_pool_resource{{}, &m_upstream_alloc};

}; // class interface

} // namespace udp