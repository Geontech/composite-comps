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

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <sys/socket.h>
#include <sys/uio.h>
#include <thread>
#include <vector>

namespace udpsrc {

class mmsgs {
public:
    using buffer_type = std::vector<uint8_t>;

    mmsgs(size_t num_msgs, size_t msg_size);

    std::vector<struct mmsghdr> msgs;
    std::vector<struct iovec> iovecs;
    std::shared_ptr<buffer_type> buffer;

}; // class mmsgs

class buffer_pool {
public:
    using size_type = std::size_t;
    using buffer_type = mmsgs;
    using value_type = std::unique_ptr<buffer_type, std::function<void(buffer_type*)>>;

    buffer_pool(size_type init_size, size_type num_msgs, size_type msg_size);
    ~buffer_pool();

    [[nodiscard]]
    auto acquire() -> value_type;
    auto release(buffer_type* buf) -> void;
    auto replenish() -> void;
    auto size() -> size_type;

private:
    std::queue<buffer_type*> m_pool;
    size_type m_num_msgs{};
    size_type m_msg_size{};
    size_type m_min_size{};
    std::mutex m_mtx;

}; // class buffer_pool

} // namespace udpsrc