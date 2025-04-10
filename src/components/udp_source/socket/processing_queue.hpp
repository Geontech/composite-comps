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
#include <mutex>
#include <queue>

template <typename T>
class processing_queue {
public:
    using size_type = std::size_t;
    using value_type = T;

    auto pop() -> value_type {
        using namespace std::chrono_literals;
        auto lock = std::unique_lock{m_mtx};
        m_cv.wait_for(lock, 1s, [this]{ return !m_queue.empty(); });
        if (m_queue.empty()) {
            return nullptr;
        }
        auto retval = std::move(m_queue.front());
        m_queue.pop_front();
        return retval;
    }

    auto push(value_type data) -> void {
        const auto lock = std::scoped_lock{m_mtx};
        m_queue.emplace_back(std::move(data));
        m_cv.notify_one();
    }

    auto clear() -> void {
        const auto lock = std::scoped_lock{m_mtx};
        return m_queue.clear();
    }

    auto empty() const -> bool {
        const auto lock = std::scoped_lock{m_mtx};
        return m_queue.empty();
    }

    auto size() const -> size_type {
        const auto lock = std::scoped_lock{m_mtx};
        return m_queue.size();
    }

private:
    std::deque<value_type> m_queue;
    std::mutex m_mtx;
    std::condition_variable m_cv;

}; // processing_queue
