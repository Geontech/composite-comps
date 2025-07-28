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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <format>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

class task_queue {
    using task_type = std::packaged_task<void()>;
public:
    task_queue() : task_queue(1) {}
    explicit task_queue(std::size_t num_workers) {
        resize(num_workers);
    }

    ~task_queue() {
        for (auto& w : m_workers) {
            w.request_stop();
        }
        m_cv.notify_all();
    }

    task_queue(const task_queue&) = delete;
    task_queue(task_queue&&) = delete;
    task_queue& operator=(const task_queue&) = delete;
    task_queue& operator=(task_queue&&) = delete;

    auto size() const noexcept -> std::size_t {
        auto lock = std::scoped_lock{m_wmtx};
        return m_workers.size();
    }

    auto resize(std::size_t size) -> void {
        auto lock = std::scoped_lock{m_wmtx};
        const auto curr_size = m_workers.size();
        if (size == curr_size) {
            return;
        }

        if (size > curr_size) {
            // Add new workers
            for (std::size_t i = curr_size; i < size; ++i) {
                m_workers.emplace_back([this](std::stop_token stoken) {
                    thread_func(stoken);
                });
                const auto name = std::format("fft_worker_{}", i);
                pthread_setname_np(m_workers.back().native_handle(), name.c_str());
            }
        } else {
            // Stop existing workers
            for (std::size_t i = size; i < curr_size; ++i) {
                m_workers[i].request_stop();
            }
            m_cv.notify_all();

            // Join excess threads and shrink vector
            for (std::size_t i = size; i < curr_size; ++i) {
                if (m_workers[i].joinable()) {
                    m_workers[i].join();
                }
            }
            m_workers.resize(size);
        }
    }

    // Submit a task and get a future
    template <typename F, typename... Args>
    requires requires(F&& f, Args&&... args) {
        std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    }
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using result_type = std::invoke_result_t<F, Args...>;
        auto task = std::packaged_task<result_type()>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        auto fut = task.get_future();
        {
            auto lock = std::scoped_lock{m_mtx};
            m_tasks.emplace(std::move(task));
        }
        m_cv.notify_one();
        return fut;
    }

private:
    auto thread_func(std::stop_token stoken) -> void {
        while (!stoken.stop_requested()) {
            using namespace std::chrono_literals;
            // Wait for task to be available on the queue
            auto lock = std::unique_lock{m_mtx};
            m_cv.wait(lock, [this, &stoken]{ return !m_tasks.empty() || stoken.stop_requested(); });
            if (m_tasks.empty()) {
                continue;
            }
            // Get task from queue
            auto task = std::move(m_tasks.front());
            m_tasks.pop();
            lock.unlock();
            // Execute task
            task();
        }
    }

    std::vector<std::jthread> m_workers;
    std::queue<task_type> m_tasks;
    std::mutex m_mtx;
    std::condition_variable_any m_cv;
    mutable std::mutex m_wmtx;

}; // class task_queue
