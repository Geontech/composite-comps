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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

template <typename T>
class window_buffer {
public:
    window_buffer() = default;
    explicit window_buffer(std::size_t len) : m_len(len) {
        if (!(len > 0)) {
            throw std::invalid_argument("window buffer length must be greater than 0");
        }
        auto m = std::size_t{};
        while ((1u << m) < m_len) {
            ++m;
        }
        m_n = 1u << m;
        m_mask = m_n - 1;
        m_buffer.resize(m_n + m_len - 1);
    }

    auto push(const T& value) -> void {
        m_read_index = (m_read_index + 1) & m_mask;
        if (m_read_index == 0) {
            std::memcpy(m_buffer.data(), m_buffer.data() + m_n, (m_len - 1) * sizeof(T));
        }
        m_buffer[m_read_index + m_len - 1] = value;
    }

    auto operator[](std::size_t idx) const -> const T& {
        return m_buffer[m_read_index + idx];
    }

    auto at(std::size_t idx) const -> const T& {
        return m_buffer.at(m_read_index + idx);
    }

    auto data() const -> const T* {
        return &m_buffer[m_read_index];
    }

    auto size() const -> std::size_t {
        return m_len;
    }

    auto reset() -> void {
        std::fill(m_buffer.begin(), m_buffer.end(), T{});
        m_read_index = 0;
    }

private:
    std::size_t m_len{};
    std::size_t m_n{};
    std::size_t m_mask{};
    std::size_t m_read_index{};
    std::vector<T> m_buffer;

}; // class window_buffer<T>
