/*
 * Copyright (C) 2024 Geon Technologies, LLC
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

#include "aligned_mem.hpp"

#include <composite/component.hpp>
#include <complex>
#include <fcntl.h>
#include <string>

template <typename T>
class aligned_mem_writer : public composite::component {
    using input_t = aligned::aligned_mem<T>;
    using input_port_t = composite::input_port<input_t>;
public:
    aligned_mem_writer() : composite::component("aligned_mem_writer") {
        add_port(m_in_port.get());
        add_property("filename", &m_filename);
        add_property("num_bytes", &m_num_bytes);
    }

    ~aligned_mem_writer() override {
        close(m_file);
    }

    auto initialize() -> void override {
        m_file = open(m_filename.c_str(), O_CREAT|O_TRUNC|O_WRONLY, 0644);
    }

    auto process() -> composite::retval override {
        using enum composite::retval;
        auto [data, ts] = m_in_port->get_data();
        if (data == nullptr) {
            return NOOP;
        }
        auto write_len = data->size_bytes();
        if (m_bytes_written + write_len >= m_num_bytes) {
            write_len -= (m_bytes_written + write_len - m_num_bytes);
        }
        if (auto num_written = write(m_file, data->data(), write_len); num_written != -1) {
            m_bytes_written += num_written;
            if (m_bytes_written >= m_num_bytes) {
                m_in_port->clear();
                return FINISH;
            }
        }
        return NORMAL;
    }

private:
    // Ports
    std::unique_ptr<input_port_t> m_in_port{std::make_unique<input_port_t>("data_in")};

    // Properties
    std::string m_filename;
    uint64_t m_num_bytes{};

    // Members
    int m_file{-1};
    uint64_t m_bytes_written{};

}; // class aligned_mem_writer
