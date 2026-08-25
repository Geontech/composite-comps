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

#include <composite/core/component.hpp>
#include <composite/metrics/metrics.hpp>
#include <array>
#include <cstdint>
#include <string>

class file_writer : public composite::component {
    // General-purpose byte-stream writer. In the redesigned pipeline every data hop is an
    // immutable_buffer<uint8_t>; pkt_parser already strips VITA-49 headers and emits payload
    // bytes, so file_writer just appends the byte stream it receives (no overlay parsing here).
    using input_port_t = composite::input_port<composite::immutable_buffer<uint8_t>>;

    // One writev() covers the whole drained batch, so this also caps the iovec count —
    // far below IOV_MAX (1024).
    static constexpr std::size_t INPUT_BATCH_SIZE{32};
public:
    explicit file_writer(std::string_view id);
    ~file_writer() override;
    auto initialize() -> void override;
    auto process() -> composite::retval override;

private:
    // Ports
    input_port_t m_in_port{"data_in"};
    std::array<input_port_t::queue_type, INPUT_BATCH_SIZE> m_input_batch;

    // Properties
    std::string m_filename;
    uint64_t m_num_bytes{};  // stop after this many bytes; 0 means write until the stream ends

    // Members
    int m_file{-1};
    uint64_t m_total_bytes{};
    composite::metrics::counter<uint64_t>* m_bytes_written{nullptr};

    // MUST be last: stops the framework worker before any member above destructs (the base
    // ~component stops too late). See component.hpp auto_stop.
    composite::component::auto_stop m_auto_stop{*this};

}; // class file_writer
