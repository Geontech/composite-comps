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

#include <composite/component.hpp>
#include <vector>

class file_writer : public composite::component {
    using input_port_t = composite::input_port<std::vector<std::vector<uint8_t>>>;
public:
    file_writer();
    ~file_writer() override;
    auto initialize() -> void override;
    auto process() -> composite::retval override;

private:
    // Ports
    std::unique_ptr<input_port_t> m_in_port{std::make_unique<input_port_t>("data_in")};

    // Properties
    std::string m_filename;
    uint64_t m_num_bytes{};

    // Members
    int m_file{-1};
    uint64_t m_total_bytes{};

}; // class file_writer
