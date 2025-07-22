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

#include <array>
#include <composite/component.hpp>
#include <memory>
#include <poll.h>
#include <string>
#include <string_view>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/uio.h>

enum class transport : uint8_t {
    unknown,
    sdds,
    vita49
}; // enum class transport

class pkt_parser : public composite::component {
    // using input_t = const uint8_t;
    // using input_port_t = composite::input_port<input_t>;
    // using output_t = input_t;
    // using output_port_t = composite::output_port<output_t>;
    using input_t = std::pmr::vector<uint8_t>;
    using input_port_t = composite::input_port<std::shared_ptr<input_t>>;
    using output_t = input_t;
    using output_port_t = composite::output_port<std::shared_ptr<output_t>>;
public:
    pkt_parser();
    ~pkt_parser() override = default;
    auto property_change_handler() -> void override;
    auto process() -> composite::retval override;

private:
    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    struct format {
        std::optional<bool> is_complex;
        std::string type;
        uint32_t bit_width{};
        std::string endianness;
    }; // struct format
    struct signal_overrides {
        std::optional<double> center_frequency;
        std::optional<double> bandwidth;
        std::optional<double> sample_rate;
        format data_format;
        std::string transport;
    }; // struct signal_overrides
    signal_overrides m_signal_overrides;

    // Members
    transport m_transport;
    composite::metadata m_metadata;
    bool m_init_metadata{};
    uint16_t m_pkt_count{};

}; // class pkt_parser
