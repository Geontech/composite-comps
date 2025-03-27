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

#include "packet_mmap.hpp"
#include "processing_queue.hpp"

#include <array>
#include <composite/component.hpp>
#include <memory>
#include <poll.h>
#include <string>
#include <string_view>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/uio.h>

class udp_source : public composite::component {
    using output_t = std::pmr::vector<uint8_t>;
    using output_port_t = composite::output_port<std::shared_ptr<output_t>>;
public:
    udp_source();
    ~udp_source() override = default;
    auto property_change_handler() -> void override;
    auto start() -> void override;
    auto stop() -> void override;
    auto process() -> composite::retval override;

private:
    // Ports
    output_port_t m_out_port{"data_out"};
    
    // Properties
    std::string m_interface;
    std::string m_ip_addr;
    uint32_t m_port{};
    std::string m_transport;
    uint32_t m_num_msgs{};
    uint32_t m_msg_size{};
    uint32_t m_recv_buf_size{};
    uint32_t m_pool_size{64};

    // Members
    std::unique_ptr<udp::packet_mmap> m_receiver;
    uint16_t m_pkt_count{};
    bool m_new_socket_required{true};
    bool m_flush_queue{true};
    uint64_t total_recvd{};
    uint64_t not_data_recvd{};
    uint64_t data_recvd{};
    uint64_t num_calls{};

}; // class udp_source
