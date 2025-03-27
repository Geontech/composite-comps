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

#include "component.hpp"
#include "overlay.hpp"

#include <arpa/inet.h>
#include <array>
#include <complex>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <netinet/in.h>
#include <net/if.h>
#include <source_location>
#include <spdlog/spdlog.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

udp_source::udp_source() : composite::component("udp_source") {
    add_port(&m_out_port);
    using enum composite::properties::config_type;
    add_property("interface", &m_interface).configurability(RUNTIME).change_listener([this]() {
        m_new_socket_required = true;
        return true;
    });
    add_property("ip_addr", &m_ip_addr).configurability(RUNTIME).change_listener([this]() {
        m_new_socket_required = true;
        return true;
    });
    add_property("port", &m_port).configurability(RUNTIME).change_listener([this]() {
        m_new_socket_required = true;
        return true;
    });
    add_property("transport", &m_transport).configurability(RUNTIME).change_listener([this]() {
        m_flush_queue = true;
        return (m_transport == "sdds") || (m_transport == "vita49");
    });
    add_property("recv_buf_size", &m_recv_buf_size).units("bytes");
    add_property("msg_size", &m_msg_size).units("bytes").configurability(RUNTIME).change_listener([this]() {
        m_flush_queue = true;
        return true;
    });
}

auto udp_source::property_change_handler() -> void {
    logger()->trace(std::source_location::current().function_name());
    if (m_new_socket_required) {
        logger()->trace("property changes indicate new socket is required; closing socket");
        m_receiver.reset();
        m_receiver = std::make_unique<udp::packet_mmap>(m_interface, m_ip_addr, m_port);
    }
}

auto udp_source::start() -> void {
    component::start();
    m_receiver->start();
    m_stat_thread = std::jthread([this](std::stop_token stoken) {
        while (!stoken.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            auto stats = m_receiver->get_stats();
            logger()->trace(
                "statistics: pkts_recvd_user={}, pkts_recvd_kernel={}, pkts_dropped_kernel={}",
                stats.pkts_recvd_user, stats.pkts_recvd_kernel, stats.pkts_dropped_kernel
            );
        }
    });
}

auto udp_source::stop() -> void {
    m_stat_thread.request_stop();
    if (m_stat_thread.joinable()) {
        m_stat_thread.join();
    }
    m_receiver->stop();
    component::stop();
}

auto udp_source::process() -> composite::retval {
    using enum composite::retval;

    // Get data from intermediate queue
    auto data = output_t{nullptr};
    if (!m_receiver->get_data(data)) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        return NORMAL;
    } else if (data == nullptr) {
        return NORMAL;
    }

    // Extract metadata
    auto ts = composite::timestamp{};
    // Parse packets based on protocol
    if (m_transport == "sdds") {
        auto packet = overlay::sdds::overlay(*data);
        auto seq_num = packet.seq_num();
        if (packet.pp_id() && ((seq_num % 32) != 31)) [[unlikely]] {
            logger()->error("invalid SDDS packet received, pp_id=true, seq_num={}", seq_num);
        } else if (!packet.pp_id() && ((seq_num % 32) == 31)) [[unlikely]] {
            logger()->error("invalid SDDS packet received pp_id=false, seq_num={}", seq_num);
        }
        auto expected_seq_num = static_cast<uint16_t>(m_pkt_count + 1);
        if ((expected_seq_num % 32) == 31) {
            ++expected_seq_num;
        }
        if (seq_num != expected_seq_num) [[unlikely]] {
            logger()->warn("dropped pkt(s) expected={}, got={}", expected_seq_num, seq_num);
        }
        m_pkt_count = seq_num;
        ts = composite::timestamp{packet.secs(), packet.psecs()};
        data->erase(data->begin(), data->begin() + 56); // move metadata off
        data->resize(1024);
    } else if (m_transport == "vita49") {
        // auto packet = overlay::v49::overlay(*data);
        // auto& header = packet.header();
        // if (overlay::v49::is_data(header)) {
        //     if (auto expected_count = ((m_pkt_count + 1) % 16); header.packet_count() != expected_count) {
        //         logger()->warn("dropped pkt(s) expected={}, got={}", expected_count, header.packet_count());   
        //     }
        //     m_pkt_count = header.packet_count();
        // }
    }

    // Send data
    m_out_port.send_data(std::move(data), ts);
    
    // Fast return
    return NO_YIELD;
}

extern "C" {
    auto create() -> std::shared_ptr<composite::component> {
        return std::make_shared<udp_source>();
    }
}
