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
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

#include "component.hpp"
#include "socket/packet_mmap.hpp"
#include "socket/recvmmsg.hpp"
#include "socket/dpdk.hpp"

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

udp_source::udp_source(std::string_view id) : composite::component(id) {
    add_port(&m_out_port);
    using enum composite::properties::config_type;
    add_property("active", m_active, RUNTIME);
    add_property("socket_type", m_socket_type).change_listener([this]() {
        return m_socket_type == RECVMMSG
            || m_socket_type == PACKET_MMAP
            || m_socket_type == DPDK;
    });
    add_property("interface", m_interface, RUNTIME);
    add_property("ip_addr", m_ip_addr, RUNTIME);
    add_property("port", m_port, RUNTIME);
    add_property("recv_buf_size", m_recv_buf_size).units("bytes");
    add_property("num_msgs", m_num_msgs, RUNTIME);
    add_property("frame_count", m_frame_count, RUNTIME);
    add_property("autodiscovery_timeout", m_autodiscovery_timeout).units("seconds");
    add_property("overrides", m_overrides, RUNTIME);
    add_property("dpdk", m_dpdk);
}

auto udp_source::property_change_handler() -> void {
    logger()->trace(std::source_location::current().function_name());

    // If we're transitioning to inactive, just stop the receiver
    if (!m_active) {
        std::scoped_lock lock(m_receiver_mtx);
        stop_receiver_locked();
        return;
    }

    // Skip receiver creation if configuration is incomplete
    if (m_ip_addr.empty() || m_port == 0) {
        logger()->debug("Skipping receiver creation: incomplete configuration (ip_addr='{}', port={})",
                       m_ip_addr, m_port);
        return;
    }

    auto config = udp::config{
        .logger = logger(),
        .interface = m_interface,
        .ip_addr = m_ip_addr,
        .port = m_port,
        .recv_buf_size = m_recv_buf_size,
        .batch_size = m_num_msgs,
        .frame_count = m_frame_count,
        .autodiscovery_timeout = m_autodiscovery_timeout
    };
    if (m_overrides.msg_size.has_value()) {
        config.msg_size = m_overrides.msg_size.value();
    }
    std::unique_ptr<udp::interface> receiver;
    if (m_socket_type == PACKET_MMAP) {
        receiver = std::make_unique<udp::packet_mmap>(config);
    } else if (m_socket_type == DPDK) {
#ifdef COMPOSITE_HAS_DPDK
        udp::dpdk::config dpdk_cfg{
            .logger = logger(),
            .interface = m_interface,
            .ip_addr = m_ip_addr,
            .port = m_port,
            .port_id = m_dpdk.port_id,
            .queue_id = m_dpdk.queue_id,
            .mempool_name = m_dpdk.mempool_name,
            .burst_size = m_dpdk.burst_size,
            .src_ip = m_dpdk.src_ip,
            .igmp_respond_to_queries = m_dpdk.igmp_respond_to_queries
        };
        receiver = std::make_unique<udp::dpdk>(dpdk_cfg);
#else
        logger()->error("DPDK support not compiled in");
        throw std::runtime_error("DPDK support not available");
#endif
    } else { // recvmmsg
        receiver = std::make_unique<udp::recvmmsg>(config);
    }
    {
        std::scoped_lock lock(m_receiver_mtx);
        stop_receiver_locked();
        m_receiver.reset();
        m_receiver = std::move(receiver);
        if (m_component_running) {
            start_receiver_locked();
        }
    }
}

auto udp_source::start() -> void {
    component::start();
    {
        std::scoped_lock lock(m_receiver_mtx);
        m_component_running = true;
        start_receiver_locked();
    }
    m_stat_thread = std::jthread([this](std::stop_token token) {
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            if (!token.stop_requested()) {
                if (logger()->should_log(spdlog::level::debug)) {
                    std::map<std::string, std::string> stats;
                    {
                        std::scoped_lock lock(m_receiver_mtx);
                        if (!m_receiver) {
                            continue;
                        }
                        stats = m_receiver->get_stats();
                    }
                    std::ostringstream oss;
                    oss << "statistics: ";
                    auto first = true;
                    for (const auto& [k ,v] : stats) {
                        if (!first) {
                            oss << ", ";
                        }
                        first = false;
                        oss << k << "=" << v;
                    }
                    logger()->debug("{}", oss.str());
                }
            }
        }
    });
    if (auto ret = pthread_setname_np(m_stat_thread.native_handle(), "udp_stats"); ret != 0) {
        logger()->warn("failed to set thread name: {}", std::string{strerror(ret)});
    }
}

auto udp_source::stop() -> void {
    {
        std::scoped_lock lock(m_receiver_mtx);
        stop_receiver_locked();
        m_component_running = false;
    }
    m_stat_thread.request_stop();
    if (m_stat_thread.joinable()) {
        m_stat_thread.join();
    }
    component::stop();
}

auto udp_source::process() -> composite::retval {
    // Output provided by the receiver classes
    // Returning FINISH to shut this thread down
    return composite::retval::FINISH;
}

auto udp_source::start_receiver_locked() -> void {
    if (!m_active || !m_receiver || m_receiver_running) {
        return;
    }
    m_receiver->start_recv(&m_out_port);
    m_receiver_running = true;
}

auto udp_source::stop_receiver_locked() -> void {
    if (!m_receiver || !m_receiver_running) {
        return;
    }
    m_receiver->stop_recv();
    m_receiver_running = false;
}

extern "C" {
    auto create(std::string_view id) -> std::shared_ptr<composite::component> {
        return std::make_shared<udp_source>(id);
    }
}
