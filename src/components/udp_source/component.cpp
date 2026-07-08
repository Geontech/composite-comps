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

#include <composite/core/register.hpp>

#include <arpa/inet.h>
#include <array>
#include <complex>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <netinet/in.h>
#include <net/if.h>
#include <source_location>
#include <sys/ioctl.h>
#include <sys/socket.h>

udp_source::udp_source(std::string_view id) : composite::component(id) {
    add_port(&m_out_port);
    using enum composite::properties::config_type;
    add_property("active", m_active, RUNTIME);
    add_property("socket_type", m_socket_type).validate([](const std::string& v) {
        return v == RECVMMSG
            || v == PACKET_MMAP
            || v == DPDK;
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

    // Create metrics via the base helpers, which label each series with this component's
    // RAW id so ~component's remove_by_label("component_id", id) cleans them up on
    // destruction (a sanitized label would never match and the series would leak).
    m_packets_received = &create_counter(
        "udp_source.packets_received", "Total UDP packets received");
    m_bytes_received = &create_counter(
        "udp_source.bytes_received", "Total bytes received from network", "bytes");
    m_packets_dropped = &create_counter(
        "udp_source.packets_dropped", "Packets dropped due to filtering or errors");
    m_batch_sizes = &create_histogram_pow2(
        "udp_source.batch_sizes", "Distribution of packets received per batch", "1",
        10);  // 10 buckets: 1, 2, 4, 8, ..., 512
}

auto udp_source::create_metrics() -> udp::metrics {
    return udp::metrics{
        .packets_received = *m_packets_received,
        .bytes_received = *m_bytes_received,
        .packets_dropped = *m_packets_dropped,
        .batch_sizes = *m_batch_sizes
    };
}

auto udp_source::property_change_handler(const composite::properties::json& diff) -> void {
    (void)diff;
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
        .autodiscovery_timeout = m_autodiscovery_timeout,
        .metrics = create_metrics()
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
            .igmp_respond_to_queries = m_dpdk.igmp_respond_to_queries,
            .metrics = create_metrics()
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
                if (logger()->should_log(composite::log_level::debug)) {
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
    // Data output is produced by the receiver classes (packet_mmap / recvmmsg / dpdk) on their
    // OWN threads, which write straight to m_out_port via start_recv(&m_out_port). This worker
    // has nothing to do. Return NOOP — NOT FINISH: a clean FINISH now auto-fires send_eos() on
    // completion (framework component completion path), which would close m_out_port at startup
    // while the receiver thread keeps pushing packets into it — an at-startup end-of-stream plus
    // two threads writing a single-producer port. NOOP parks this worker on the idle cadence (it
    // has no input ring to wake it); stop()'s stop-token wakes it immediately. A live UDP stream
    // has no natural EOF, so udp_source never sends EOS — an external stop() does not (only a
    // self-FINISH would).
    return composite::retval::NOOP;
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

COMPOSITE_REGISTER_SIMPLE(udp_source)
