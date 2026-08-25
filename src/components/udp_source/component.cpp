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
#include "net/utils.hpp"

#include <random>
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
#include <sys/eventfd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

udp_source::abort_event::abort_event() : m_fd(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
    if (m_fd < 0) {
        throw std::runtime_error(std::format("udp_source: failed to create abort event: {}",
                                             std::string{strerror(errno)}));
    }
}

udp_source::abort_event::~abort_event() {
    ::close(m_fd);
}

auto udp_source::abort_event::signal() const noexcept -> void {
    // EAGAIN (counter saturated) still leaves the fd readable, which is all a waiter needs.
    (void)::eventfd_write(m_fd, 1);
}

auto udp_source::abort_event::drain() const noexcept -> void {
    eventfd_t value{};
    while (::eventfd_read(m_fd, &value) == 0) {}
}

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
    add_property("recvmmsg", m_recvmmsg, RUNTIME);
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
    m_kernel_drops = &create_counter(
        "udp_source.kernel_drops", "Packets dropped by the kernel receive queue");
    m_batch_sizes = &create_histogram_pow2(
        "udp_source.batch_sizes", "Distribution of packets received per batch", "1",
        10);  // 10 buckets: 1, 2, 4, 8, ..., 512
}

auto udp_source::create_metrics() -> udp::metrics {
    return udp::metrics{
        .packets_received = *m_packets_received,
        .bytes_received = *m_bytes_received,
        .packets_dropped = *m_packets_dropped,
        .kernel_drops = *m_kernel_drops,
        .batch_sizes = *m_batch_sizes
    };
}

auto udp_source::property_change_handler(const composite::properties::json& diff) -> void {
    (void)diff;
    logger()->trace(std::source_location::current().function_name());

    // If we're transitioning to inactive, stop and release the receiver. A later
    // activation reconstructs it from the current properties, and releasing it
    // here drops this component's ownership of backend resources (including the
    // recvmmsg slab pool once any downstream buffers still in flight are freed).
    if (!m_active) {
        std::scoped_lock lock(m_receiver_mtx);
        stop_receiver_locked();
        m_receiver.reset();
        return;
    }

    // Skip receiver creation if configuration is incomplete
    if (m_ip_addr.empty() || m_port == 0) {
        logger()->debug("Skipping receiver creation: incomplete configuration (ip_addr='{}', port={})",
                       m_ip_addr, m_port);
        return;
    }

    // IN-BAND stream-boundary signal: every receiver (re)construction is a new stream
    // session (ip/port/config change, reactivation). The annotation rides every emitted
    // packet as one latched metadata instance (a refcount bump per batch), so a downstream
    // pkt_parser resets protocol detection causally ordered with the FIRST packet of the new
    // stream — no orchestration race, no failure-counting loss window.
    // Session tokens must be unique across component lifetimes AND module reloads: component
    // ids are unique among concurrently-live components only, and the control plane
    // dlclose()s this module when its last instance is removed — so ANY counter held in this
    // library (instance member or process-wide static) restarts on reload and can repeat a
    // token ("udp0:1" twice), leaving a parser that outlived the swap locked on the old
    // protocol. A per-session random nonce needs no surviving state at all: collision odds
    // are 2^-64 per pair. The id prefix stays for operators reading logs; receiver
    // construction is control-plane, so the random_device cost is irrelevant.
    std::random_device rd;
    const auto nonce = (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
    composite::metadata session_md;
    session_md.annotations["stream_session"] = std::format("{}:{:016x}", id(), nonce);

    auto config = udp::config{
        .logger = logger(),
        .session_metadata = composite::make_metadata(std::move(session_md)),
        .interface = m_interface,
        .ip_addr = m_ip_addr,
        .port = m_port,
        .recv_buf_size = m_recv_buf_size,
        .batch_size = m_num_msgs,
        .frame_count = m_frame_count,
        .autodiscovery_timeout = m_autodiscovery_timeout,
        .receive_batch_wait_us = m_recvmmsg.receive_batch_wait_us,
        .output_batch_size = m_recvmmsg.output_batch_size,
        .max_batch_delay_us = m_recvmmsg.max_batch_delay_us,
        .metrics = create_metrics(),
        .abort_fd = m_abort.fd()
    };
    if (m_overrides.msg_size.has_value()) {
        config.msg_size = m_overrides.msg_size.value();
    }
    std::unique_ptr<udp::interface> receiver;
    if (m_socket_type == PACKET_MMAP) {
        if (!m_overrides.packet_fd_path.empty()) {
            // Unprivileged mode: adopt an AF_PACKET fd from a privileged helper (see
            // overrides.packet_fd_path). Bounded wait so a missing helper fails loudly.
            config.packet_fd = net::receive_fd(m_overrides.packet_fd_path, 5000);
            logger()->info("adopted AF_PACKET fd from '{}'", m_overrides.packet_fd_path);
        }
        receiver = std::make_unique<udp::packet_mmap>(config);
    } else if (m_socket_type == DPDK) {
#ifdef COMPOSITE_HAS_DPDK
        udp::dpdk::config dpdk_cfg{
            .logger = logger(),
            .session_metadata = config.session_metadata,
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

auto udp_source::on_worker_start() -> void {
    // Stale abort signals are drained inside start_receiver_locked(), at the last moment
    // before start_recv() — the one drain point every start path shares.
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

auto udp_source::on_park_requested() -> void {
    // The framework calls this when a stop or a property writer needs the worker to yield
    // promptly. Packet-size autodiscovery can be running ON the worker (a reactivation inside
    // property_change_handler), and it is the one long wait in this component the park cannot
    // interrupt by itself — so signal it here, exactly what this hook exists for.
    // NOTE this fires for every park, writes included, so the fd accumulates counts that mean
    // "yield", not "stop" — which is why start_receiver_locked() drains immediately before
    // starting a receiver: the write that triggers a reactivation parks THIS hook first, and
    // its own signal must not abort the discovery it is about to start.
    m_abort.signal();
}

auto udp_source::on_worker_stop() -> void {
    // Signal BEFORE taking m_receiver_mtx, and without touching m_receiver (the fd is
    // component-owned, so this races nothing). In the current framework this is mostly
    // belt-and-braces — on_worker_start/on_worker_stop are serialized by the lifecycle lock,
    // so a stop cannot get here while an on_worker_start discovery holds m_receiver_mtx (that
    // case is bounded by the discovery deadline; on_park_requested() covers the worker-thread
    // reactivation case). It is kept unconditional so any future caller of on_worker_stop that
    // is NOT lifecycle-serialized still cuts the wait short rather than deadlocking on the
    // mutex below.
    m_abort.signal();
    {
        std::scoped_lock lock(m_receiver_mtx);
        stop_receiver_locked();
        m_component_running = false;
    }
    m_stat_thread.request_stop();
    if (m_stat_thread.joinable()) {
        m_stat_thread.join();
    }
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
    // Drain the abort event at the LAST moment before starting: any count that predates a
    // deliberate start is stale by definition. This matters because on_park_requested() fires
    // for EVERY park — property writes included — and a reactivation runs inside the very park
    // whose poke signalled the fd, so without this drain the write's own signal aborted the
    // discovery it was starting: any runtime write (active, ip_addr, overrides, ...) left the
    // source with no running receiver behind a single warning. A stop signalled AFTER this
    // drain still aborts discovery; one signalled before is swallowed, serialized behind
    // m_receiver_mtx, and bounded by the discovery deadline — the same limitation already
    // documented for the on_worker_start path.
    m_abort.drain();
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
