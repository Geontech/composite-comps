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

#include "component.hpp"
#include "socket/send.hpp"
#include "socket/sendmmsg.hpp"

#include <composite/core/register.hpp>

#include <arpa/inet.h>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <netinet/in.h>
#include <optional>
#include <sstream>
#include <format>

namespace {
// Ceilings for the sendmmsg staging buffer's two dimensions. A UDP datagram cannot exceed
// 65507 bytes of payload, so a larger max_packet_size can only waste memory. The batch ceiling
// is well above any useful sendmmsg vector (the kernel caps a single call at UIO_MAXIOV = 1024)
// and exists so batch_size * max_packet_size cannot be driven into an absurd allocation.
constexpr uint32_t MAX_BATCH_SIZE = 1024;
constexpr uint32_t MAX_PACKET_SIZE_LIMIT = 65507;
} // namespace

namespace {

// Read an annotation as an unsigned integer: typed integers directly, exact-integral
// doubles accepted (JSON-sourced metadata often types numbers as double), strings parsed.
auto annotation_as_uint(const composite::annotation_value& v) -> std::optional<uint64_t> {
    if (v.holds<std::int64_t>()) {
        const auto i = v.get<std::int64_t>();
        return i >= 0 ? std::optional<uint64_t>{static_cast<uint64_t>(i)} : std::nullopt;
    }
    if (v.holds<double>()) {
        const auto d = v.get<double>();
        if (d >= 0.0 && d <= 65535.0 * 65536.0 && d == std::floor(d)) {
            return static_cast<uint64_t>(d);
        }
        return std::nullopt;
    }
    const auto s = v.to_string();
    uint64_t out{};
    const auto* end = s.data() + s.size();
    if (auto [p, ec] = std::from_chars(s.data(), end, out); ec == std::errc{} && p == end) {
        return out;
    }
    return std::nullopt;
}

// A destination-IP string is usable only if inet_pton accepts it: validating here (at latch
// and property time) is what lets the senders treat an unparsable IP as exceptional.
auto valid_ipv4(const std::string& ip) -> bool {
    struct in_addr addr{};
    return inet_pton(AF_INET, ip.c_str(), &addr) == 1;
}

} // namespace

udp_sink::udp_sink(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    using enum composite::properties::config_type;
    add_property("socket_type", m_socket_type)
        .validate([](const std::string& v) { return v == udp_sink::SEND || v == udp_sink::SENDMMSG; });
    add_property("dest_ip_key", m_dest_ip_key, RUNTIME);
    add_property("dest_port_key", m_dest_port_key, RUNTIME);
    add_property("stream_id_key", m_stream_id_key, RUNTIME);
    add_property("socket_timeout_s", m_socket_timeout_s, RUNTIME).units("s");
    add_property("send_buf_size", m_send_buf_size).units("bytes");
    // batch_size and max_packet_size size the sendmmsg staging buffer
    // (batch_size * max_packet_size). Zero for either makes the buffer unusable -- a zero
    // batch_size allocates nothing and flushes on every packet -- and unbounded values invite
    // an allocation the process cannot satisfy. batch_size is RUNTIME, so these are reachable
    // from a property write on a running graph, not only from the initial config.
    //
    // The sender re-checks all of this itself: it must not trust a config it did not build.
    add_property("batch_size", m_batch_size, RUNTIME)
        .validate([](const uint32_t& v) { return v > 0 && v <= MAX_BATCH_SIZE; },
                  std::format("batch_size must be in [1, {}]", MAX_BATCH_SIZE));
    add_property("batch_timeout_us", m_batch_timeout_us, RUNTIME).units("us");
    add_property("max_packet_size", m_max_packet_size)
        .validate([](const uint32_t& v) { return v > 0 && v <= MAX_PACKET_SIZE_LIMIT; },
                  std::format("max_packet_size must be in [1, {}] bytes", MAX_PACKET_SIZE_LIMIT))
        .units("bytes");
    add_property("bind_interface", m_bind_interface);
    // An unparsable default would fail every send: reject it at the property boundary.
    add_property("default_dest_ip", m_default_dest_ip, RUNTIME)
        .validate([](const std::string& v) { return v.empty() || valid_ipv4(v); },
                  "default_dest_ip must be empty or a valid IPv4 address");
    add_property("default_dest_port", m_default_dest_port, RUNTIME);

    m_packets_sent = &create_counter("udp_sink.packets_sent", "Datagrams handed to the kernel");
    m_bytes_sent = &create_counter("udp_sink.bytes_sent", "Payload bytes handed to the kernel");
    m_send_errors = &create_counter(
        "udp_sink.send_errors", "Datagrams not sent (invalid destination, oversize, syscall failure)");
    m_packets_dropped = &create_counter(
        "udp_sink.packets_dropped", "Packets dropped before send: no destination resolved");
}

auto udp_sink::property_change_handler(const composite::properties::json& diff) -> void {
    (void)diff;
    // (Re)build the sender with the current configuration. Runs after the INITIALIZE batch
    // and on runtime changes; the worker is parked for the write, but the cleanup thread is
    // not — hence the lock. Flush the outgoing batch through the old sender first so a
    // config change never drops buffered packets.
    std::scoped_lock lock(m_sender_mtx);
    if (m_sender) {
        m_sender->flush();
    }
    m_sender = create_sender();
    // Keys or defaults may have changed; re-resolve the destination on the next packet
    // (the cached endpoint must be dropped too, or a null-metadata stream would keep it).
    m_last_meta = nullptr;
    m_last_ip.clear();
    m_last_port = 0;
    m_resolve_attempted = false;
    m_no_dest_warned = false;
    logger()->info("udp_sink: sender configured (socket_type={})", m_socket_type);
}

auto udp_sink::create_sender() -> std::unique_ptr<udp_tx::interface> {
    auto config = udp_tx::config{
        .logger = logger(),
        .send_buf_size = m_send_buf_size,
        .socket_timeout_s = m_socket_timeout_s,
        .batch_size = m_batch_size,
        .batch_timeout_us = m_batch_timeout_us,
        .max_packet_size = m_max_packet_size,
        .bind_interface = m_bind_interface,
        .packets_sent = m_packets_sent,
        .bytes_sent = m_bytes_sent,
        .send_errors = m_send_errors
    };

    if (m_socket_type == SENDMMSG) {
        return std::make_unique<udp_tx::sendmmsg_tx>(config);
    }
    return std::make_unique<udp_tx::send_tx>(config);
}

auto udp_sink::on_worker_start() -> void {
    // A sender may not exist yet if no property batch ran (all-defaults start).
    {
        std::scoped_lock lock(m_sender_mtx);
        if (!m_sender) {
            m_sender = create_sender();
        }
    }
    // Idle-socket cleanup + periodic stats, off the data path.
    m_cleanup_thread = std::jthread([this](std::stop_token token) {
        // The wait must be stop-interruptible: a plain sleep_for(5s) made every component
        // stop/reconfigure block in join() for up to the full period.
        std::mutex wait_mtx;
        std::condition_variable_any wait_cv;
        while (!token.stop_requested()) {
            {
                std::unique_lock lk(wait_mtx);
                wait_cv.wait_for(lk, token, std::chrono::seconds(5), [] { return false; });
            }
            if (token.stop_requested()) {
                break;
            }
            std::scoped_lock lock(m_sender_mtx);
            if (m_sender) {
                m_sender->cleanup_idle_sockets();
                if (logger()->should_log(composite::log_level::debug)) {
                    auto stats = m_sender->get_stats();
                    std::ostringstream oss;
                    oss << "statistics: ";
                    auto first = true;
                    for (const auto& [k, v] : stats) {
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
}

auto udp_sink::on_worker_stop() -> void {
    m_cleanup_thread.request_stop();
    if (m_cleanup_thread.joinable()) {
        m_cleanup_thread.join();
    }
    // Flush any pending batch so a stop never strands buffered packets.
    std::scoped_lock lock(m_sender_mtx);
    if (m_sender) {
        m_sender->flush();
    }
}

auto udp_sink::on_end_of_stream() -> void {
    std::scoped_lock lock(m_sender_mtx);
    if (m_sender) {
        m_sender->flush();
    }
}

auto udp_sink::process() -> composite::retval {
    using enum composite::retval;

    // Drain a bounded batch with one ring-head publication (the fleet ingest pattern);
    // destination resolution is per packet — metadata can change inside the batch.
    const auto count = m_in_port.get_batch(std::span{m_input_batch});
    if (count == 0) {
        // Going idle: flush the outgoing batch before parking on the doorbell, so a paused
        // stream's tail is not held hostage to the next packet's arrival.
        {
            std::scoped_lock lock(m_sender_mtx);
            if (m_sender) {
                m_sender->flush();
            }
        }
        return NOOP;
    }

    for (std::size_t i = 0; i < count; ++i) {
        auto& [data, ts, meta] = m_input_batch[i];
        (void)ts;

        // Resolve the destination. Steady state: the same shared metadata instance as the
        // previous packet — reuse the resolved endpoint without touching the annotations.
        // A FAILED resolution is cached too (m_resolve_attempted): the same instance
        // resolves the same way every time, and re-resolving (and re-warning) per packet
        // turned a misconfigured stream into a log flood.
        if (meta != m_last_meta || !m_resolve_attempted) {
            const auto metadata = meta ? *meta : composite::metadata{};
            auto stream_id = get_stream_id(metadata);
            auto [ip, port] = get_destination(stream_id, metadata);
            m_last_meta = meta;
            m_last_ip = std::move(ip);
            m_last_port = port;
            m_resolve_attempted = true;
            if (m_last_ip.empty() || m_last_port == 0) {
                if (!m_no_dest_warned) {
                    m_no_dest_warned = true;
                    logger()->warn("udp_sink: no destination in metadata and no default "
                                   "configured; dropping (counted in udp_sink.packets_dropped)");
                }
            } else {
                m_no_dest_warned = false;  // re-arm for the next unresolvable stream
            }
        }

        if (m_last_ip.empty() || m_last_port == 0) {
            if (m_packets_dropped != nullptr) { m_packets_dropped->inc(); }
            continue;
        }

        {
            std::scoped_lock lock(m_sender_mtx);
            if (m_sender) {
                auto sent = m_sender->send(m_last_ip, m_last_port, std::span{data.data(), data.size()});
                if (sent < 0) {
                    logger()->trace("udp_sink: send failed to {}:{}", m_last_ip, m_last_port);
                }
            }
        }
    }
    // Release the drained slots now (the senders copied what they needed): entries left in
    // the member array would otherwise pin upstream pool buffers until a later batch
    // happens to overwrite them.
    for (std::size_t i = 0; i < count; ++i) {
        m_input_batch[i] = {};
    }

    return NORMAL;
}

auto udp_sink::get_stream_id(const composite::metadata& metadata) -> uint32_t {
    if (auto it = metadata.annotations.find(m_stream_id_key); it != metadata.annotations.end()) {
        if (auto v = annotation_as_uint(it->second)) {
            return static_cast<uint32_t>(*v);
        }
    }
    return 0; // Default stream ID
}

auto udp_sink::get_destination(uint32_t stream_id, const composite::metadata& metadata)
    -> std::pair<std::string, uint16_t> {
    // Bound the per-stream state: stream ids arrive in untrusted metadata, so an upstream
    // cycling them must not grow this map without limit. At the cap, evict the
    // least-recently-resolved stream (an O(size) scan, but only on insert at the cap —
    // resolution itself only runs on a metadata change). Mirrors pkt_builder's LRU cap.
    if (auto it = m_stream_states.find(stream_id); it == m_stream_states.end()
        && m_stream_states.size() >= MAX_STREAM_STATES) {
        auto lru = m_stream_states.begin();
        for (auto cur = m_stream_states.begin(); cur != m_stream_states.end(); ++cur) {
            if (cur->second.last_used < lru->second.last_used) {
                lru = cur;
            }
        }
        logger()->debug("udp_sink: stream-state cap ({}) reached; evicting stream {}",
                        MAX_STREAM_STATES, lru->first);
        m_stream_states.erase(lru);
    }
    auto& state = m_stream_states[stream_id];
    state.last_used = ++m_stream_use_tick;

    // Latch any destination info this metadata carries. The IP is validated HERE, at the
    // boundary: latching an unparsable string would fail every send for the stream (and a
    // one-shot error deep in the sender), when falling back to the previous latch or the
    // configured default is both safer and observable.
    if (auto it = metadata.annotations.find(m_dest_ip_key); it != metadata.annotations.end()) {
        if (auto ip = it->second.to_string(); !ip.empty()) {
            if (valid_ipv4(ip)) {
                state.last_dest_ip = ip;
            } else {
                logger()->warn("udp_sink: invalid {} value in metadata: '{}' (ignored)",
                               m_dest_ip_key, ip);
            }
        }
    }
    if (auto it = metadata.annotations.find(m_dest_port_key); it != metadata.annotations.end()) {
        if (auto v = annotation_as_uint(it->second); v && *v != 0 && *v <= 65535) {
            state.last_dest_port = static_cast<uint16_t>(*v);
        } else {
            logger()->warn("udp_sink: invalid {} value in metadata: '{}'", m_dest_port_key,
                           it->second.to_string());
        }
    }

    // Latched values win; fall back to the configured defaults.
    auto ip = state.last_dest_ip.empty() ? m_default_dest_ip : state.last_dest_ip;
    auto port = state.last_dest_port == 0 ? m_default_dest_port : state.last_dest_port;
    return {std::move(ip), port};
}

COMPOSITE_REGISTER_SIMPLE(udp_sink)
