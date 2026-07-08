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

#include <charconv>
#include <chrono>
#include <optional>
#include <sstream>

namespace {

// Read an annotation as an unsigned integer: typed integers directly, strings parsed.
auto annotation_as_uint(const composite::annotation_value& v) -> std::optional<uint64_t> {
    if (v.holds<std::int64_t>()) {
        const auto i = v.get<std::int64_t>();
        return i >= 0 ? std::optional<uint64_t>{static_cast<uint64_t>(i)} : std::nullopt;
    }
    const auto s = v.to_string();
    uint64_t out{};
    const auto* end = s.data() + s.size();
    if (auto [p, ec] = std::from_chars(s.data(), end, out); ec == std::errc{} && p == end) {
        return out;
    }
    return std::nullopt;
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
    add_property("batch_size", m_batch_size, RUNTIME);
    add_property("batch_timeout_us", m_batch_timeout_us, RUNTIME).units("us");
    add_property("max_packet_size", m_max_packet_size).units("bytes");
    add_property("bind_interface", m_bind_interface);
    add_property("default_dest_ip", m_default_dest_ip, RUNTIME);
    add_property("default_dest_port", m_default_dest_port, RUNTIME);
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
        .bind_interface = m_bind_interface
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
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
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

    auto pkt = m_in_port.try_get();
    if (!pkt) {
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
    auto& [data, ts, meta] = *pkt;
    (void)ts;

    // Resolve the destination. Steady state: the same shared metadata instance as the
    // previous packet with an endpoint already resolved — reuse it without touching the
    // annotations. The cache must be treated as empty until a resolution succeeded, or a
    // first packet with no metadata (null == null) would skip resolution entirely.
    if (meta != m_last_meta || m_last_ip.empty() || m_last_port == 0) {
        const auto metadata = meta ? *meta : composite::metadata{};
        auto stream_id = get_stream_id(metadata);
        auto [ip, port] = get_destination(stream_id, metadata);
        m_last_meta = meta;
        m_last_ip = std::move(ip);
        m_last_port = port;
    }

    if (m_last_ip.empty() || m_last_port == 0) {
        logger()->warn("udp_sink: no destination in metadata and no default configured; dropping packet");
        return NORMAL;
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
    auto& state = m_stream_states[stream_id];

    // Latch any destination info this metadata carries.
    if (auto it = metadata.annotations.find(m_dest_ip_key); it != metadata.annotations.end()) {
        if (auto ip = it->second.to_string(); !ip.empty()) {
            state.last_dest_ip = ip;
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
