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

#include <source_location>

udp_sink::udp_sink(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    using enum composite::properties::config_type;
    add_property("socket_type", m_socket_type).change_listener([this]() {
        return m_socket_type == SEND || m_socket_type == SENDMMSG;
    });
    add_property("dest_ip_key", m_dest_ip_key, RUNTIME);
    add_property("dest_port_key", m_dest_port_key, RUNTIME);
    add_property("socket_timeout_s", m_socket_timeout_s, RUNTIME).units("s");
    add_property("send_buf_size", m_send_buf_size).units("bytes");
    add_property("batch_size", m_batch_size, RUNTIME);
    add_property("batch_timeout_us", m_batch_timeout_us, RUNTIME).units("us");
    add_property("bind_interface", m_bind_interface);
    add_property("default_dest_ip", m_default_dest_ip, RUNTIME);
    add_property("default_dest_port", m_default_dest_port, RUNTIME);
}

udp_sink::~udp_sink() {
    stop();
}

auto udp_sink::property_change_handler() -> void {
    logger()->trace(std::source_location::current().function_name());

    auto config = udp_tx::config{
        .logger = logger(),
        .send_buf_size = m_send_buf_size,
        .socket_timeout_s = m_socket_timeout_s,
        .batch_size = m_batch_size,
        .batch_timeout_us = m_batch_timeout_us,
        .bind_interface = m_bind_interface
    };

    std::unique_ptr<udp_tx::interface> sender;
    if (m_socket_type == SENDMMSG) {
        sender = std::make_unique<udp_tx::sendmmsg_tx>(config);
    } else {
        sender = std::make_unique<udp_tx::send_tx>(config);
    }

    {
        std::scoped_lock lock(m_sender_mtx);
        stop_sender_locked();
        m_sender = std::move(sender);
        if (m_component_running) {
            start_sender_locked();
        }
    }
}

auto udp_sink::start() -> void {
    component::start();
    {
        std::scoped_lock lock(m_sender_mtx);
        m_component_running = true;
        start_sender_locked();
    }

    // Start cleanup thread
    m_cleanup_thread = std::jthread([this](std::stop_token token) {
        while (!token.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            if (!token.stop_requested()) {
                std::scoped_lock lock(m_sender_mtx);
                if (m_sender) {
                    m_sender->cleanup_idle_sockets();

                    // Log stats periodically
                    if (logger()->should_log(spdlog::level::debug)) {
                        auto stats = m_sender->get_stats();
                        std::ostringstream oss;
                        oss << "statistics: ";
                        auto first = true;
                        for (const auto& [k, v] : stats) {
                            if (!first) oss << ", ";
                            first = false;
                            oss << k << "=" << v;
                        }
                        logger()->debug("{}", oss.str());
                    }
                }
            }
        }
    });
}

auto udp_sink::stop() -> void {
    // Stop cleanup thread
    m_cleanup_thread.request_stop();
    if (m_cleanup_thread.joinable()) {
        m_cleanup_thread.join();
    }

    {
        std::scoped_lock lock(m_sender_mtx);
        stop_sender_locked();
        m_component_running = false;
    }

    component::stop();
}

auto udp_sink::process() -> composite::retval {
    using enum composite::retval;

    // Get input data
    auto [data, timestamp, metadata_opt] = m_in_port.get_data();
    if (!data) {
        // Flush any pending batch when no data
        std::scoped_lock lock(m_sender_mtx);
        if (m_sender) {
            m_sender->flush();
        }
        return NORMAL;
    }

    // Get destination from metadata
    auto metadata = metadata_opt.value_or(composite::metadata{});
    auto [dest_ip, dest_port] = get_destination(metadata);

    if (dest_ip.empty() || dest_port == 0) {
        logger()->warn("No destination specified in metadata and no default configured");
        return NORMAL;
    }

    // Send the packet
    {
        std::scoped_lock lock(m_sender_mtx);
        if (m_sender) {
            auto sent = m_sender->send(dest_ip, dest_port, std::span{data.data(), data.size()});
            if (sent < 0) {
                logger()->trace("Send failed to {}:{}", dest_ip, dest_port);
            }
        }
    }

    return NORMAL;
}

auto udp_sink::get_destination(const composite::metadata& metadata) -> std::pair<std::string, uint16_t> {
    std::string ip;
    uint16_t port = 0;

    // Try to get IP from metadata
    if (auto it = metadata.annotations.find(m_dest_ip_key); it != metadata.annotations.end()) {
        ip = it->second;
    } else {
        ip = m_default_dest_ip;
    }

    // Try to get port from metadata
    if (auto it = metadata.annotations.find(m_dest_port_key); it != metadata.annotations.end()) {
        try {
            port = static_cast<uint16_t>(std::stoul(it->second));
        } catch (...) {
            logger()->warn("Invalid {} value in metadata: '{}'", m_dest_port_key, it->second);
            port = m_default_dest_port;
        }
    } else {
        port = m_default_dest_port;
    }

    return {ip, port};
}

auto udp_sink::start_sender_locked() -> void {
    // Sender is already started when created
    logger()->info("UDP sink started");
}

auto udp_sink::stop_sender_locked() -> void {
    if (m_sender) {
        m_sender->flush();
        logger()->info("UDP sink stopped");
    }
}

extern "C" {
    auto create(std::string_view id) -> std::shared_ptr<composite::component> {
        return std::make_shared<udp_sink>(id);
    }
}
