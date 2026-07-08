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

#pragma once

#include <composite/core/logger.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

// Forward declarations
class websocket_session;
class websocket_server_impl;

// Message data type for WebSocket communication
using ws_message_data_t = std::variant<std::string, std::shared_ptr<const std::vector<uint8_t>>>;

// WebSocket session abstraction
class websocket_session {
  public:
    using message_data_t = ws_message_data_t;
    using queue_size_callback_t = std::function<void(const std::string& client_id, size_t queue_size)>;

    virtual ~websocket_session() = default;
    virtual auto get_id() const -> std::string = 0;
    virtual auto send(message_data_t data, bool is_binary) -> void = 0;
    virtual auto close() -> void = 0;
    // Rebind where this session reports its send-queue size. Call before any send()
    // is issued to the session (there is no synchronization against in-flight sends).
    virtual auto set_queue_size_callback(queue_size_callback_t callback) -> void = 0;
};

// WebSocket server abstraction
class websocket_server {
  public:
    using connect_handler_t = std::function<void(std::shared_ptr<websocket_session>)>;
    using disconnect_handler_t = std::function<void(const std::string& client_id)>;
    using message_handler_t =
        std::function<void(const std::string& client_id, const std::string& message)>;
    using queue_size_callback_t =
        std::function<void(const std::string& client_id, size_t queue_size)>;

    websocket_server();
    ~websocket_server();

    // Logging goes through the framework's logger facade (spdlog is not part of the
    // composite 0.5 public API). Set before start(); defaults to a no-op logger.
    auto set_logger(std::shared_ptr<composite::logger> logger) -> void;

    // Lifecycle. max_connections caps concurrent clients at the accept layer.
    auto start(uint16_t port, const std::string& bind_address, bool enable_compression,
               size_t max_connections) -> void;
    auto stop() -> void;

    // Callbacks
    auto set_connect_handler(connect_handler_t handler) -> void;
    auto set_disconnect_handler(disconnect_handler_t handler) -> void;
    auto set_message_handler(message_handler_t handler) -> void;
    auto set_queue_size_callback(queue_size_callback_t callback) -> void;

    // Sending
    auto send_to_client(const std::string& client_id, ws_message_data_t data, bool is_binary)
        -> void;

  private:
    std::unique_ptr<websocket_server_impl> m_impl;
};
