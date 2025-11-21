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

#include "websocket_server.hpp"

#include <ixwebsocket/IXWebSocketServer.h>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <queue>
#include <shared_mutex>
#include <spdlog/spdlog.h>
#include <thread>
#include <unordered_map>

// Message type for queue
struct queued_message {
    std::variant<std::string, std::shared_ptr<const std::vector<uint8_t>>> data;
    bool is_binary;
};

// WebSocket session implementation using IXWebSocket
class websocket_session_impl : public websocket_session {
  public:
    websocket_session_impl(std::shared_ptr<ix::WebSocket> socket, const std::string& id)
        : m_socket(std::move(socket)), m_id(id) {
        // Start worker thread (jthread auto-joins on destruction)
        m_worker_thread = std::jthread([this](std::stop_token stoken) { worker_loop(stoken); });
    }

    ~websocket_session_impl() = default;

    auto get_id() const -> std::string override {
        return m_id;
    }

    auto send(message_data_t data, bool is_binary) -> void override {
        size_t queue_size;
        {
            std::lock_guard lock(m_queue_mutex);
            m_send_queue.push({std::move(data), is_binary});
            queue_size = m_send_queue.size();
            m_queue_cv.notify_one();
        }

        // Update queue size for backpressure detection
        m_queue_size_callback(m_id, queue_size);
    }

    auto close() -> void override {
        if (m_socket) {
            m_socket->close();
        }
    }

    auto set_queue_size_callback(std::function<void(const std::string&, size_t)> callback) -> void {
        m_queue_size_callback = std::move(callback);
    }

  private:
    auto worker_loop(std::stop_token stoken) -> void {
        // Set thread name for debugging (Linux limit: 15 chars)
        size_t start = m_id.length() > 8 ? m_id.length() - 8 : 0;
        std::string thread_name = "ws_clt_" + m_id.substr(start);
        pthread_setname_np(pthread_self(), thread_name.c_str());

        while (!stoken.stop_requested()) {
            queued_message msg;
            size_t queue_size;

            // Wait for message or shutdown signal
            {
                std::unique_lock lock(m_queue_mutex);
                m_queue_cv.wait(lock, stoken, [this] { return !m_send_queue.empty(); });

                if (stoken.stop_requested() && m_send_queue.empty()) {
                    break;
                }

                if (m_send_queue.empty()) {
                    continue;
                }

                msg = std::move(m_send_queue.front());
                m_send_queue.pop();
                queue_size = m_send_queue.size();
            }

            // Send without holding lock (this may block)
            try {
                if (std::holds_alternative<std::string>(msg.data)) {
                    auto& str = std::get<std::string>(msg.data);
                    if (msg.is_binary) {
                        m_socket->sendBinary(str);
                    } else {
                        m_socket->sendText(str);
                    }
                } else {
                    auto& vec = std::get<std::shared_ptr<const std::vector<uint8_t>>>(msg.data);
                    std::string binary_str(reinterpret_cast<const char*>(vec->data()), vec->size());
                    m_socket->sendBinary(binary_str);
                }
            } catch (const std::exception& e) {
                spdlog::error("WebSocket send error for client {}: {}", m_id, e.what());
            }

            // Update queue size for backpressure (outside lock)
            m_queue_size_callback(m_id, queue_size);
        }
    }

    std::string m_id;
    std::shared_ptr<ix::WebSocket> m_socket;
    std::queue<queued_message> m_send_queue;
    std::mutex m_queue_mutex;
    std::condition_variable_any m_queue_cv;
    std::jthread m_worker_thread;
    std::function<void(const std::string&, size_t)> m_queue_size_callback;
};

// Server implementation using IXWebSocket
class websocket_server_impl {
  public:
    websocket_server_impl() {
        m_server = std::make_unique<ix::WebSocketServer>(8080, "0.0.0.0");
    }

    ~websocket_server_impl() {
        stop();
    }

    auto start(uint16_t port, const std::string& bind_address, bool enable_compression) -> void {
        m_server = std::make_unique<ix::WebSocketServer>(port, bind_address);

        // Configure per-message deflate compression
        if (!enable_compression) {
            m_server->disablePerMessageDeflate();
        }

        m_server->setOnClientMessageCallback(
            [this](std::shared_ptr<ix::ConnectionState> connectionState, ix::WebSocket& webSocket,
                   const ix::WebSocketMessagePtr& msg) {
                if (msg->type == ix::WebSocketMessageType::Open) {
                    on_open(connectionState, webSocket);
                } else if (msg->type == ix::WebSocketMessageType::Close) {
                    on_close(connectionState);
                } else if (msg->type == ix::WebSocketMessageType::Message) {
                    on_message(connectionState, msg->str);
                } else if (msg->type == ix::WebSocketMessageType::Error) {
                    spdlog::error("WebSocket error: {}", msg->errorInfo.reason);
                } else if (msg->type == ix::WebSocketMessageType::Ping) {
                    // Ping frames are auto-handled by IXWebSocket
                    spdlog::trace("Received ping from client");
                } else if (msg->type == ix::WebSocketMessageType::Pong) {
                    // Pong frames are auto-handled by IXWebSocket
                    spdlog::trace("Received pong from client");
                }
            });

        auto result = m_server->listen();
        if (!result.first) {
            throw std::runtime_error("Failed to start WebSocket server: " + result.second);
        }

        m_server->start();
        spdlog::info("WebSocket server listening on {}:{}", bind_address, port);
    }

    auto stop() -> void {
        if (m_server) {
            m_server->stop();
        }
    }

    auto set_connect_handler(websocket_server::connect_handler_t handler) -> void {
        m_connect_handler = std::move(handler);
    }

    auto set_disconnect_handler(websocket_server::disconnect_handler_t handler) -> void {
        m_disconnect_handler = std::move(handler);
    }

    auto set_message_handler(websocket_server::message_handler_t handler) -> void {
        m_message_handler = std::move(handler);
    }

    auto set_queue_size_callback(websocket_server::queue_size_callback_t callback) -> void {
        m_queue_size_callback = std::move(callback);
    }

    auto send_to_client(const std::string& client_id, ws_message_data_t data, bool is_binary)
        -> void {
        std::shared_lock lock(m_sessions_mutex);
        if (auto it = m_sessions.find(client_id); it != m_sessions.end()) {
            it->second->send(std::move(data), is_binary);
        }
    }

  private:
    auto on_open(std::shared_ptr<ix::ConnectionState> connectionState, ix::WebSocket& webSocket)
        -> void {
        std::string client_id = std::to_string(reinterpret_cast<uintptr_t>(connectionState.get()));

        // Create wrapped socket
        auto socket_ptr = std::shared_ptr<ix::WebSocket>(&webSocket, [](ix::WebSocket*) {});
        auto session = std::make_shared<websocket_session_impl>(socket_ptr, client_id);

        // Set queue size callback
        session->set_queue_size_callback(m_queue_size_callback);

        // Store session
        {
            std::unique_lock lock(m_sessions_mutex);
            m_sessions[client_id] = session;
        }

        // Notify connect handler
        m_connect_handler(session);

        spdlog::debug("Client {} connected", client_id);
    }

    auto on_close(std::shared_ptr<ix::ConnectionState> connectionState) -> void {
        std::string client_id = std::to_string(reinterpret_cast<uintptr_t>(connectionState.get()));

        {
            std::unique_lock lock(m_sessions_mutex);
            m_sessions.erase(client_id);
        }

        m_disconnect_handler(client_id);

        spdlog::debug("Client {} disconnected", client_id);
    }

    auto on_message(std::shared_ptr<ix::ConnectionState> connectionState,
                    const std::string& message) -> void {
        std::string client_id = std::to_string(reinterpret_cast<uintptr_t>(connectionState.get()));
        m_message_handler(client_id, message);
    }

    std::unique_ptr<ix::WebSocketServer> m_server;
    std::unordered_map<std::string, std::shared_ptr<websocket_session_impl>> m_sessions;
    std::shared_mutex m_sessions_mutex;

    websocket_server::connect_handler_t m_connect_handler;
    websocket_server::disconnect_handler_t m_disconnect_handler;
    websocket_server::message_handler_t m_message_handler;
    websocket_server::queue_size_callback_t m_queue_size_callback;
};

// ========== websocket_server public API ==========

websocket_server::websocket_server() : m_impl(std::make_unique<websocket_server_impl>()) {
}

websocket_server::~websocket_server() = default;

auto websocket_server::start(uint16_t port, const std::string& bind_address,
                             bool enable_compression) -> void {
    m_impl->start(port, bind_address, enable_compression);
}

auto websocket_server::stop() -> void {
    m_impl->stop();
}

auto websocket_server::set_connect_handler(connect_handler_t handler) -> void {
    m_impl->set_connect_handler(std::move(handler));
}

auto websocket_server::set_disconnect_handler(disconnect_handler_t handler) -> void {
    m_impl->set_disconnect_handler(std::move(handler));
}

auto websocket_server::set_message_handler(message_handler_t handler) -> void {
    m_impl->set_message_handler(std::move(handler));
}

auto websocket_server::set_queue_size_callback(queue_size_callback_t callback) -> void {
    m_impl->set_queue_size_callback(std::move(callback));
}

auto websocket_server::send_to_client(const std::string& client_id, ws_message_data_t data,
                                      bool is_binary) -> void {
    m_impl->send_to_client(client_id, std::move(data), is_binary);
}
