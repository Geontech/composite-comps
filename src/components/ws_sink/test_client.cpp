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
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */

#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <ixwebsocket/IXWebSocket.h>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

using json = nlohmann::json;

struct test_stats {
    uint64_t headers_received = 0;
    uint64_t binary_received = 0;
    uint64_t histograms_received = 0;
    uint64_t metrics_received = 0;
    uint64_t bytes_received = 0;
    std::chrono::steady_clock::time_point start_time;
    bool expecting_binary = false;
};

auto print_usage(const char* program_name) -> void {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Test WebSocket client for ws_sink component\n\n"
              << "Options:\n"
              << "  -s, --server HOST     WebSocket server hostname (default: localhost)\n"
              << "  -p, --port PORT       WebSocket server port (default: 8081)\n"
              << "  -d, --duration SEC    Test duration in seconds (default: 30)\n"
              << "  -f, --fps FPS         Request specific FPS from server\n"
              << "  -h, --help            Show this help message\n\n"
              << "Examples:\n"
              << "  " << program_name << "\n"
              << "  " << program_name << " -s 10.0.0.5 -p 8081\n"
              << "  " << program_name << " -f 30 -d 60\n";
}

auto format_timestamp() -> std::string {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&time, &tm_buf);
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%H:%M:%S");
    return oss.str();
}

auto format_bytes(uint64_t bytes) -> std::string {
    std::ostringstream oss;
    oss.imbue(std::locale(""));
    oss << bytes;
    return oss.str();
}

auto main(int argc, char* argv[]) -> int {
    // Parse command line arguments
    std::string host = "localhost";
    int port = 8081;
    int duration = 30;
    float requested_fps = -1.0f;

    static struct option long_options[] = {
        {"server", required_argument, 0, 's'},   {"port", required_argument, 0, 'p'},
        {"duration", required_argument, 0, 'd'}, {"fps", required_argument, 0, 'f'},
        {"help", no_argument, 0, 'h'},           {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "s:p:d:f:h", long_options, nullptr)) != -1) {
        switch (opt) {
        case 's':
            host = optarg;
            break;
        case 'p':
            port = std::atoi(optarg);
            break;
        case 'd':
            duration = std::atoi(optarg);
            break;
        case 'f':
            requested_fps = std::atof(optarg);
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    std::string url = "ws://" + host + ":" + std::to_string(port);
    std::cout << "Connecting to " << url << "...\n";

    test_stats stats;
    stats.start_time = std::chrono::steady_clock::now();

    ix::WebSocket ws;
    ws.setUrl(url);

    // Setup callbacks
    ws.setOnMessageCallback([&stats, duration](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Message) {
            if (msg->binary) {
                // Binary spectrum data
                stats.binary_received++;
                stats.bytes_received += msg->str.size();
                stats.expecting_binary = false;

                size_t num_samples = msg->str.size() / 8; // 8 bytes per complex float32
                std::cout << "  Binary data: " << msg->str.size() << " bytes (" << num_samples
                          << " complex samples)\n";
            } else {
                // JSON message
                try {
                    auto j = json::parse(msg->str);
                    std::string event = j.contains("event") ? j["event"].get<std::string>()
                                                            : j.value("type", "unknown");

                    if (event == "header") {
                        stats.headers_received++;
                        stats.expecting_binary = true;

                        auto payload = j["payload"];
                        std::cout << "\n[" << format_timestamp() << "] Header received:\n"
                                  << "  Format: " << payload.value("format", "") << "\n"
                                  << "  Sample Rate: " << std::fixed << std::setprecision(1)
                                  << payload.value("sampleRate", 0.0) / 1e6 << " MHz\n"
                                  << "  Bandwidth: " << payload.value("bandwidth", 0.0) / 1e6
                                  << " MHz\n"
                                  << "  Frequency: " << payload.value("frequency", 0.0) / 1e6
                                  << " MHz\n"
                                  << "  FFT Size: " << payload.value("fftsize", 0) << "\n";

                    } else if (event == "histogram") {
                        stats.histograms_received++;
                        auto bins = j["payload"];
                        uint64_t total = 0;
                        for (const auto& bin : bins) {
                            total += bin.get<uint64_t>();
                        }
                        std::cout << "\n[" << format_timestamp() << "] Histogram: " << bins.size()
                                  << " bins, " << total << " total samples\n";

                    } else if (event == "metrics") {
                        stats.metrics_received++;

                        auto payload = j.value("payload", json::object());

                        std::cout << "\n[" << format_timestamp() << "] Metrics:\n"
                                  << "  Active connections: "
                                  << payload.value("activeConnections", 0) << "\n"
                                  << "  Dropping frames: "
                                  << (payload.value("droppingFrames", false) ? "yes" : "no") << "\n"
                                  << "  Frames per second: "
                                  << payload.value("framesPerSecond", "0.0") << "\n"
                                  << "  Frames sent: " << payload.value("framesSent", 0) << "\n"
                                  << "  Frames dropped: " << payload.value("framesDropped", 0)
                                  << "\n"
                                  << "  Drop ratio: " << payload.value("dropRatio", "0.0") << "%\n";

                    } else if (event == "stream_fps_updated") {
                        std::cout << "\n[" << format_timestamp() << "] FPS Updated:\n"
                                  << "  Requested: " << j.value("requested_fps", 0.0f) << "\n"
                                  << "  Actual: " << j.value("fps", 0.0f) << "\n";

                    } else {
                        std::cout << "\n[" << format_timestamp()
                                  << "] Unknown message type: " << event << "\n"
                                  << "  " << j.dump(2) << "\n";
                    }

                } catch (const json::exception& e) {
                    std::cerr << "Error parsing JSON: " << e.what() << "\n"
                              << "Message: "
                              << msg->str.substr(0, std::min(size_t(200), msg->str.size())) << "\n";
                }
            }

        } else if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "Connected!\n";
            std::cout << "Listening for " << duration << " seconds (Ctrl+C to stop early)...\n";

        } else if (msg->type == ix::WebSocketMessageType::Close) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::steady_clock::now() - stats.start_time)
                               .count();

            std::cout << "\n\nConnection closed (code: " << msg->closeInfo.code << ")\n";
            std::cout << "\nTest Summary (" << elapsed << "s):\n"
                      << "  Headers received: " << stats.headers_received << "\n"
                      << "  Binary frames received: " << stats.binary_received << "\n"
                      << "  Histograms received: " << stats.histograms_received << "\n"
                      << "  Metrics received: " << stats.metrics_received << "\n"
                      << "  Total bytes: " << format_bytes(stats.bytes_received) << "\n";

            if (elapsed > 0) {
                std::cout << "  Average throughput: " << std::fixed << std::setprecision(2)
                          << static_cast<double>(stats.bytes_received) / elapsed / 1024 / 1024
                          << " MB/s\n";
                if (stats.headers_received > 0) {
                    std::cout << "  Average stream FPS: "
                              << static_cast<double>(stats.headers_received) / elapsed << "\n";
                }
            }

        } else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cerr << "Error: " << msg->errorInfo.reason << "\n";
        }
    });

    // Start connection
    ws.start();

    // Wait for connection
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Request specific FPS if provided
    if (requested_fps > 0) {
        json fps_request = {{"type", "set_stream_fps"}, {"fps", requested_fps}};
        std::string request_str = fps_request.dump();
        std::cout << "\nRequesting " << requested_fps << " FPS...\n";
        ws.send(request_str);
    }

    // Run for specified duration
    std::this_thread::sleep_for(std::chrono::seconds(duration));

    // Cleanup
    std::cout << "\nTest duration reached, closing...\n";
    ws.stop();

    return 0;
}
