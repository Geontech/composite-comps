#
# Copyright (C) 2025 Geon Technologies, LLC
#
# This file is part of composite-comps.
#
# composite-comps is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# composite-comps is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/.
#

cmake_minimum_required(VERSION 3.15)
project(ws_sink VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)

# Set the C++ version required
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile flags
set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Custom compile options
add_compile_options(-march=x86-64-v2 -mtune=generic)

# Find dependencies
# IXWebSocket - fetch if not found
find_package(IXWebSocket QUIET)
if(NOT IXWebSocket_FOUND)
    include(FetchContent)
    FetchContent_Declare(IXWebSocket
        GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
        GIT_TAG v11.4.6
    )
    set(USE_TLS ON CACHE BOOL "" FORCE)
    set(USE_OPEN_SSL ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(IXWebSocket)
endif()

# Library
add_library(ws_sink MODULE
    component.cpp
    websocket_server.cpp
)

# Includes
target_include_directories(ws_sink
    PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include
)

target_link_libraries(ws_sink
    PRIVATE
    composite::composite
    spdlog::spdlog_header_only
    ixwebsocket
    nlohmann_json::nlohmann_json
    z
)

# Install
install(TARGETS ws_sink
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Optional test client
option(BUILD_WS_SINK_TEST_CLIENT "Build WebSocket test client executable" OFF)

if(BUILD_WS_SINK_TEST_CLIENT)
    add_executable(ws_sink_test_client
        test_client.cpp
    )

    target_link_libraries(ws_sink_test_client
        PRIVATE
        ixwebsocket
        nlohmann_json::nlohmann_json
        z
    )

    install(TARGETS ws_sink_test_client
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    message(STATUS "Building ws_sink test client")
endif()
