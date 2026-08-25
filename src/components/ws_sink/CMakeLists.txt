#
# Copyright (C) 2024 Geon Technologies, LLC
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
# along with this program. If not, see http://www.gnu.org/licenses/.
#

# nlohmann_json arrives via composite's package config; ixwebsocket via the top-level project.
comps_add_component(ws_sink
    SOURCES component.cpp
            websocket_server.cpp
    LINK    nlohmann_json::nlohmann_json ixwebsocket z
    ABI_TYPE cf32
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Optional test client
option(BUILD_WS_SINK_TEST_CLIENT "Build WebSocket test client executable" OFF)

if(BUILD_WS_SINK_TEST_CLIENT)
    add_executable(ws_sink_test_client test_client.cpp)
    target_link_libraries(ws_sink_test_client
        PRIVATE
        ixwebsocket
        nlohmann_json::nlohmann_json
        z
        comps_baseline
        comps_warnings
    )
    install(TARGETS ws_sink_test_client
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT Tests
    )
    message(STATUS "Building ws_sink test client")
endif()
