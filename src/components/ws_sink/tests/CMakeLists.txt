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

# ws_sink loopback integration tests (real 127.0.0.1 websocket server + client)

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

find_package(Threads REQUIRED)

add_executable(ws_sink_integration_tests
    ws_sink_integration_tests.cpp
    ../component.cpp
    ../websocket_server.cpp
)
target_link_libraries(ws_sink_integration_tests
    PRIVATE
        Catch2::Catch2WithMain
        composite::composite
        nlohmann_json::nlohmann_json
        ixwebsocket
        z
        Threads::Threads
)
target_include_directories(ws_sink_integration_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_compile_features(ws_sink_integration_tests PRIVATE cxx_std_23)

include(CTest)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(ws_sink_integration_tests)

install(TARGETS
    ws_sink_integration_tests
    RUNTIME DESTINATION bin/tests
    OPTIONAL
)
