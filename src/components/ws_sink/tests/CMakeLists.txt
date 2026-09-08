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

# ws_sink loopback integration tests (real 127.0.0.1 websocket server + client).
# nlohmann_json, ixwebsocket, and z arrive through ws_sink_objs.
add_executable(ws_sink_integration_tests ws_sink_integration_tests.cpp)
target_link_libraries(ws_sink_integration_tests
    PRIVATE
        ws_sink_objs
        Catch2::Catch2WithMain
        Threads::Threads
)

# Under TSan these report a lock-order-inversion inside IXWebSocket itself
# (IXWebSocketTransport.cpp:212, ix::WebSocketTransport::setReadyState) -- third-party code,
# not fleet code. The framework takes the same position on its httplib-based integration test
# (`ctest --preset tsan -E TestHttpIntegration`). The target still BUILDS under TSan, so
# compile coverage is kept; only registration is skipped.
if(COMPS_SANITIZER MATCHES "thread")
    message(STATUS "composite-comps: ws_sink integration tests not registered under TSan "
                   "(lock-order-inversion inside IXWebSocket; third-party)")
else()
    catch_discover_tests(ws_sink_integration_tests)
endif()

install(TARGETS
    ws_sink_integration_tests
    RUNTIME DESTINATION bin/tests
    COMPONENT Tests
    OPTIONAL
)
