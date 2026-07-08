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

# pkt_builder integration tests, including a loopback through the real vita49_parser:
# what the builder emits must round-trip through the parser (one wire-format truth).

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

find_package(Threads REQUIRED)

add_executable(pkt_builder_integration_tests
    pkt_builder_integration_tests.cpp
    ../component.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pkt_parser/overlay/vita49.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pkt_parser/parsers/vita49_parser.cpp
)
target_link_libraries(pkt_builder_integration_tests
    PRIVATE
        Catch2::Catch2WithMain
        composite::composite
        Threads::Threads
)
target_include_directories(pkt_builder_integration_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../pkt_parser
        ${PROJECT_SOURCE_DIR}/include
        ${vrtgen_SOURCE_DIR}/include
)
target_compile_features(pkt_builder_integration_tests PRIVATE cxx_std_23)

include(CTest)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(pkt_builder_integration_tests)

install(TARGETS
    pkt_builder_integration_tests
    RUNTIME DESTINATION bin/tests
    OPTIONAL
)
