# SPDX-License-Identifier: LGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

# pkt_builder integration tests

# Fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

# Integration tests for pkt_builder -> pkt_debug pipeline
add_executable(pkt_builder_integration_tests
    pkt_builder_integration_tests.cpp
    ../component.cpp
    ../../pkt_debug/component.cpp
)
target_link_libraries(pkt_builder_integration_tests
    PRIVATE
        Catch2::Catch2WithMain
        composite::composite
        spdlog::spdlog_header_only
)
target_include_directories(pkt_builder_integration_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../pkt_debug
        ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(pkt_builder_integration_tests PRIVATE cxx_std_23)
target_compile_options(pkt_builder_integration_tests PRIVATE -march=x86-64-v2)
target_compile_definitions(pkt_builder_integration_tests PRIVATE UNIT_TESTS)

# Register tests with CTest
include(CTest)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(pkt_builder_integration_tests)

# Install test executables (optional)
install(TARGETS pkt_builder_integration_tests
    RUNTIME DESTINATION bin/tests
    OPTIONAL
)