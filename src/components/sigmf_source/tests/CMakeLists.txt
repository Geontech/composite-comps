# SPDX-License-Identifier: LGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

# sigmf_source tests

# Fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(sigmf_source_destination_tests
    sigmf_source_destination_tests.cpp
    ../component.cpp
)
target_link_libraries(sigmf_source_destination_tests
    PRIVATE
        Catch2::Catch2WithMain
        composite::composite
        spdlog::spdlog_header_only
        nlohmann_json::nlohmann_json
)
target_include_directories(sigmf_source_destination_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(sigmf_source_destination_tests PRIVATE cxx_std_23)
target_compile_options(sigmf_source_destination_tests PRIVATE -march=x86-64-v2)
target_compile_definitions(sigmf_source_destination_tests PRIVATE UNIT_TESTS)

# Register tests with CTest
include(CTest)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(sigmf_source_destination_tests)

install(TARGETS sigmf_source_destination_tests
    RUNTIME DESTINATION bin/tests
    OPTIONAL
)
