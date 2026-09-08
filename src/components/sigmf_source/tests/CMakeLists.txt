# SPDX-License-Identifier: LGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

# sigmf_source tests
#
# Catch2 is NOT declared here. cmake/CompsDependencies.cmake is the single place
# it is fetched -- when five test directories each declared it, only the first
# took effect and the other four versions were silently ignored, and Catch2 also
# inherited whichever component's directory-scope compile flags got there first.

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
        ${COMPS_SOURCE_DIR}/include
)
target_compile_features(sigmf_source_destination_tests PRIVATE cxx_std_23)
target_compile_definitions(sigmf_source_destination_tests PRIVATE UNIT_TESTS)

include(Catch)
catch_discover_tests(sigmf_source_destination_tests)
