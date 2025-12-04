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

# Framer component unit tests

# Fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

# Find threading library (needed for std::call_once in converters)
find_package(Threads REQUIRED)

# Helper function to create a pool test target
function(add_pool_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})
    target_link_libraries(${TEST_NAME}
        PRIVATE
            Catch2::Catch2WithMain
            composite::composite
            Threads::Threads
    )
    target_include_directories(${TEST_NAME}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/..
            ${PROJECT_SOURCE_DIR}/include
    )
    target_compile_features(${TEST_NAME} PRIVATE cxx_std_23)
    target_compile_options(${TEST_NAME} PRIVATE -mavx2 -mfma)

    # Check if AVX512 is available
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        target_compile_options(${TEST_NAME} PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq)
    endif()
endfunction()

# Pool test suite (comprehensive)
add_pool_test(pool_basic_tests pool_basic_tests.cpp)
add_pool_test(pool_multiframe_tests pool_multiframe_tests.cpp)
add_pool_test(pool_wraparound_tests pool_wraparound_tests.cpp)
add_pool_test(pool_conflict_tests pool_conflict_tests.cpp)
add_pool_test(pool_edge_cases_tests pool_edge_cases_tests.cpp)

# Converter tests (variant-based converters)
add_pool_test(converter_tests converter_tests.cpp)

# Integration tests
add_executable(framer_integration_tests framer_integration_tests.cpp ../component.cpp)
target_link_libraries(framer_integration_tests
    PRIVATE
        Catch2::Catch2WithMain
        composite::composite
        spdlog::spdlog_header_only
        Threads::Threads
)
target_include_directories(framer_integration_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${PROJECT_SOURCE_DIR}/include
)
target_compile_features(framer_integration_tests PRIVATE cxx_std_23)
target_compile_options(framer_integration_tests PRIVATE -mavx2 -mfma)

# Check AVX512 for integration tests
if(COMPILER_SUPPORTS_AVX512)
    target_compile_options(framer_integration_tests PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq)
endif()

# Register tests with CTest
include(CTest)
include(Catch)

# Pool test suite
catch_discover_tests(pool_basic_tests)
catch_discover_tests(pool_multiframe_tests)
catch_discover_tests(pool_wraparound_tests)
catch_discover_tests(pool_conflict_tests)
catch_discover_tests(pool_edge_cases_tests)

# Additional tests
catch_discover_tests(converter_tests)
catch_discover_tests(framer_integration_tests)

# Install test executables (optional)
install(TARGETS
    pool_basic_tests
    pool_multiframe_tests
    pool_wraparound_tests
    pool_conflict_tests
    pool_edge_cases_tests
    converter_tests
    framer_integration_tests
    RUNTIME DESTINATION bin/tests
    OPTIONAL
)
