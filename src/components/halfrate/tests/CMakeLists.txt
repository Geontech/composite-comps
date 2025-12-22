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

# Halfrate component tests

# Fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

# Kernel unit tests (header-only kernels)
add_executable(kernel_tests kernel_tests.cpp)
target_link_libraries(kernel_tests
    PRIVATE
        Catch2::Catch2WithMain
)
target_include_directories(kernel_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_compile_features(kernel_tests PRIVATE cxx_std_23)
target_compile_options(kernel_tests PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq)

# Performance benchmarks
add_executable(performance_benchmarks performance_benchmarks.cpp ../component.cpp)
target_link_libraries(performance_benchmarks
    PRIVATE
        Catch2::Catch2WithMain
        composite::composite
        spdlog::spdlog_header_only
)
target_include_directories(performance_benchmarks
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(performance_benchmarks PRIVATE cxx_std_23)
target_compile_options(performance_benchmarks PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq -O3 -march=native)

# Integration tests (requires component.cpp)
add_executable(halfrate_integration_tests halfrate_integration_tests.cpp ../component.cpp)
target_link_libraries(halfrate_integration_tests
    PRIVATE
        Catch2::Catch2WithMain
        composite::composite
        spdlog::spdlog_header_only
)
target_include_directories(halfrate_integration_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(halfrate_integration_tests PRIVATE cxx_std_23)
target_compile_options(halfrate_integration_tests PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq)

# Register tests with CTest
include(CTest)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

catch_discover_tests(kernel_tests)
catch_discover_tests(halfrate_integration_tests)
# Note: performance_benchmarks are not auto-registered with CTest
# Run manually with: ./performance_benchmarks --benchmark-samples 100

# Install test executables (optional)
install(TARGETS
    kernel_tests
    halfrate_integration_tests
    performance_benchmarks
    RUNTIME DESTINATION bin/tests
    OPTIONAL
)
