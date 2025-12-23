# SPDX-License-Identifier: LGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

# Halfrate component tests

# Fetch Catch2 for testing
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

# Fetch Google Benchmark for performance testing
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

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

# Google Benchmark kernel benchmarks
add_executable(kernel_benchmarks kernel_benchmarks.cpp)
target_link_libraries(kernel_benchmarks
    PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
)
target_include_directories(kernel_benchmarks
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_compile_features(kernel_benchmarks PRIVATE cxx_std_23)
target_compile_options(kernel_benchmarks PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq -O3 -march=native)

# Register tests with CTest
include(CTest)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

catch_discover_tests(kernel_tests)
catch_discover_tests(halfrate_integration_tests)
# Note: Google Benchmark kernel_benchmarks are not auto-registered with CTest
# Run manually with: ./kernel_benchmarks [--benchmark_filter=pattern] [--benchmark_repetitions=N]

# Install test executables (optional)
install(TARGETS
    kernel_tests
    halfrate_integration_tests
    kernel_benchmarks
    RUNTIME DESTINATION bin/tests
    OPTIONAL
)
