# SPDX-License-Identifier: LGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

# PFBC component unit tests

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

# Find required packages
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)

# Helper function to create a pfbc test target
function(add_pfbc_test TEST_NAME SOURCE_FILE)
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
            ${PROJECT_SOURCE_DIR}/../../../include
    )
    target_compile_features(${TEST_NAME} PRIVATE cxx_std_23)
    target_compile_options(${TEST_NAME} PRIVATE -march=x86-64-v3 -mtune=generic)
endfunction()

# Unit tests for new SIMD kernels (commutator, filter, gather, fft8)
add_executable(pfbc_kernel_tests kernel_tests.cpp)
target_link_libraries(pfbc_kernel_tests
    PRIVATE
        Catch2::Catch2WithMain
        composite::composite
        Threads::Threads
        ${FFTW3F_LIBRARIES}
)
target_include_directories(pfbc_kernel_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${PROJECT_SOURCE_DIR}/../../../include
        ${FFTW3F_INCLUDE_DIRS}
)
target_compile_features(pfbc_kernel_tests PRIVATE cxx_std_23)
target_compile_options(pfbc_kernel_tests PRIVATE -march=x86-64-v3 -mtune=generic)

# Integration tests (requires component.cpp and FFTW)
add_executable(pfbc_integration_tests
    pfbc_integration_tests.cpp
    ../component.cpp
)
target_link_libraries(pfbc_integration_tests
    PRIVATE
        Catch2::Catch2WithMain
        composite::composite
        spdlog::spdlog_header_only
        Threads::Threads
        ${FFTW3F_LIBRARIES}
        fftw3f_threads
)
target_include_directories(pfbc_integration_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${PROJECT_SOURCE_DIR}/../../../include
        ${FFTW3F_INCLUDE_DIRS}
)
target_compile_features(pfbc_integration_tests PRIVATE cxx_std_23)
target_compile_options(pfbc_integration_tests PRIVATE -march=x86-64-v3 -mtune=generic)

# Google Benchmark kernel benchmarks
add_executable(pfbc_kernel_benchmarks kernel_benchmarks.cpp)
target_link_libraries(pfbc_kernel_benchmarks
    PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
        composite::composite
        Threads::Threads
        ${FFTW3F_LIBRARIES}
        fftw3f_threads
)
target_include_directories(pfbc_kernel_benchmarks
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${PROJECT_SOURCE_DIR}/../../../include
        ${FFTW3F_INCLUDE_DIRS}
)
target_compile_features(pfbc_kernel_benchmarks PRIVATE cxx_std_23)
target_compile_options(pfbc_kernel_benchmarks PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq -O3 -march=native)

# Register tests with CTest
include(CTest)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

catch_discover_tests(pfbc_kernel_tests
    PROPERTIES LABELS pfbc
)
catch_discover_tests(pfbc_integration_tests
    PROPERTIES LABELS pfbc
)

# Install test executables (optional)
install(TARGETS
    pfbc_kernel_tests
    pfbc_integration_tests
    pfbc_kernel_benchmarks
    RUNTIME DESTINATION bin/tests
    OPTIONAL
)
