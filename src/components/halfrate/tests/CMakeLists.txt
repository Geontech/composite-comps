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
# We create three versions to test different ISA implementations:
# 1. Scalar-only (baseline x86-64)
# 2. AVX2 (with FMA)
# 3. AVX-512

# Scalar version - forces baseline x86-64, disables all SIMD
add_executable(kernel_tests_scalar kernel_tests.cpp)
target_link_libraries(kernel_tests_scalar
    PRIVATE
        Catch2::Catch2WithMain
)
target_include_directories(kernel_tests_scalar
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_compile_features(kernel_tests_scalar PRIVATE cxx_std_23)
target_compile_options(kernel_tests_scalar PRIVATE
    -march=x86-64-v2        # x86-64-v2: SSE up to SSE4.2, no AVX
    -mno-avx -mno-avx2      # Explicitly disable AVX/AVX2
    -mno-avx512f            # Explicitly disable AVX-512
)
target_compile_definitions(kernel_tests_scalar PRIVATE
    KERNEL_TEST_ISA="scalar"
    KERNEL_TEST_SCALAR=1
)

# AVX2 version - enables AVX2/FMA, disables AVX-512
add_executable(kernel_tests_avx2 kernel_tests.cpp)
target_link_libraries(kernel_tests_avx2
    PRIVATE
        Catch2::Catch2WithMain
)
target_include_directories(kernel_tests_avx2
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_compile_features(kernel_tests_avx2 PRIVATE cxx_std_23)
target_compile_options(kernel_tests_avx2 PRIVATE
    -march=haswell          # Haswell has AVX2 + FMA
    -mno-avx512f            # Explicitly disable AVX-512
)
target_compile_definitions(kernel_tests_avx2 PRIVATE
    KERNEL_TEST_ISA="avx2"
    KERNEL_TEST_AVX2=1
)

# AVX-512 version - enables full AVX-512 instruction set
add_executable(kernel_tests_avx512 kernel_tests.cpp)
target_link_libraries(kernel_tests_avx512
    PRIVATE
        Catch2::Catch2WithMain
)
target_include_directories(kernel_tests_avx512
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_compile_features(kernel_tests_avx512 PRIVATE cxx_std_23)
target_compile_options(kernel_tests_avx512 PRIVATE
    -mavx512f -mavx512bw -mavx512vl -mavx512dq
)
target_compile_definitions(kernel_tests_avx512 PRIVATE
    KERNEL_TEST_ISA="avx512"
    KERNEL_TEST_AVX512=1
)

# Convenience alias: kernel_tests -> kernel_tests_avx512
add_executable(kernel_tests ALIAS kernel_tests_avx512)

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

# Google Benchmark kernel benchmarks - AVX-512 version
add_executable(kernel_benchmarks_avx512 kernel_benchmarks.cpp)
target_link_libraries(kernel_benchmarks_avx512
    PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
)
target_include_directories(kernel_benchmarks_avx512
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_compile_features(kernel_benchmarks_avx512 PRIVATE cxx_std_23)
target_compile_options(kernel_benchmarks_avx512 PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512dq -O3)

# Google Benchmark kernel benchmarks - AVX2 version
add_executable(kernel_benchmarks_avx2 kernel_benchmarks.cpp)
target_link_libraries(kernel_benchmarks_avx2
    PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
)
target_include_directories(kernel_benchmarks_avx2
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_compile_features(kernel_benchmarks_avx2 PRIVATE cxx_std_23)
target_compile_options(kernel_benchmarks_avx2 PRIVATE -march=haswell -mno-avx512f -O3)

# Convenience alias
add_executable(kernel_benchmarks ALIAS kernel_benchmarks_avx512)

# Register tests with CTest
include(CTest)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

# Register all three ISA variants of kernel tests with automatic test discovery
catch_discover_tests(kernel_tests_scalar
    TEST_PREFIX "scalar:"
    PROPERTIES LABELS "scalar;kernels"
)
catch_discover_tests(kernel_tests_avx2
    TEST_PREFIX "avx2:"
    PROPERTIES LABELS "avx2;kernels"
)
catch_discover_tests(kernel_tests_avx512
    TEST_PREFIX "avx512:"
    PROPERTIES LABELS "avx512;kernels"
)
catch_discover_tests(halfrate_integration_tests)
# Note: Google Benchmark kernel_benchmarks are not auto-registered with CTest
# Run manually with: ./kernel_benchmarks [--benchmark_filter=pattern] [--benchmark_repetitions=N]

# Install test executables (optional)
install(TARGETS
    kernel_tests_scalar
    kernel_tests_avx2
    kernel_tests_avx512
    halfrate_integration_tests
    kernel_benchmarks_avx512
    kernel_benchmarks_avx2
    RUNTIME DESTINATION bin/tests
    OPTIONAL
)
