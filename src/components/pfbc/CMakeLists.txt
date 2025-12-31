# SPDX-License-Identifier: LGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

cmake_minimum_required(VERSION 3.15)
project(pfbc VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)

# Set the C++ version required
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile flags
set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Custom compile options - enable AVX-512 for SIMD kernels
add_compile_options(-march=native -mtune=native)

# Find FFTW3 (single-precision)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)

# Library
add_library(pfbc MODULE
    component.cpp
)

# Includes
target_include_directories(pfbc
    PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include
    ${FFTW3F_INCLUDE_DIRS}
)

# Link
target_link_libraries(pfbc
    PRIVATE
    composite::composite
    spdlog::spdlog_header_only
    ${FFTW3F_LIBRARIES}
    fftw3f_threads
)

# Install
install(TARGETS pfbc
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Tests
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
