# SPDX-License-Identifier: LGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

cmake_minimum_required(VERSION 3.20)
project(halfrate VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)

# Set the C++ version required
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile flags
set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -ffast-math -funroll-loops")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Custom compile options
add_compile_options(-march=x86-64-v3 -mtune=generic)

# Library
add_library(halfrate MODULE
    component.cpp
)
# Includes
target_include_directories(halfrate
    PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include
)
# Link
target_link_libraries(halfrate
    PRIVATE
    composite::composite
    spdlog::spdlog_header_only
)
# Install
install(TARGETS halfrate
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Tests
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
