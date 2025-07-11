#
# Copyright (C) 2024 Geon Technologies, LLC
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
# along with this program. If not, see http://www.gnu.org/licenses/.
#

cmake_minimum_required(VERSION 3.20)
project(udp_source VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)

# Set the C++ version required
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile flags
set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-march=x86-64-v2 -mtune=generic)

include(FetchContent)

FetchContent_Declare(
  readerwriterqueue
  GIT_REPOSITORY    https://github.com/cameron314/readerwriterqueue
  GIT_TAG           master
)

FetchContent_MakeAvailable(readerwriterqueue)
# DPDK
find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED libdpdk)

# Library
add_library(udp_source MODULE
    component.cpp
    socket/dpdk.cpp
    socket/packet_mmap.cpp
    socket/recvmmsg.cpp
    helpers.cpp
    overlay.cpp
    pmr/ring_resource.cpp
    # socket/pcapng.hpp
    socket/pcap_writer_thread.hpp
)
# Includes
target_include_directories(udp_source
    PRIVATE
    ${vrtgen_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DPDK_INCLUDE_DIRS}
)
target_link_directories(udp_source PRIVATE ${DPDK_LIBRARY_DIRS})
target_compile_options(udp_source PRIVATE ${DPDK_CFLAGS_OTHER})
# Link
target_link_libraries(udp_source
    PRIVATE
    composite::composite
    spdlog::spdlog_header_only
    ${DPDK_LIBRARIES}
    readerwriterqueue
)
# Install
install(TARGETS udp_source
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
