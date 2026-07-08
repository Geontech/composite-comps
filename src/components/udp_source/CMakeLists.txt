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
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-march=x86-64-v2 -mtune=generic)

# Check if DPDK is available from composite
get_target_property(COMPOSITE_COMPILE_DEFS composite::composite INTERFACE_COMPILE_DEFINITIONS)
if(COMPOSITE_COMPILE_DEFS MATCHES "COMPOSITE_USE_DPDK")
    set(COMPOSITE_HAS_DPDK ON)
    message(STATUS "udp_source: DPDK support enabled")
else()
    set(COMPOSITE_HAS_DPDK OFF)
    message(STATUS "udp_source: DPDK support disabled")
endif()

# Library sources
set(UDP_SOURCE_SRCS
    component.cpp
    socket/packet_mmap.cpp
    socket/recvmmsg.cpp
    net/utils.cpp
)

# Add DPDK backend if available
if(COMPOSITE_HAS_DPDK)
    list(APPEND UDP_SOURCE_SRCS
        socket/dpdk.cpp
        socket/igmp.cpp
    )
endif()

add_library(udp_source MODULE ${UDP_SOURCE_SRCS})

# Compile definitions
if(COMPOSITE_HAS_DPDK)
    target_compile_definitions(udp_source PRIVATE COMPOSITE_HAS_DPDK=1)
endif()

# Includes
target_include_directories(udp_source
    PRIVATE
    ${vrtgen_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link
target_link_libraries(udp_source
    PRIVATE
    composite::composite
)

# Add DPDK includes and linkage if available
if(COMPOSITE_HAS_DPDK)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DPDK REQUIRED libdpdk)
    target_include_directories(udp_source PRIVATE ${DPDK_INCLUDE_DIRS})
    target_link_libraries(udp_source PRIVATE ${DPDK_LIBRARIES})
    target_link_directories(udp_source PRIVATE ${DPDK_LIBRARY_DIRS})
endif()
# Install
install(TARGETS udp_source
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
