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
# along with this program.  If not, see http://www.gnu.org/licenses/.
#

cmake_minimum_required(VERSION 3.15)
project(pcap_writer VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)

# Set the C++ version required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile flags
set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PCAP REQUIRED libpcap)
# Custom compile options
add_compile_options(-march=x86-64-v2 -mtune=generic -ffast-math)

# Library
add_library(pcap_writer MODULE
    component.cpp
)
# Includes
target_include_directories(pcap_writer
    PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include
    ${PCAP_INCLUDE_DIRS}
)
target_link_libraries(pcap_writer
    PRIVATE
    composite::composite
    spdlog::spdlog_header_only
    ${PCAP_LIBRARIES}
)
# Install
install(TARGETS pcap_writer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
