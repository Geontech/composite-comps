#
# Copyright (C)
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
project(simple_writer VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)

# Set the C++ version required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile flags
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${hdf5_SOURCE_DIR}/include -DH5_BUILT_AS_DYNAMIC_LIB -L${hdf5_SOURCE_DIR}/lib -lhdf5 -lhiredis")
set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic ")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -ggdb -O0 ")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# Library
add_library(simple_writer MODULE
    component.cpp
    helpers.cpp
)
# Includes
target_include_directories(simple_writer
    PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# Link
target_link_libraries(simple_writer
    PRIVATE
    composite::composite
    spdlog::spdlog_header_only
    nlohmann_json::nlohmann_json
)
# Install
install(TARGETS simple_writer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
