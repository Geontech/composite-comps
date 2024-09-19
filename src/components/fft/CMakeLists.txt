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
project(fft VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)

# Set the C++ version required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile flags
set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Custom compile options
add_compile_options(-march=native -mtune=native -ffast-math)

# Library
add_library(fft MODULE
    component.cpp
)
# Includes
target_include_directories(fft
    PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include
    ${vrtgen_SOURCE_DIR}/include
)
target_link_libraries(fft
    PRIVATE
    composite::composite
    fftw3f
    fftw3f_threads
    fftw3
    fftw3_threads
)
# Install
install(TARGETS fft
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
