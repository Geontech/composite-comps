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
project(composite-comps VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# composite dependency
find_package(composite)

# Use FetchContent for other dependencies
include(FetchContent)
# vrtgen dependency
FetchContent_Declare(vrtgen
    GIT_REPOSITORY https://github.com/geontech/vrtgen.git
    GIT_TAG v0.7.12
)
# Make all deps available
FetchContent_MakeAvailable(vrtgen)

add_subdirectory(src)
