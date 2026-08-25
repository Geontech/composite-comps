#
# Copyright (C) 2024-2026 Geon Technologies, LLC
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

# 3.22 matches the framework's CMakePresets floor. Rocky 9's appstream cmake is
# 3.31.8, so presets are usable in the container as well as in dev and CI.
cmake_minimum_required(VERSION 3.22)

# ONE project() for the whole fleet. Components are targets within it, not nested
# projects.
#
# VERSIONING: this version is the FLEET's own, deliberately independent of the
# framework's. The fleet changes far more often than composite does, and adding a
# component is a fleet-only event, so the two version lines diverge immediately and
# tying them would make the fleet number lie about what it contains.
#
# The framework compatibility contract is expressed separately, by
# COMPS_COMPOSITE_VERSION below. Because composite's package config declares
# SameMinorVersion compatibility, a non-EXACT find_package(composite 0.5.0)
# accepts any 0.5.x and rejects 0.6 -- which is exactly the range the fleet means.
#
# Two consequences worth keeping in mind:
#   * the fleet version cannot tell you which modules an image contains, so the
#     generated manifest (module list + framework version) is the source of truth,
#     not a convenience;
#   * an image's framework pairing must be discoverable from the image itself --
#     see the com.geontech.composite.version label in docker/Dockerfile.rocky9.
project(composite-comps VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# A consuming overlay project decides whether the FOSS modules install into its image, so this
# defaults to the master-project case and is overridable. Mirrors COMPOSITE_INSTALL.
if(NOT DEFINED COMPS_INSTALL)
    if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
        set(COMPS_INSTALL ON)
    else()
        set(COMPS_INSTALL OFF)
    endif()
endif()
option(COMPS_INSTALL "Generate install rules for the fleet's modules" ${COMPS_INSTALL})

# include(CTest) defines the BUILD_TESTING option (default ON) AND calls enable_testing(), so the
# component test dirs guarded by if(BUILD_TESTING) build by default. Plain enable_testing() left
# BUILD_TESTING undefined, so `ctest` found no tests unless -DBUILD_TESTING=ON was passed.
include(CTest)

# An empty CMAKE_BUILD_TYPE means no optimization flag at all -- not even -O0 --
# and both checked-in build trees were configured that way. This fleet's open
# performance questions are throughput-sensitive, so a bare `cmake -B build`
# must not quietly produce an unoptimized build. Matches the framework.
# Use the `release` preset for anything being measured: this default is -O2 -g.
# GENERATOR_IS_MULTI_CONFIG is a global property, not a variable -- testing it
# as a variable would always read as unset and force a build type on Ninja
# Multi-Config / Visual Studio, where CMAKE_BUILD_TYPE is meaningless.
get_property(_comps_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(PROJECT_IS_TOP_LEVEL AND NOT _comps_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
    message(STATUS "composite-comps: CMAKE_BUILD_TYPE not set; defaulting to RelWithDebInfo")
endif()

# The framework release this fleet is qualified against. Enforced rather than
# hoped for -- see COMPS_REQUIRE_EXACT_COMPOSITE in cmake/CompsDependencies.cmake.
set(COMPS_COMPOSITE_VERSION "0.5.0" CACHE STRING "composite framework version to build against")

# Captured because the helpers are usable from a CONSUMING project (an overlay repo that
# FetchContent's this one and calls comps_add_component() for its own components). Inside such a
# call PROJECT_SOURCE_DIR is the CONSUMER's project dir, not this one, so the helpers must not
# use it -- they would hand the consumer its own include/ and lose simd_fmv.hpp / windows.hpp.
set(COMPS_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "composite-comps source root")
set(COMPS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "composite-comps binary root")
set(COMPS_VERSION "${PROJECT_VERSION}" CACHE INTERNAL "composite-comps version")

list(APPEND CMAKE_MODULE_PATH ${COMPS_SOURCE_DIR}/cmake)
include(CompsFlags)          # the single authority for compile/link policy
include(CompsDependencies)   # composite, vrtgen, Catch2, IXWebSocket, Threads
include(CompsAddComponent)   # comps_add_component()

add_subdirectory(src)

# Fleet-level tests (ABI/dlopen gate, single-component pipeline smokes, race regression).
# Added after src/ so COMPS_ENABLED_MODULES is populated.
if(BUILD_TESTING)
    add_subdirectory(test)
endif()

# Release manifest: generated from the modules the build produced, never hand-maintained.
include(CompsManifest)
