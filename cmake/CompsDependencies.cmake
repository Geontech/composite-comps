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

# ---------------------------------------------------------------------------
# External dependencies, declared once for the whole project.
#
# Supersedes the former cmake/composite.cmake and cmake/vrtgen.cmake, and is the
# only place composite, vrtgen, Catch2, IXWebSocket and Threads are declared.
# ---------------------------------------------------------------------------

include(FetchContent)

# --- composite framework -------------------------------------------------
# How the framework is acquired determines what a qualification run actually
# qualified, so it is an explicit choice rather than a fallback chain:
#
#   package  find_package() only. A missing framework is a hard error. This is
#            what release, container, and CI builds must use -- the fleet is
#            then provably built against one known framework install.
#   fetch    FetchContent only, at tag v${COMPS_COMPOSITE_VERSION}.
#   auto     find_package(), falling back to FetchContent. The default, for the
#            convenience of a developer without an installed framework. The
#            hazard it carries is that a build can succeed against a framework
#            nobody chose, so release and container builds set `package`.
#
# The tag must be an immutable ref that actually ships the API the fleet is
# written against (typed_property/COMPOSITE_STRUCT, 3-arg send_data, typed
# annotations). A missing tag fails loudly here rather than silently building an
# incompatible framework.
set(COMPS_COMPOSITE_PROVIDER "auto" CACHE STRING
    "How to acquire the composite framework: package | fetch | auto")
set_property(CACHE COMPS_COMPOSITE_PROVIDER PROPERTY STRINGS package fetch auto)

# Enforce the framework/fleet compatibility contract at configure time instead of
# discovering it at dlopen time. Release and container builds set this ON.
option(COMPS_REQUIRE_EXACT_COMPOSITE
    "Require exactly composite ${COMPS_COMPOSITE_VERSION}" OFF)

function(_comps_find_composite required)
    set(_args ${COMPS_COMPOSITE_VERSION})
    if(COMPS_REQUIRE_EXACT_COMPOSITE)
        list(APPEND _args EXACT)
    endif()
    if(required)
        list(APPEND _args REQUIRED)
    else()
        list(APPEND _args QUIET)
    endif()
    find_package(composite ${_args})
    set(composite_FOUND ${composite_FOUND} PARENT_SCOPE)
    set(composite_VERSION ${composite_VERSION} PARENT_SCOPE)
endfunction()

function(_comps_fetch_composite)
    FetchContent_Declare(composite
        GIT_REPOSITORY https://github.com/geontech/composite.git
        GIT_TAG        v${COMPS_COMPOSITE_VERSION}
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(composite)
endfunction()

# find_package() creates composite::composite as an IMPORTED target, and imported targets are
# DIRECTORY-scoped: visible here and below, but NOT in a sibling directory of a consuming project.
# An overlay repo calling comps_add_component() for its own component would fail with
# 'links to target "composite::composite" but the target was not found'. Promote it.
#
# find_package(... GLOBAL) and CMAKE_FIND_PACKAGE_TARGETS_GLOBAL both need CMake 3.24; this works
# from 3.11 and must run in the directory that created the target, which is here.
function(_comps_promote_imported tgt)
    if(NOT TARGET ${tgt})
        return()
    endif()
    get_target_property(_alias ${tgt} ALIASED_TARGET)
    if(_alias)
        return()   # aliases of non-imported targets are already globally visible
    endif()
    get_target_property(_imported ${tgt} IMPORTED)
    get_target_property(_global ${tgt} IMPORTED_GLOBAL)
    if(_imported AND NOT _global)
        set_target_properties(${tgt} PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
endfunction()

if(COMPS_COMPOSITE_PROVIDER STREQUAL "package")
    _comps_find_composite(TRUE)
    message(STATUS "composite-comps: using installed composite ${composite_VERSION}")
elseif(COMPS_COMPOSITE_PROVIDER STREQUAL "fetch")
    message(STATUS "composite-comps: fetching composite v${COMPS_COMPOSITE_VERSION}")
    _comps_fetch_composite()
elseif(COMPS_COMPOSITE_PROVIDER STREQUAL "auto")
    _comps_find_composite(FALSE)
    if(TARGET composite::composite)
        message(STATUS "composite-comps: using installed composite ${composite_VERSION}")
    else()
        message(STATUS "composite not found, falling back to FetchContent")
        _comps_fetch_composite()
    endif()
else()
    message(FATAL_ERROR
        "COMPS_COMPOSITE_PROVIDER must be package, fetch, or auto "
        "(got '${COMPS_COMPOSITE_PROVIDER}')")
endif()

# Record how the framework was actually resolved, in the CACHE rather than as a directory-scoped
# variable. comps_write_manifest() may be called from a CONSUMING project's scope, where
# composite_VERSION set by find_package() here is invisible -- which made the manifest report a
# found framework as "(fetched)". A provenance artifact that misreports its own inputs is worse
# than none.
set(COMPS_FRAMEWORK_RESOLVED "${composite_VERSION}" CACHE INTERNAL "resolved composite version")
set(COMPS_FRAMEWORK_PROVIDER "${COMPS_COMPOSITE_PROVIDER}" CACHE INTERNAL "how composite was acquired")
if(NOT COMPS_FRAMEWORK_RESOLVED)
    set(COMPS_FRAMEWORK_RESOLVED "${COMPS_COMPOSITE_VERSION} (fetched)" CACHE INTERNAL "" FORCE)
endif()

# The helper links composite::composite into every component, including a consumer's, so it must
# be visible outside this directory. Threads and nlohmann_json are promoted too: they are the
# imported targets an overlay component is most likely to want.
_comps_promote_imported(composite::composite)

# --- threads -------------------------------------------------------------
# Used by the converters' std::call_once and by every concurrency-touching test.
# Was found separately in four test directories; rely on the imported target
# rather than an implicit -pthread.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

_comps_promote_imported(Threads::Threads)
_comps_promote_imported(nlohmann_json::nlohmann_json)

# --- IXWebSocket ---------------------------------------------------------
# Hoisted out of src/components/ws_sink. The TLS options must be set before the
# subproject is configured, which is why they are cache-FORCEd here. It stays at THIS
# scope rather than moving back down: CompsFlags adds -fsanitize at directory scope
# here, so a dependency configured from a deeper directory would miss the
# instrumentation the sanitize jobs rely on.
#
# ws_sink is the only consumer, so an overlay that takes the FOSS modules from the
# published image (-DCOMPS_BUILD_ALL=OFF) has no use for it -- and was cloning and
# building it anyway, which on the FetchContent path costs a git clone plus a static
# library, and drags openssl-devel/zlib-devel into the builder image for nothing.
#
# The per-component options are not declared until src/components/CMakeLists.txt,
# which runs after this file, so decide the same way that file will: an explicit
# -DCOMPS_BUILD_WS_SINK wins, else -DCOMPS_BUILD_ALL, else that option's ON default.
# Both arrive in the cache before any option() call, so a first configure straight
# from the command line reads correctly.
if(DEFINED CACHE{COMPS_BUILD_WS_SINK})
    set(_comps_want_ws_sink "$CACHE{COMPS_BUILD_WS_SINK}")
elseif(DEFINED COMPS_BUILD_ALL)
    set(_comps_want_ws_sink "${COMPS_BUILD_ALL}")
else()
    set(_comps_want_ws_sink ON)
endif()

if(_comps_want_ws_sink)
    find_package(IXWebSocket QUIET)
    if(NOT IXWebSocket_FOUND AND NOT TARGET ixwebsocket)
        FetchContent_Declare(IXWebSocket
            GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
            GIT_TAG v11.4.6
        )
        set(USE_TLS ON CACHE BOOL "" FORCE)
        set(USE_OPEN_SSL ON CACHE BOOL "" FORCE)
        # IXWEBSOCKET_INSTALL defaults TRUE, which put its entire SDK into our install
        # tree: 51 headers, libixwebsocket.a, a CMake package config, and a .pc file.
        # The fleet installs component modules, nothing else -- and shipping a
        # dependency's headers in the container image is redistribution we would then
        # owe notices for.
        set(IXWEBSOCKET_INSTALL OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(IXWebSocket)
    endif()
else()
    message(STATUS "composite-comps: ws_sink disabled; skipping IXWebSocket")
endif()
unset(_comps_want_ws_sink)

# --- Catch2 --------------------------------------------------------------
# Was declared five times, in five test directories. Only the first declaration
# ever took effect, so bumping the version in the other four did nothing.
#
# Declaring it here also stops it inheriting a component's directory-scope
# add_compile_options(): configured from src/components/framer/tests, Catch2 was
# being compiled with the fleet's -march=x86-64-v2 baseline. Third-party code
# should not be built with our flags.
if(BUILD_TESTING)
    find_package(Catch2 3 QUIET)
    if(NOT TARGET Catch2::Catch2WithMain)
        FetchContent_Declare(Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.11.0
        )
        FetchContent_MakeAvailable(Catch2)
        # catch_discover_tests() lives in Catch2's extras/, which is not on the
        # module path for a FetchContent build. Appending here (root scope) makes
        # it available to every test directory, replacing four copies of this line.
        list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    endif()
    include(Catch)
endif()

# --- vrtgen --------------------------------------------------------------
# comps::vrtgen exists so components never have to care which acquisition path ran.
# They previously added ${vrtgen_SOURCE_DIR}/include by hand, which only works on
# the FetchContent path: if find_package(vrtgen) succeeds, vrtgen_SOURCE_DIR is
# undefined, the include expands to a bare -I/include, and nothing links
# vrtgen::vrtgen -- so the headers are simply absent.
find_package(vrtgen 0.7.12 QUIET)
if(NOT TARGET vrtgen::vrtgen)
    message(STATUS "vrtgen not found, falling back to FetchContent")
    FetchContent_Declare(vrtgen
        GIT_REPOSITORY https://github.com/geontech/vrtgen.git
        GIT_TAG v0.7.12
    )
    FetchContent_MakeAvailable(vrtgen)
else()
    message(STATUS "Using vrtgen version: ${vrtgen_VERSION}")
endif()

add_library(comps_vrtgen INTERFACE)
add_library(comps::vrtgen ALIAS comps_vrtgen)
if(TARGET vrtgen::vrtgen)
    target_link_libraries(comps_vrtgen INTERFACE vrtgen::vrtgen)
else()
    # Header-only fallback: assert the path exists rather than quietly emitting
    # an -I to a directory that is not there.
    if(NOT EXISTS "${vrtgen_SOURCE_DIR}/include")
        message(FATAL_ERROR
            "vrtgen headers not found at '${vrtgen_SOURCE_DIR}/include'")
    endif()
    # SYSTEM: vrtgen's headers are third-party, so our -Wall -Wextra -Wpedantic must not
    # fire on them -- with COMPS_WERROR=ON that would fail the build on someone else's code.
    target_include_directories(comps_vrtgen SYSTEM INTERFACE ${vrtgen_SOURCE_DIR}/include)
endif()

# --- DPDK (a property of the framework, not of one component) --------------
# udp_source's DPDK backend compiles only when the framework itself was built
# with COMPOSITE_USE_DPDK, so the probe belongs where the framework is resolved.
# It used to live inside src/components/udp_source, which made a fleet-wide fact
# look like one component's private business -- and pkt_parser and udp_sink will
# both eventually want to ask the same question.
get_target_property(_composite_defs composite::composite INTERFACE_COMPILE_DEFINITIONS)
if(_composite_defs MATCHES "COMPOSITE_USE_DPDK")
    set(COMPS_HAS_DPDK ON)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DPDK REQUIRED libdpdk)
    message(STATUS "composite-comps: framework has DPDK; udp_source DPDK backend enabled")
else()
    set(COMPS_HAS_DPDK OFF)
    message(STATUS "composite-comps: framework has no DPDK; udp_source DPDK backend disabled")
endif()
