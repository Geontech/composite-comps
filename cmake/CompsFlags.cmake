#
# Copyright (C) 2026 Geon Technologies, LLC
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
# The single authority for fleet compile/link policy.
#
# Policy is expressed as INTERFACE targets driven by cache variables, and is
# linked into fleet targets by comps_add_component(). Components consume policy;
# they never declare it. Three properties follow from that:
#
#   * every knob is discoverable (`cmake -LAH`) and settable from a preset;
#   * warning flags reach fleet code only, never a FetchContent dependency, so a
#     future -Werror cannot be tripped by Catch2 or IXWebSocket;
#   * there is exactly one place to look when the compiler line is not what you
#     expected.
#
# The flags this replaces were not merely scattered, they were inert. Each
# component set CMAKE_CXX_FLAGS_INIT after the root project() call had already
# enabled CXX, and *_INIT is consulted only when a language is first enabled --
# so -Wall -Wextra -Wpedantic never reached the compiler in any component. That
# is why the whole fleet was warning-clean by accident rather than by discipline.
# ---------------------------------------------------------------------------

# --- language standard ---------------------------------------------------
# Applied per-target by comps_add_component(), NOT as a directory-scope
# CMAKE_CXX_STANDARD. Setting it at directory scope here would also apply to
# every FetchContent dependency configured from this scope (Catch2, IXWebSocket,
# vrtgen), silently changing how third-party code is compiled.
#
# CXX_EXTENSIONS is deliberately left at CMake's default (ON, i.e. -std=gnu++23)
# because that is what the fleet compiles with today; moving to -std=c++23 is a
# separate, testable decision.
set(COMPS_CXX_STANDARD "23" CACHE STRING "C++ standard for fleet targets")

# --- warnings ------------------------------------------------------------
# Linked PRIVATE to fleet targets, so warning level does not propagate to
# anything that merely links against them.
option(COMPS_WERROR "Treat compiler warnings as errors (CI)" OFF)
add_library(comps_warnings INTERFACE)
add_library(comps::warnings ALIAS comps_warnings)
target_compile_options(comps_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
if(COMPS_WERROR)
    target_compile_options(comps_warnings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Werror>)
endif()

# --- CPU baseline --------------------------------------------------------
# One variable instead of nine copies of -march scattered through the component
# directories. Those copies were also unoverridable: directory-scope
# add_compile_options() is appended AFTER CMAKE_CXX_FLAGS, and for conflicting
# -march the last flag wins, so -DCMAKE_CXX_FLAGS="-march=native" was silently
# discarded. Overriding the baseline is now one explicit, total choice.
#
# Set COMPS_CPU_BASELINE to "" for a portable build, or "native" for a tuned
# local one.
set(COMPS_CPU_BASELINE "x86-64-v2" CACHE STRING "-march value for fleet targets ('' disables)")
set(COMPS_CPU_TUNE     "generic"   CACHE STRING "-mtune value for fleet targets ('' disables)")
add_library(comps_baseline INTERFACE)
add_library(comps::baseline ALIAS comps_baseline)
if(COMPS_CPU_BASELINE)
    target_compile_options(comps_baseline INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-march=${COMPS_CPU_BASELINE}>)
endif()
if(COMPS_CPU_TUNE)
    target_compile_options(comps_baseline INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-mtune=${COMPS_CPU_TUNE}>)
endif()

# --- fast math (opt-in, per component) -----------------------------------
# Currently enabled for fft, psd, and histogram. Kept as a named, defeatable
# policy rather than an inherited habit: -ffast-math changes NaN/Inf and
# denormal handling, which matters for components whose numerical output is
# still under review.
option(COMPS_FAST_MATH "Allow -ffast-math on components that request it" ON)
add_library(comps_fastmath INTERFACE)
add_library(comps::fastmath ALIAS comps_fastmath)
if(COMPS_FAST_MATH)
    target_compile_options(comps_fastmath INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>)
endif()

# --- sanitizers ----------------------------------------------------------
# Mirrors the framework's COMPOSITE_SANITIZER. Applied at directory scope on
# purpose: every translation unit built below -- including Catch2 -- must be
# instrumented consistently, which is exactly what tsan-test's hand-rolled SAN
# variable could not achieve for a single test binary.
set(COMPS_SANITIZER "" CACHE STRING "Sanitizer: '' (off), 'thread', 'address,undefined'")
if(COMPS_SANITIZER)
    message(STATUS "composite-comps: building with -fsanitize=${COMPS_SANITIZER}")
    add_compile_options(-fsanitize=${COMPS_SANITIZER} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${COMPS_SANITIZER})
endif()

# --- link-time optimization ----------------------------------------------
# udp_source enables this for Release today, silently and only for itself.
# Applied per-target by comps_add_component() once components migrate.
option(COMPS_LTO "Enable interprocedural optimization for Release fleet targets" OFF)

# --- function multiversioning under ThreadSanitizer ----------------------
# exp_smooth, fft, and psd use GCC's native C++ function multiversioning for their SIMD
# kernels. GCC implements it with an IFUNC resolver, and IFUNC resolvers run during dynamic
# relocation -- before the ThreadSanitizer runtime initializes -- so a multiversioned module
# used to kill any TSan binary inside ld.so, before main.
#
# That is now handled in the source, not here: include/simd_fmv.hpp compiles the non-default
# versions out under __SANITIZE_THREAD__ and drops the attribute from the default one, so a
# TSan build emits no resolver while every shipping build keeps full multiversioning. See that
# header for the backtrace and the reasoning.
#
# COMPS_IFUNC_MODULES is retained for reference: these are the components whose ISA dispatch
# depends on that guard staying in place.
set(COMPS_IFUNC_MODULES exp_smooth fft psd)
