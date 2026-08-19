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

# Where component modules install. composite-cli finds them through its embedded
# RUNPATH ($ORIGIN/../lib64), so the default must match the framework's libdir.
set(COMPS_MODULE_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}" CACHE STRING
    "Install destination for component modules")

# Reject a module with unresolved symbols at link time instead of at dlopen time.
option(COMPS_MODULE_NO_UNDEFINED "Link component modules with --no-undefined" ON)

# ---------------------------------------------------------------------------
# comps_add_component(<name>
#     SOURCES  <...>          component translation units
#     OBJECTS  <...>          sibling OBJECT libraries whose code this module contains
#     LINK     <...>          extra libraries (PUBLIC on the object library)
#     INCLUDES <...>          extra include directories (PUBLIC)
#     DEFINES  <...>          extra compile definitions (PUBLIC)
#     STANDARD <n>            override COMPS_CXX_STANDARD
#     ABI_TYPE <t>            data type create() needs, for templated components
#                             (COMPOSITE_REGISTER_COMPONENT); omit for COMPOSITE_REGISTER_SIMPLE
#     FAST_MATH               opt into comps_fastmath
#     NO_ARCH_BASELINE        opt out of comps_baseline
#     LTO_RELEASE             interprocedural optimization for Release
# )
#
# Creates two targets:
#
#   <name>_objs   OBJECT library -- the compiled component plus its usage
#                 requirements. Tests, smoke binaries, and cross-component test
#                 fixtures link THIS, so the tested code is the same objects,
#                 built with the same flags, as the code that ships.
#   <name>        MODULE library -- the installed .so. The name is load-bearing:
#                 pipeline JSON refers to modules by file name, so it must not
#                 drift from the target name.
#
# The alternative -- relisting ../component.cpp in each test target -- is what
# the fleet did before, in four places, with three different flag sets for
# psd/component.cpp alone.
# ---------------------------------------------------------------------------
function(comps_add_component name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG
        "FAST_MATH;NO_ARCH_BASELINE;LTO_RELEASE"   # options
        "STANDARD;ABI_TYPE"                         # one-value
        "SOURCES;OBJECTS;LINK;INCLUDES;DEFINES")    # multi-value

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "comps_add_component(${name}): SOURCES is required")
    endif()
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "comps_add_component(${name}): unrecognized arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    set(_std "${COMPS_CXX_STANDARD}")
    if(ARG_STANDARD)
        set(_std "${ARG_STANDARD}")
    endif()

    # --- the compiled component ------------------------------------------
    add_library(${name}_objs OBJECT ${ARG_SOURCES})
    set_target_properties(${name}_objs PROPERTIES
        POSITION_INDEPENDENT_CODE ON     # the objects feed a MODULE and test executables
        CXX_STANDARD ${_std}
        CXX_STANDARD_REQUIRED ON)
    # CXX_STANDARD is a target property, not a usage requirement: it does NOT travel
    # through target_link_libraries(). Export the equivalent compile feature so anything
    # linking these objects compiles at the same standard the component was built with.
    target_compile_features(${name}_objs PUBLIC cxx_std_${_std})

    # PUBLIC so the MODULE and every test target inherit them identically.
    target_include_directories(${name}_objs PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${COMPS_SOURCE_DIR}/include      # fleet-wide shared headers (simd_fmv.hpp, windows.hpp)
        ${ARG_INCLUDES})
    if(ARG_DEFINES)
        target_compile_definitions(${name}_objs PUBLIC ${ARG_DEFINES})
    endif()
    target_link_libraries(${name}_objs PUBLIC composite::composite ${ARG_LINK})
    if(NOT ARG_NO_ARCH_BASELINE)
        target_link_libraries(${name}_objs PUBLIC comps_baseline)
    endif()
    if(ARG_FAST_MATH)
        target_link_libraries(${name}_objs PUBLIC comps_fastmath)
    endif()
    # PRIVATE: our warning level is ours, not our consumers' problem.
    target_link_libraries(${name}_objs PRIVATE comps_warnings)

    # --- the installed module --------------------------------------------
    # No sources of its own; the objects arrive through the link.
    add_library(${name} MODULE)
    target_link_libraries(${name} PRIVATE ${name}_objs)

    # OBJECT-library object files are NOT propagated transitively. Linking a sibling
    # object library to ${name}_objs conveys its include dirs, flags, and compile
    # features, but NOT its .o files -- so the module must link it directly as well.
    #
    # Getting this wrong fails silently: a MODULE tolerates undefined symbols (they are
    # meant to resolve against the host process at dlopen), so the .so links "fine" and
    # only breaks when composite-cli loads it. Which is why test/abi/ must dlopen every
    # shipped module rather than trusting a green build.
    if(ARG_OBJECTS)
        target_link_libraries(${name}_objs PUBLIC ${ARG_OBJECTS})   # usage requirements
        target_link_libraries(${name}      PRIVATE ${ARG_OBJECTS})  # the object files
    endif()

    # Turn that silent failure into a link error: with composite and the component's own
    # dependencies linked, a correctly-built module has nothing left unresolved.
    if(COMPS_MODULE_NO_UNDEFINED)
        target_link_options(${name} PRIVATE LINKER:--no-undefined)
    endif()

    if(ARG_LTO_RELEASE OR COMPS_LTO)
        set_property(TARGET ${name}_objs PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set_property(TARGET ${name}      PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    if(COMPS_INSTALL)
        install(TARGETS ${name}
            LIBRARY DESTINATION ${COMPS_MODULE_INSTALL_DIR}
            COMPONENT Runtime)
    endif()

    # Consumed by the generated release manifest, so it cannot claim a module the build did not
    # produce.
    set_property(GLOBAL APPEND PROPERTY COMPS_ENABLED_MODULES ${name})

    # --- the ABI gate, registered here rather than from a table -----------
    #
    # Components ship as MODULE DSOs, so an ABI or link error is invisible until dlopen. Registering
    # the test at DECLARATION means it cannot be outrun by ordering: a table in test/abi/ is
    # evaluated before a consuming overlay project has declared its own components, so an overlay's
    # module silently escaped the gate. Now every component that exists gets the test, wherever it
    # was declared.
    #
    # comps_abi_smoke is defined later (test/ is added after src/); add_test resolves target names
    # at generate time, so a forward reference is fine.
    if(BUILD_TESTING)
        set(_abi_args $<TARGET_FILE:${name}> abi_${name})
        if(ARG_ABI_TYPE)
            list(APPEND _abi_args ${ARG_ABI_TYPE})
        endif()
        add_test(NAME abi_smoke.${name} COMMAND comps_abi_smoke ${_abi_args})
        set_tests_properties(abi_smoke.${name} PROPERTIES LABELS abi)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# comps_add_object_library(<name>
#     SOURCES <...> [LINK <...>] [INCLUDES <...>] [DEFINES <...>]
#     [STANDARD <n>] [FAST_MATH] [NO_ARCH_BASELINE]
# )
#
# A policy-carrying OBJECT library with no module and no install: for code shared
# between components, or between a component and a sibling component's tests.
#
# This exists because COMPOSITE_REGISTER_SIMPLE() defines the module entry points
# `create` and `composite_abi_version`. Two components' registrations cannot be
# linked into one executable -- so shared code that a second component's tests need
# has to live in a target that carries no registration.
# ---------------------------------------------------------------------------
function(comps_add_object_library name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG
        "FAST_MATH;NO_ARCH_BASELINE" "STANDARD" "SOURCES;LINK;INCLUDES;DEFINES")

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "comps_add_object_library(${name}): SOURCES is required")
    endif()
    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR
            "comps_add_object_library(${name}): unrecognized arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    set(_std "${COMPS_CXX_STANDARD}")
    if(ARG_STANDARD)
        set(_std "${ARG_STANDARD}")
    endif()

    add_library(${name} OBJECT ${ARG_SOURCES})
    set_target_properties(${name} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD ${_std}
        CXX_STANDARD_REQUIRED ON)
    target_compile_features(${name} PUBLIC cxx_std_${_std})
    target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                        ${COMPS_SOURCE_DIR}/include ${ARG_INCLUDES})
    if(ARG_DEFINES)
        target_compile_definitions(${name} PUBLIC ${ARG_DEFINES})
    endif()
    target_link_libraries(${name} PUBLIC composite::composite ${ARG_LINK})
    if(NOT ARG_NO_ARCH_BASELINE)
        target_link_libraries(${name} PUBLIC comps_baseline)
    endif()
    if(ARG_FAST_MATH)
        target_link_libraries(${name} PUBLIC comps_fastmath)
    endif()
    target_link_libraries(${name} PRIVATE comps_warnings)
endfunction()
