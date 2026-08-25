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

# The protocol layer is shared code, deliberately separate from component.cpp:
#   * pkt_builder's round-trip test parses what it builds using these exact objects, so one
#     wire-format truth is enforced by the build rather than by two copies of a source list;
#   * the fuzz harness drives the parsers with no component attached.
# It cannot simply be part of pkt_parser_objs, because COMPOSITE_REGISTER_SIMPLE() in
# component.cpp defines the module entry points `create` and `composite_abi_version` --
# linking two components' registrations into one test executable is a duplicate-symbol error.
comps_add_object_library(pkt_protocols
    SOURCES overlay/sdds.cpp
            overlay/vita49.cpp
            parsers/sdds_parser.cpp
            parsers/vita49_parser.cpp
            parsers/vita49dot1_parser.cpp
    LINK    comps::vrtgen
)

# Downstream extension point for protocol parsers.
#
# Point COMPS_LOCAL_PARSERS_DIR at a directory containing parsers/local_parsers.inc (plus whatever
# headers it includes) and parser_table.hpp picks it up through __has_include -- no FOSS file is
# edited. An overlay project sets this before adding this repo, and adds its parser's .cpp with
# target_sources(pkt_protocols PRIVATE ...).
#
# PUBLIC on pkt_protocols on purpose: parser_table() is an inline function, so EVERY translation
# unit that includes parser_table.hpp must see the same hook state. One TU seeing the local parsers
# and another not seeing them is an ODR violation, and the symptom would be a parser that exists in
# the table the tests observe but not the one the component builds.
if(COMPS_LOCAL_PARSERS_DIR)
    if(NOT EXISTS "${COMPS_LOCAL_PARSERS_DIR}/parsers/local_parsers.inc")
        message(FATAL_ERROR
            "COMPS_LOCAL_PARSERS_DIR='${COMPS_LOCAL_PARSERS_DIR}' does not contain "
            "parsers/local_parsers.inc")
    endif()
    target_include_directories(pkt_protocols PUBLIC ${COMPS_LOCAL_PARSERS_DIR})
    message(STATUS "composite-comps: local protocol parsers from ${COMPS_LOCAL_PARSERS_DIR}")
endif()

# pkt_protocols above is declared unconditionally -- see src/components/CMakeLists.txt. Only the
# component module and its tests are gated.
if(COMPS_BUILD_PKT_PARSER)
    comps_add_component(pkt_parser
        SOURCES component.cpp
        OBJECTS pkt_protocols
        LINK    comps::vrtgen
    )

    # Untrusted-input parser fuzz/regression (built only when testing is enabled).
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
