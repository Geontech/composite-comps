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

# The fleet's components, and the switches that select them.
#
# COMPS_BUILD_ALL sets the default for every per-component option. An overlay project that consumes
# this repo only for its flag policy and comps_add_component() -- building its own components and
# taking the FOSS modules from the published image instead of recompiling them -- configures with
# -DCOMPS_BUILD_ALL=OFF and compiles no component code here.
#
# NOTE on the cache: the per-component values are cached, so changing COMPS_BUILD_ALL later does not
# retroactively move options already written to the cache. Set the specific option, or reconfigure
# from a clean build directory.
set_property(GLOBAL PROPERTY COMPS_COMPONENTS_PROCESSED TRUE)

option(COMPS_BUILD_ALL "Default for every COMPS_BUILD_<COMPONENT> option" ON)

set(COMPS_COMPONENTS
    exp_smooth
    fft
    file_writer
    framer
    histogram
    pkt_builder
    pkt_parser
    psd
    udp_sink
    udp_source
    ws_sink
    CACHE INTERNAL "All components in the fleet"
)

foreach(_comp IN LISTS COMPS_COMPONENTS)
    string(TOUPPER ${_comp} _COMP)
    if(NOT DEFINED COMPS_BUILD_${_COMP})
        set(COMPS_BUILD_${_COMP} ${COMPS_BUILD_ALL} CACHE BOOL "Build the ${_comp} component")
    endif()
endforeach()

# pkt_parser is added unconditionally even when its component is switched off, because its directory
# also declares pkt_protocols -- the shared VITA 49 / SDDS protocol layer that pkt_builder's
# round-trip test links, and that a downstream extension parser links to reuse the v49 overlay
# instead of reimplementing it. The component and its tests are gated inside.
add_subdirectory(pkt_parser)

foreach(_comp IN LISTS COMPS_COMPONENTS)
    if(_comp STREQUAL "pkt_parser")
        continue()
    endif()
    string(TOUPPER ${_comp} _COMP)
    if(COMPS_BUILD_${_COMP})
        add_subdirectory(${_comp})
    else()
        message(STATUS "composite-comps: ${_comp} disabled (COMPS_BUILD_${_COMP}=OFF)")
    endif()
endforeach()
