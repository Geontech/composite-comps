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

set(UDP_SOURCE_SRCS
    component.cpp
    socket/packet_mmap.cpp
    socket/recvmmsg.cpp
    net/utils.cpp
)

# The DPDK backend is gated on the framework having been built with DPDK; the
# probe (and the libdpdk pkg-config lookup) lives in cmake/CompsDependencies.cmake.
if(COMPS_HAS_DPDK)
    list(APPEND UDP_SOURCE_SRCS
        socket/dpdk.cpp
        socket/igmp.cpp
    )
endif()

# NOTE: udp_source carried ${vrtgen_SOURCE_DIR}/include for no reason -- there is
# not a single vrtgen reference anywhere under this component. It links no vrtgen.
comps_add_component(udp_source
    SOURCES  ${UDP_SOURCE_SRCS}
    SYSTEM_INCLUDES $<$<BOOL:${COMPS_HAS_DPDK}>:${DPDK_INCLUDE_DIRS}>
    LINK     $<$<BOOL:${COMPS_HAS_DPDK}>:${DPDK_LIBRARIES}>
    DEFINES  $<$<BOOL:${COMPS_HAS_DPDK}>:COMPOSITE_HAS_DPDK=1>
    LTO_RELEASE
)

if(COMPS_HAS_DPDK)
    target_link_directories(udp_source_objs PUBLIC ${DPDK_LIBRARY_DIRS})

    # DPDK's headers cannot satisfy -Wpedantic: they use flexible array members, compound
    # literals, anonymous structs, and a zero-size array member not at the end of a struct.
    #
    # SYSTEM_INCLUDES above is necessary but NOT sufficient. GCC suppresses ordinary warnings from
    # system headers -- that is what silenced -Wvolatile from rte_spinlock.h -- but it does NOT
    # suppress -Wpedantic diagnostics there, so with COMPS_WERROR=ON the build still failed on
    # someone else's headers. Relax exactly one flag, on exactly the two translation units that
    # include DPDK, so -Wall and -Wextra still apply to this code and the rest of the fleet keeps
    # -Wpedantic.
    set_source_files_properties(socket/dpdk.cpp socket/igmp.cpp
        TARGET_DIRECTORY udp_source_objs
        PROPERTIES COMPILE_OPTIONS "-Wno-pedantic")
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
