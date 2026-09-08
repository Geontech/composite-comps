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

# The fleet's most important gate: every shipped module must dlopen and honor create().
#
# Components ship as MODULE DSOs, so an ABI or link mistake is invisible until load -- a .so with
# unresolved symbols links with exit 0. That has already happened here once -- libpkt_parser.so
# built clean while missing every parser constructor -- and only an executable's link failure
# caught it. COMPS_MODULE_NO_UNDEFINED now rejects it at link time; this gate is the backstop.
#
# The per-module tests are registered by comps_add_component() itself, not from a table here. A
# table was evaluated before a consuming overlay project declared its components, so a downstream
# module silently escaped the gate. Registering at declaration makes that impossible.

add_executable(comps_abi_smoke abi_smoke.cpp)
target_link_libraries(comps_abi_smoke
    PRIVATE
        composite::composite
        comps_baseline
        comps_warnings
        ${CMAKE_DL_LIBS}
)
target_compile_features(comps_abi_smoke PRIVATE cxx_std_${COMPS_CXX_STANDARD})

# Negative case: an unknown type must fail cleanly (throw -> null), not crash.
if(TARGET fft)
    add_test(NAME abi_smoke.reject_bad_type
             COMMAND comps_abi_smoke $<TARGET_FILE:fft> abi_bad bogus --expect-reject)
    set_tests_properties(abi_smoke.reject_bad_type PROPERTIES LABELS abi)
endif()
