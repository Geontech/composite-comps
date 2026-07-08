#
# Copyright (C) 2025 Geon Technologies, LLC
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

# Untrusted-input fuzz/regression for the protocol parsers. Compiles the overlay +
# parser translation units directly (no DSO) so it can be built under sanitizers.
# For real coverage of the OOB-read guards, configure with
#   -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
add_executable(pkt_parser_fuzz
    fuzz_parsers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../overlay/sdds.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../overlay/vita49.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../parsers/sdds_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../parsers/vita49_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../parsers/vita49dot1_parser.cpp
)
target_include_directories(pkt_parser_fuzz
    PRIVATE
    ${vrtgen_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(pkt_parser_fuzz
    PRIVATE
    composite::composite
)
add_test(NAME pkt_parser_fuzz COMMAND pkt_parser_fuzz)

# Protocol auto-re-detection: drives the real component's process() through connected
# ports and asserts on the detection state machine. Compiles the component + parser TUs
# directly (no DSO) so it links standalone against composite.
add_executable(pkt_parser_redetect
    redetect_test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../component.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../overlay/sdds.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../overlay/vita49.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../parsers/sdds_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../parsers/vita49_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../parsers/vita49dot1_parser.cpp
)
target_include_directories(pkt_parser_redetect
    PRIVATE
    ${vrtgen_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(pkt_parser_redetect
    PRIVATE
    composite::composite
)
add_test(NAME pkt_parser_redetect COMMAND pkt_parser_redetect)
