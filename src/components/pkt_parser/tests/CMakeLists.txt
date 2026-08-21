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

# Untrusted-input fuzz/regression for the protocol parsers. Links pkt_protocols (no
# component, no module registration) so it can be built under sanitizers.
# Real coverage of the OOB-read guards comes from the asan-ubsan preset.
add_executable(pkt_parser_fuzz fuzz_parsers.cpp)
target_link_libraries(pkt_parser_fuzz PRIVATE pkt_protocols)
add_test(NAME pkt_parser_fuzz COMMAND pkt_parser_fuzz)

# Protocol auto-re-detection: drives the real component's process() through connected
# ports and asserts on the detection state machine, so it needs the component objects too.
add_executable(pkt_parser_redetect redetect_test.cpp)
# Both object libraries directly: object files do not propagate transitively.
target_link_libraries(pkt_parser_redetect PRIVATE pkt_parser_objs pkt_protocols)
add_test(NAME pkt_parser_redetect COMMAND pkt_parser_redetect)

# The parser table's probe order is behavior; pin it.
add_executable(parser_table_tests parser_table_tests.cpp)
target_link_libraries(parser_table_tests PRIVATE pkt_protocols Catch2::Catch2WithMain)
catch_discover_tests(parser_table_tests)

# VITA 49 packet-type contract: what vita49_parser claims, and what it does with a packet type it
# cannot decode.
add_executable(vita49_packet_type_tests vita49_packet_type_tests.cpp)
target_link_libraries(vita49_packet_type_tests PRIVATE pkt_protocols Catch2::Catch2WithMain)
catch_discover_tests(vita49_packet_type_tests)

# Correctness/hardening regressions: CIF0 64-bit field widths, empty-payload rejection,
# one-shot sequence-gap warnings + per-gap flags, SDDS parity handling, and the PLRV
# pooled-byteswap round-trip (including pool exhaustion).
add_executable(parser_hardening_tests parser_hardening_tests.cpp)
target_link_libraries(parser_hardening_tests PRIVATE pkt_protocols Catch2::Catch2WithMain)
catch_discover_tests(parser_hardening_tests)
