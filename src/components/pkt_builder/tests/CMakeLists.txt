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

# pkt_builder integration tests, including a loopback through the real vita49 parser:
# what the builder emits must round-trip through the parser (one wire-format truth).
#
# pkt_protocols, not pkt_parser_objs: the latter carries pkt_parser's COMPOSITE_REGISTER_SIMPLE,
# whose create()/composite_abi_version() would collide with pkt_builder's.
add_executable(pkt_builder_integration_tests pkt_builder_integration_tests.cpp)
target_link_libraries(pkt_builder_integration_tests
    PRIVATE
        pkt_builder_objs
        pkt_protocols
        Catch2::Catch2WithMain
        Threads::Threads
)

catch_discover_tests(pkt_builder_integration_tests)

install(TARGETS
    pkt_builder_integration_tests
    RUNTIME DESTINATION bin/tests
    COMPONENT Tests
    OPTIONAL
)
