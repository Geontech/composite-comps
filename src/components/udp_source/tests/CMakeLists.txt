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

# Deterministic unit tests for the fixed-window recvmmsg batching policy.
#
# This one deliberately does NOT link udp_source_objs: receive_batch_policy.hpp is a
# self-contained policy header, and keeping the test free of the component (and therefore
# of composite and the socket backends) is what makes it a fast, hermetic unit test.
# It links comps_baseline so the header is still compiled at the flags it ships with.
add_executable(receive_batch_policy_tests receive_batch_policy_tests.cpp)
target_link_libraries(receive_batch_policy_tests
    PRIVATE
        Catch2::Catch2WithMain
        comps_baseline
)
target_include_directories(receive_batch_policy_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(receive_batch_policy_tests PRIVATE cxx_std_23)

catch_discover_tests(receive_batch_policy_tests TEST_PREFIX "receive_batch_policy: ")

install(TARGETS
    receive_batch_policy_tests
    RUNTIME DESTINATION bin/tests
    COMPONENT Tests
    OPTIONAL
)
