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

# Deterministic unit tests for the frame-size autodiscovery classifier.
#
# Hermetic for the same reason as above: discovery_filter.hpp is a self-contained predicate, so
# the test needs neither the component nor a socket. comps_baseline compiles it at ship flags.
add_executable(discovery_filter_tests discovery_filter_tests.cpp)
target_link_libraries(discovery_filter_tests
    PRIVATE
        Catch2::Catch2WithMain
        comps_baseline
)
target_include_directories(discovery_filter_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(discovery_filter_tests PRIVATE cxx_std_23)

catch_discover_tests(discovery_filter_tests TEST_PREFIX "discovery_filter: ")

# Loopback integration test for the runtime-reactivation path: an `active=true` RUNTIME write
# must leave the source RECEIVING when autodiscovery is in play. Links the real component (and
# so the socket backends): the regression it pins lives in the interplay of the park hook, the
# abort eventfd, and start_receiver_locked(), which no hermetic unit can exercise.
add_executable(udp_source_reactivation_tests reactivation_test.cpp)
target_link_libraries(udp_source_reactivation_tests
    PRIVATE
        udp_source_objs
        Catch2::Catch2WithMain
        Threads::Threads
)
target_include_directories(udp_source_reactivation_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(udp_source_reactivation_tests PRIVATE cxx_std_23)

catch_discover_tests(udp_source_reactivation_tests TEST_PREFIX "udp_source: ")

install(TARGETS
    receive_batch_policy_tests
    discovery_filter_tests
    udp_source_reactivation_tests
    RUNTIME DESTINATION bin/tests
    COMPONENT Tests
    OPTIONAL
)
