#
# Copyright (C) 2026 Geon Technologies, LLC
# SPDX-License-Identifier: LGPL-3.0-or-later
#

# Deterministic unit tests for whole-frame decimation. Header-only policy, so the test needs
# neither the component nor the framework -- the same shape as udp_source's batching-policy tests.
add_executable(frame_decimator_tests frame_decimator_tests.cpp)
target_link_libraries(frame_decimator_tests PRIVATE Catch2::Catch2WithMain comps_baseline)
target_include_directories(frame_decimator_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(frame_decimator_tests PRIVATE cxx_std_23)

catch_discover_tests(frame_decimator_tests TEST_PREFIX "frame_decimator: ")

install(TARGETS frame_decimator_tests
    RUNTIME DESTINATION bin/tests
    COMPONENT Tests
    OPTIONAL
)

# Component-level tests: real ports, real process(), pinning the binning math, threshold
# emission, EOS flush, and the no-sample-rate discard path.
add_executable(histogram_integration_tests histogram_integration_tests.cpp)
target_link_libraries(histogram_integration_tests
    PRIVATE
        histogram_objs
        Catch2::Catch2WithMain
        Threads::Threads
)
target_include_directories(histogram_integration_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(histogram_integration_tests PRIVATE cxx_std_23)

catch_discover_tests(histogram_integration_tests TEST_PREFIX "histogram: ")

install(TARGETS histogram_integration_tests
    RUNTIME DESTINATION bin/tests
    COMPONENT Tests
    OPTIONAL
)
