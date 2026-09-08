#
# Copyright (C) 2026 Geon Technologies, LLC
# SPDX-License-Identifier: LGPL-3.0-or-later
#

# Component-level tests: real ports, real process(), verifying the bytes on disk — the
# writev gather ordering, the num_bytes cap (mid-buffer and mid-batch), re-initialize
# truncation, and the loud open failure.
add_executable(file_writer_integration_tests file_writer_integration_tests.cpp)
target_link_libraries(file_writer_integration_tests
    PRIVATE
        file_writer_objs
        Catch2::Catch2WithMain
        Threads::Threads
)
target_include_directories(file_writer_integration_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(file_writer_integration_tests PRIVATE cxx_std_23)

catch_discover_tests(file_writer_integration_tests TEST_PREFIX "file_writer: ")

install(TARGETS file_writer_integration_tests
    RUNTIME DESTINATION bin/tests
    COMPONENT Tests
    OPTIONAL
)
