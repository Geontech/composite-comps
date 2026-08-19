#
# Copyright (C) 2026 Geon Technologies, LLC
# SPDX-License-Identifier: LGPL-3.0-or-later
#

# Fleet-level tests. These three directories were previously standalone CMake projects
# (abi-test/, pipeline-test/, tsan-test/) that were never added to the build: each had its
# own project() and find_package(composite), two referenced an undefined ${SPDLOG_INC}, and
# pipeline-test/ and tsan-test/ compiled fft/psd component.cpp by relative path with
# different flags than the modules they were meant to mirror. They are targets now.

add_subdirectory(abi)
add_subdirectory(smoke)
add_subdirectory(race)
