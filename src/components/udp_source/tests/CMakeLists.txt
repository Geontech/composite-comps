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
# along with this program.  If not, see http://www.gnu.org/licenses/.
#

# Deterministic unit tests for the fixed-window recvmmsg batching policy.

include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(receive_batch_policy_tests
    receive_batch_policy_tests.cpp
)
target_link_libraries(receive_batch_policy_tests
    PRIVATE
        Catch2::Catch2WithMain
)
target_include_directories(receive_batch_policy_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_compile_features(receive_batch_policy_tests PRIVATE cxx_std_23)

include(CTest)
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(receive_batch_policy_tests TEST_PREFIX "receive_batch_policy: ")

install(TARGETS
    receive_batch_policy_tests
    RUNTIME DESTINATION bin/tests
    OPTIONAL
)
