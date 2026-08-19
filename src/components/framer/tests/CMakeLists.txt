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

# Framer component unit tests.
#
# Both targets link framer_objs rather than recompiling ../component.cpp, so the tested
# code is the same objects, built with the same flags, as libframer.so ships.
#
# The -mavx2/-mfma/-mavx512* options these targets used to carry are gone on purpose.
# convert_variant.hpp selects its SIMD path at RUNTIME via __builtin_cpu_supports(), and
# each kernel carries its own [[gnu::target("avx2")]] / [[gnu::target("avx512f,...")]]
# attribute, so the ISA-specific code compiles and is reachable at any -march. The flags
# therefore never gated which paths were tested -- they only let the compiler emit AVX
# elsewhere in the TU, which (a) made the tested component.cpp differ in codegen from the
# shipped module and (b) risked SIGILL on a runner without AVX-512, since
# check_cxx_compiler_flag only proves the COMPILER accepts the flag. Neither test source
# uses an intrinsic directly.

add_executable(converter_tests converter_tests.cpp)
target_link_libraries(converter_tests
    PRIVATE
        framer_objs
        Catch2::Catch2WithMain
        Threads::Threads
)

# NOTE: the overlapped framing ring was lifted into the framework (composite::overlap_ring);
# its unit tests now live there as test_overlap_ring. Only framer-specific tests remain here.
add_executable(framer_integration_tests framer_integration_tests.cpp)
target_link_libraries(framer_integration_tests
    PRIVATE
        framer_objs
        Catch2::Catch2WithMain
        Threads::Threads
)

catch_discover_tests(converter_tests)
catch_discover_tests(framer_integration_tests)

install(TARGETS
    converter_tests
    framer_integration_tests
    RUNTIME DESTINATION bin/tests
    COMPONENT Tests
    OPTIONAL
)
