#
# Copyright (C) 2026 Geon Technologies, LLC
# SPDX-License-Identifier: LGPL-3.0-or-later
#

# Single-component pipeline smokes: drive the real component through connected ports and
# assert the outputs are ordered, lossless, correctly sized, and annotated.
#
# These link fft_objs / psd_objs. Previously (as pipeline-test/) they recompiled
# ../src/components/{fft,psd}/component.cpp by relative path with their own -march/-ffast-math
# lines, so they exercised a differently-compiled copy of the component than the one that ships.

add_executable(fft_smoke fft_smoke.cpp)
target_link_libraries(fft_smoke PRIVATE fft_objs)
add_test(NAME fft_smoke COMMAND fft_smoke)
set_tests_properties(fft_smoke PROPERTIES LABELS smoke)

add_executable(psd_smoke psd_smoke.cpp)
target_link_libraries(psd_smoke PRIVATE psd_objs)
add_test(NAME psd_smoke COMMAND psd_smoke)
set_tests_properties(psd_smoke PROPERTIES LABELS smoke)

install(TARGETS fft_smoke psd_smoke
    RUNTIME DESTINATION bin/tests
    COMPONENT Tests
    OPTIONAL
)
