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

# Module-loaded chain coverage for intentionally lossy runtime FFT-size transitions. Loading the
# DSOs is necessary because each module exports the same create()/ABI entry points and their object
# libraries therefore cannot be linked into one executable.
add_executable(fft_size_runtime_chain fft_size_runtime_chain.cpp)
target_link_libraries(fft_size_runtime_chain
    PRIVATE composite::composite comps_baseline comps_warnings ${CMAKE_DL_LIBS})
target_compile_features(fft_size_runtime_chain PRIVATE cxx_std_${COMPS_CXX_STANDARD})
add_dependencies(fft_size_runtime_chain framer fft psd)
add_test(NAME fft_size_runtime_chain
    COMMAND fft_size_runtime_chain $<TARGET_FILE:framer> $<TARGET_FILE:fft> $<TARGET_FILE:psd>)
set_tests_properties(fft_size_runtime_chain PROPERTIES LABELS smoke)

# Numeric correctness against a naive reference DFT, across every fft work() path (folded
# fftshift, rotate, unaligned-copy) and through output-pool exhaustion.
add_executable(fft_numeric fft_numeric.cpp)
target_link_libraries(fft_numeric PRIVATE fft_objs)
add_test(NAME fft_numeric COMMAND fft_numeric)
set_tests_properties(fft_numeric PROPERTIES LABELS smoke)

# Numeric correctness for psd against a double-precision reference: normalization contracts
# (name-keyed window, declared sum(w^2), no window), malformed-metadata degradation, pbn
# annotation coherence across runtime flips, and pooled-output exhaustion.
add_executable(psd_numeric psd_numeric.cpp)
target_link_libraries(psd_numeric PRIVATE psd_objs)
add_test(NAME psd_numeric COMMAND psd_numeric)
set_tests_properties(psd_numeric PROPERTIES LABELS smoke)

# Numeric correctness for exp_smooth against a double-precision EWMA reference: recursion +
# one-frame delay + EOS flush, two-sided NaN/Inf sanitization (incl. the -inf baseline
# latch regression), size-change re-baselining, and pass-through mode.
add_executable(exp_smooth_numeric exp_smooth_numeric.cpp)
target_link_libraries(exp_smooth_numeric PRIVATE exp_smooth_objs)
add_test(NAME exp_smooth_numeric COMMAND exp_smooth_numeric)
set_tests_properties(exp_smooth_numeric PROPERTIES LABELS smoke)

install(TARGETS fft_smoke psd_smoke fft_size_runtime_chain fft_numeric psd_numeric exp_smooth_numeric
    RUNTIME DESTINATION bin/tests
    COMPONENT Tests
    OPTIONAL
)
