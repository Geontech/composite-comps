#
# Copyright (C) 2026 Geon Technologies, LLC
# SPDX-License-Identifier: LGPL-3.0-or-later
#

# Concurrency regression for psd: property writes racing the worker's process().
#
# It builds and runs without a sanitizer (a plain smoke of the same path), but it only has
# teeth under one: the process always exits 0 and relies on the sanitizer to report. Drive it
# with -DCOMPS_SANITIZER=thread (the tsan preset), whose TSAN_OPTIONS set halt_on_error=1.
#
# Previously (as tsan-test/) this carried its own SAN cache variable and applied -fsanitize to
# one target, which cannot instrument the framework or Catch2 it links against. COMPS_SANITIZER
# applies at directory scope from the top level, so every TU in the build is instrumented
# consistently -- the same approach the framework takes.

add_executable(tsan_psd_race tsan_psd_race.cpp)
target_link_libraries(tsan_psd_race PRIVATE psd_objs)
add_test(NAME psd_race COMMAND tsan_psd_race)
set_tests_properties(psd_race PROPERTIES LABELS race)

if(NOT COMPS_SANITIZER)
    message(STATUS "composite-comps: psd_race built without a sanitizer; it will pass vacuously "
                   "(configure with -DCOMPS_SANITIZER=thread for the real gate)")
endif()
