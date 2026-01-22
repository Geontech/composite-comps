# SPDX-License-Identifier: LGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

# rate_monitor - Test harness to measure observed sample rate

add_library(rate_monitor SHARED
    component.cpp
)
target_link_libraries(rate_monitor
    PRIVATE
        composite::composite
        spdlog::spdlog_header_only
)
target_include_directories(rate_monitor
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(rate_monitor PRIVATE cxx_std_23)
target_compile_options(rate_monitor PRIVATE -march=x86-64-v2)

set_target_properties(rate_monitor PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/components
)

install(TARGETS rate_monitor
    LIBRARY DESTINATION lib/composite/components
)
