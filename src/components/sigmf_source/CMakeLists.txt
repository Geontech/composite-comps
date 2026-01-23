# SPDX-License-Identifier: LGPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

# sigmf_source - SigMF file source component with rate-controlled playback

include(GNUInstallDirs)

add_library(sigmf_source SHARED
    component.cpp
)
target_link_libraries(sigmf_source
    PRIVATE
        composite::composite
        spdlog::spdlog_header_only
        nlohmann_json::nlohmann_json
)
target_include_directories(sigmf_source
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(sigmf_source PRIVATE cxx_std_23)
target_compile_options(sigmf_source PRIVATE -march=x86-64-v2)

set_target_properties(sigmf_source PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/components
)

install(TARGETS sigmf_source
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)