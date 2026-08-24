#
# Copyright (C) 2026 Geon Technologies, LLC
# SPDX-License-Identifier: LGPL-3.0-or-later
#

# The fleet's release manifest.
#
# With the fleet versioned independently of the framework, the fleet version cannot tell you
# which modules an image contains or which framework it runs on. This file is where that is
# recorded, and it is GENERATED from the module list the build actually produced -- the same
# global property test/abi drives its per-module tests from -- so it cannot claim a component
# that failed to build, and it cannot drift from what shipped.
#
# Must be included AFTER add_subdirectory(src).
#
# On `project` vs `composite_comps`. In THIS project's own manifest the two carry the same
# version and the pair looks redundant. It is not: `project` is whoever ran the build, and
# `composite_comps` is the composite-comps release the build was made against. They diverge in
# an OVERLAY build, where `project` is the overlay (program-comps 1.0.0, say) while
# `composite_comps` still reports 0.1.0 -- which is what lets a consumer tell that an overlay's
# replaced module was built against the same fleet as the modules it ships beside. The inner
# `name` was dropped when this was renamed from `fleet`: the key already states it.

# Deferred into a function on purpose.
#
# When an OVERLAY project consumes this one, this file is evaluated while the FOSS root
# CMakeLists runs -- which is BEFORE the overlay has added its own components. Generating the
# manifest there produced an image whose manifest listed 11 modules while 12 were installed:
# the overlay's component shipped unrecorded. For a fleet whose whole provenance story is "the
# manifest is the source of truth", that is the one thing it must not do.
#
# So: the FOSS root calls this only when it is the top-level project. An overlay calls
# comps_write_manifest() itself, at the end of its own top-level CMakeLists, after all of its
# components are declared.
# comps_write_manifest([FILENAME <name>])
#
# FILENAME exists so a CONSUMER does not overwrite the fleet's manifest with a partial one. An
# overlay that rebuilds two modules and takes the other nine from the published image must not
# install a manifest.json listing only its two -- that is worse than no manifest, because it reads
# as authoritative. Overlays pass their own filename; both files then sit side by side in the image.
function(comps_write_manifest)
    cmake_parse_arguments(PARSE_ARGV 0 ARG "" "FILENAME" "")
    set(_manifest_name "manifest.json")
    if(ARG_FILENAME)
        set(_manifest_name "${ARG_FILENAME}")
    endif()
    # An EMPTY module list is legitimate: an overlay project configures with COMPS_BUILD_ALL=OFF to
    # get the flag policy and comps_add_component() while compiling no FOSS component. What is not
    # legitimate is being called before the components are added at all, which would silently write
    # a manifest missing everything -- so key the error on the marker, not on the list being empty.
    get_property(_processed GLOBAL PROPERTY COMPS_COMPONENTS_PROCESSED)
    if(NOT _processed)
        message(FATAL_ERROR
            "comps_write_manifest(): called before the components were added. Call it after "
            "add_subdirectory(src) (and, in an overlay project, after your own components).")
    endif()
    get_property(_comps_modules GLOBAL PROPERTY COMPS_ENABLED_MODULES)
    if(_comps_modules)
        list(SORT _comps_modules)
    endif()
    set(COMPS_MODULE_JSON "")
    foreach(_m IN LISTS _comps_modules)
        string(APPEND COMPS_MODULE_JSON "\n    \"${_m}\",")
    endforeach()
    string(REGEX REPLACE ",$" "\n  " COMPS_MODULE_JSON "${COMPS_MODULE_JSON}")

    # Booleans have to reach the template as JSON literals, not CMake's ON/OFF.
    foreach(_pair "COMPS_FAST_MATH" "COMPS_HAS_DPDK")
        if(${_pair})
            set(${_pair}_JSON "true")
        else()
            set(${_pair}_JSON "false")
        endif()
    endforeach()

    set(COMPS_GIT_SHA "unknown")
    find_package(Git QUIET)
    if(GIT_FOUND)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE _sha OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET RESULT_VARIABLE _rc)
        if(_rc EQUAL 0)
            set(COMPS_GIT_SHA "${_sha}")
        endif()
    endif()

    configure_file("${COMPS_SOURCE_DIR}/cmake/manifest.json.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/${_manifest_name}" @ONLY)
    if(COMPS_INSTALL)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${_manifest_name}"
                DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/composite-comps"
                COMPONENT Runtime)
    endif()

endfunction()

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    comps_write_manifest()
endif()
