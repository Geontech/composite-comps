find_package(composite 0.5.0 QUIET)
if(NOT TARGET composite::composite)
    message(STATUS "composite not found, falling back to FetchContent")
    # Pin to the release tag carrying the inverted-core API these components are
    # written against (typed_property/COMPOSITE_STRUCT, 3-arg send_data, typed
    # annotations). NOTE: must be an immutable ref that actually ships that API —
    # the old '57-add-metrics-and-enable-otel-exporting' branch PREDATES the
    # redesign and will fail to compile the fleet. Bump in lockstep with framework
    # releases; a missing tag fails loudly here rather than silently building an
    # incompatible framework.
    FetchContent_Declare(composite
        GIT_REPOSITORY https://github.com/geontech/composite.git
        GIT_TAG v0.5.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(composite)
else()
    message(STATUS "Using composite version: ${composite_VERSION}")
endif()
