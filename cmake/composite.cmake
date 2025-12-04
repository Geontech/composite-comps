find_package(composite 0.5.0 QUIET)
if(NOT TARGET composite::composite)
    message(STATUS "composite not found, falling back to FetchContent")
    FetchContent_Declare(composite
        GIT_REPOSITORY https://github.com/geontech/composite.git
        GIT_TAG develop
    )
    FetchContent_MakeAvailable(composite)
else()
    message(STATUS "Using composite version: ${composite_VERSION}")
endif()
