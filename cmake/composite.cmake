find_package(composite 0.4.0 QUIET)
if(NOT TARGET composite::composite)
    message(STATUS "composite not found, falling back to FetchContent")
    FetchContent_Declare(composite
        GIT_REPOSITORY https://github.com/geontech/composite.git
        GIT_TAG v0.4.0
    )
    FetchContent_MakeAvailable(composite)
else()
    message(STATUS "Using composite version: ${composite_VERSION}")
endif()

