find_package(vrtgen 0.7.12 QUIET)
if(NOT TARGET vrtgen::vrtgen)
    message(STATUS "vrtgen not found, falling back to FetchContent")
    FetchContent_Declare(vrtgen
        GIT_REPOSITORY https://github.com/geontech/vrtgen.git
        GIT_TAG v0.7.12
    )
    FetchContent_MakeAvailable(vrtgen)
else()
    message(STATUS "Using vrtgen version: ${vrtgen_VERSION}")
endif()

