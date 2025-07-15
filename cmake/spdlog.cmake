find_package(spdlog 1.14.1 QUIET)
if(NOT TARGET spdlog::spdlog)
    message(STATUS "spdlog not found, falling back to FetchContent")
    FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.14.1
    )
    FetchContent_MakeAvailable(spdlog)
else()
    message(STATUS "spdlog found: using installed version")
endif()

