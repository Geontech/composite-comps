find_package(nlohmann_json 3.11.0 QUIET)
if(NOT TARGET nlohmann_json::nlohmann_json)
    message(STATUS "nlohmann_json not found, falling back to FetchContent")
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
else()
    message(STATUS "nlohmann_json found: using installed version")
endif()
