cmake_minimum_required(VERSION 3.15)
project(file_writer VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)

# Set the C++ version required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compile flags
set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use FetchContent for dependencies
include(FetchContent)
# vrtgen dependency
FetchContent_Declare(vrtgen
    GIT_REPOSITORY https://github.com/geontech/vrtgen.git
    GIT_TAG v0.7.12
)
# Make all deps available
FetchContent_MakeAvailable(vrtgen)

# Library
add_library(file_writer MODULE
    component.cpp
)
# Includes
target_include_directories(file_writer
    PRIVATE
    ${PROJECT_SOURCE_DIR}/../../../include
    ${vrtgen_SOURCE_DIR}/include
)
# Link
target_link_libraries(file_writer
    PRIVATE
    caddie::caddie
)
# Install
install(TARGETS file_writer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
