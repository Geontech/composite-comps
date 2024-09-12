cmake_minimum_required(VERSION 3.15)
project(sticks VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
# Set the C++ version required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Set compile flags
set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# caddie dependency
find_package(caddie)

add_subdirectory(src)
