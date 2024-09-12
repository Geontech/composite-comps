cmake_minimum_required(VERSION 3.15)
project(udp_source VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
# Set the C++ version required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Set compile flags
set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -ggdb -O0")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Library
add_library(udp_source MODULE
    component.cpp
)
# Link
target_link_libraries(udp_source
    PRIVATE
    caddie::caddie
)
# Install
install(TARGETS udp_source
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
