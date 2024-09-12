cmake_minimum_required(VERSION 3.15)
project(sticks VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# caddie dependency
find_package(caddie)

add_subdirectory(src)
