cmake_minimum_required(VERSION 3.20)
project(abi_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(composite REQUIRED)

add_executable(abi_smoke abi_smoke.cpp)
target_include_directories(abi_smoke PRIVATE ${SPDLOG_INC})
target_link_libraries(abi_smoke PRIVATE composite::composite ${CMAKE_DL_LIBS})
