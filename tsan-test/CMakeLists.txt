cmake_minimum_required(VERSION 3.20)
project(tsan_psd_race LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(composite REQUIRED)

add_executable(tsan_psd_race
    tsan_psd_race.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/components/psd/component.cpp
)
target_include_directories(tsan_psd_race PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/components/psd
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${SPDLOG_INC}
)
# Match psd's component build flags + a configurable sanitizer (SAN="thread"|"address"|"").
set(SAN "thread" CACHE STRING "sanitizer")
target_compile_options(tsan_psd_race PRIVATE
    -fno-omit-frame-pointer -g -O1
    -march=x86-64-v2 -mtune=generic -ffast-math
    $<$<BOOL:${SAN}>:-fsanitize=${SAN}>
)
target_link_options(tsan_psd_race PRIVATE $<$<BOOL:${SAN}>:-fsanitize=${SAN}>)
target_link_libraries(tsan_psd_race PRIVATE composite::composite)
