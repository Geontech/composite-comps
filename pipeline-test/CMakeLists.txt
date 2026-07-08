cmake_minimum_required(VERSION 3.20)
project(pipeline_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(composite REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
pkg_check_modules(FFTW3 REQUIRED fftw3)

add_executable(fft_smoke
    fft_smoke.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/components/fft/component.cpp
)
target_include_directories(fft_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/components/fft
    ${CMAKE_CURRENT_SOURCE_DIR}/../include   # windows.hpp
    ${FFTW3F_INCLUDE_DIRS}
    ${FFTW3_INCLUDE_DIRS}
)
target_compile_options(fft_smoke PRIVATE -march=x86-64-v2 -mtune=generic)
target_link_libraries(fft_smoke PRIVATE
    composite::composite
    ${FFTW3F_LIBRARIES} ${FFTW3_LIBRARIES}
    fftw3f_threads fftw3_threads
)

add_executable(psd_smoke
    psd_smoke.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/components/psd/component.cpp
)
target_include_directories(psd_smoke PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/components/psd
    ${CMAKE_CURRENT_SOURCE_DIR}/../include   # windows.hpp
)
target_compile_options(psd_smoke PRIVATE -march=x86-64-v2 -mtune=generic -ffast-math)
target_link_libraries(psd_smoke PRIVATE composite::composite)
