#
# Copyright (C) 2024 Geon Technologies, LLC
#
# This file is part of composite-comps.
#
# composite-comps is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# composite-comps is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
# for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.
#

# FFTW via pkg-config. Kept in variable form (not IMPORTED_TARGET) so the link line
# stays as it was; converting to IMPORTED_TARGET / PkgConfig:: targets is a separate change.
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
pkg_check_modules(FFTW3 REQUIRED fftw3)

comps_add_component(fft
    SOURCES  component.cpp
    SYSTEM_INCLUDES ${FFTW3F_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIRS}
    LINK     ${FFTW3F_LIBRARIES} ${FFTW3_LIBRARIES} fftw3f_threads fftw3_threads
    FAST_MATH
    ABI_TYPE cf32
)
