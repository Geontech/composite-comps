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

# NOTE: never add FAST_MATH here. The kernels' NaN/Inf sanitization (work.hpp) relies on
# std::isfinite and IEEE comparisons in the scalar paths, which -ffast-math constant-folds
# away — a -inf PSD bin would then poison the running average.
comps_add_component(exp_smooth
    SOURCES component.cpp
    ABI_TYPE f32
)
