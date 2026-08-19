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

# NOTE: file_writer carries no -march baseline, unlike every other component. Preserved rather
# than silently changed -- it is a byte-copy path with no SIMD or math, so the baseline buys it
# nothing, but the inconsistency is worth a deliberate decision rather than drift.
comps_add_component(file_writer
    SOURCES component.cpp
    NO_ARCH_BASELINE
)
