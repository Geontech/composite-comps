/*
 * Copyright (C) 2024 Geon Technologies, LLC
 *
 * This file is part of composite-comps.
 *
 * composite-comps is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * composite-comps is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 */

#pragma once

#include <immintrin.h>

auto apply_window(const float* data, const float* window, float* dst) -> void {
    // Load payload data
    auto payload_m512 = _mm512_load_ps(data);
    // Load window floats
    auto window_m512 = _mm512_load_ps(window);
    // Multiply payload by window
    payload_m512 = _mm512_mul_ps(payload_m512, window_m512);
    // Stored result into dst
    _mm512_store_ps(dst, payload_m512);
}

auto apply_window(const double* data, const double* window, double* dst) -> void {
    // Load payload data
    auto payload_m512d = _mm512_load_pd(data);
    // Load window doubles
    auto window_m512d = _mm512_load_pd(window);
    // Multiply payload by window
    payload_m512d = _mm512_mul_pd(payload_m512d, window_m512d);
    // Stored result into dst
    _mm512_store_pd(dst, payload_m512d);
}