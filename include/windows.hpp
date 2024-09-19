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

#include "aligned_mem.hpp"

#include <cmath>
#include <cstdlib>
#include <memory>

namespace windows {

constexpr std::size_t ALIGNMENT = 64;

/*
 * Reference: https://www.mathworks.com/help/signal/ref/blackmanharris.html
 */
template <typename T>
auto blackman_harris(const std::size_t length, bool complex=true) {
    auto window = aligned::make_aligned<T>(ALIGNMENT, length * (complex ? 2 : 1));

    constexpr T a0 = 0.35875;
    constexpr T a1 = 0.48829;
    constexpr T a2 = 0.14128;
    constexpr T a3 = 0.01168;

    const T N = length;
    for (auto n=0; n<length; ++n) {
        const auto tn = static_cast<T>(n);
        auto val = a0
            - a1*cos(2. * M_PI * tn / N)
            + a2*cos(4. * M_PI * tn / N)
            - a3*cos(6. * M_PI * tn / N);
        if (complex) {
            window->at(n * 2) = val;
            window->at(n * 2 + 1) = val;
        } else {
            window->at(n) = val;
        }
    }

    return window;
}

/*
 * Reference: https://www.mathworks.com/help/signal/ref/hamming.html
 */
template <typename T>
auto hamming(const std::size_t length, bool complex=true) {
    auto window = aligned::make_aligned<T>(ALIGNMENT, length *  (complex ? 2 : 1));

    constexpr T a0 = 0.54;
    constexpr T a1 = 0.46;
    
    const T N = length - 1;
    for (auto n=0; n<length; ++n) {
        const auto tn = static_cast<T>(n);
        auto val = a0 - a1*cos(2. * M_PI * tn / N);
        if (complex) {
            window->at(n * 2) = val;
            window->at(n * 2 + 1) = val;
        } else {
            window->at(n) = val;
        }
    }

    return window;
}

} // namespace windows
