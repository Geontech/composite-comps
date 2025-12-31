// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#pragma once

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace pfbc {

// Generate a default sinc-windowed lowpass prototype of length M * K.
inline auto generate_default_prototype(std::size_t num_channels,
                                       std::size_t taps_per_phase) -> std::vector<float> {
    const auto total_taps = num_channels * taps_per_phase;
    auto prototype = std::vector<float>(total_taps);

    const auto center = static_cast<double>(total_taps - 1) / 2.0;
    const auto cutoff = 1.0 / static_cast<double>(num_channels);

    for (std::size_t i = 0; i < total_taps; ++i) {
        const auto x = static_cast<double>(i) - center;

        double sinc = 1.0;
        if (std::abs(x) > 1e-10) {
            const auto arg = std::numbers::pi * x * cutoff;
            sinc = std::sin(arg) / arg;
        }

        const auto t = static_cast<double>(i) / static_cast<double>(total_taps - 1);
        const auto window = 0.35875
                          - 0.48829 * std::cos(2.0 * std::numbers::pi * t)
                          + 0.14128 * std::cos(4.0 * std::numbers::pi * t)
                          - 0.01168 * std::cos(6.0 * std::numbers::pi * t);

        prototype[i] = static_cast<float>(sinc * window * cutoff);
    }

    return prototype;
}

// Decompose prototype into polyphase taps: phase_coeffs[phase][tap].
inline auto build_phase_coeffs(const std::vector<float>& prototype,
                               std::size_t num_channels,
                               std::size_t taps_per_phase) -> std::vector<std::vector<float>> {
    const auto total_taps = num_channels * taps_per_phase;
    std::vector<std::vector<float>> phase_coeffs(num_channels, std::vector<float>(taps_per_phase));

    for (std::size_t n = 0; n < total_taps; ++n) {
        const auto phase = n % num_channels;
        const auto tap = n / num_channels;
        phase_coeffs[phase][taps_per_phase - 1 - tap] = prototype[n];
    }

    return phase_coeffs;
}

// Reorganize coefficients from [phase][tap] to [tap][phase*2] (doubled).
inline auto prepare_interleaved_coeffs(
    const std::vector<std::vector<float>>& phase_coeffs,
    std::size_t num_channels,
    std::size_t taps_per_phase
) -> std::vector<float>
{
    std::vector<float> result(taps_per_phase * num_channels * 2);

    for (std::size_t k = 0; k < taps_per_phase; ++k) {
        float* row = result.data() + k * num_channels * 2;
        for (std::size_t p = 0; p < num_channels; ++p) {
            const float h = phase_coeffs[p][k];
            row[p * 2 + 0] = h;
            row[p * 2 + 1] = h;
        }
    }

    return result;
}

} // namespace pfbc
