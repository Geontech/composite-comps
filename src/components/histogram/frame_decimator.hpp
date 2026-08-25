/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * Whole-frame decimation by an arbitrary fraction.
 *
 * The histogram used to decimate with an integer threshold:
 *
 *     m_skip_threshold = static_cast<uint32_t>(1.0F / percent_sampled);
 *     if (++m_skip_counter >= m_skip_threshold) { keep }
 *
 * which disagrees with its own contract for every ratio above 0.5. percent_sampled is validated
 * as a fraction in (0, 1], but 1/0.75 truncates to 1, so 0.75 keeps EVERY frame -- 100% sampling
 * under a property that says 75%. 0.6 and 0.9 likewise become 100%; 0.4 becomes 50%. Only the
 * reciprocals of integers were ever honored.
 *
 * This is a fixed-point phase accumulator instead: add the ratio to a phase each frame and keep a
 * frame whenever the phase crosses one. It honors any ratio, and the long-run kept fraction is the
 * requested one.
 *
 * Fixed point rather than float on purpose: the phase is a long-running accumulator, and repeated
 * float addition of a value like 0.1 drifts. A 32-bit-fraction phase in a uint64 is exact under
 * addition and subtraction, so a stream that runs for days keeps sampling at the rate asked for.
 */
#pragma once

#include <cmath>
#include <cstdint>

namespace histogram_detail {

class frame_decimator {
public:
    /// One whole frame, in phase units. 2^32 gives ~2.3e-10 ratio resolution.
    static constexpr std::uint64_t PHASE_ONE = std::uint64_t{1} << 32;

    frame_decimator() = default;
    explicit frame_decimator(double ratio) { set_ratio(ratio); }

    /// Set the fraction of frames to keep. Values are clamped into (0, 1]: a non-finite or
    /// non-positive ratio would otherwise stall the stream forever, which is worse than sampling.
    auto set_ratio(double ratio) -> void {
        if (!std::isfinite(ratio) || ratio <= 0.0) {
            m_increment = 1;                       // ~never, but never *stalled*
        } else if (ratio >= 1.0) {
            m_increment = PHASE_ONE;               // keep everything
        } else {
            const auto scaled = std::llround(ratio * static_cast<double>(PHASE_ONE));
            m_increment = scaled <= 0 ? std::uint64_t{1} : static_cast<std::uint64_t>(scaled);
        }
        reset();
    }

    /// Advance one frame. True when this frame should be binned.
    [[nodiscard]] auto keep() -> bool {
        m_phase += m_increment;
        if (m_phase < PHASE_ONE) {
            return false;
        }
        m_phase -= PHASE_ONE;
        return true;
    }

    /// Drop the accumulated phase (a new stream starts at a frame boundary).
    auto reset() -> void { m_phase = 0; }

    [[nodiscard]] auto increment() const -> std::uint64_t { return m_increment; }

private:
    std::uint64_t m_increment{PHASE_ONE};  // keep everything until told otherwise
    std::uint64_t m_phase{0};
};

} // namespace histogram_detail
