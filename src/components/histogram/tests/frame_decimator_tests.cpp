/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */
#include "frame_decimator.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

namespace {

// Fraction of frames kept over n frames.
auto kept_fraction(double ratio, std::uint64_t frames) -> double {
    histogram_detail::frame_decimator dec{ratio};
    std::uint64_t kept = 0;
    for (std::uint64_t i = 0; i < frames; ++i) {
        if (dec.keep()) { ++kept; }
    }
    return static_cast<double>(kept) / static_cast<double>(frames);
}

} // namespace

TEST_CASE("every ratio is honored, not just reciprocals of integers", "[histogram][decimator]") {
    // The integer-threshold implementation this replaces truncated 1/ratio, so 0.75, 0.6 and 0.9
    // all collapsed to "keep every frame" and 0.4 to "keep every other".
    for (double ratio : {1.0, 0.9, 0.75, 0.6, 0.5, 0.4, 0.3333, 0.25, 0.1, 0.01}) {
        const auto got = kept_fraction(ratio, 100000);
        CHECK(got > ratio - 0.001);
        CHECK(got < ratio + 0.001);
    }
}

TEST_CASE("ratio 1.0 keeps every frame", "[histogram][decimator]") {
    histogram_detail::frame_decimator dec{1.0};
    for (int i = 0; i < 1000; ++i) {
        CHECK(dec.keep());
    }
}

TEST_CASE("0.75 keeps three of every four", "[histogram][decimator]") {
    // Exact pattern, not just the long-run average: skip, keep, keep, keep.
    histogram_detail::frame_decimator dec{0.75};
    CHECK_FALSE(dec.keep());
    CHECK(dec.keep());
    CHECK(dec.keep());
    CHECK(dec.keep());
    CHECK_FALSE(dec.keep());
}

TEST_CASE("the phase does not drift over a long run", "[histogram][decimator]") {
    // The reason for fixed point: repeated float addition of 0.1 accumulates error, and this
    // accumulator is meant to run for the life of a stream.
    const auto got = kept_fraction(0.1, 10'000'000);
    CHECK(got > 0.0999);
    CHECK(got < 0.1001);
}

TEST_CASE("a degenerate ratio never stalls the stream", "[histogram][decimator]") {
    // Non-finite or non-positive ratios are rejected by the property validator, but the decimator
    // must not deadlock the component if one reaches it: sample rarely, never never.
    for (double bad : {0.0, -1.0, std::nan(""), -std::nan("")}) {
        histogram_detail::frame_decimator dec{bad};
        CHECK(dec.increment() >= 1);
    }
    histogram_detail::frame_decimator dec{0.0};
    bool any = false;
    for (std::uint64_t i = 0; i < histogram_detail::frame_decimator::PHASE_ONE + 1; ++i) {
        if (dec.keep()) { any = true; break; }
    }
    CHECK(any);
}

TEST_CASE("reset drops accumulated phase", "[histogram][decimator]") {
    histogram_detail::frame_decimator dec{0.75};
    CHECK_FALSE(dec.keep());   // phase 0.75
    dec.reset();
    CHECK_FALSE(dec.keep());   // back to 0.75, so still a skip
}
