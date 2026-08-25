/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

// Component-level tests for the histogram: drive the real component's process() through
// connected ports and verify binning correctness, the send-threshold emission, the
// end-of-stream flush, and the no-sample-rate discard path. Until now the component had
// only decimator unit coverage — nothing pinned the value/bit binning math end to end.

#include "component.hpp"

#include <catch2/catch_test_macros.hpp>

#include <composite/composite.hpp>
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

// Pack int16 samples (host endian) into the raw byte stream the component consumes.
auto make_i16_frame(const std::vector<int16_t>& samples) -> composite::immutable_buffer<uint8_t> {
    auto v = std::make_shared<std::vector<uint8_t>>(samples.size() * sizeof(int16_t));
    std::memcpy(v->data(), samples.data(), v->size());
    return composite::immutable_buffer<uint8_t>(v);
}

auto make_meta(double sample_rate, uint32_t bit_width, bool is_complex)
    -> composite::metadata_ptr {
    composite::metadata md;
    md.sample_rate = sample_rate;
    md.format.bit_width = bit_width;
    md.format.is_complex = is_complex;
    md.format.endianness = std::endian::native;
    return composite::make_metadata(std::move(md));
}

struct histogram_fixture {
    static auto next_id() -> std::string {
        static std::atomic<int> n{0};
        return "histogram_test_" + std::to_string(n.fetch_add(1));
    }

    std::shared_ptr<histogram> uut{std::make_shared<histogram>(next_id())};
    composite::output_port<composite::immutable_buffer<uint8_t>> src{"src"};
    composite::input_port<composite::mutable_buffer<uint64_t>> capture{"capture"};

    histogram_fixture() {
        src.connect(uut->get_port<composite::input_port<composite::immutable_buffer<uint8_t>>>(
            "data_in"));
        uut->get_port<composite::output_port<composite::mutable_buffer<uint64_t>>>("data_out")
            ->connect(&capture);
    }

    void configure(const composite::properties::json& props) {
        uut->set_properties(props, composite::properties::config_type::INITIALIZE);
        uut->initialize();
    }

    void feed(composite::immutable_buffer<uint8_t> frame, composite::metadata_ptr md) {
        src.send_data(std::move(frame), composite::timestamp{7, 9}, std::move(md));
        REQUIRE(uut->process() == composite::retval::NORMAL);
    }

    auto take_histogram() -> std::optional<composite::mutable_buffer<uint64_t>> {
        auto out = capture.try_get();
        if (!out) {
            return std::nullopt;
        }
        return std::get<0>(std::move(*out));
    }
};

} // namespace

TEST_CASE_METHOD(histogram_fixture, "value histogram bins int16 samples at the right indices",
                 "[histogram][integration]") {
    // rate 8, percent 1.0 -> threshold 8 samples: one 8-sample frame emits immediately.
    configure({{"sample_rate", 8.0}, {"percent_sampled", 1.0}});
    const auto md = make_meta(8.0, 16, /*is_complex=*/false);

    feed(make_i16_frame({0, 0, 5, 5, 5, -3, 32767, -32768}), md);

    auto hist = take_histogram();
    REQUIRE(hist.has_value());
    REQUIRE(hist->size() == 65536);  // 1 << 16 value bins
    // bin = uint16(sample) ^ 0x8000 — i.e. offset-binary: -32768 -> 0, 0 -> 32768.
    CHECK((*hist)[32768] == 2);          // the zeros
    CHECK((*hist)[32768 + 5] == 3);      // the fives
    CHECK((*hist)[32768 - 3] == 1);      // the -3
    CHECK((*hist)[65535] == 1);          // 32767
    CHECK((*hist)[0] == 1);              // -32768
}

TEST_CASE_METHOD(histogram_fixture, "send threshold accumulates across frames",
                 "[histogram][integration]") {
    // threshold = 16 samples: the first 8-sample frame accumulates, the second emits.
    configure({{"sample_rate", 16.0}, {"percent_sampled", 1.0}});
    const auto md = make_meta(16.0, 16, false);

    feed(make_i16_frame(std::vector<int16_t>(8, 1)), md);
    CHECK_FALSE(take_histogram().has_value());

    feed(make_i16_frame(std::vector<int16_t>(8, 1)), md);
    auto hist = take_histogram();
    REQUIRE(hist.has_value());
    CHECK((*hist)[32769] == 16);  // all 16 ones, both frames in one accumulator
}

TEST_CASE_METHOD(histogram_fixture, "complex input bins only the real component",
                 "[histogram][integration]") {
    configure({{"sample_rate", 4.0}, {"percent_sampled", 1.0}});
    const auto md = make_meta(4.0, 16, /*is_complex=*/true);

    // Four I/Q pairs: I = 2, Q = 99. The 99s must not appear anywhere.
    feed(make_i16_frame({2, 99, 2, 99, 2, 99, 2, 99}), md);

    auto hist = take_histogram();
    REQUIRE(hist.has_value());
    CHECK((*hist)[32768 + 2] == 4);
    CHECK((*hist)[32768 + 99] == 0);
}

TEST_CASE_METHOD(histogram_fixture, "end-of-stream flushes the below-threshold remainder",
                 "[histogram][integration]") {
    configure({{"sample_rate", 1000.0}, {"percent_sampled", 1.0}});
    const auto md = make_meta(1000.0, 16, false);

    feed(make_i16_frame(std::vector<int16_t>(4, 7)), md);  // far below the 1000-sample threshold
    CHECK_FALSE(take_histogram().has_value());

    uut->on_end_of_stream();
    auto hist = take_histogram();
    REQUIRE(hist.has_value());
    CHECK((*hist)[32768 + 7] == 4);

    // The flush left a clean accumulator: a fresh frame does not inherit old counts.
    feed(make_i16_frame(std::vector<int16_t>(4, 7)), md);
    uut->on_end_of_stream();
    auto hist2 = take_histogram();
    REQUIRE(hist2.has_value());
    CHECK((*hist2)[32768 + 7] == 4);
}

TEST_CASE_METHOD(histogram_fixture, "a rate-less stream does not inherit the previous stream's rate",
                 "[histogram][integration]") {
    // The rate used to update only when metadata carried a VALID one, so a transition from
    // a 1 MHz stream to a rate-less stream kept binning at 1 MHz — bypassing both the
    // configured fallback and the no-rate discard signal.
    SECTION("no config fallback: the new stream lands in the discard path") {
        configure({{"percent_sampled", 1.0}});  // config sample_rate unset (0)

        feed(make_i16_frame(std::vector<int16_t>(8, 2)), make_meta(8.0, 16, false));
        REQUIRE(take_histogram().has_value());  // old stream emits at its own rate

        // New stream, metadata present but rate-less: nothing may emit, at ANY volume.
        for (int i = 0; i < 4; ++i) {
            feed(make_i16_frame(std::vector<int16_t>(8, 2)), make_meta(0.0, 16, false));
        }
        CHECK_FALSE(take_histogram().has_value());
    }
    SECTION("with a config fallback: the new stream uses it, not the old stream's rate") {
        configure({{"sample_rate", 16.0}, {"percent_sampled", 1.0}});

        feed(make_i16_frame(std::vector<int16_t>(8, 2)), make_meta(8.0, 16, false));
        REQUIRE(take_histogram().has_value());  // metadata rate 8 wins while it is valid

        // Rate-less stream falls back to the CONFIG threshold (16 samples): one 8-sample
        // frame must not emit, the second must.
        feed(make_i16_frame(std::vector<int16_t>(8, 2)), make_meta(0.0, 16, false));
        CHECK_FALSE(take_histogram().has_value());
        feed(make_i16_frame(std::vector<int16_t>(8, 2)), make_meta(0.0, 16, false));
        CHECK(take_histogram().has_value());
    }
}

TEST_CASE_METHOD(histogram_fixture, "input with no known sample rate is consumed, not emitted",
                 "[histogram][integration]") {
    // No config rate and metadata without one: the send threshold cannot be sized, so
    // frames are discarded (counted + one-shot warned) — and process() reports NORMAL,
    // because a consumed packet is work done even when its frame is discarded.
    configure({{"percent_sampled", 1.0}});
    composite::metadata md;   // sample_rate defaults to 0
    md.format.bit_width = 16;
    md.format.is_complex = false;
    md.format.endianness = std::endian::native;

    feed(make_i16_frame(std::vector<int16_t>(8, 3)), composite::make_metadata(std::move(md)));
    CHECK_FALSE(take_histogram().has_value());

    // Once a rate arrives, binning resumes from a clean slate.
    feed(make_i16_frame(std::vector<int16_t>(8, 3)), make_meta(8.0, 16, false));
    auto hist = take_histogram();
    REQUIRE(hist.has_value());
    CHECK((*hist)[32768 + 3] == 8);  // only the second frame's samples
}
