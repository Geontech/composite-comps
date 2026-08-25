/*
 * Copyright (C) 2025 Geon Technologies, LLC
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
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

// Integration tests for the VITA-49 packet builder, including a LOOPBACK through the
// real vita49_parser: builder and parser are built on the same vrtgen packing classes,
// and these tests pin that what one emits the other reads back — headers, class id,
// timestamps, context fields (bandwidth / rf / sample rate / payload format), payload
// bytes (with the byteswap-to-wire round trip), and the shared-metadata invariants.

#include "component.hpp"
#include "parsers/vita49_parser.hpp"

#include <catch2/catch_test_macros.hpp>

#include <composite/composite.hpp>
#include <composite/metrics/registry.hpp>
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>

#include <algorithm>

#include <atomic>
#include <bit>
#include <complex>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

// Read big-endian words from raw packet bytes (header field spot-checks).
auto read_be32(const uint8_t* p) -> uint32_t {
    uint32_t v{};
    std::memcpy(&v, p, sizeof(v));
    if constexpr (std::endian::native == std::endian::little) {
        v = std::byteswap(v);
    }
    return v;
}

auto make_ci16_payload(std::size_t num_samples) -> std::shared_ptr<std::vector<std::byte>> {
    auto v = std::make_shared<std::vector<std::byte>>(num_samples * 2 * sizeof(int16_t));
    auto* s = reinterpret_cast<int16_t*>(v->data());
    for (std::size_t i = 0; i < num_samples; ++i) {
        s[i * 2] = static_cast<int16_t>(i);
        s[i * 2 + 1] = static_cast<int16_t>(-static_cast<int>(i));
    }
    return v;
}

auto ci16_metadata(double sample_rate = 1e6) -> composite::metadata {
    composite::metadata md;
    md.format.is_complex = true;
    md.format.type = composite::data_type::signed_integer;
    md.format.bit_width = 16;
    md.format.endianness = std::endian::native;
    md.sample_rate = sample_rate;
    md.center_frequency = 915e6;
    md.bandwidth = 20e6;
    return md;
}

} // namespace

// Global scope so the `friend struct PktBuilderTestFixture;` in pkt_builder grants access.
struct PktBuilderTestFixture {
    static auto next_id() -> std::string {
        static std::atomic<int> n{0};
        return "pkt_builder_test_" + std::to_string(n.fetch_add(1));
    }

    std::shared_ptr<pkt_builder> uut{std::make_shared<pkt_builder>(next_id())};
    composite::output_port<composite::immutable_buffer<std::byte>> src{"src"};
    composite::input_port<composite::immutable_buffer<uint8_t>> sink{"sink"};

    PktBuilderTestFixture() {
        src.connect(&uut->m_in_port);
        uut->m_out_port.connect(&sink);
        // Long periodic interval so only first-packet/change-driven context packets appear
        // unless a test configures otherwise.
        configure({{"context_interval_ms", 3600000}});
    }

    void configure(const composite::properties::json& props) {
        uut->set_properties(props, composite::properties::config_type::INITIALIZE);
    }

    void feed(std::shared_ptr<std::vector<std::byte>> payload, composite::metadata_ptr md) {
        src.send_data(composite::immutable_buffer<std::byte>(std::move(payload)),
                      composite::timestamp{123, 456'000}, std::move(md));
        REQUIRE(uut->process() == composite::retval::NORMAL);
    }

    struct captured {
        std::vector<uint8_t> bytes;
        composite::metadata_ptr md;
        composite::timestamp ts;
    };
    auto capture() -> std::vector<captured> {
        std::vector<captured> packets;
        while (sink.size() > 0) {
            auto [data, ts, md] = sink.get_data();
            if (data) {
                packets.push_back({{data.begin(), data.end()}, md, ts});
            }
        }
        return packets;
    }
    auto drops() -> uint64_t {
        return composite::metrics::registry::instance()
            .get_or_create_counter("pkt_builder.packets_dropped", "", "1",
                                   {{"component_id", std::string{uut->id()}}}).value();
    }
    auto dropped_bytes() -> uint64_t {
        return composite::metrics::registry::instance()
            .get_or_create_counter("pkt_builder.bytes_dropped", "", "1",
                                   {{"component_id", std::string{uut->id()}}}).value();
    }
};

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder emits a context packet then data packets",
                 "[pkt_builder][integration]") {
    feed(make_ci16_payload(64), composite::make_metadata(ci16_metadata()));
    auto packets = capture();
    REQUIRE(packets.size() == 2);

    // Context packet: type nibble 0x4; data packet: type nibble 0x1 (with stream id).
    REQUIRE((packets[0].bytes[0] >> 4) == 0x4);
    REQUIRE((packets[1].bytes[0] >> 4) == 0x1);
    // Header-declared packet size matches the actual byte count for both.
    for (const auto& p : packets) {
        auto words = read_be32(p.bytes.data()) & 0xFFFF;
        REQUIRE(words * 4 == p.bytes.size());
    }
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder takes the stream id from metadata annotations",
                 "[pkt_builder][integration]") {
    auto md = ci16_metadata();
    md.annotations["stream_id"] = 42;
    feed(make_ci16_payload(16), composite::make_metadata(std::move(md)));
    auto packets = capture();
    REQUIRE(packets.size() == 2);
    for (const auto& p : packets) {
        REQUIRE(read_be32(p.bytes.data() + 4) == 42); // stream id follows the header word
    }
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder uses the configured default stream id",
                 "[pkt_builder][integration]") {
    configure({{"default_stream_id", 99}});
    feed(make_ci16_payload(16), composite::make_metadata(ci16_metadata()));
    auto packets = capture();
    REQUIRE(packets.size() == 2);
    REQUIRE(read_be32(packets[1].bytes.data() + 4) == 99);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder data sequence numbers increment modulo 16",
                 "[pkt_builder][integration]") {
    auto md = composite::make_metadata(ci16_metadata());
    for (int i = 0; i < 20; ++i) {
        feed(make_ci16_payload(8), md);
    }
    auto packets = capture();
    std::vector<int> data_seqs;
    for (const auto& p : packets) {
        if ((p.bytes[0] >> 4) == 0x1) {
            data_seqs.push_back((p.bytes[1] >> 0) & 0x0F); // packet count: bits 19-16 of word 0
        }
    }
    REQUIRE(data_seqs.size() == 20);
    for (std::size_t i = 0; i < data_seqs.size(); ++i) {
        REQUIRE(data_seqs[i] == static_cast<int>(i % 16));
    }
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder splits large payloads into multiple packets",
                 "[pkt_builder][integration]") {
    configure({{"max_payload_size", 16}}); // samples per packet
    feed(make_ci16_payload(64), composite::make_metadata(ci16_metadata()));
    auto packets = capture();
    std::size_t data_packets = 0;
    for (const auto& p : packets) {
        if ((p.bytes[0] >> 4) == 0x1) {
            ++data_packets;
        }
    }
    REQUIRE(data_packets == 4); // 64 samples / 16 per packet
}

TEST_CASE_METHOD(PktBuilderTestFixture,
                 "pkt_builder resends context on metadata change but not on the steady state",
                 "[pkt_builder][integration]") {
    auto md1 = composite::make_metadata(ci16_metadata(1e6));
    feed(make_ci16_payload(8), md1);
    feed(make_ci16_payload(8), md1); // same shared instance: steady state
    feed(make_ci16_payload(8), md1);
    auto packets = capture();
    std::size_t contexts = 0;
    for (const auto& p : packets) {
        if ((p.bytes[0] >> 4) == 0x4) {
            ++contexts;
        }
    }
    REQUIRE(contexts == 1); // only the first packet triggered context

    // A genuinely changed instance triggers a fresh context packet.
    feed(make_ci16_payload(8), composite::make_metadata(ci16_metadata(2e6)));
    packets = capture();
    contexts = 0;
    for (const auto& p : packets) {
        if ((p.bytes[0] >> 4) == 0x4) {
            ++contexts;
        }
    }
    REQUIRE(contexts == 1);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder shares one output metadata instance per stream",
                 "[pkt_builder][integration]") {
    auto md = composite::make_metadata(ci16_metadata());
    feed(make_ci16_payload(8), md);
    feed(make_ci16_payload(8), md);
    auto packets = capture();
    REQUIRE(packets.size() >= 3);
    for (const auto& p : packets) {
        REQUIRE(p.md != nullptr);
        REQUIRE(p.md.get() == packets[0].md.get()); // pointer identity across every emitted packet
    }
}

namespace {
auto read_be64(const uint8_t* p) -> uint64_t {
    uint64_t v{};
    std::memcpy(&v, p, sizeof(v));
    if constexpr (std::endian::native == std::endian::little) {
        v = std::byteswap(v);
    }
    return v;
}
} // namespace

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder stamps context packets with the stream time",
                 "[pkt_builder][integration]") {
    feed(make_ci16_payload(8), composite::make_metadata(ci16_metadata()));
    auto packets = capture();
    REQUIRE(packets.size() == 2);
    REQUIRE((packets[0].bytes[0] >> 4) == 0x4);
    // Context layout: header(4) | stream id(4) | class id(8) | integer ts(4) | fractional ts(8).
    // Zeros here used to declare TSI::UTC and then stamp the 1970 epoch.
    REQUIRE(read_be32(packets[0].bytes.data() + 16) == 123);
    REQUIRE(read_be64(packets[0].bytes.data() + 20) == 456'000);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder advances per-chunk timestamps by emitted samples",
                 "[pkt_builder][integration]") {
    configure({{"max_payload_size", 16}});
    feed(make_ci16_payload(64), composite::make_metadata(ci16_metadata(1e6)));  // 1 MHz
    auto packets = capture();

    parsers::vita49_parser parser{struct_props::signal_overrides{}};
    composite::metadata current{};
    std::vector<composite::timestamp> stamps;
    for (const auto& p : packets) {
        auto buf = composite::immutable_buffer<uint8_t>(std::make_shared<std::vector<uint8_t>>(p.bytes));
        auto result = parser.parse(buf, current);
        if (result.metadata_changed) { current = std::move(result.metadata); }
        if (result.should_send) { stamps.push_back(result.timestamp); }
    }
    REQUIRE(stamps.size() == 4);  // 64 samples / 16 per packet
    // Each chunk's timestamp is the time of ITS first sample: base + k * 16 samples @ 1 MHz
    // (16 us = 16e6 ps). Identical stamps on every chunk was the old, wrong behavior.
    for (std::size_t k = 0; k < stamps.size(); ++k) {
        REQUIRE(stamps[k].seconds == 123);
        REQUIRE(stamps[k].picoseconds == 456'000 + k * 16'000'000);
    }
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder declares f64 payloads double-precision and round-trips them",
                 "[pkt_builder][integration][loopback]") {
    const std::size_t num_samples = 16;
    auto payload = std::make_shared<std::vector<std::byte>>(num_samples * 2 * sizeof(double));
    auto* d = reinterpret_cast<double*>(payload->data());
    for (std::size_t i = 0; i < num_samples * 2; ++i) { d[i] = static_cast<double>(i) * 0.5 - 3.25; }
    const std::vector<std::byte> original(*payload);

    composite::metadata md;
    md.format.is_complex = true;
    md.format.type = composite::data_type::floating_point;
    md.format.bit_width = 64;
    md.format.endianness = std::endian::native;
    md.sample_rate = 1e6;
    feed(std::move(payload), composite::make_metadata(std::move(md)));
    auto packets = capture();
    REQUIRE(packets.size() == 2);

    parsers::vita49_parser parser{struct_props::signal_overrides{}};
    composite::metadata current{};
    std::vector<uint8_t> received;
    for (const auto& p : packets) {
        auto buf = composite::immutable_buffer<uint8_t>(std::make_shared<std::vector<uint8_t>>(p.bytes));
        auto result = parser.parse(buf, current);
        if (result.metadata_changed) { current = std::move(result.metadata); }
        if (result.should_send) { received.insert(received.end(), result.payload.begin(), result.payload.end()); }
    }
    // The context must declare floating point at 64 bits (IEEE754_DOUBLE_PRECISION — the old
    // single-precision label parsed back as bit_width 64 but format code 0x0e).
    REQUIRE(current.format.type == composite::data_type::floating_point);
    REQUIRE(current.format.bit_width == 64);
    // Payload round-trip through the fused 64-bit copy-swap.
    REQUIRE(received.size() == original.size());
    for (std::size_t i = 0; i + 8 <= received.size(); i += 8) {
        if constexpr (std::endian::native == std::endian::little) {
            std::reverse(received.begin() + i, received.begin() + i + 8);
        }
    }
    REQUIRE(std::memcmp(received.data(), original.data(), original.size()) == 0);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder rejects an out-of-range stream id annotation",
                 "[pkt_builder][integration]") {
    auto md = ci16_metadata();
    md.annotations["stream_id"] = static_cast<std::int64_t>(0x1'0000'0002LL);  // > uint32 max
    feed(make_ci16_payload(8), composite::make_metadata(std::move(md)));
    auto packets = capture();
    REQUIRE(packets.size() == 2);
    // Silent truncation would alias this onto stream 2; the default (0) must be used instead.
    REQUIRE(read_be32(packets[1].bytes.data() + 4) == 0);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder drops payloads with an unsupported bit width",
                 "[pkt_builder][integration]") {
    auto md = ci16_metadata();
    md.format.bit_width = 24;  // no big-endian serialization path for 24-bit elements
    feed(make_ci16_payload(8), composite::make_metadata(std::move(md)));
    auto packets = capture();
    std::size_t data_packets = 0;
    for (const auto& p : packets) {
        if ((p.bytes[0] >> 4) == 0x1) { ++data_packets; }
    }
    REQUIRE(data_packets == 0);  // context may still describe the stream; data must not go out mislabeled
    auto& registry = composite::metrics::registry::instance();
    REQUIRE(registry.get_or_create_counter("pkt_builder.packets_dropped", "", "1",
                                           {{"component_id", std::string{uut->id()}}}).value() == 1);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder output round-trips through vita49_parser",
                 "[pkt_builder][integration][loopback]") {
    const std::size_t num_samples = 32;
    auto payload = make_ci16_payload(num_samples);
    const std::vector<std::byte> original(*payload); // keep a copy; feed consumes it
    feed(std::move(payload), composite::make_metadata(ci16_metadata(5e6)));
    auto packets = capture();
    REQUIRE(packets.size() == 2);

    // Parse everything the builder emitted with the REAL vita49 parser.
    parsers::vita49_parser parser{struct_props::signal_overrides{}};
    composite::metadata current{};
    std::vector<uint8_t> received_payload;
    for (const auto& p : packets) {
        auto buf = composite::immutable_buffer<uint8_t>(
            std::make_shared<std::vector<uint8_t>>(p.bytes));
        auto result = parser.parse(buf, current);
        if (result.metadata_changed) {
            current = std::move(result.metadata);
        }
        if (result.should_send) {
            received_payload.insert(received_payload.end(), result.payload.begin(),
                                    result.payload.end());
            // The data packet timestamp must round-trip too.
            REQUIRE(result.timestamp.seconds == 123);
            REQUIRE(result.timestamp.picoseconds == 456'000);
        }
    }

    // Context metadata round-trip: format + signal parameters as sent.
    REQUIRE(current.format.is_complex);
    REQUIRE(current.format.type == composite::data_type::signed_integer);
    REQUIRE(current.format.bit_width == 16);
    REQUIRE(current.format.endianness == std::endian::big);
    REQUIRE(current.sample_rate == 5e6);
    REQUIRE(current.center_frequency == 915e6);
    REQUIRE(current.bandwidth == 20e6);

    // Payload round-trip: wire bytes are big-endian; swap back and compare to the input.
    REQUIRE(received_payload.size() == original.size());
    for (std::size_t i = 0; i + 1 < received_payload.size(); i += 2) {
        if constexpr (std::endian::native == std::endian::little) {
            std::swap(received_payload[i], received_payload[i + 1]);
        }
    }
    REQUIRE(std::memcmp(received_payload.data(), original.data(), original.size()) == 0);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder pauses on a full downstream and resumes losslessly",
                 "[pkt_builder][integration][backpressure]") {
    configure({{"max_payload_size", 8}});
    sink.depth(2);  // tiny downstream: 9 packets (1 context + 8 data) cannot fit at once

    src.send_data(composite::immutable_buffer<std::byte>(make_ci16_payload(64)),
                  composite::timestamp{123, 456'000}, composite::make_metadata(ci16_metadata()));

    std::vector<captured> all;
    bool saw_await = false;
    for (int spins = 0; spins < 32 && all.size() < 9; ++spins) {
        const auto rv = uut->process();
        REQUIRE((rv == composite::retval::NORMAL || rv == composite::retval::AWAIT_OUTPUT ||
                 rv == composite::retval::NOOP));
        saw_await |= (rv == composite::retval::AWAIT_OUTPUT);
        for (auto& p : capture()) { all.push_back(std::move(p)); }
    }
    REQUIRE(saw_await);           // the loop genuinely hit the full downstream
    REQUIRE(all.size() == 9);     // ...and NOTHING was dropped
    std::vector<int> seqs;
    for (const auto& p : all) {
        if ((p.bytes[0] >> 4) == 0x1) { seqs.push_back(p.bytes[1] & 0x0F); }
    }
    REQUIRE(seqs.size() == 8);
    for (std::size_t i = 0; i < seqs.size(); ++i) { REQUIRE(seqs[i] == static_cast<int>(i)); }
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder sanitizes non-finite RF metadata before fixed-point packing",
                 "[pkt_builder][integration]") {
    auto md = ci16_metadata();
    md.center_frequency = std::numeric_limits<double>::quiet_NaN();
    md.bandwidth = std::numeric_limits<double>::infinity();
    md.sample_rate = 1e300;  // far beyond the 44.20 fixed-point range
    feed(make_ci16_payload(8), composite::make_metadata(std::move(md)));
    auto packets = capture();
    REQUIRE(packets.size() == 2);

    parsers::vita49_parser parser{struct_props::signal_overrides{}};
    composite::metadata current{};
    for (const auto& p : packets) {
        auto buf = composite::immutable_buffer<uint8_t>(std::make_shared<std::vector<uint8_t>>(p.bytes));
        auto result = parser.parse(buf, current);
        if (result.metadata_changed) { current = std::move(result.metadata); }
    }
    REQUIRE(current.center_frequency == 0.0);
    REQUIRE(current.bandwidth == 0.0);
    REQUIRE(current.sample_rate == 0.0);

    // Semantically invalid values are rejected too: negative bandwidth, and a rate too small
    // to represent at 20 fractional bits (the wire would say 0 while metadata said nonzero).
    auto md2 = ci16_metadata();
    md2.bandwidth = -5e6;
    md2.sample_rate = 1e-8;
    feed(make_ci16_payload(8), composite::make_metadata(std::move(md2)));
    composite::metadata current2{};
    for (const auto& p : capture()) {
        auto buf = composite::immutable_buffer<uint8_t>(std::make_shared<std::vector<uint8_t>>(p.bytes));
        auto result = parser.parse(buf, current2);
        if (result.metadata_changed) { current2 = std::move(result.metadata); }
    }
    REQUIRE(current2.bandwidth == 0.0);
    REQUIRE(current2.sample_rate == 0.0);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder carries sub-word remainders instead of padding or dropping",
                 "[pkt_builder][integration]") {
    // Real i8: 37 bytes per buffer is not word-alignable. The remainder must be CARRIED into
    // the next buffer's first packet — never zero-padded into fabricated samples, never
    // dropped at the buffer boundary. Distinct fill patterns pin ordering across the carry.
    auto md = composite::make_metadata([] {
        composite::metadata m;
        m.format.is_complex = false;
        m.format.type = composite::data_type::signed_integer;
        m.format.bit_width = 8;
        m.format.endianness = std::endian::native;
        m.sample_rate = 1e6;
        return m;
    }());
    feed(std::make_shared<std::vector<std::byte>>(37, std::byte{0x5A}), md);
    feed(std::make_shared<std::vector<std::byte>>(37, std::byte{0xA5}), md);
    auto packets = capture();

    parsers::vita49_parser parser{struct_props::signal_overrides{}};
    composite::metadata current{};
    std::vector<uint8_t> received;
    for (const auto& p : packets) {
        auto buf = composite::immutable_buffer<uint8_t>(std::make_shared<std::vector<uint8_t>>(p.bytes));
        auto result = parser.parse(buf, current);
        if (result.metadata_changed) { current = std::move(result.metadata); }
        if (result.should_send) {
            received.insert(received.end(), result.payload.begin(), result.payload.end());
        }
    }
    // 74 bytes in, 72 emitted so far (word-granular), 2 CARRIED (not dropped) for the next buffer.
    REQUIRE(received.size() == 72);
    for (std::size_t i = 0; i < 37; ++i) { REQUIRE(received[i] == 0x5A); }   // incl. the carried byte
    for (std::size_t i = 37; i < 72; ++i) { REQUIRE(received[i] == 0xA5); }
    REQUIRE(dropped_bytes() == 0);

    // At end-of-stream the < 4-byte residue cannot be emitted without fabricating pad
    // samples: counted as dropped, so the loss is visible.
    uut->on_end_of_stream();
    REQUIRE(dropped_bytes() == 2);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder drops a trailing partial sample instead of shipping it unswapped",
                 "[pkt_builder][integration]") {
    // Real i16: 5 bytes = 2 whole samples + 1 partial byte.
    auto payload = std::make_shared<std::vector<std::byte>>(5, std::byte{0x11});
    composite::metadata md;
    md.format.is_complex = false;
    md.format.type = composite::data_type::signed_integer;
    md.format.bit_width = 16;
    md.format.endianness = std::endian::native;
    feed(std::move(payload), composite::make_metadata(std::move(md)));
    REQUIRE(dropped_bytes() == 1);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder declares f16 half-precision and rejects f8",
                 "[pkt_builder][integration]") {
    // f16 -> IEEE754_HALF_PRECISION on the wire.
    auto md16 = ci16_metadata();
    md16.format.type = composite::data_type::floating_point;
    md16.format.bit_width = 16;
    feed(make_ci16_payload(8), composite::make_metadata(std::move(md16)));
    auto packets = capture();
    parsers::vita49_parser parser{struct_props::signal_overrides{}};
    composite::metadata current{};
    for (const auto& p : packets) {
        auto buf = composite::immutable_buffer<uint8_t>(std::make_shared<std::vector<uint8_t>>(p.bytes));
        auto result = parser.parse(buf, current);
        if (result.metadata_changed) { current = std::move(result.metadata); }
    }
    REQUIRE(current.format.type == composite::data_type::floating_point);
    REQUIRE(current.format.bit_width == 16);

    // f8 has no IEEE wire representation: dropped + counted, and no context for it either.
    auto md8 = ci16_metadata();
    md8.format.type = composite::data_type::floating_point;
    md8.format.bit_width = 8;
    feed(make_ci16_payload(8), composite::make_metadata(std::move(md8)));
    auto rejected = capture();
    REQUIRE(rejected.empty());
    REQUIRE(drops() == 1);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder evicts the least-recent stream at the cap instead of locking out",
                 "[pkt_builder][integration]") {
    // 65 distinct ids: the 65th evicts the least-recently-seen (id 1) — a hard reject would
    // let 64 disposable ids permanently deny state to every later legitimate stream.
    for (int i = 0; i < 65; ++i) {
        auto md = ci16_metadata();
        md.annotations["stream_id"] = i + 1;
        feed(make_ci16_payload(4), composite::make_metadata(std::move(md)));
    }
    auto packets = capture();
    std::size_t stream65_data = 0;
    for (const auto& p : packets) {
        if ((p.bytes[0] >> 4) == 0x1 && read_be32(p.bytes.data() + 4) == 65) { ++stream65_data; }
    }
    REQUIRE(stream65_data == 1);  // the new stream is served, nothing dropped
    REQUIRE(drops() == 0);

    // The evicted stream re-enters cleanly: fresh state, context resent as a first packet.
    auto md1 = ci16_metadata();
    md1.annotations["stream_id"] = 1;
    feed(make_ci16_payload(4), composite::make_metadata(std::move(md1)));
    auto reentry = capture();
    REQUIRE(reentry.size() == 2);
    REQUIRE((reentry[0].bytes[0] >> 4) == 0x4);  // context first
    REQUIRE(read_be32(reentry[0].bytes.data() + 4) == 1);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder re-resolves stream ids after a property change",
                 "[pkt_builder][integration]") {
    auto md = ci16_metadata();
    md.annotations["stream_id"] = 42;
    auto shared = composite::make_metadata(std::move(md));
    feed(make_ci16_payload(4), shared);
    REQUIRE(read_be32(capture().at(1).bytes.data() + 4) == 42);

    // The SAME metadata instance must re-resolve under the new key (the instance-keyed
    // cache is invalidated by the property change).
    configure({{"stream_id_key", "other_key"}, {"default_stream_id", 7}});
    feed(make_ci16_payload(4), shared);
    auto packets = capture();
    REQUIRE(read_be32(packets.back().bytes.data() + 4) == 7);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder builds bare packets from defaults, not stale stream state",
                 "[pkt_builder][integration]") {
    feed(make_ci16_payload(8), composite::make_metadata(ci16_metadata()));
    REQUIRE(capture().size() == 2);

    // A bare packet (nullptr metadata) must NOT inherit the stream's previous format — with
    // no configured default format it is unserializable: dropped + counted.
    feed(make_ci16_payload(8), nullptr);
    REQUIRE(capture().empty());
    REQUIRE(drops() == 1);
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder advances port timestamps even without wire timestamps",
                 "[pkt_builder][integration]") {
    configure({{"max_payload_size", 16}, {"include_timestamp", false}});
    feed(make_ci16_payload(64), composite::make_metadata(ci16_metadata(1e6)));
    auto packets = capture();
    std::vector<composite::timestamp> stamps;
    for (const auto& p : packets) {
        if ((p.bytes[0] >> 4) == 0x1) { stamps.push_back(p.ts); }
    }
    REQUIRE(stamps.size() == 4);
    for (std::size_t k = 0; k < stamps.size(); ++k) {
        REQUIRE(stamps[k].picoseconds == 456'000 + k * 16'000'000);
    }
}

TEST_CASE_METHOD(PktBuilderTestFixture, "pkt_builder bounds the packets emitted per process() call",
                 "[pkt_builder][integration]") {
    configure({{"max_payload_size", 8}});
    // 520 samples / 8 per packet = 65 data packets: one more than the per-call budget (64).
    src.send_data(composite::immutable_buffer<std::byte>(make_ci16_payload(520)),
                  composite::timestamp{123, 456'000}, composite::make_metadata(ci16_metadata()));

    REQUIRE(uut->process() == composite::retval::NORMAL);
    const auto first = capture();
    REQUIRE(first.size() == 65);  // 1 context + 64 data: budget reached, pending retained

    REQUIRE(uut->process() == composite::retval::NORMAL);  // resumes WITHOUT new input
    const auto second = capture();
    REQUIRE(second.size() == 1);  // the 65th data packet
    REQUIRE((second[0].bytes[0] >> 4) == 0x1);
    REQUIRE((second[0].bytes[1] & 0x0F) == 64 % 16);  // sequence continues across the budget pause
}
