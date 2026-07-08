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
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>

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
    };
    auto capture() -> std::vector<captured> {
        std::vector<captured> packets;
        while (sink.size() > 0) {
            auto [data, ts, md] = sink.get_data();
            if (data) {
                packets.push_back({{data.begin(), data.end()}, md});
            }
        }
        return packets;
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
