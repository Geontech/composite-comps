/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * VITA 49 packet-type contract for vita49_parser.
 *
 * can_parse() used to accept extension-data packets while parse() handled only data and context.
 * An extension-data packet therefore fell through to a default-constructed parse_result, whose
 * should_send defaults to TRUE -- so the component forwarded an EMPTY payload with a zero
 * timestamp downstream, silently and uncounted, as if it were valid data. Not a crash: silent
 * corruption of the stream.
 */
#include "parsers/vita49_parser.hpp"

#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace {

// A minimal VITA 49 packet: packet_type occupies the top nibble of word 0, and the packet size
// (in 32-bit words) the bottom 16 bits. Big-endian on the wire.
auto make_packet(uint8_t packet_type, std::size_t words = 4)
    -> composite::immutable_buffer<uint8_t> {
    auto storage = std::make_shared<std::vector<uint8_t>>(words * 4, 0);
    const uint32_t word0 = (static_cast<uint32_t>(packet_type & 0x0F) << 28)
                         | static_cast<uint32_t>(words & 0xFFFF);
    const uint32_t be = std::byteswap(word0);
    std::memcpy(storage->data(), &be, sizeof(be));
    return composite::immutable_buffer<uint8_t>{storage};
}

constexpr uint8_t SIGNAL_DATA            = 0b0000;
constexpr uint8_t SIGNAL_DATA_STREAM_ID  = 0b0001;
constexpr uint8_t EXTENSION_DATA         = 0b0010;
constexpr uint8_t EXTENSION_DATA_STREAM  = 0b0011;
constexpr uint8_t CONTEXT                = 0b0100;
constexpr uint8_t COMMAND                = 0b0110;

auto make_parser() -> parsers::vita49_parser {
    return parsers::vita49_parser{struct_props::signal_overrides{}, {}};
}

} // namespace

TEST_CASE("vita49_parser claims data and context packets", "[pkt_parser][packet_type]") {
    auto parser = make_parser();
    CHECK(parser.can_parse(composite::immutable_buffer<uint8_t>{}) == false);  // empty
    for (auto type : {SIGNAL_DATA, SIGNAL_DATA_STREAM_ID, CONTEXT}) {
        auto pkt = make_packet(type);
        CHECK(parser.can_parse(pkt));
    }
}

TEST_CASE("vita49_parser does NOT claim extension-data packets", "[pkt_parser][packet_type]") {
    // It cannot decode them, so claiming them would win detection for a stream it would then
    // drop -- and would shut out a downstream parser that does implement the extension format.
    auto parser = make_parser();
    for (auto type : {EXTENSION_DATA, EXTENSION_DATA_STREAM}) {
        auto pkt = make_packet(type);
        CHECK_FALSE(parser.can_parse(pkt));
    }
}

TEST_CASE("an unsupported packet on a locked stream is dropped, not forwarded empty",
          "[pkt_parser][packet_type]") {
    // Protocol lock-in keeps calling parse() for every packet once vita49 is active, so an
    // extension-data or command packet can still reach parse(). It must NOT come back with
    // should_send set and an empty payload.
    auto parser = make_parser();
    const composite::metadata current{};

    for (auto type : {EXTENSION_DATA, EXTENSION_DATA_STREAM, COMMAND}) {
        auto pkt = make_packet(type);
        auto result = parser.parse(pkt, current);
        CHECK_FALSE(result.should_send);
        CHECK(result.payload.size() == 0);
    }
}

TEST_CASE("the unsupported-packet warning is one-shot", "[pkt_parser][packet_type]") {
    // A steady stream of unsupported packets must not flood the log.
    auto parser = make_parser();
    const composite::metadata current{};
    auto pkt = make_packet(EXTENSION_DATA);

    auto first = parser.parse(pkt, current);
    CHECK(first.warning.has_value());
    for (int i = 0; i < 5; ++i) {
        auto again = parser.parse(pkt, current);
        CHECK_FALSE(again.warning.has_value());
    }
    // on_activated() re-arms it for the next stream.
    parser.on_activated();
    auto rearmed = parser.parse(pkt, current);
    CHECK(rearmed.warning.has_value());
}
