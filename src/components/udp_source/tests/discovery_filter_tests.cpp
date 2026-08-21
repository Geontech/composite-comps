/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "socket/discovery_filter.hpp"

#include <catch2/catch_test_macros.hpp>

using udp::discovery_filter;

namespace {

constexpr auto SDDS = discovery_filter::SDDS_PACKET_SIZE;
constexpr auto MIN = discovery_filter::MIN_DATA_PACKET_SIZE;

// An SDDS byte 0 with standard_format (0x80) clear, data_mode in the low three bits, and the
// remaining flag bits set so the high nibble collides with a V49 packet-type value.
constexpr auto sdds_byte0(uint8_t high_nibble, uint8_t data_mode) -> uint8_t {
    return static_cast<uint8_t>((high_nibble & 0xF0) | (data_mode & 0x07));
}

} // namespace

TEST_CASE("undersized datagrams are never sizing evidence") {
    // The boundary is exclusive: the original code used `recvd > MIN_DATA_PACKET_SIZE`.
    CHECK_FALSE(discovery_filter::is_sizing_candidate(MIN, 0x00));
    CHECK(discovery_filter::is_sizing_candidate(MIN + 1, 0x00));
    CHECK_FALSE(discovery_filter::is_sizing_candidate(0, 0x00));
    CHECK_FALSE(discovery_filter::is_sizing_candidate(1, 0x00));
}

TEST_CASE("a recvfrom error is rejected rather than promoted to a huge size") {
    // Guards the signedness of the size comparison: as std::size_t, -1 becomes SIZE_MAX and
    // would sail past the minimum-size check.
    CHECK_FALSE(discovery_filter::is_sizing_candidate(-1, 0x00));
}

TEST_CASE("V49 context packets are skipped so the pool is sized from data") {
    CHECK_FALSE(discovery_filter::is_sizing_candidate(4096, discovery_filter::V49_CONTEXT_PACKET));
    CHECK_FALSE(discovery_filter::is_sizing_candidate(4096, discovery_filter::V49_EXT_CONTEXT_PACKET));
    // The test is on the high nibble only; the low nibble carries other header bits.
    CHECK_FALSE(discovery_filter::is_sizing_candidate(4096, 0x4F));
    CHECK_FALSE(discovery_filter::is_sizing_candidate(4096, 0x5A));
}

TEST_CASE("V49 data packets size the pool") {
    CHECK(discovery_filter::is_sizing_candidate(4096, 0x10));
    CHECK(discovery_filter::is_sizing_candidate(4096, 0x60));
    CHECK(discovery_filter::is_sizing_candidate(1024, 0x00));
}

// REGRESSION: discovery applied the V49 packet-type test to every datagram, so an SDDS packet
// whose byte-0 flags happened to form a 0x40/0x50 high nibble was discarded. On a pure-SDDS
// stream every packet was discarded, discovery exhausted its window and start_recv threw --
// while sdds_parser::can_parse would have accepted those same packets, since it validates
// data_mode/bps and never requires standard_format.
TEST_CASE("SDDS packets are exempt from the V49 packet-type test") {
    for (uint8_t nibble : {0x40, 0x50}) {
        for (uint8_t dm : {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07}) {
            CHECK(discovery_filter::is_sizing_candidate(SDDS, sdds_byte0(nibble, dm)));
        }
    }
}

TEST_CASE("the SDDS exemption is exact, not a range") {
    // Only the fixed SDDS size is exempt: one byte either side is classified as V49 again, so
    // the exemption cannot swallow arbitrary context packets.
    CHECK_FALSE(discovery_filter::is_sizing_candidate(SDDS - 1, 0x40));
    CHECK_FALSE(discovery_filter::is_sizing_candidate(SDDS + 1, 0x40));
    CHECK(discovery_filter::is_sizing_candidate(SDDS, 0x40));
}

TEST_CASE("SDDS-sized packets with non-colliding flags still pass") {
    // The common case, which worked before the fix and must keep working.
    CHECK(discovery_filter::is_sizing_candidate(SDDS, sdds_byte0(0x80, 0x01)));
    CHECK(discovery_filter::is_sizing_candidate(SDDS, sdds_byte0(0x00, 0x01)));
}

TEST_CASE("the classifier is usable in a constant expression") {
    // constexpr keeps it inlinable in the receive path and free of hidden state.
    static_assert(discovery_filter::is_sizing_candidate(SDDS, 0x40));
    static_assert(!discovery_filter::is_sizing_candidate(4096, 0x40));
    static_assert(!discovery_filter::is_sizing_candidate(-1, 0x00));
    SUCCEED();
}
