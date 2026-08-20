/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * Bounds tests for the sendmmsg staging buffer.
 *
 * These exist because send() used to check capacity, flush on failure, and then memcpy
 * UNCONDITIONALLY without re-checking -- so a datagram larger than the whole staging buffer ran
 * off the end of the allocation with an upstream-controlled length. batch_size and
 * max_packet_size were also unvalidated, and batch_size == 0 sized the buffer to nothing, making
 * every send an overflow.
 *
 * Run under the asan-ubsan preset for these to have teeth: a regression here is a heap write past
 * the end of a std::vector, which an unsanitized build will usually absorb silently.
 */
#include "socket/sendmmsg.hpp"

#include <catch2/catch_test_macros.hpp>

#include <composite/core/logger.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

// One shared logger for the whole file, quiet: these tests deliberately drive the error paths,
// which log at error level, and the expected noise would otherwise bury a real failure.
auto test_logger() -> std::shared_ptr<composite::logger> {
    static auto log = [] {
        auto l = std::make_shared<composite::logger>("sendmmsg_bounds_tests");
        l->set_level(composite::log_level::critical);
        return l;
    }();
    return log;
}

auto make_config(uint32_t batch_size, uint32_t max_packet_size) -> udp_tx::config {
    return udp_tx::config{
        .logger = test_logger(),
        .send_buf_size = 0,
        .socket_timeout_s = 30,
        .batch_size = batch_size,
        .batch_timeout_us = 0,          // never flush on time; the tests drive flushing by size
        .max_packet_size = max_packet_size,
        .bind_interface = {},
    };
}

// Loopback discard port: these tests exercise the staging-buffer arithmetic, not delivery.
constexpr const char* DEST_IP = "127.0.0.1";
constexpr uint16_t DEST_PORT = 9;

} // namespace

TEST_CASE("zero staging-buffer dimensions are rejected", "[udp_sink][bounds]") {
    // batch_size == 0 previously resized the data buffer to 0, so the very first send overflowed.
    CHECK_THROWS_AS(udp_tx::sendmmsg_tx{make_config(0, 4096)}, std::invalid_argument);
    CHECK_THROWS_AS(udp_tx::sendmmsg_tx{make_config(64, 0)}, std::invalid_argument);
    CHECK_THROWS_AS(udp_tx::sendmmsg_tx{make_config(0, 0)}, std::invalid_argument);
}

TEST_CASE("absurd staging-buffer dimensions are rejected", "[udp_sink][bounds]") {
    // Bounding both dimensions is what makes batch_size * max_packet_size safe to compute; an
    // unbounded product is an allocation the process cannot satisfy.
    CHECK_THROWS_AS(udp_tx::sendmmsg_tx{make_config(0xFFFFFFFFU, 4096)}, std::invalid_argument);
    CHECK_THROWS_AS(udp_tx::sendmmsg_tx{make_config(64, 0xFFFFFFFFU)}, std::invalid_argument);
    // 65507 is the largest possible UDP payload and must remain acceptable.
    CHECK_NOTHROW(udp_tx::sendmmsg_tx{make_config(1, 65507)});
}

TEST_CASE("a datagram larger than max_packet_size is refused, not copied", "[udp_sink][bounds]") {
    // THE original defect. With batch_size == 1 the staging buffer is exactly one datagram, so a
    // larger datagram cannot be staged even after a flush -- which is precisely the case the old
    // code fell through into memcpy.
    constexpr uint32_t max_packet = 1024;
    auto tx = udp_tx::sendmmsg_tx{make_config(1, max_packet)};

    SECTION("one byte over the per-datagram budget") {
        std::vector<uint8_t> data(max_packet + 1, 0xAB);
        CHECK(tx.send(DEST_IP, DEST_PORT, data) == -1);
    }
    SECTION("far larger than the entire staging buffer") {
        std::vector<uint8_t> data(max_packet * 64, 0xCD);
        CHECK(tx.send(DEST_IP, DEST_PORT, data) == -1);
    }
    SECTION("exactly at the budget is accepted") {
        std::vector<uint8_t> data(max_packet, 0xEF);
        CHECK(tx.send(DEST_IP, DEST_PORT, data) >= 0);
    }
}

TEST_CASE("filling the batch exactly does not overflow", "[udp_sink][bounds]") {
    // Near-overflow: N datagrams of exactly max_packet_size fill the buffer to its last byte,
    // and the next send must flush and restart at offset 0 rather than write past the end.
    constexpr uint32_t batch = 8;
    constexpr uint32_t max_packet = 512;
    auto tx = udp_tx::sendmmsg_tx{make_config(batch, max_packet)};

    std::vector<uint8_t> data(max_packet, 0x5A);
    for (uint32_t i = 0; i < batch * 3; ++i) {
        CHECK(tx.send(DEST_IP, DEST_PORT, data) >= 0);
    }
    tx.flush();
}

TEST_CASE("mixed sizes across a batch boundary stay in bounds", "[udp_sink][bounds]") {
    // Uneven sizes make the write position land at offsets that are not multiples of
    // max_packet_size, so the remaining-capacity arithmetic is exercised rather than the
    // convenient aligned case.
    constexpr uint32_t batch = 4;
    constexpr uint32_t max_packet = 300;
    auto tx = udp_tx::sendmmsg_tx{make_config(batch, max_packet)};

    for (uint32_t i = 0; i < 40; ++i) {
        std::vector<uint8_t> data((i * 37) % max_packet + 1, static_cast<uint8_t>(i));
        CHECK(tx.send(DEST_IP, DEST_PORT, data) >= 0);
    }
    tx.flush();
}
