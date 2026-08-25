/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <catch2/catch_test_macros.hpp>

#include "socket/receive_batch_policy.hpp"

#include <chrono>

namespace {

using namespace std::chrono_literals;
using policy = udp::receive_batch_policy;
using reason = policy::flush_reason;
using event = policy::event;

constexpr auto t0 = policy::clock::time_point{};

auto make_policy(std::chrono::microseconds receive_wait = 100us,
                 std::chrono::microseconds max_delay = 1000us) -> policy {
    return policy(policy::settings{
        .output_batch_size = 64,
        .receive_vector_size = 128,
        .receive_batch_wait = receive_wait,
        .max_batch_delay = max_delay,
    });
}

} // namespace

TEST_CASE("fixed receive window is bounded by the total batch deadline") {
    auto p = make_policy(250us, 1000us);
    REQUIRE(p.receive_wait(t0) == 250us);

    p.note_batch_started(t0);
    REQUIRE(p.receive_wait(t0 + 800us) == 200us);
    REQUIRE(p.receive_wait(t0 + 1000us) == 0us);
}

TEST_CASE("a full receive vector bypasses the fixed window until backlog clears") {
    auto p = make_policy();
    p.note_receive_result(128);
    REQUIRE(p.backlog_likely());
    REQUIRE(p.receive_wait(t0) == 0us);

    p.note_receive_result(17);
    REQUIRE_FALSE(p.backlog_likely());
    REQUIRE(p.receive_wait(t0) == 100us);
}

TEST_CASE("output flush decisions cover size deadline pool stall and shutdown") {
    auto p = make_policy();
    REQUIRE(p.flush_for(0, t0) == reason::none);
    REQUIRE(p.flush_for(64, t0) == reason::size);

    p.note_batch_started(t0);
    REQUIRE(p.flush_for(12, t0 + 999us) == reason::none);
    REQUIRE(p.flush_for(12, t0 + 1000us) == reason::deadline);
    REQUIRE(p.flush_for(12, t0, event::pool_stall) == reason::pool_stall);
    REQUIRE(p.flush_for(12, t0, event::shutdown) == reason::shutdown);
}

TEST_CASE("flushing resets the oldest packet deadline for the next remainder") {
    auto p = make_policy();
    p.note_batch_started(t0);
    p.note_batch_flushed();

    REQUIRE_FALSE(p.deadline_remaining(t0 + 2ms).has_value());
    p.note_batch_started(t0 + 2ms);
    REQUIRE(p.deadline_remaining(t0 + 2500us) == 500us);
}

TEST_CASE("zero delay mode performs no fixed wait and expires a partial batch immediately") {
    auto p = make_policy(100us, 0us);
    REQUIRE(p.receive_wait(t0) == 0us);

    p.note_batch_started(t0);
    REQUIRE(p.flush_for(1, t0) == reason::deadline);
}

TEST_CASE("zero receive window keeps output deadline batching enabled") {
    auto p = make_policy(0us, 1000us);
    REQUIRE(p.receive_wait(t0) == 0us);

    p.note_batch_started(t0);
    REQUIRE(p.flush_for(10, t0 + 999us) == reason::none);
    REQUIRE(p.flush_for(10, t0 + 1000us) == reason::deadline);
}
