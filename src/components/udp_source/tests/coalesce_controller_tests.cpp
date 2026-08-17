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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "socket/coalesce_controller.hpp"

#include <chrono>

using udp::coalesce_controller;
using namespace std::chrono_literals;

namespace {

using clock_t_ = coalesce_controller::clock;
using reason = coalesce_controller::congestion_reason;
using zero_reason = coalesce_controller::zero_sleep_reason;

// The live-cluster scenario from the v0.5.0 coalescing investigation: a steady ~97.66 kpps
// digitizer stream, receive vector 128, target batch 64, no minimum, 1000 us maximum.
constexpr double STEADY_PPS = 97'660.0;
constexpr auto WINDOW = 250ms;
// Packets arriving in one 250 ms adaptation window at the steady rate.
constexpr std::size_t WINDOW_PACKETS = 24'415;
// Steady-state delay to accumulate target-1 = 63 packets: ~645 us.
constexpr double STEADY_DELAY_US = 63.0 * 1'000'000.0 / STEADY_PPS;

// Generous socket-memory state: nothing allocated against a 256 MiB limit.
constexpr uint64_t RMEM_LIMIT = 268'435'456;
constexpr uint64_t RMEM_IDLE = 4'096;
constexpr uint64_t PACKET_CHARGE = 4'096;

auto live_settings() -> coalesce_controller::settings {
    return {
        .target_batch = 64,
        .batch_size = 128,
        .min_coalesce = 0us,
        .max_coalesce = 1000us,
        .adaptation_interval = 250ms,
        .latch_hold = 250ms,
    };
}

auto t0() -> clock_t_::time_point { return clock_t_::time_point{} + 1h; }

// Feed complete adaptation windows of steady traffic, returning the time after the last one.
auto feed_steady_windows(coalesce_controller& ctrl, clock_t_::time_point start, int windows,
                         std::size_t packets_per_window = WINDOW_PACKETS)
    -> clock_t_::time_point {
    auto now = start;
    for (int i = 0; i < windows; ++i) {
        now += WINDOW;
        ctrl.on_packets(packets_per_window, now);
    }
    return now;
}

auto sleep_us(coalesce_controller& ctrl, clock_t_::time_point now) -> int64_t {
    return ctrl.compute_sleep(now, RMEM_IDLE, RMEM_LIMIT, PACKET_CHARGE).count();
}

} // namespace

TEST_CASE("steady stream: nominal seeds directly and converges within one window") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    const auto now = feed_steady_windows(ctrl, t0(), 1);

    // Seeding must be direct, not an EWMA ramp from zero (the 129/232/315... live pathology).
    REQUIRE_THAT(ctrl.estimated_pps(),
                 Catch::Matchers::WithinRel(STEADY_PPS, 0.01));
    REQUIRE_THAT(ctrl.nominal_us(),
                 Catch::Matchers::WithinRel(STEADY_DELAY_US, 0.02));
    REQUIRE(sleep_us(ctrl, now) > 0);
}

TEST_CASE("steady stream: nominal stays within 10% across many windows (deadband holds)") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 1);
    const auto seeded = ctrl.nominal_us();
    now = feed_steady_windows(ctrl, now, 40);
    REQUIRE_THAT(ctrl.nominal_us(), Catch::Matchers::WithinRel(seeded, 0.10));
    REQUIRE_THAT(static_cast<double>(sleep_us(ctrl, now)),
                 Catch::Matchers::WithinRel(STEADY_DELAY_US, 0.10));
}

TEST_CASE("full receive vector is load information only: no latch, no model reset") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 4);
    const auto learned = ctrl.nominal_us();
    REQUIRE(learned > 0.0);

    for (int i = 0; i < 5; ++i) { ctrl.note_full_vector(); }

    REQUIRE(ctrl.counters().full_vector == 5);
    REQUIRE(ctrl.counters().latch_entries == 0);
    REQUIRE(ctrl.nominal_us() == learned);
    REQUIRE_FALSE(ctrl.latch_active(now));
    REQUIRE(sleep_us(ctrl, now) > 0);
}

TEST_CASE("safety latch suppresses sleep but retains the learned nominal delay") {
    const auto trigger = GENERATE_COPY(reason::kernel_drop, reason::rmem_pressure,
                                       reason::pool_stall);
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 4);
    const auto learned = ctrl.nominal_us();

    REQUIRE(ctrl.enter_congestion(trigger, now));
    REQUIRE(ctrl.latch_active(now));
    REQUIRE(ctrl.latch_reason() == trigger);
    REQUIRE(ctrl.counters().latch_entries == 1);

    // Zero sleep for the latch duration, with the reason exported.
    REQUIRE(sleep_us(ctrl, now + 1ms) == 0);
    REQUIRE(ctrl.zero_reason() == zero_reason::latched);
    REQUIRE(ctrl.nominal_us() == learned);

    // Bounded recovery: after the hold expires, sleep resumes from the retained nominal
    // immediately — no relearn ramp.
    const auto after = now + 251ms;
    const auto resumed = sleep_us(ctrl, after);
    REQUIRE_THAT(static_cast<double>(resumed), Catch::Matchers::WithinRel(learned, 0.01));
    REQUIRE_FALSE(ctrl.latch_active(after));
}

TEST_CASE("re-triggering extends the latch; a second entry is not double-counted") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 2);

    REQUIRE(ctrl.enter_congestion(reason::kernel_drop, now));
    REQUIRE_FALSE(ctrl.enter_congestion(reason::kernel_drop, now + 200ms));
    REQUIRE(ctrl.counters().latch_entries == 1);
    REQUIRE(ctrl.counters().kernel_drop == 2);

    REQUIRE(sleep_us(ctrl, now + 300ms) == 0);   // extended past the first hold
    REQUIRE(sleep_us(ctrl, now + 452ms) > 0);    // expired 250ms after the re-trigger
}

TEST_CASE("a signal arriving after latch expiry is a new entry, not an extension") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 2);

    REQUIRE(ctrl.enter_congestion(reason::kernel_drop, now));
    // No controller calls happen between expiry (now+250ms) and the next signal; it must
    // still be counted as a distinct latch episode with a fresh hold.
    REQUIRE(ctrl.enter_congestion(reason::kernel_drop, now + 400ms));
    REQUIRE(ctrl.counters().latch_entries == 2);
    REQUIRE(sleep_us(ctrl, now + 500ms) == 0);
    REQUIRE(sleep_us(ctrl, now + 651ms) > 0);
}

TEST_CASE("a window that begins while latched cannot train, even if the latch expires mid-window") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 4);
    const auto pps = ctrl.estimated_pps();

    // Latch until now+260ms: still active when the window rolls at now+250ms, expired by the
    // middle of the following window. That following window was partly observed under
    // suppression and must not teach the model.
    ctrl.enter_congestion(reason::pool_stall, now + 10ms);
    ctrl.on_packets(WINDOW_PACKETS, now + WINDOW);
    ctrl.on_packets(WINDOW_PACKETS / 2, now + 2 * WINDOW);
    REQUIRE(ctrl.estimated_pps() == pps);

    // The first window born after expiry learns again.
    ctrl.on_packets(WINDOW_PACKETS / 2, now + 3 * WINDOW);
    REQUIRE(ctrl.estimated_pps() != pps);
}

TEST_CASE("learning is frozen while latched and for the tainted window") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 4);
    const auto pps = ctrl.estimated_pps();
    const auto learned = ctrl.nominal_us();

    // A pool stall stretches wall-clock time: the same packets appear over a longer window,
    // which would understate the rate and overstate the delay if it were learned.
    ctrl.enter_congestion(reason::pool_stall, now);
    now += 2 * WINDOW;
    ctrl.on_packets(WINDOW_PACKETS, now); // rolls a stalled window at half the true rate
    REQUIRE(ctrl.estimated_pps() == pps);
    REQUIRE(ctrl.nominal_us() == learned);

    // After the latch expires, an untainted window teaches again.
    now = feed_steady_windows(ctrl, now, 2, WINDOW_PACKETS / 2);
    REQUIRE(ctrl.estimated_pps() != pps);
}

TEST_CASE("idle reset discards the model and recovery is bounded by one window") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 4);

    ctrl.on_idle_reset(now);
    REQUIRE(ctrl.counters().idle_resets == 1);
    REQUIRE(ctrl.estimated_pps() == 0.0);
    REQUIRE(sleep_us(ctrl, now) == 0);
    REQUIRE(ctrl.zero_reason() == zero_reason::warming_up);

    // One full window of the new stream reseeds the nominal directly.
    now = feed_steady_windows(ctrl, now, 1);
    REQUIRE_THAT(ctrl.nominal_us(),
                 Catch::Matchers::WithinRel(STEADY_DELAY_US, 0.02));
}

TEST_CASE("deadband ignores small rate noise but tracks a real rate change") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 4);
    const auto learned = ctrl.nominal_us();

    // +3% rate windows: inside the 5% deadband once the EWMA smooths them, nominal holds.
    now = feed_steady_windows(ctrl, now, 10,
        static_cast<std::size_t>(WINDOW_PACKETS * 1.03));
    REQUIRE_THAT(ctrl.nominal_us(), Catch::Matchers::WithinRel(learned, 0.05));

    // Rate halves: nominal must follow toward the max-clamped delay (1000 us).
    now = feed_steady_windows(ctrl, now, 40, WINDOW_PACKETS / 2);
    REQUIRE(ctrl.nominal_us() > learned * 1.4);
    REQUIRE(ctrl.nominal_us() <= 1000.0);
}

TEST_CASE("bursty stream: alternating windows never reset the model to zero") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 1);
    for (int i = 0; i < 20; ++i) {
        const auto packets = (i % 2 == 0)
            ? static_cast<std::size_t>(WINDOW_PACKETS * 1.2)
            : static_cast<std::size_t>(WINDOW_PACKETS * 0.8);
        now += WINDOW;
        ctrl.on_packets(packets, now);
        REQUIRE(ctrl.nominal_us() > 0.0);
    }
    REQUIRE_THAT(ctrl.nominal_us(), Catch::Matchers::WithinRel(STEADY_DELAY_US, 0.25));
}

TEST_CASE("jitter-headroom cap: delay never plans past half the receive vector") {
    // Default 75% target (96 of 128) would want ~973 us; the cap holds it at the
    // half-vector fill time (~655 us) so one modest scheduling delay cannot fill the vector.
    auto settings = live_settings();
    settings.target_batch = 96;
    auto ctrl = coalesce_controller(settings, t0());
    feed_steady_windows(ctrl, t0(), 8);
    const auto half_vector_fill_us = 64.0 * 1'000'000.0 / STEADY_PPS;
    REQUIRE(ctrl.nominal_us() <= half_vector_fill_us * 1.01);
    REQUIRE(ctrl.nominal_us() > half_vector_fill_us * 0.9);
}

TEST_CASE("max_coalesce clamps the nominal delay at low rates") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    // 10 kpps wants 6300 us for 63 packets; max is 1000 us.
    feed_steady_windows(ctrl, t0(), 4, 2'500);
    REQUIRE(ctrl.nominal_us() <= 1000.0);
    REQUIRE(ctrl.nominal_us() > 900.0);
}

TEST_CASE("socket occupancy: capacity gate and ceiling override the nominal delay") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 4);

    SECTION("occupancy at the 25% budget yields zero sleep") {
        const auto sleep = ctrl.compute_sleep(now, RMEM_LIMIT / 4, RMEM_LIMIT, PACKET_CHARGE);
        REQUIRE(sleep.count() == 0);
        REQUIRE(ctrl.zero_reason() == zero_reason::capacity);
    }
    SECTION("occupancy near the budget clamps the sleep below nominal") {
        // Room for ~16 more packet charges: ~163 us of budget at the steady rate.
        const auto allocated = RMEM_LIMIT / 4 - 16 * PACKET_CHARGE;
        const auto sleep = ctrl.compute_sleep(now, allocated, RMEM_LIMIT, PACKET_CHARGE);
        REQUIRE(sleep.count() > 0);
        REQUIRE(static_cast<double>(sleep.count()) < ctrl.nominal_us());
    }
    SECTION("a capacity-limited sleep below min_coalesce becomes zero, never a violation") {
        auto settings = live_settings();
        settings.min_coalesce = 300us;
        auto strict = coalesce_controller(settings, t0());
        const auto then = feed_steady_windows(strict, t0(), 4);
        const auto allocated = RMEM_LIMIT / 4 - 16 * PACKET_CHARGE; // ~163 us of budget
        const auto sleep = strict.compute_sleep(then, allocated, RMEM_LIMIT, PACKET_CHARGE);
        REQUIRE(sleep.count() == 0);
        REQUIRE(strict.zero_reason() == zero_reason::below_min);
    }
}

TEST_CASE("disabled coalescing always sleeps zero with the reason exported") {
    auto settings = live_settings();
    settings.max_coalesce = 0us;
    auto ctrl = coalesce_controller(settings, t0());
    const auto now = feed_steady_windows(ctrl, t0(), 4);
    REQUIRE(sleep_us(ctrl, now) == 0);
    REQUIRE(ctrl.zero_reason() == zero_reason::disabled);
}

TEST_CASE("interval statistics summarize the previous adaptation window") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 2);

    // Five coalesced cycles and one latched (zero-sleep) cycle inside the next window.
    for (int i = 0; i < 5; ++i) { REQUIRE(sleep_us(ctrl, now + i * 1ms) > 0); }
    ctrl.enter_congestion(reason::kernel_drop, now + 6ms);
    REQUIRE(sleep_us(ctrl, now + 7ms) == 0);

    now = feed_steady_windows(ctrl, now, 1);
    const auto& interval = ctrl.interval();
    REQUIRE(interval.cycles == 6);
    REQUIRE(interval.coalesced_cycles == 5);
    REQUIRE(interval.min_sleep_us > 0);
    REQUIRE(interval.max_sleep_us >= interval.mean_sleep_us);
    REQUIRE(interval.mean_sleep_us >= interval.min_sleep_us);
}

TEST_CASE("latch remaining time is exported and reaches zero") {
    auto ctrl = coalesce_controller(live_settings(), t0());
    auto now = feed_steady_windows(ctrl, t0(), 2);
    ctrl.enter_congestion(reason::rmem_pressure, now);
    REQUIRE(ctrl.latch_remaining_us(now) == 250'000);
    REQUIRE(ctrl.latch_remaining_us(now + 100ms) == 150'000);
    REQUIRE(ctrl.latch_remaining_us(now + 251ms) == 0);
}

TEST_CASE("reason strings are stable for telemetry") {
    REQUIRE(coalesce_controller::to_string(reason::none) == "none");
    REQUIRE(coalesce_controller::to_string(reason::kernel_drop) == "kernel_drop");
    REQUIRE(coalesce_controller::to_string(reason::rmem_pressure) == "rmem_pressure");
    REQUIRE(coalesce_controller::to_string(reason::pool_stall) == "pool_stall");
    REQUIRE(coalesce_controller::to_string(zero_reason::none) == "none");
    REQUIRE(coalesce_controller::to_string(zero_reason::disabled) == "disabled");
    REQUIRE(coalesce_controller::to_string(zero_reason::latched) == "latched");
    REQUIRE(coalesce_controller::to_string(zero_reason::warming_up) == "warming_up");
    REQUIRE(coalesce_controller::to_string(zero_reason::below_min) == "below_min");
    REQUIRE(coalesce_controller::to_string(zero_reason::capacity) == "capacity");
}
