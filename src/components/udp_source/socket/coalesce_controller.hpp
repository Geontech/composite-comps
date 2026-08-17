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

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace udp {

/**
 * @brief Deterministic policy for adaptive recvmmsg coalescing.
 *
 * Estimates the stream packet rate with an EWMA and derives the nominal delay needed to
 * accumulate approximately target_batch datagrams before the first receive call of a drain.
 * Safety signals (kernel drops, receive-memory pressure, pool stalls) suppress the actual
 * sleep through a time-bounded latch but never erase the learned nominal delay, so recovery
 * after a transient event is bounded by the latch duration instead of a full EWMA relearn.
 * A full receive vector is recorded as load information only: the caller's drain loop already
 * continues nonblocking to EAGAIN, and sustained overload surfaces through the receive-memory
 * and kernel-drop signals instead.
 *
 * Single-threaded by design: every method must be called from the receive thread with an
 * explicit time point, which keeps the policy deterministic and directly unit-testable.
 */
class coalesce_controller {
public:
    using clock = std::chrono::steady_clock;

    /// Intentional coalescing may plan to occupy at most 1/RMEM_DIVISOR of SO_RCVBUF.
    static constexpr uint64_t RMEM_DIVISOR = 4;

    enum class congestion_reason : uint8_t {
        none = 0,
        kernel_drop = 1,
        rmem_pressure = 2,
        pool_stall = 3,
    };

    enum class zero_sleep_reason : uint8_t {
        none = 0,        ///< last computed sleep was nonzero
        disabled = 1,    ///< max_coalesce is zero
        latched = 2,     ///< a safety latch is holding sleep at zero
        warming_up = 3,  ///< no learned rate/nominal delay yet
        below_min = 4,   ///< capacity ceiling pushed the sleep under min_coalesce
        capacity = 5,    ///< socket occupancy leaves no safe coalescing budget
    };

    struct settings {
        std::size_t target_batch{1};
        std::size_t batch_size{1};
        std::chrono::microseconds min_coalesce{0};
        std::chrono::microseconds max_coalesce{0};
        std::chrono::milliseconds adaptation_interval{250};
        /// How long a safety signal holds the actual sleep at zero. Re-triggering extends it.
        clock::duration latch_hold{std::chrono::milliseconds{250}};
        double rate_alpha{0.2};
        /// Relative deadband: nominal is held when |desired - nominal| < fraction * nominal,
        /// so packet-rate noise does not produce continuous small delay changes.
        double deadband_fraction{0.05};
    };

    struct event_counters {
        uint64_t full_vector{};
        uint64_t pool_stall{};
        uint64_t kernel_drop{};
        uint64_t rmem_pressure{};
        uint64_t latch_entries{};
        uint64_t idle_resets{};
    };

    /// Snapshot of the previous adaptation window's computed sleeps. min/mean/max cover only
    /// coalesced (nonzero-sleep) cycles; coalesced_cycles/cycles gives the coalesced fraction.
    struct interval_stats {
        uint64_t cycles{};
        uint64_t coalesced_cycles{};
        uint64_t min_sleep_us{};
        uint64_t mean_sleep_us{};
        uint64_t max_sleep_us{};
    };

    coalesce_controller(const settings& config, clock::time_point now)
      : m_cfg(config), m_window_start(now) {
        if (!(m_cfg.rate_alpha > 0.0) || m_cfg.rate_alpha > 1.0) { m_cfg.rate_alpha = 0.2; }
        if (m_cfg.deadband_fraction < 0.0) { m_cfg.deadband_fraction = 0.0; }
        if (m_cfg.batch_size == 0) { m_cfg.batch_size = 1; }
        if (m_cfg.target_batch == 0) { m_cfg.target_batch = 1; }
        if (m_cfg.min_coalesce > m_cfg.max_coalesce) { m_cfg.min_coalesce = m_cfg.max_coalesce; }
        if (m_cfg.adaptation_interval <= std::chrono::milliseconds::zero()) {
            m_cfg.adaptation_interval = std::chrono::milliseconds{250};
        }
        if (m_cfg.latch_hold <= clock::duration::zero()) {
            m_cfg.latch_hold = m_cfg.adaptation_interval;
        }
    }

    /**
     * @brief Record datagrams received at @p now; roll the adaptation window when it elapses.
     *
     * Learning is skipped for any window during which a safety latch was entered or active:
     * backpressure stretches wall-clock receive intervals (pool-stall backoff) and drops make
     * the measured receipt rate understate the offered rate, and either would mistrain the
     * feed-forward model.
     */
    auto on_packets(std::size_t packets, clock::time_point now) -> void {
        m_window_packets += static_cast<uint64_t>(packets);
        update_latch(now);
        const auto elapsed = now - m_window_start;
        if (elapsed < m_cfg.adaptation_interval) { return; }

        if (!m_latch_active && !m_window_tainted) {
            const auto elapsed_seconds = std::chrono::duration<double>(elapsed).count();
            const auto measured_pps =
                static_cast<double>(m_window_packets) / elapsed_seconds;
            m_estimated_pps = m_estimated_pps == 0.0
                ? measured_pps
                : m_cfg.rate_alpha * measured_pps
                    + (1.0 - m_cfg.rate_alpha) * m_estimated_pps;

            double desired_us = static_cast<double>(m_cfg.min_coalesce.count());
            if (m_estimated_pps > 0.0 && m_cfg.target_batch > 1) {
                desired_us = (static_cast<double>(m_cfg.target_batch - 1) * 1'000'000.0)
                    / m_estimated_pps;
            }
            // Jitter-headroom cap: never plan to accumulate more than half the receive vector
            // during the sleep. Beyond that, one modest scheduling delay fills the vector and
            // costs an extra syscall for no coalescing benefit.
            if (m_estimated_pps > 0.0 && m_cfg.batch_size > 1) {
                const auto half_vector_fill_us =
                    (static_cast<double>(m_cfg.batch_size) / 2.0) * 1'000'000.0
                        / m_estimated_pps;
                desired_us = std::min(desired_us, half_vector_fill_us);
            }
            desired_us = std::clamp(desired_us,
                static_cast<double>(m_cfg.min_coalesce.count()),
                static_cast<double>(m_cfg.max_coalesce.count()));

            if (m_nominal_us == 0.0) {
                // Seed directly: the desired value is already known, so recovering from a
                // legitimate reset must not crawl up an EWMA ramp from zero.
                m_nominal_us = desired_us;
            } else if (std::abs(desired_us - m_nominal_us)
                       >= m_cfg.deadband_fraction * m_nominal_us) {
                m_nominal_us += m_cfg.rate_alpha * (desired_us - m_nominal_us);
            }
        }

        roll_interval_stats();
        // A window that begins while the latch is still active is born tainted: even if the
        // latch expires mid-window, part of its traffic was observed under suppression and
        // must not train the model.
        m_window_tainted = m_latch_active;
        m_window_packets = 0;
        m_window_start = now;
    }

    /// A receive call returned the entire vector. Load information only; never a latch.
    auto note_full_vector() noexcept -> void { ++m_counters.full_vector; }

    /**
     * @brief Enter (or extend) the safety latch. The learned nominal delay is retained.
     * @return true when this call newly entered the latch (useful for one-shot logging).
     */
    auto enter_congestion(congestion_reason reason, clock::time_point now) -> bool {
        // Expire a stale latch first so a signal arriving after the hold elapsed counts as a
        // new episode instead of silently extending the old one.
        update_latch(now);
        switch (reason) {
            case congestion_reason::kernel_drop: ++m_counters.kernel_drop; break;
            case congestion_reason::rmem_pressure: ++m_counters.rmem_pressure; break;
            case congestion_reason::pool_stall: ++m_counters.pool_stall; break;
            case congestion_reason::none: return false;
        }
        m_window_tainted = true;
        const bool newly_entered = !m_latch_active;
        if (newly_entered) { ++m_counters.latch_entries; }
        m_latch_active = true;
        m_latch_reason = reason;
        m_latch_until = now + m_cfg.latch_hold;
        return newly_entered;
    }

    /// The stream went idle: discard the rate model and relearn from the new stream.
    auto on_idle_reset(clock::time_point now) -> void {
        ++m_counters.idle_resets;
        m_estimated_pps = 0.0;
        m_nominal_us = 0.0;
        m_window_packets = 0;
        m_window_start = now;
        reset_interval_accumulators();
        update_latch(now);
        m_window_tainted = m_latch_active;
    }

    /**
     * @brief Compute this cycle's actual sleep from the nominal delay and live socket state.
     *
     * The socket-capacity ceiling and the safety latch have absolute priority over the
     * learned nominal value. Also records the cycle into the interval statistics.
     */
    auto compute_sleep(clock::time_point now, uint64_t rmem_allocated, uint64_t rmem_limit,
                       uint64_t packet_charge) -> std::chrono::microseconds {
        update_latch(now);
        double actual_us = 0.0;
        if (m_cfg.max_coalesce <= std::chrono::microseconds::zero()) {
            m_zero_reason = zero_sleep_reason::disabled;
        } else if (m_latch_active) {
            m_zero_reason = zero_sleep_reason::latched;
        } else if (m_estimated_pps <= 0.0 || m_nominal_us <= 0.0) {
            m_zero_reason = zero_sleep_reason::warming_up;
        } else if (rmem_limit / RMEM_DIVISOR <= rmem_allocated || packet_charge == 0) {
            m_zero_reason = zero_sleep_reason::capacity;
        } else {
            const auto safe_bytes = rmem_limit / RMEM_DIVISOR - rmem_allocated;
            const auto capacity_us = static_cast<double>(safe_bytes) * 1'000'000.0
                / (m_estimated_pps * static_cast<double>(packet_charge));
            actual_us = std::min(m_nominal_us, capacity_us);
            if (actual_us < static_cast<double>(m_cfg.min_coalesce.count())) {
                // Never violate the capacity ceiling to satisfy a configured minimum.
                actual_us = 0.0;
                m_zero_reason = zero_sleep_reason::below_min;
            } else {
                actual_us = std::min(actual_us,
                    static_cast<double>(m_cfg.max_coalesce.count()));
                m_zero_reason = zero_sleep_reason::none;
            }
        }
        record_cycle(actual_us);
        m_last_actual_us = actual_us;
        return std::chrono::microseconds{static_cast<int64_t>(actual_us)};
    }

    [[nodiscard]] auto nominal_us() const noexcept -> double { return m_nominal_us; }
    [[nodiscard]] auto last_actual_us() const noexcept -> double { return m_last_actual_us; }
    [[nodiscard]] auto estimated_pps() const noexcept -> double { return m_estimated_pps; }
    [[nodiscard]] auto counters() const noexcept -> const event_counters& { return m_counters; }
    [[nodiscard]] auto interval() const noexcept -> const interval_stats& { return m_interval; }
    [[nodiscard]] auto zero_reason() const noexcept -> zero_sleep_reason { return m_zero_reason; }
    [[nodiscard]] auto latch_reason() const noexcept -> congestion_reason {
        return m_latch_active ? m_latch_reason : congestion_reason::none;
    }
    [[nodiscard]] auto latch_active(clock::time_point now) const noexcept -> bool {
        return m_latch_active && now < m_latch_until;
    }
    [[nodiscard]] auto latch_remaining_us(clock::time_point now) const noexcept -> uint64_t {
        if (!latch_active(now)) { return 0; }
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(m_latch_until - now).count());
    }

    static constexpr auto to_string(congestion_reason reason) noexcept -> std::string_view {
        switch (reason) {
            case congestion_reason::kernel_drop: return "kernel_drop";
            case congestion_reason::rmem_pressure: return "rmem_pressure";
            case congestion_reason::pool_stall: return "pool_stall";
            case congestion_reason::none: break;
        }
        return "none";
    }

    static constexpr auto to_string(zero_sleep_reason reason) noexcept -> std::string_view {
        switch (reason) {
            case zero_sleep_reason::disabled: return "disabled";
            case zero_sleep_reason::latched: return "latched";
            case zero_sleep_reason::warming_up: return "warming_up";
            case zero_sleep_reason::below_min: return "below_min";
            case zero_sleep_reason::capacity: return "capacity";
            case zero_sleep_reason::none: break;
        }
        return "none";
    }

private:
    auto update_latch(clock::time_point now) noexcept -> void {
        if (m_latch_active && now >= m_latch_until) {
            m_latch_active = false;
            m_latch_reason = congestion_reason::none;
        }
    }

    auto record_cycle(double actual_us) noexcept -> void {
        ++m_acc_cycles;
        if (actual_us > 0.0) {
            const auto us = static_cast<uint64_t>(actual_us);
            ++m_acc_coalesced;
            m_acc_sleep_sum_us += us;
            m_acc_sleep_min_us = std::min(m_acc_sleep_min_us, us);
            m_acc_sleep_max_us = std::max(m_acc_sleep_max_us, us);
        }
    }

    auto roll_interval_stats() noexcept -> void {
        m_interval.cycles = m_acc_cycles;
        m_interval.coalesced_cycles = m_acc_coalesced;
        m_interval.min_sleep_us = m_acc_coalesced > 0 ? m_acc_sleep_min_us : 0;
        m_interval.max_sleep_us = m_acc_sleep_max_us;
        m_interval.mean_sleep_us = m_acc_coalesced > 0 ? m_acc_sleep_sum_us / m_acc_coalesced : 0;
        reset_interval_accumulators();
    }

    auto reset_interval_accumulators() noexcept -> void {
        m_acc_cycles = 0;
        m_acc_coalesced = 0;
        m_acc_sleep_sum_us = 0;
        m_acc_sleep_min_us = std::numeric_limits<uint64_t>::max();
        m_acc_sleep_max_us = 0;
    }

    settings m_cfg;
    clock::time_point m_window_start;
    uint64_t m_window_packets{0};
    bool m_window_tainted{false};

    double m_estimated_pps{0.0};
    double m_nominal_us{0.0};
    double m_last_actual_us{0.0};

    bool m_latch_active{false};
    congestion_reason m_latch_reason{congestion_reason::none};
    clock::time_point m_latch_until{};
    zero_sleep_reason m_zero_reason{zero_sleep_reason::none};

    event_counters m_counters{};
    interval_stats m_interval{};
    uint64_t m_acc_cycles{0};
    uint64_t m_acc_coalesced{0};
    uint64_t m_acc_sleep_sum_us{0};
    uint64_t m_acc_sleep_min_us{std::numeric_limits<uint64_t>::max()};
    uint64_t m_acc_sleep_max_us{0};

}; // class coalesce_controller

} // namespace udp
