/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <optional>

namespace udp {

/**
 * Pure timing/state policy for fixed-window recvmmsg batching.
 *
 * The policy owns no buffers and performs no I/O. Keeping its clock injectable makes the
 * latency accounting and backlog fast path deterministic in unit tests.
 */
class receive_batch_policy {
public:
    using clock = std::chrono::steady_clock;
    using duration = std::chrono::microseconds;

    struct settings {
        std::size_t output_batch_size{1};
        std::size_t receive_vector_size{1};
        duration receive_batch_wait{};
        duration max_batch_delay{};
    };

    enum class event {
        none,
        pool_stall,
        shutdown,
    };

    enum class flush_reason {
        none,
        size,
        deadline,
        pool_stall,
        shutdown,
    };

    explicit receive_batch_policy(settings config) : m_config(config) {
        m_config.output_batch_size = std::max<std::size_t>(1, m_config.output_batch_size);
        m_config.receive_vector_size = std::max<std::size_t>(1, m_config.receive_vector_size);
        m_config.receive_batch_wait = std::max(duration::zero(), m_config.receive_batch_wait);
        m_config.max_batch_delay = std::max(duration::zero(), m_config.max_batch_delay);
    }

    auto note_batch_started(clock::time_point socket_readable_at) noexcept -> void {
        if (!m_oldest_packet) { m_oldest_packet = socket_readable_at; }
    }

    auto note_batch_flushed() noexcept -> void { m_oldest_packet.reset(); }

    auto note_receive_result(std::size_t count) noexcept -> void {
        m_backlog_likely = count == m_config.receive_vector_size;
    }

    [[nodiscard]] auto backlog_likely() const noexcept -> bool { return m_backlog_likely; }

    [[nodiscard]] auto receive_wait(clock::time_point now) const noexcept -> duration {
        if (m_backlog_likely || m_config.receive_batch_wait == duration::zero()) {
            return duration::zero();
        }
        auto wait = m_config.receive_batch_wait;
        if (const auto remaining = deadline_remaining(now)) {
            wait = std::min(wait, *remaining);
        } else {
            // The first socket-readability timestamp becomes the batch start after receive.
            // Bound this pre-read wait by the same total batch-latency budget.
            wait = std::min(wait, m_config.max_batch_delay);
        }
        return wait;
    }

    [[nodiscard]] auto deadline_remaining(clock::time_point now) const noexcept
        -> std::optional<duration> {
        if (!m_oldest_packet) { return std::nullopt; }
        const auto deadline = *m_oldest_packet + m_config.max_batch_delay;
        if (now >= deadline) { return duration::zero(); }
        return std::chrono::duration_cast<duration>(deadline - now);
    }

    [[nodiscard]] auto flush_for(std::size_t pending, clock::time_point now,
                                 event current_event = event::none) const noexcept
        -> flush_reason {
        if (pending == 0) { return flush_reason::none; }
        if (pending >= m_config.output_batch_size) { return flush_reason::size; }
        if (current_event == event::pool_stall) { return flush_reason::pool_stall; }
        if (current_event == event::shutdown) { return flush_reason::shutdown; }
        if (const auto remaining = deadline_remaining(now);
            remaining && *remaining == duration::zero()) {
            return flush_reason::deadline;
        }
        return flush_reason::none;
    }

private:
    settings m_config;
    std::optional<clock::time_point> m_oldest_packet;
    bool m_backlog_likely{false};
};

} // namespace udp
