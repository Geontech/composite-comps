// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include "component.hpp"
#include "kernels.hpp"
#include <windows.hpp>

halfrate::halfrate(std::string_view id) : composite::component(id) {
    add_port(&m_in_port);
    add_port(&m_out_port);
    add_property("filter_semi_length", m_filter_semi_length);
    add_property("window", m_window_type).change_listener([this]() {
        return m_window_type.empty() || (m_window_type == "BLACKMAN_HARRIS") || (m_window_type == "HAMMING");
    });
}

auto halfrate::property_change_handler() -> void {
    // Generate filter coefficients
    generate_coeffs();

    // 1. Calculate history requirements
    // Defensive max: ensure we satisfy both FIR history (taps-1) and delay history (offset)
    m_taps_needed = m_coeffs.size();
    std::size_t fir_req = m_taps_needed > 0 ? m_taps_needed - 1 : 0;
    std::size_t delay_req = (m_filter_semi_length > 0) ? m_filter_semi_length - 1 : 0;
    m_history_len = std::max(fir_req, delay_req);

    // 2. Pre-allocate memory
    // Reserve space for history + 8192 samples
    // This prevents .resize() in the hot path from triggering malloc.
    constexpr size_t RESERVE_CAPACITY = 8192;
    std::size_t total_capacity = m_history_len + RESERVE_CAPACITY;

    // Reset vectors
    m_even_lane.clear();
    m_odd_lane.clear();

    // Enforce capacity
    if (m_even_lane.capacity() < total_capacity) {
        m_even_lane.reserve(total_capacity);
        m_odd_lane.reserve(total_capacity);
    }

    // Set initial size to just history (zeroed out)
    m_even_lane.assign(m_history_len, {0.0f, 0.0f});
    m_odd_lane.assign(m_history_len, {0.0f, 0.0f});
}

auto halfrate::process() -> composite::retval {
    using enum composite::retval;
    auto [data, ts, meta] = m_in_port.get_data();
    if (!data) {
        return NORMAL;
    }

    // Iterate over samples to produce output
    const auto n_in = data.size();
    const auto n_out = n_in / 2;

    // 1. Buffer management
    // Resize lanes if needed
    if (m_even_lane.size() < m_history_len + n_out) {
        m_even_lane.resize(m_history_len + n_out);
        m_odd_lane.resize(m_history_len + n_out);
    }

    // 2. De-Interleave with avx-enabled kernel
    auto* even_ptr = m_even_lane.data() + m_history_len;
    auto* odd_ptr = m_odd_lane.data() + m_history_len;
    kernels::deinterleave_block(data.data(), even_ptr, odd_ptr, n_out);

    // 3. Process vertical filter kernel
    auto delay_offset = (m_filter_semi_length > 0) ? m_filter_semi_length - 1 : 0;
    kernels::halfband_filter_vertical(
        m_even_lane.data(),
        m_odd_lane.data(),
        m_coeffs.data(), m_coeffs.size(),
        m_center_tap, delay_offset,
        data.data(), // destination
        n_out
    );

    // 4. Update history
    // Copy the last m_history_len samples to the front of the buffer for the next process call
    if (m_history_len > 0) {
        auto* src_even = m_even_lane.data() + n_out;
        auto* src_odd  = m_odd_lane.data() + n_out;

        std::memcpy(m_even_lane.data(), src_even, m_history_len * sizeof(cf32_t));
        std::memcpy(m_odd_lane.data(), src_odd, m_history_len * sizeof(cf32_t));
    }

    // Send data
    data.resize(n_out);
    m_out_port.send_data(std::move(data).to_immutable(), ts);

    return NORMAL;
}

auto halfrate::generate_coeffs() -> void {
    auto L = 4 * m_filter_semi_length + 1; // full length
    auto center = L / 2;
    auto h = std::vector<float>(L);

    auto window = (m_window_type == "BLACKMAN_HARRIS")
        ? windows::blackman_harris<float>(L, false)
        : windows::hamming<float>(L, false);

    for (int n = 0; n < L; ++n) {
        int k = n - center;
        if (k == 0) {
            h[n] = 0.5;  // half-band filter: center tap is always 0.5
            m_center_tap = h[n];
        } else if (k % 2 != 0) {
            auto sinc_val = std::sin(0.5f * std::numbers::pi_v<float> * k) / (std::numbers::pi_v<float> * k);
            h[n] = sinc_val * window->at(n);
        } else {
            h[n] = 0.0f;
        }
    }

    m_coeffs.clear();
    // Pack the non-zero coefficients into the dense vector
    // Iterate backwards to match convolution order
    // Extract odd taps
    for (int i = h.size() - 1; i >= 0; --i) {
        int k = i - center;
        if (k == 0) { continue; } // skip center tap
        // keep non-zero odd-offset taps (even offsets are zeroed)
        if (std::abs(k) % 2 == 1) {
            m_coeffs.push_back(h[i]);
        }
    }
}

extern "C" {
    auto create(std::string_view id) -> std::shared_ptr<composite::component> {
        return std::make_shared<halfrate>(id);
    }
}
