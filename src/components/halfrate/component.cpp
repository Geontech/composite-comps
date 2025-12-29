// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include "component.hpp"
#include "kernels.hpp"
#include <windows.hpp>

// AVX-512 requires 64-byte alignment
static constexpr std::size_t ALIGNMENT = 64;

halfrate::halfrate(std::string_view id) :
  composite::component(id),
  m_coeffs(ALIGNMENT, 0),
  m_even_lane(ALIGNMENT, 0),
  m_odd_lane(ALIGNMENT, 0) {
    add_port(&m_in_port);
    add_port(&m_out_port);
    add_property("filter_semi_length", m_filter_semi_length).change_listener([this]() {
        return m_filter_semi_length >= 1;
    });
    add_property("window", m_window_type).change_listener([this]() {
        return m_window_type.empty() || (m_window_type == "BLACKMAN_HARRIS") || (m_window_type == "HAMMING");
    });
}

auto halfrate::property_change_handler() -> void {
    // Generate filter coefficients
    generate_coeffs();

    // 1. Calculate history requirements
    m_history_len = std::max(m_coeffs.size(), std::size_t{m_filter_semi_length - 1});

    // 2. Pre-allocate memory
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

    // Efficient zeroing
    std::memset(m_even_lane.data(), 0, m_history_len * sizeof(cf32_t));
    std::memset(m_odd_lane.data(), 0, m_history_len * sizeof(cf32_t));
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
    // aligned_mem::resize is smart; it won't realloc if capacity is sufficient.
    if (m_even_lane.size() < m_history_len + n_out) {
        m_even_lane.resize(m_history_len + n_out);
        m_odd_lane.resize(m_history_len + n_out);
    }

    // 2. De-Interleave with AVX-enabled kernel
    // Note: We write *after* the history
    auto* even_ptr = m_even_lane.data() + m_history_len;
    auto* odd_ptr = m_odd_lane.data() + m_history_len;
    kernels::deinterleave_block(data.data(), even_ptr, odd_ptr, n_out);

    // 3. Process vertical filter kernel
    auto delay_offset = m_filter_semi_length;
    kernels::halfband_filter_vertical(
        m_even_lane.data(),
        m_odd_lane.data(),
        m_coeffs.data(), m_coeffs.size(),
        m_center_tap, delay_offset,
        data.data(), // Write output directly back to input buffer (in-place safe for decimation)
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

    // Send data (with updated metadata if present)
    data.resize(n_out);
    if (meta) {
        // Update metadata for decimation: sample rate is halved
        auto updated_meta = *meta;
        updated_meta.sample_rate = updated_meta.sample_rate / 2.0;
        m_out_port.send_metadata(updated_meta);
    }
    m_out_port.send_data(std::move(data).to_immutable(), ts);

    return NORMAL;
}

auto halfrate::generate_coeffs() -> void {
    auto L = 4 * m_filter_semi_length + 1; // full length
    auto center = L / 2;

    // 1. Pre-calculate the exact number of coefficients we will store.
    // We only store the non-zero odd taps.
    // Taps are at indices [0 ... 4*M]. Center is 2*M.
    // We skip center. We skip evens.
    // Indices i != center where (i - center) is odd.
    std::size_t num_coeffs = 0;
    for (int i = 0; i < L; ++i) {
        int k = i - center;
        if (k == 0) { continue; }
        if (std::abs(k) % 2 == 1) { num_coeffs++; }
    }

    // 2. Resize aligned memory once (avoids push_back)
    m_coeffs.resize(num_coeffs);

    auto window = (m_window_type == "BLACKMAN_HARRIS")
        ? windows::blackman_harris<float>(L, false)
        : windows::hamming<float>(L, false);

    // 3. Fill coefficients
    // Iterate backwards to match convolution order
    std::size_t write_idx = 0;

    for (int i = L - 1; i >= 0; --i) {
        int k = i - center;
        if (k == 0) {
            m_center_tap = 0.5f; // Center tap is implicit
            continue;
        }

        // Calculate and store only if it's an odd offset
        if (std::abs(k) % 2 == 1) {
            auto sinc_val = std::sin(0.5f * std::numbers::pi_v<float> * k) / (std::numbers::pi_v<float> * k);
            float val = sinc_val * window->at(i);

            // Direct access - no push_back
            m_coeffs[write_idx++] = val;
        }
    }
}

extern "C" {
    auto create(std::string_view id) -> std::shared_ptr<composite::component> {
        return std::make_shared<halfrate>(id);
    }
}
