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
        // Min 1, max 128 (kernel limit: history_len = 2*semi_length <= 256)
        return m_filter_semi_length >= 1 && m_filter_semi_length <= 128;
    });
    add_property("window", m_window_type).change_listener([this]() {
        return m_window_type.empty() || (m_window_type == "BLACKMAN_HARRIS") || (m_window_type == "HAMMING");
    });
}

auto halfrate::property_change_handler() -> void {
    // Generate filter coefficients
    generate_coeffs();

    // Calculate history requirements
    // Need max of: num_taps (for FIR), filter_semi_length (for delay)
    m_history_len = std::max(m_coeffs.size(), static_cast<std::size_t>(m_filter_semi_length));

    // Resize history buffers (fused kernel only needs history_len, not full working set)
    m_even_lane.resize(m_history_len);
    m_odd_lane.resize(m_history_len);

    // Zero-initialize history for clean startup
    std::memset(m_even_lane.data(), 0, m_history_len * sizeof(cf32_t));
    std::memset(m_odd_lane.data(), 0, m_history_len * sizeof(cf32_t));
}

auto halfrate::process() -> composite::retval {
    using enum composite::retval;
    auto [data, ts, meta] = m_in_port.get_data();
    if (!data) {
        return NORMAL;
    }

    const auto n_in = data.size();
    const auto n_out = n_in / 2;

    // Fused kernel handles deinterleave + filter + history update in one pass
    // History buffers only need to hold history_len samples (not full working set)
    kernels::halfband_filter_fused(
        data.data(),                          // Interleaved input
        m_even_lane.data(),                   // Even history buffer
        m_odd_lane.data(),                    // Odd history buffer
        m_coeffs.data(), m_coeffs.size(),
        m_center_tap, m_filter_semi_length,   // delay_offset = filter_semi_length
        data.data(),                          // Output (in-place safe for decimation)
        n_out,
        m_history_len
    );

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
