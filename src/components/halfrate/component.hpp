// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include <composite/composite.hpp>

#include <complex>
#include <cstdint>
#include <vector>

class halfrate : public composite::component {
    using cf32_t = std::complex<float>;
    using input_t = composite::mutable_buffer<cf32_t>;
    using output_t = composite::immutable_buffer<cf32_t>;
    using input_port_t = composite::input_port<input_t>;
    using output_port_t = composite::output_port<output_t>;
public:
    explicit halfrate(std::string_view);
    ~halfrate() override = default;

    auto property_change_handler() -> void override;
    auto process() -> composite::retval override;

private:
    auto generate_coeffs() -> void;

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    uint32_t m_filter_semi_length{3};
    std::string m_window_type{"HAMMING"};

    // Members
    std::vector<float> m_coeffs;
    float m_center_tap{};
    std::size_t m_taps_needed{};

    std::vector<cf32_t> m_even_lane;
    std::vector<cf32_t> m_odd_lane;
    std::size_t m_history_len{};

}; // class halfrate
