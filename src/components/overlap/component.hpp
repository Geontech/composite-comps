/*
* Copyright (C) 2024 Geon Technologies, LLC
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
* along with this program.  If not, see http://www.gnu.org/licenses/.
*/
#include "convert.hpp"
#include <aligned_mem.hpp>
#include <readerwriterqueue.h>

#include <algorithm>
#include <composite/component.hpp>
#include <complex>
#include <cstdint>
#include <memory_resource>
#include <spdlog/spdlog.h>
#include <variant>
#include <vector>
#include <cmath>  // for std::fmod and std::round

#include <chrono>

template <typename T>
class overlap : public composite::component {
    using overlap_t = aligned::aligned_mem<std::complex<T>>;
    using input_port_t = composite::input_port<std::unique_ptr<overlap_t>>;
    using output_port_t = composite::output_port<std::unique_ptr<overlap_t>>;
    using enum composite::properties::config_type;
public:
    overlap() : composite::component("overlap") {
        add_port(&m_in_port);
        add_port(&m_out_port);
        // add_property("byteswap", &m_byteswap).configurability(RUNTIME);
        add_property("window_size", &m_window_size);
        add_property("overlap_percentage", &m_overlap_percentage);
    }

    ~overlap() override = default;

    auto property_change_handler() -> void override {
        logger()->trace("window_size: {}", m_window_size);
        logger()->trace("overlap_percentage: {}", m_overlap_percentage);
        m_in_port.clear();
    }

    auto process() -> composite::retval override {
        // using namespace std::chrono;
        using enum composite::retval;
    
        auto [data, ts, meta] = m_in_port.get_data();
        if (!data) return NORMAL;
    
        if (meta.has_value()) {
            if (m_metadata != meta.value()) {
                m_metadata = meta.value();
                // if packet_drop
                // m_output_idx = 0
                m_out_port.send_metadata(m_metadata);
            }
        }
    
        auto* input_ptr = data->data();
        auto input_len = data->size();
        size_t offset = 0;

        if (!m_overlap_packet_boundary_checked){
            logger()->info("Checking packet boundary");
            check_overlap_alignment(input_len);
            m_overlap_packet_boundary_checked = true;
        }

        while (offset < input_len) {
            if (!m_output_buf) {
                m_output_buf = aligned::make_aligned<typename overlap_t::value_type>(64, m_window_size);
                m_output_idx = 0;
                m_output_ts = ts;
            }
            auto space_left = m_window_size - m_output_idx;
            auto copy_count = std::min<std::size_t>(
                static_cast<std::size_t>(space_left),
                input_len - offset
            );

            std::memcpy(
                m_output_buf->data() + m_output_idx,
                input_ptr + offset,
                copy_count * sizeof(typename overlap_t::value_type)
            );

            if (m_output_idx == m_overlap_count){
                m_next_frame_ts = ts;
            }
            m_output_idx += copy_count;
            offset += copy_count;
            

            if (m_output_idx == m_window_size) {
                auto new_buf = aligned::make_aligned<typename overlap_t::value_type>(64, m_window_size);

                std::memcpy(
                    new_buf->data(),
                    m_output_buf->data() + (m_window_size - m_overlap_count),
                    m_overlap_count * sizeof(typename overlap_t::value_type)
                );                

                m_out_port.send_data(std::move(m_output_buf), m_output_ts);

                m_output_buf = std::move(new_buf);
                m_output_idx = m_overlap_count;
                m_output_ts = m_next_frame_ts;
            }
        }
    
        return NORMAL;
    }

    

private:
    void check_overlap_alignment(uint32_t input_len) {
        double percentage = m_overlap_percentage / 100.0;
        double overlap_samples = percentage * static_cast<double>(m_window_size);

        // Only adjust if not already aligned
        if (std::fmod(overlap_samples, static_cast<double>(input_len)) != 0.0) {
            // Compute nearest higher alignment
            double remainder = std::fmod(overlap_samples, static_cast<double>(input_len));
            double adjusted_overlap = overlap_samples - remainder + input_len;

            m_overlap_percentage = (adjusted_overlap / static_cast<double>(m_window_size)) * 100.0;

            logger()->info("Auto-adjusted overlap from {:.4f}% to {:.4f}% for packet-aligned stride ({} samples)",
                        percentage * 100.0,
                        m_overlap_percentage,
                        input_len);
        } else {
            logger()->info("Overlap {:.4f}% already aligned to packet stride ({} samples)",
                        percentage * 100.0,
                        input_len);
        }

        // Always recalculate overlap_count
        m_overlap_count = static_cast<uint32_t>(
            std::round(static_cast<double>(m_window_size) * (m_overlap_percentage / 100.0))
        );
    }

    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Members
    uint32_t m_output_idx{};
    composite::metadata m_metadata;
    std::unique_ptr<overlap_t> m_output_buf;
    composite::timestamp m_output_ts;
    composite::timestamp m_next_frame_ts;
    

    // Properties
    uint32_t m_window_size{65536};
    double m_overlap_percentage{50};
    uint32_t m_overlap_count = m_window_size * m_overlap_percentage / 100;
    bool m_overlap_packet_boundary_checked{false};

}; // class overlap
