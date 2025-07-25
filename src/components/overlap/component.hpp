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

#include <aligned_mem.hpp>

#include <algorithm>
#include <composite/component.hpp>
#include <complex>
#include <cstdint>
#include <memory_resource>
#include <spdlog/spdlog.h>
#include <variant>
#include <vector>
#include <cmath>

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
        using enum composite::retval;
    
        auto [data, ts, meta] = m_in_port.get_data();
        if (!data) return NORMAL;
        
        if (meta.has_value() && m_metadata != meta.value()) {
            m_metadata = meta.value();
            // logger()->trace("received metadata:\n{}", meta->to_string());
                if (meta->sample_rate < 0.0) {
                    return NORMAL;
                }
            // if (meta->WHATEVER_VALUE_FOR_BREAK_IN_SEQUENCE_NUMBERS){
            //     reset_ring(ts);
            // }
            m_out_port.send_metadata(meta.value());
        }
        if (m_metadata.sample_rate == 0.0) {
            return NORMAL;
        }
    
        auto* input_ptr = data->data();
        auto input_len = data->size();
    
        // Initialize ring and state
        if (!m_ring_initialized) {
            m_ring_size = m_window_size * 2; // Headroom for wrap
            m_ring_initialized = true;
            m_ring_buf = aligned::make_aligned<std::complex<T>>(64, m_ring_size);
            reset_ring(ts);
            // m_overlap_count = static_cast<uint32_t>(
            //     std::round(m_window_size * m_overlap_percentage / 100.0)
            // );            
            // m_ring_head = 0;
            // m_ring_count = 0;
            // m_stride = m_window_size - m_overlap_count;
            // m_output_ts = ts;
            // m_next_window_start_sample = 0;
            // m_picoseconds_per_sample = (1. / m_metadata.sample_rate) * 1'000'000'000'000;
        }

        size_t packet_start_sample = m_total_samples_seen;
        size_t packet_end_sample = m_total_samples_seen + input_len;
        // Check if the window starting point lands inside this packet
        if (m_next_window_start_sample >= packet_start_sample &&
            m_next_window_start_sample < packet_end_sample) {
                const auto ps_per_sec = uint64_t{1'000'000'000'000};
                size_t offset_within_packet = m_next_window_start_sample - packet_start_sample;
                const auto sample_ps = offset_within_packet * m_picoseconds_per_sample;
                m_next_output_ts.seconds = ts.seconds;
                m_next_output_ts.picoseconds = ts.picoseconds + sample_ps;
                if (m_next_output_ts.picoseconds > ps_per_sec) {
                    m_next_output_ts.seconds += 1;
                    m_next_output_ts.picoseconds = m_next_output_ts.picoseconds % ps_per_sec;
                }
            }      

        size_t offset = 0;
        size_t samples_remaining = input_len;

        while (samples_remaining > 0) {
            size_t space_to_end = m_ring_size - m_ring_head;

            if (samples_remaining <= space_to_end) {
                // Contiguous write
                std::memcpy(
                    m_ring_buf->data() + m_ring_head,
                    input_ptr + offset,
                    samples_remaining * sizeof(std::complex<T>)
                );
                m_ring_head += samples_remaining;
                m_ring_count += samples_remaining;
                offset += samples_remaining;
                samples_remaining = 0;
            } else {
                // Wraparound write
                std::memcpy(
                    m_ring_buf->data() + m_ring_head,
                    input_ptr + offset,
                    space_to_end * sizeof(std::complex<T>)
                );
                std::memcpy(
                    m_ring_buf->data(),
                    input_ptr + offset + space_to_end,
                    (samples_remaining - space_to_end) * sizeof(std::complex<T>)
                );
                m_ring_head = samples_remaining - space_to_end;
                m_ring_count += samples_remaining;
                offset += samples_remaining;
                samples_remaining = 0;
            }

            m_ring_head %= m_ring_size;

            // Drain full windows
            while (m_ring_count >= m_window_size) {
                auto output = aligned::make_aligned<std::complex<T>>(64, m_window_size);
                size_t start = (m_ring_head + m_ring_size - m_window_size) % m_ring_size;

                if (start + m_window_size <= m_ring_size) {
                    std::memcpy(
                        output->data(),
                        m_ring_buf->data() + start,
                        m_window_size * sizeof(std::complex<T>)
                    );
                } else {
                    size_t first = m_ring_size - start;
                    size_t second = m_window_size - first;
                    std::memcpy(
                        output->data(),
                        m_ring_buf->data() + start,
                        first * sizeof(std::complex<T>)
                    );
                    std::memcpy(
                        output->data() + first,
                        m_ring_buf->data(),
                        second * sizeof(std::complex<T>)
                    );
                }

                m_out_port.send_data(std::move(output), m_output_ts);
                m_output_ts = m_next_output_ts;
                m_ring_count -= m_stride;
            }
        }

        return NORMAL;
    }

private:
    void reset_ring(composite::timestamp ts){
        m_overlap_count = static_cast<uint32_t>(
            std::round(m_window_size * m_overlap_percentage / 100.0)
        );            
        m_ring_head = 0;
        m_ring_count = 0;
        m_stride = m_window_size - m_overlap_count;
        m_output_ts = ts;
        m_next_window_start_sample = 0;
        m_total_samples_seen = 0;
        m_picoseconds_per_sample = (1. / m_metadata.sample_rate) * 1'000'000'000'000;
    }
    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Members
    uint32_t m_output_idx{};
    composite::metadata m_metadata;
    std::unique_ptr<overlap_t> m_output_buf;
    composite::timestamp m_output_ts;
    composite::timestamp m_next_output_ts;
    

    // Properties
    uint32_t m_window_size{65536};
    double m_overlap_percentage{50};
    uint32_t m_overlap_count = m_window_size * m_overlap_percentage / 100;
    // bool m_overlap_packet_boundary_checked{false};
    std::unique_ptr<aligned::aligned_mem<std::complex<T>>> m_ring_buf;
    std::size_t m_ring_size = 0;
    std::size_t m_ring_head = 0;
    std::size_t m_ring_count = 0; 
    std::size_t m_stride = 0; 
    bool m_ring_initialized = false;
    double m_sample_rate{};
    
    size_t m_next_window_start_sample = 0;
    size_t m_total_samples_seen{0};
    double m_picoseconds_per_sample{0};

}; // class overlap
