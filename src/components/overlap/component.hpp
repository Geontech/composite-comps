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

#include <typeinfo>

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
        add_property("window_size", &m_window_size).configurability(RUNTIME);
        add_property("overlap_percentage", &m_overlap_percentage).configurability(RUNTIME);
    }

    ~overlap() override = default;

    auto property_change_handler() -> void override {
        logger()->trace("window_size: {}", m_window_size);
        logger()->trace("overlap_percentage: {}", m_overlap_percentage);
        m_output_buf.reset();
        m_in_port.clear();
    }

    auto process() -> composite::retval override {
        using enum composite::retval;
    
        auto [data, ts, meta] = m_in_port.get_data();
        if (!data) return NORMAL;
        
        if (meta.has_value() && m_metadata != meta.value()) {
            m_metadata = meta.value();
            logger()->trace("RX metadata:\n{}", m_metadata.to_string());
            if (meta->sample_rate <= 0.0) return NORMAL;
            if (meta->sample_rate != m_sample_rate){
                m_sample_rate = meta->sample_rate;
                m_picoseconds_per_sample = (1. / m_metadata.sample_rate) * 1'000'000'000'000;
            }
            auto it = m_metadata.annotations.find("contiguous");
            if (it != m_metadata.annotations.end() && it->second == "false") {
                reset_state(ts);
                logger()->warn("Non-contiguous stream detected — resetting overlap state for next window.");
            } else {
                m_contiguous = true;
            }
            m_out_port.send_metadata(meta.value());
        }
     
        if (m_metadata.sample_rate == 0.0) return NORMAL;
    
        if (!m_output_buf || !m_contiguous) {
            reset_state(ts);
        }
    
        auto* input_ptr = data->data();
        size_t input_len = data->size();
        size_t offset = 0;
    
        const size_t packet_start_sample = m_total_samples_seen;
        const size_t packet_end_sample   = m_total_samples_seen + input_len;
    
        if (m_next_window_start_sample >= packet_start_sample &&
            m_next_window_start_sample <  packet_end_sample) {
            constexpr uint64_t ps_per_sec = 1'000'000'000'000ULL;
            const size_t offset_within_packet = m_next_window_start_sample - packet_start_sample;
            const uint64_t sample_ps = static_cast<uint64_t>(offset_within_packet * m_picoseconds_per_sample);
    
            m_next_output_ts.seconds     = ts.seconds;
            m_next_output_ts.picoseconds = ts.picoseconds + sample_ps;
            if (m_next_output_ts.picoseconds >= ps_per_sec) {
                m_next_output_ts.seconds += 1;
                m_next_output_ts.picoseconds %= ps_per_sec;
            }
        }
    
        while (offset < input_len) {
            size_t space_left = m_window_size - m_output_buf_count;
            size_t to_copy = std::min(space_left, input_len - offset);
    
            std::memcpy(
                m_output_buf->data() + m_output_buf_count,
                input_ptr + offset,
                to_copy * sizeof(std::complex<T>)
            );
    
            m_output_buf_count += to_copy;
            offset += to_copy;
    
            while (m_output_buf_count >= m_window_size) {
                auto new_buf = aligned::make_aligned<std::complex<T>>(64, m_window_size);
                std::memcpy(
                    new_buf->data(),
                    m_output_buf->data() + m_stride,
                    m_overlap_count * sizeof(std::complex<T>)
                );
    
                size_t available_after = input_len - offset;
                size_t room_after_overlap = m_window_size - m_overlap_count;
                size_t take_now = std::min(available_after, room_after_overlap);
    
                if (take_now > 0) {
                    std::memcpy(
                        new_buf->data() + m_overlap_count,
                        input_ptr + offset,
                        take_now * sizeof(std::complex<T>)
                    );
                    offset += take_now;
                }
    
                m_out_port.send_data(std::move(m_output_buf), m_output_ts);
    
                m_output_ts = m_next_output_ts;
                m_next_window_start_sample += m_stride;
    
                m_output_buf = std::move(new_buf);
                m_output_buf_count = m_overlap_count + take_now;
            }
        }
        m_total_samples_seen += input_len;
    
        return NORMAL;
    }
    

private:
    void reset_state(composite::timestamp ts) {
        m_overlap_count = static_cast<uint32_t>(
            std::round(m_window_size * m_overlap_percentage / 100.0)
        );            
        m_stride = m_window_size - m_overlap_count;
        m_output_ts = ts;
        if (!m_output_buf){
            m_output_buf = aligned::make_aligned<std::complex<T>>(64, m_window_size);
        }
        m_output_buf_count = 0;
        m_contiguous = false;
        m_total_samples_seen = 0;
        m_next_window_start_sample = 0;
    }
    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Members
    uint32_t m_output_idx{};
    composite::metadata m_metadata;
    std::unique_ptr<overlap_t> m_output_buf;
    std::size_t m_output_buf_count = 0;
    composite::timestamp m_output_ts;
    composite::timestamp m_next_output_ts;
    bool m_contiguous = true;

    // Properties
    uint32_t m_window_size{65536};
    double m_overlap_percentage{50};
    uint32_t m_overlap_count = static_cast<uint32_t>( std::round(m_window_size * m_overlap_percentage / 100.0));    
    std::size_t m_stride = 0; 
    double m_sample_rate{};
    
    size_t m_next_window_start_sample = 0;
    size_t m_total_samples_seen{0};
    double m_picoseconds_per_sample{0};

}; // class overlap
