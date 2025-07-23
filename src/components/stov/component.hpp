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

#include <algorithm>
#include <composite/component.hpp>
#include <complex>
#include <cstdint>
#include <memory_resource>
#include <spdlog/spdlog.h>
#include <variant>
#include <vector>

template <typename T>
class stov : public composite::component {
    using input_t = std::pmr::vector<uint8_t>;
    using input_port_t = composite::input_port<std::shared_ptr<input_t>>;
    using output_t = aligned::aligned_mem<T>;
    using output_port_t = composite::output_port<std::unique_ptr<output_t>>;
    using enum composite::properties::config_type;
public:
    stov() : composite::component("stov") {
        add_port(&m_in_port);
        add_port(&m_out_port);
        add_property("output_size", &m_output_size).configurability(RUNTIME);
        add_property("byteswap", &m_byteswap).configurability(RUNTIME);
    }

    ~stov() override = default;

    auto property_change_handler() -> void override {
        if constexpr (std::is_same_v<T, std::complex<float>>) {
            m_converter = converter<int16_t, float>(m_byteswap);
        } else {
            m_converter = converter<int16_t, int16_t>(m_byteswap);
        }
        // flush input port
        m_in_port.clear();
    }

    auto process() -> composite::retval override {
        using enum composite::retval;
        auto [data, ts, meta] = m_in_port.get_data();
        if (data == nullptr) {
            return NORMAL;
        }
        if (meta.has_value()) {
            logger()->trace("received metadata:\n{}", meta->to_string());
            if (m_byteswap) {
                meta->format.endianness = (meta->format.endianness == std::endian::big) ? std::endian::little : std::endian::big;
            }
            meta->format.bit_width = sizeof(T) * 8;
            if (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<int16_t>>) {
                meta->format.is_complex = true;
                meta->format.bit_width = meta->format.bit_width / 2;
            }
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) {
                meta->format.type = composite::data_type::floating_point;
            } else {
                meta->format.type = composite::data_type::signed_integer;
            }
            if (m_metadata != meta.value()) {
                m_metadata = meta.value();
                logger()->trace("sending updated metadata:\n{}", m_metadata.to_string());
                m_out_port.send_metadata(m_metadata);
            }
        }
        auto bit_len = __builtin_cpu_supports("avx512f") ? std::size_t{512} : std::size_t{256};
        auto stride = size_t{bit_len / 8u / sizeof(T)};
        auto data_ci16 = reinterpret_cast<std::complex<int16_t>*>(data->data());
        auto data_ci16_len = data->size() / sizeof(std::complex<int16_t>);
        for (auto i=0u; i < data_ci16_len; i += stride) {
            if (m_output_buf == nullptr) {
                m_output_buf = aligned::make_aligned<typename output_t::value_type>(64, m_output_size);
                if (i == 0) {
                    m_output_ts = ts;
                } else {
                    // calculate timestamp
                    const auto ps_per_sec = uint64_t{1'000'000'000'000};
                    const auto xdelta = 1. / m_metadata.sample_rate;
                    const auto sample_ps = i * xdelta * ps_per_sec;
                    m_output_ts.seconds = ts.seconds;
                    m_output_ts.picoseconds = ts.picoseconds + sample_ps;
                    if (m_output_ts.picoseconds > ps_per_sec) {
                        m_output_ts.seconds += 1;
                        m_output_ts.picoseconds = m_output_ts.picoseconds % ps_per_sec;
                    }
                    logger()->info("xdelta: {}, curr_idx={}, ts={}.{} sample_ps={} new_ts={}.{}", xdelta, i, ts.seconds, ts.picoseconds, sample_ps, m_output_ts.seconds, m_output_ts.picoseconds);
                }
            }
            if constexpr (std::is_same_v<T, std::complex<float>>) {
                std::get<0>(m_converter).process(
                    reinterpret_cast<const int16_t*>(data_ci16 + i),
                    reinterpret_cast<float*>(m_output_buf->data() + m_output_idx)
                );
            } else { // int16_t or std::complex<int16_t>
                std::get<1>(m_converter).process(
                    reinterpret_cast<const int16_t*>(data_ci16 + i),
                    reinterpret_cast<int16_t*>(m_output_buf->data() + m_output_idx)
                );
            }
            m_output_idx += stride;
            if (m_output_idx == m_output_size) {
                m_out_port.send_data(std::move(m_output_buf), m_output_ts);
                m_output_buf.reset();
                m_output_idx = 0;
            }
        }
        return NORMAL;
    }

private:
    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    uint32_t m_output_size{};
    bool m_byteswap{true};

    // Members
    std::variant<converter<int16_t, float>, converter<int16_t, int16_t>, converter<float, float>> m_converter;
    std::unique_ptr<output_t> m_output_buf;
    composite::metadata m_metadata;
    uint32_t m_output_idx{};
    composite::timestamp m_output_ts;
    uint8_t m_pkt_count{};

}; // class stov
