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

#include "f32_to_i16.hpp"
#include "i8_to_f32.hpp"
#include "i8_to_i16.hpp"
#include "i16_to_f32.hpp"

#include <aligned_mem_resource.hpp>

#include <algorithm>
#include <composite/component.hpp>
#include <complex>
#include <cstdint>
#include <memory_resource>
#include <spdlog/spdlog.h>
#include <vector>

template <typename T>
class type_converter : public composite::component {
    using input_t = std::pmr::vector<uint8_t>;
    using input_port_t = composite::input_port<std::shared_ptr<input_t>>;
    using output_t = std::pmr::vector<T>;
    using output_port_t = composite::output_port<std::shared_ptr<output_t>>;
    using enum composite::properties::config_type;
public:
    type_converter() : composite::component("type_converter") {
        add_port(&m_in_port);
        add_port(&m_out_port);
        add_property("m_output_endianness", &m_output_endianness).configurability(RUNTIME);
    }

    ~type_converter() override = default;

    auto property_change_handler() -> void override {
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
            m_metadata = meta.value();
            meta->format.bit_width = sizeof(T) * 8;
            if constexpr (std::is_floating_point_v<T>) {
                meta->format.type = composite::data_type::floating_point;
            } else if constexpr (std::is_integral_v<T>) {
                if constexpr (std::is_signed_v<T>) {
                    meta->format.type = composite::data_type::signed_integer;
                } else {
                    meta->format.type = composite::data_type::unsigned_integer;
                }
            }
        }

        // Create aligned output
        auto output = std::make_shared<std::pmr::vector<T>>(&m_allocator);
        output->resize(data->size());

        // Convert between input and output
        if constexpr (std::is_same_v<T, int8_t>) { // 8-bit output
            // TODO
        } else if constexpr (std::is_same_v<T, int16_t>) { // 16-bit output
            if (m_metadata.format.type == composite::data_type::signed_integer) {
                if (m_metadata.format.bit_width == 8) { // char-to-short
                    convert(
                        {reinterpret_cast<const int8_t*>(data->data()), data->size()},
                        *output,
                        m_metadata.format.is_complex
                    );
                }
            } else if (m_metadata.format.type == composite::data_type::floating_point) {
                if (m_metadata.format.bit_width == 32) { // float-to-short
                    convert(
                        {reinterpret_cast<const float*>(data->data()), data->size()},
                        *output,
                        m_metadata.format.is_complex
                    );
                }
            }
        } else if constexpr (std::is_same_v<T, float>) {
            if (m_metadata.format.type == composite::data_type::signed_integer) {
                if (m_metadata.format.bit_width == 8) { // char-to-float
                    convert(
                        {reinterpret_cast<const int8_t*>(data->data()), data->size()},
                        *output,
                        m_metadata.format.is_complex
                    );
                } else if (m_metadata.format.bit_width == 16) { // short-to-float
                    convert(
                        {reinterpret_cast<const int16_t*>(data->data()), data->size()},
                        *output,
                        m_metadata.format.is_complex
                    );
                }
            }
        }

        // Send updated metadata
        if (meta.has_value()) {
            m_out_port.send_metadata(meta.value());
        }

        // Send data and timestamp
        m_out_port.send_data(std::move(output), ts);
        
        return NORMAL;
    }

private:
    // Ports
    input_port_t m_in_port{"data_in"};
    output_port_t m_out_port{"data_out"};

    // Properties
    uint32_t m_output_size{};
    std::string m_output_endianness{"native"};

    // Members
    composite::metadata m_metadata;
    aligned_mem_resource m_allocator{64};

}; // class stov
