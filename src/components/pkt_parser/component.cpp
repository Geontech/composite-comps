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

#include "component.hpp"
#include "overlay.hpp"

#include <arpa/inet.h>
#include <array>
#include <complex>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <netinet/in.h>
#include <net/if.h>
#include <source_location>
#include <spdlog/spdlog.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

pkt_parser::pkt_parser() : composite::component("pkt_parser") {
    add_port(&m_in_port);
    add_port(&m_out_port);
    using enum composite::properties::config_type;
    add_struct_property("signal_overrides", &m_signal_overrides, [this](auto& set, auto* prop) {
        set.add_property("center_frequency", &prop->center_frequency);
        set.add_property("bandwidth", &prop->bandwidth);
        set.add_property("sample_rate", &prop->sample_rate);
        set.add_property("is_complex", &prop->is_complex);
        set.add_property("transport", &prop->transport).change_listener([this]() {
            return (m_signal_overrides.transport == "sdds") || (m_signal_overrides.transport == "vita49");
        });
    });
}

auto pkt_parser::property_change_handler() -> void {
    logger()->trace(std::source_location::current().function_name());
    if (m_signal_overrides.transport == "sdds") {
        m_transport = transport::sdds;
    } else if (m_signal_overrides.transport == "vita49") {
        m_transport = transport::vita49;
    } else {
        m_transport = transport::unknown;
    }
}

auto pkt_parser::process() -> composite::retval {
    using enum composite::retval;

    // Get input data if available
    auto [data, _, __] = m_in_port.get_data();
    if (data == nullptr) {
        return NORMAL;
    }

    // Have we determined the protocol?
    if (m_transport == transport::unknown) {

    }

    // Parse packets based on protocol
    auto meta = m_metadata;
    auto ts = composite::timestamp{};
    if (m_transport == transport::sdds) {
        auto packet = overlay::sdds::overlay(*data);
        auto seq_num = packet.seq_num();
        if (packet.pp_id() && ((seq_num % 32) != 31)) [[unlikely]] {
            logger()->error("invalid SDDS packet received, pp_id=true, seq_num={}", seq_num);
        } else if (!packet.pp_id() && ((seq_num % 32) == 31)) [[unlikely]] {
            logger()->error("invalid SDDS packet received pp_id=false, seq_num={}", seq_num);
        }
        auto expected_seq_num = static_cast<uint16_t>(m_pkt_count + 1);
        if ((expected_seq_num % 32) == 31) {
            ++expected_seq_num;
        }
        if (seq_num != expected_seq_num) [[unlikely]] {
            logger()->warn("dropped pkt(s) expected={}, got={}", expected_seq_num, seq_num);
        }
        m_pkt_count = seq_num;
        meta.format.is_complex = packet.complex();
        meta.format.type = composite::data_type::signed_integer;
        meta.format.endianness = std::endian::big;
        meta.format.bit_width = packet.bps();
        meta.sample_rate = packet.sample_rate();
        if (m_signal_overrides.is_complex.has_value()) {
            meta.format.is_complex = m_signal_overrides.is_complex.value();
        }
        if (m_signal_overrides.center_frequency.has_value()) {
            meta.center_frequency = m_signal_overrides.center_frequency.value();
        }
        if (m_signal_overrides.bandwidth.has_value()) {
            meta.bandwidth = m_signal_overrides.bandwidth.value();
        }
        if (m_signal_overrides.sample_rate.has_value()) {
            meta.sample_rate = m_signal_overrides.sample_rate.value();
        }
        meta.annotations["protocol"] = "sdds";
        ts = composite::timestamp{packet.secs(), packet.psecs()};
        std::copy(data->begin() + 56, data->end(), data->begin()); // move metadata off
        data->resize(1024);
    } else if (m_transport == transport::vita49) {
        auto packet = overlay::v49::overlay(*data);
        if (packet.is_data()) [[likely]] {
            auto& header = packet.header();
            if (auto expected_count = ((m_pkt_count + 1) % 16); header.packet_count() != expected_count) {
                logger()->warn("dropped pkt(s) expected={}, got={}", expected_count, header.packet_count());
            }
            m_pkt_count = header.packet_count();
            if (auto int_ts = packet.integer_timestamp()) {
                ts.seconds = int_ts.value();
            }
            if (auto frac_ts = packet.fractional_timestamp()) {
                ts.picoseconds = frac_ts.value();
            }
            std::copy(data->begin() + packet.payload_start(), data->end(), data->begin()); // move metadata off
            data->resize(packet.payload_size());
        } else if (packet.is_context()) {
            if (auto format = packet.signal_data_format()) {
                meta.format.is_complex = format->real_complex_type() != vrtgen::packing::DataSampleType::REAL;
                if (std::to_underlying(format->data_item_format()) <= 0x07) { // signed enumerations
                    meta.format.type = composite::data_type::signed_integer;
                } else if (std::to_underlying(format->data_item_format()) >= 0x10) { // unsigned enumerations
                    meta.format.type = composite::data_type::unsigned_integer;
                } else {
                    meta.format.type = composite::data_type::floating_point;
                }
                meta.format.bit_width = format->data_item_size();
                meta.format.endianness = packet.endianness();
            }
            meta.center_frequency = packet.rf_frequency().value_or(0);
            meta.bandwidth = packet.bandwidth().value_or(0);
            meta.sample_rate = packet.sample_rate().value_or(0);
        }
        if (m_signal_overrides.is_complex.has_value()) {
            meta.format.is_complex = m_signal_overrides.is_complex.value();
        }
        if (m_signal_overrides.center_frequency.has_value()) {
            meta.center_frequency = m_signal_overrides.center_frequency.value();
        }
        if (m_signal_overrides.bandwidth.has_value()) {
            meta.bandwidth = m_signal_overrides.bandwidth.value();
        }
        if (m_signal_overrides.sample_rate.has_value()) {
            meta.sample_rate = m_signal_overrides.sample_rate.value();
        }
        meta.annotations["protocol"] = "v49";
    }

    // Send metadata on changes
    if (m_metadata != meta) {
        m_metadata = meta;
        logger()->trace("sending updated metadata:\n{}", m_metadata.to_string());
        m_out_port.send_metadata(m_metadata);
        m_init_metadata = true;
    }

    // Send data
    if (m_init_metadata) [[likely]] {
        m_out_port.send_data(std::move(data), ts);
    }

    return NORMAL;
}

extern "C" {
    auto create() -> std::shared_ptr<composite::component> {
        return std::make_shared<pkt_parser>();
    }
}
