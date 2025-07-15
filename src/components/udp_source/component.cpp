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
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

#include "component.hpp"
#include "overlay.hpp"
#include "socket/dpdk.hpp"
#include "socket/packet_mmap.hpp"
#include "socket/recvmmsg.hpp"

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
#include <stdexcept>

udp_source::udp_source() : composite::component("udp_source") {
    add_port(&m_out_port);
    using enum composite::properties::config_type;
    add_property("socket_type", &m_socket_type).change_listener([this]() {
        return m_socket_type == RECVMMSG
            || m_socket_type == PACKET_MMAP
            || m_socket_type == DPDK;
    });
    add_property("interface", &m_interface).configurability(RUNTIME);
    add_property("ip_addr", &m_ip_addr).configurability(RUNTIME);
    add_property("port", &m_port).configurability(RUNTIME);
    add_property("recv_buf_size", &m_recv_buf_size).units("bytes");
    add_property("num_msgs", &m_num_msgs).configurability(RUNTIME);
    add_property("frame_count", &m_frame_count).configurability(RUNTIME);
    add_struct_property("signal_overrides", &m_signal_overrides, [this](auto& set, auto* prop) {
        set.add_property("center_frequency", &prop->center_frequency);
        set.add_property("bandwidth", &prop->bandwidth);
        set.add_property("sample_rate", &prop->sample_rate);
        set.add_property("is_complex", &prop->is_complex);
        set.add_property("msg_size", &prop->msg_size).units("bytes");
        set.add_property("transport", &prop->transport).change_listener([this]() {
            return (m_signal_overrides.transport == "sdds") || (m_signal_overrides.transport == "vita49");
        });
    });
    add_property("socket_mem", &m_socket_mem).configurability(RUNTIME);
    add_property("rx_ring_size", &m_rx_ring_size).configurability(RUNTIME);
    add_property("num_mbufs", &m_num_mbufs).configurability(RUNTIME);
    add_property("mbuf_cache_size", &m_mbuf_cache_size).configurability(RUNTIME);
    add_property("burst_size", &m_burst_size).configurability(RUNTIME);
    add_property("write_pcap", &m_write_pcap);
    add_property("packets_per_pcap", &m_packets_per_pcap);
    add_property("m_dir_size", &m_dir_size);
}

auto udp_source::property_change_handler() -> void {
    logger()->trace(std::source_location::current().function_name());
    m_receiver.reset();
    auto config = udp::config{
        .logger = logger(),
        .interface = m_interface,
        .ip_addr = m_ip_addr,
        .port = m_port,
        .recv_buf_size = m_recv_buf_size,
        .batch_size = m_num_msgs,
        .frame_count = m_frame_count,
        .rx_ring_size = m_rx_ring_size,
        .num_mbufs = m_num_mbufs,
        .mbuf_cache_size = m_mbuf_cache_size,
        .burst_size = m_burst_size,
        .socket_mem = m_socket_mem,
        .write_pcap = m_write_pcap,
        // .dir_size_GB = m_dir_size,
        // .write_directory = m_writer_directory,
        // .packets_per_pcap = m_packets_per_pcap
    };
    if (!m_signal_overrides.transport.empty()) {
        config.transport = m_signal_overrides.transport;
    }
    if (m_signal_overrides.msg_size.has_value()) {
        config.msg_size = m_signal_overrides.msg_size.value();
    }
    if (m_socket_type == PACKET_MMAP) {
        m_receiver = std::make_unique<udp::packet_mmap>(config);
    } else if (m_socket_type == DPDK) {
        try {
            m_receiver = std::make_unique<udp::dpdk_udp>(config);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to initialize dpdk source");
        }
    } else { // recvmmsg
        m_receiver = std::make_unique<udp::recvmmsg>(config);
    }
}

auto udp_source::start() -> void {
    component::start();
    m_receiver->start_recv();
    m_stat_thread = std::jthread([this](std::stop_token token) {
        while (!token.stop_requested()) {
            
            std::this_thread::sleep_for(std::chrono::seconds(5));
            auto stats = m_receiver->get_stats();
            if (m_socket_type == DPDK) {
                logger()->trace("Statistics: pkts_recvd_user={}, pkts_rcvd_nic={}, pkts_dropped_nic={}, rx_nombuf={}, cycles_per_packet={}, pkts_per_burst={}, iterations={}, total_nb_rx={}, no_queue={}",
                                stats.pkts_recvd_user, stats.pkts_recvd_nic, stats.pkts_dropped_nic, stats.rx_nombuf, stats.cycles_per_packet, stats.avg_pkts_per_burst, stats.iterations, stats.total_nb_rx, stats.no_queue);
            } else {
                stats.pkts_processed = m_pkts_processed.exchange(0);
                logger()->trace(
                    "statistics: pkts_recvd_user={}, pkts_recvd_kernel={}, pkts_dropped_kernel={}",
                    stats.pkts_recvd_user, stats.pkts_recvd_kernel, stats.pkts_dropped_kernel);
            }
        }
    });
    pthread_setname_np(m_stat_thread.native_handle(), "udp_stats");
}

auto udp_source::stop() -> void {
    m_stat_thread.request_stop();
    if (m_stat_thread.joinable()) {
        m_stat_thread.join();
    }
    m_receiver->stop_recv();
    component::stop();
}

auto udp_source::process() -> composite::retval {
    using enum composite::retval;

    // Get data from intermediate queue
    auto data = output_t{nullptr};
    if (!m_receiver->get_data(data)) {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        return NORMAL;
    } else if (data == nullptr) {
        return NORMAL;
    }

    // Parse packets based on protocol
    auto meta = m_metadata;
    auto ts = composite::timestamp{};
    if (m_receiver->get_transport() == udp::transport::sdds) {
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
    } else if (m_receiver->get_transport() == udp::transport::vita49) {
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
            // std::copy(data->begin() + packet.payload_start(), data->end(), data->begin()); // move metadata off
            std::copy(data->begin() + packet.payload_start(), data->end(), data->begin());
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
    m_pkts_processed.fetch_add(1, std::memory_order_relaxed);

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
        return std::make_shared<udp_source>();
    }
}
