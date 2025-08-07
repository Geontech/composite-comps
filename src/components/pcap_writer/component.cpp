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

#include "component.hpp"
#include <nlohmann/json.hpp>
#include <pcap/pcap.h>
#include <composite/component.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <filesystem>
#include <utility>
#include <helpers.hpp>

pcap_writer::pcap_writer() : composite::component("pcap_writer") {
    add_port(&m_in_port);
    add_property("rotate_every", &m_rotate_every);
    add_property("write_dir", &m_write_dir);    
    add_property("duration_seconds", &m_duration_seconds);
    add_property("mission", &m_mission);
}

pcap_writer::~pcap_writer() {
    close_file();
}

auto pcap_writer::process() -> composite::retval {
    using enum composite::retval;

    auto [data, ts, meta] = m_in_port.get_data();

    if (m_expired) {
        return NORMAL;
    }

    if (meta.has_value() && m_metadata != meta.value()) {
        composite::metadata temp_meta = meta.value();
        logger()->trace("metadata:\n{}", temp_meta.to_string());
        if (temp_meta.sample_rate == 0.0) {
            // Skip writing if metadata is not set
            return NORMAL;
        }
        m_metadata = meta.value();
        m_meta = sigmf::build_sigmf_json(m_metadata);
        m_meta["global"]["rfnext:mission"] = m_mission;
        logger()->trace("metadata:\n{}", m_metadata.to_string());
        m_started = true;
        m_start_time = std::chrono::steady_clock::now();
    }
    if (!m_started) {
        return NORMAL;
    }
    if (!data) return NORMAL;
    
    if (!m_dumper) {
        open_new_file();
        m_current_file_first_ts = ts;
    }
    if (m_current_file_first_ts.seconds == 0){
        logger()->trace("First packet is context, replacing timestamp with next available");
        m_current_file_first_ts = ts;
    }
    m_current_file_last_ts = ts;

    gettimeofday(&hdr.ts, nullptr);
    hdr.caplen = data->size();
    hdr.len = data->size();

    pcap_dump(reinterpret_cast<u_char*>(m_dumper), &hdr, data->data());
    ++m_counter;

    if (m_counter >= m_rotate_every) {
        close_file();
        // m_current_file_first_ts.seconds = 0;
        // open_new_file();
        m_counter = 0;
    }
    // Check for duration expiry
    if (m_duration_seconds > 0) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_start_time).count();
        if (elapsed >= m_duration_seconds) {
            close_file();
            m_expired = true;
            logger()->info("No longer writing pcaps");
            return NORMAL;
        }
    }
    return NORMAL;
}

void pcap_writer::open_new_file() {
    namespace fs = std::filesystem;

    std::stringstream filename;
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    filename // << 
            // "capture_"
             << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S")
             << "_" << m_file_number++ << ".pcap";

    fs::path full_path = fs::path(m_write_dir) / filename.str();
    m_current_filename = full_path.string();
    

    m_pcap = pcap_open_dead(DLT_EN10MB, 65535);
    if (!m_pcap) {
        throw std::runtime_error("Failed to open pcap_dead handle");
    }

    m_dumper = pcap_dump_open(m_pcap, m_current_filename.c_str());
    if (!m_dumper) {
        pcap_close(m_pcap);
        m_pcap = nullptr;
        throw std::runtime_error("Failed to open pcap file: " + m_current_filename);
    }
}

void pcap_writer::close_file() {
    if (!m_current_filename.empty()) {
        std::ostringstream key_ss;
        key_ss << m_current_file_first_ts.seconds << "." << std::setw(12) << std::setfill('0') << m_current_file_first_ts.picoseconds;
        std::string key = key_ss.str();

        std::ostringstream ts_end_ss;
        ts_end_ss << m_current_file_last_ts.seconds << "." << std::setw(12) << std::setfill('0') << m_current_file_last_ts.picoseconds;
        std::string ts_end = ts_end_ss.str();
        m_timestamp_map[key] = {
            {"filename", std::filesystem::path(m_current_filename).filename().string()},
            {"ts_end", ts_end}
        };

        m_meta["global"]["rfnext:timestamp"] = m_timestamp_map;

        std::ofstream out(std::filesystem::path(m_write_dir) / m_meta_file);
        out << std::setw(2) << m_meta << std::endl;
        m_current_filename.clear();
    }

    if (m_dumper) {
        pcap_dump_close(m_dumper);
        m_dumper = nullptr;
    }

    if (m_pcap) {
        pcap_close(m_pcap);
        m_pcap = nullptr;
    }
}

extern "C" {
    auto create() -> std::shared_ptr<composite::component> {
        return std::make_shared<pcap_writer>();
    }
}