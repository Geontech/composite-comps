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
// #include "aligned_mem.hpp"
// //  #include "fft_plan.hpp"
// #include "windows.hpp"
// #include <chrono>
// #include <iostream>
// #include <bit>
// #include <composite/component.hpp>
// #include <complex>
// #include <fftw3.h>
// #include <immintrin.h>
// #include <memory>
// #include <queue>
// #include <vector>

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

template <typename T>
class pcap_writer : public composite::component {
    using buffer_t = std::pmr::vector<uint8_t>;
    using buffer_ptr_t = std::shared_ptr<buffer_t>;
    using input_port_t = composite::input_port<buffer_ptr_t>;

public:
    pcap_writer() : composite::component("pcap_writer") {
        add_port(&m_in_port);
        add_property("rotate_every", &m_rotate_every);
        add_property("write_dir", &m_write_dir);    
        add_property("duration_seconds", &m_duration_seconds);
    
    }

    ~pcap_writer() override {
        close_file();
    }

    auto process() -> composite::retval override {
        using enum composite::retval;

        auto [data, ts, meta] = m_in_port.get_data();
        if (!data) return NORMAL;
        if (!m_started) {
            m_start_time = std::chrono::steady_clock::now();
            m_started = true;
        }
    
        if (!m_started) {
            m_start_time = std::chrono::steady_clock::now();
            m_started = true;
        }
        
        // Check for duration expiry
        if (!m_expired && m_duration_seconds > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_start_time).count();
            if (elapsed >= m_duration_seconds) {
                close_file();
                m_expired = true;
                logger()->info("No longer writing pcaps");
                return NORMAL;
            }
        }

        // if (meta.has_value() && m_metadata != meta.value()) {
        //     m_metadata = meta.value();
        // }

        // if (m_metadata.sample_rate == 0.0) {
        //     // Skip writing if metadata is not set
        //     return NORMAL;
        // }
        if (!m_dumper) {
            open_new_file();
        }

        gettimeofday(&hdr.ts, nullptr);
        hdr.caplen = data->size();
        hdr.len = data->size();

        pcap_dump(reinterpret_cast<u_char*>(m_dumper), &hdr, data->data());
        ++m_counter;

        if (m_counter >= m_rotate_every) {
            close_file();
            open_new_file();
            m_counter = 0;
        }

        return NORMAL;
    }

private:
    input_port_t m_in_port{"data_in"};

    pcap_t* m_pcap = nullptr;
    pcap_dumper_t* m_dumper = nullptr;

    uint32_t m_rotate_every{1'000};
    uint64_t m_counter{0};
    uint32_t m_file_number{0};
    std::string m_write_dir{"/data/output/"};
    std::string m_current_filename{};
    composite::metadata m_metadata{};
    struct pcap_pkthdr hdr;

    std::chrono::steady_clock::time_point m_start_time{};
    uint32_t m_duration_seconds{15}; // 0 = no timeout
    bool m_started{false};
    bool m_expired{false};


    void open_new_file() {
        namespace fs = std::filesystem;

        std::stringstream filename;
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        filename << "capture_"
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

    void close_file() {
        if (m_dumper) {
            pcap_dump_close(m_dumper);
            m_dumper = nullptr;
        }

        if (m_pcap) {
            pcap_close(m_pcap);
            m_pcap = nullptr;
        }
    }
};
