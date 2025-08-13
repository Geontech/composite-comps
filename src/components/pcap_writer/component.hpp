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
#include <nlohmann/json.hpp>
#include <pcap/pcap.h>
#include <composite/component.hpp>
#include <chrono>
// #include <iomanip>
// #include <sstream>
// #include <vector>
// #include <memory>
// #include <thread>
// #include <atomic>
// #include <filesystem>
// #include <utility>
// #include <helpers.hpp>

class pcap_writer : public composite::component {
    using buffer_t = std::pmr::vector<uint8_t>;
    using buffer_ptr_t = std::shared_ptr<buffer_t>;
    using input_port_t = composite::input_port<buffer_ptr_t>;

public:
    pcap_writer();
    ~pcap_writer() override;
    auto process() -> composite::retval override;

private:
    input_port_t m_in_port{"data_in"};

    pcap_t* m_pcap = nullptr;
    pcap_dumper_t* m_dumper = nullptr;

    uint32_t m_rotate_every{100'000};
    uint64_t m_counter{0};
    uint32_t m_file_number{0};
    std::string m_write_dir{"/data/output/"};
    std::string m_current_filename{};
    composite::metadata m_metadata{};
    nlohmann::json m_meta;
    struct pcap_pkthdr hdr;

    composite::timestamp m_current_file_first_ts{};
    composite::timestamp m_current_file_last_ts{};
    std::map<std::string, nlohmann::json> m_timestamp_map;
    std::string m_meta_file{"meta.sigmf-collection"};
    std::string m_mission{"rfnext_pcap"};
    std::string m_polarization{"UNKOWN"};

    std::chrono::steady_clock::time_point m_start_time{};
    uint32_t m_duration_seconds{10}; // 0 = no timeout
    bool m_started{false};
    bool m_expired{false};

    void open_new_file();
    void close_file();
};
