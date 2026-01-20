/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Simple SigMF file source component for testing.
 */

#include "component.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <format>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

// Simple JSON value extraction (avoids adding a JSON library dependency)
std::string extract_json_string(const std::string& json, const std::string& key) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (std::regex_search(json, match, pattern) && match.size() > 1) {
        return match[1].str();
    }
    return "";
}

double extract_json_number(const std::string& json, const std::string& key) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9.eE+-]+)");
    std::smatch match;
    if (std::regex_search(json, match, pattern) && match.size() > 1) {
        try {
            return std::stod(match[1].str());
        } catch (...) {
            return 0.0;
        }
    }
    return 0.0;
}

} // namespace

template<typename T>
sigmf_source<T>::sigmf_source(std::string_view id)
    : composite::component(id) {
    using enum composite::properties::config_type;
    add_port(&m_out_port);
    add_property("file_path", m_file_path, INITIALIZE);
    add_property("chunk_samples", m_chunk_samples, INITIALIZE);
    add_property("loop", m_loop, RUNTIME);
    add_property("stream_id", m_stream_id, INITIALIZE);
}

template<typename T>
auto sigmf_source<T>::initialize() -> void {
    if (m_file_path.empty()) {
        throw std::runtime_error("sigmf_source: file_path property is required");
    }

    parse_metadata();

    // Open data file
    std::string data_path = m_file_path;
    if (data_path.ends_with(".sigmf-meta")) {
        data_path = data_path.substr(0, data_path.length() - 11) + ".sigmf-data";
    } else if (!data_path.ends_with(".sigmf-data")) {
        data_path = data_path + ".sigmf-data";
    }

    m_data_file.open(data_path, std::ios::binary);
    if (!m_data_file) {
        throw std::runtime_error(std::format("sigmf_source: cannot open data file: {}", data_path));
    }

    // Get file size to calculate total samples
    m_data_file.seekg(0, std::ios::end);
    auto file_size = m_data_file.tellg();
    m_data_file.seekg(0, std::ios::beg);
    m_total_samples = static_cast<std::size_t>(file_size) / sizeof(T);

    logger()->info("sigmf_source initialized: {} samples, sr={} Hz, cf={} Hz",
                   m_total_samples, m_sample_rate, m_center_frequency);
}

template<typename T>
void sigmf_source<T>::parse_metadata() {
    std::string meta_path = m_file_path;
    if (meta_path.ends_with(".sigmf-data")) {
        meta_path = meta_path.substr(0, meta_path.length() - 11) + ".sigmf-meta";
    } else if (!meta_path.ends_with(".sigmf-meta")) {
        meta_path = meta_path + ".sigmf-meta";
    }

    std::ifstream meta_file(meta_path);
    if (!meta_file) {
        logger()->warn("sigmf_source: no metadata file found at {}, using defaults", meta_path);
        return;
    }

    std::stringstream buffer;
    buffer << meta_file.rdbuf();
    std::string json = buffer.str();

    // Extract global fields
    m_sample_rate = extract_json_number(json, "core:sample_rate");
    m_description = extract_json_string(json, "core:description");

    // Extract capture fields (first capture segment)
    m_center_frequency = extract_json_number(json, "core:frequency");

    // Verify datatype matches template
    std::string datatype = extract_json_string(json, "core:datatype");
    if (!datatype.empty()) {
        logger()->debug("sigmf_source: SigMF datatype is '{}'", datatype);
    }
}

template<typename T>
auto sigmf_source<T>::start() -> void {
    logger()->info("sigmf_source starting playback");
    m_samples_read = 0;
    m_eof = false;
    if (m_data_file.is_open()) {
        m_data_file.clear();
        m_data_file.seekg(0, std::ios::beg);
    }
    send_metadata_to_port();
    composite::component::start();  // Start the processing thread
}

template<typename T>
auto sigmf_source<T>::stop() -> void {
    if (m_data_file.is_open()) {
        m_data_file.close();
    }
    composite::component::stop();
}

template<typename T>
void sigmf_source<T>::send_metadata_to_port() {
    composite::metadata meta;
    meta.annotations["stream_id"] = std::to_string(m_stream_id);
    meta.sample_rate = m_sample_rate;
    meta.center_frequency = m_center_frequency;
    meta.bandwidth = m_sample_rate;  // Default bandwidth to sample rate
    if (!m_description.empty()) {
        meta.annotations["description"] = m_description;
    }
    m_out_port.send_metadata(meta);
}

template<typename T>
auto sigmf_source<T>::process() -> composite::retval {
    using enum composite::retval;

    if (m_eof && !m_loop) {
        return NOOP;
    }

    if (m_eof && m_loop) {
        // Reset to beginning for looping
        logger()->trace("sigmf_source: looping back to start");
        m_data_file.clear();
        m_data_file.seekg(0, std::ios::beg);
        m_samples_read = 0;
        m_eof = false;
        send_metadata_to_port();
    }

    // Calculate samples to read
    std::size_t samples_remaining = m_total_samples - m_samples_read;
    std::size_t samples_to_read = std::min(m_chunk_samples, samples_remaining);

    if (samples_to_read == 0) {
        m_eof = true;
        if (!m_loop) {
            logger()->info("sigmf_source: reached end of file after {} samples", m_samples_read);
        }
        return NOOP;
    }

    // Read data
    auto data_vec = std::make_shared<std::vector<T>>(samples_to_read);
    m_data_file.read(reinterpret_cast<char*>(data_vec->data()), samples_to_read * sizeof(T));
    auto actually_read = m_data_file.gcount() / sizeof(T);

    if (actually_read == 0) {
        m_eof = true;
        return NOOP;
    }

    if (actually_read < samples_to_read) {
        data_vec->resize(actually_read);
    }

    m_samples_read += actually_read;

    // Create timestamp (sample-based)
    composite::timestamp ts;
    ts.seconds = static_cast<uint32_t>(m_samples_read / static_cast<uint64_t>(m_sample_rate > 0 ? m_sample_rate : 1));
    ts.picoseconds = 0;

    // Send data
    composite::immutable_buffer<T> buf(data_vec);
    m_out_port.send_data(std::move(buf), ts);

    return NORMAL;
}

// Explicit template instantiations
template class sigmf_source<std::complex<float>>;
template class sigmf_source<std::complex<int16_t>>;
template class sigmf_source<std::complex<int8_t>>;
template class sigmf_source<float>;
template class sigmf_source<int16_t>;

#ifndef UNIT_TESTS
extern "C" {
auto create(std::string_view id, std::string_view type) -> std::shared_ptr<composite::component> {
    if (type == "cf32" || type.empty()) {
        return std::make_shared<sigmf_source_cf32>(id);
    } else if (type == "ci16") {
        return std::make_shared<sigmf_source_ci16>(id);
    } else if (type == "ci8") {
        return std::make_shared<sigmf_source_ci8>(id);
    } else if (type == "f32") {
        return std::make_shared<sigmf_source_f32>(id);
    } else if (type == "i16") {
        return std::make_shared<sigmf_source_i16>(id);
    }
    throw std::runtime_error(std::format("sigmf_source: unknown type '{}'", type));
}
}
#endif