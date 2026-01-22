/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * SigMF file source component with rate-controlled playback.
 */

#include "component.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <stdexcept>
#include <thread>

template<typename T>
sigmf_source<T>::sigmf_source(std::string_view id)
    : composite::component(id) {
    using enum composite::properties::config_type;
    add_port(&m_out_port);

    // File configuration
    add_property("file_path", m_file_path, INITIALIZE);
    add_property("chunk_samples", m_chunk_samples, INITIALIZE);
    add_property("loop", m_loop, RUNTIME);
    add_property("stream_id", m_stream_id, INITIALIZE);

    // Rate control
    add_property("rate_control", m_rate_control, RUNTIME);
    add_property("max_sample_rate", m_max_sample_rate, RUNTIME);
}

template<typename T>
auto sigmf_source<T>::parse_datatype(std::string_view sv) -> std::optional<SigmfFormat> {
    // Parse SigMF datatype format: [r|c][f|i|u]{bitwidth}[_le|_be]
    // Examples: cf32_le, ri16_be, cu8
    SigmfFormat fmt;
    fmt.datatype_str = std::string(sv);
    fmt.is_big_endian = false;  // Default to little-endian
    fmt.is_complex = true;      // Default to complex

    // Check for endianness suffix
    if (sv.ends_with("_le")) {
        sv.remove_suffix(3);
        fmt.is_big_endian = false;
    } else if (sv.ends_with("_be")) {
        sv.remove_suffix(3);
        fmt.is_big_endian = true;
    }

    if (sv.empty()) return std::nullopt;

    // First char: real/complex specifier
    switch (sv.front()) {
        case 'c': case 'C': fmt.is_complex = true; break;
        case 'r': case 'R': fmt.is_complex = false; break;
        default: return std::nullopt;
    }
    sv.remove_prefix(1);

    if (sv.empty()) return std::nullopt;

    // Second char: data type specifier
    switch (sv.front()) {
        case 'f': case 'F': fmt.datatype = SigmfFormat::DataType::FLOAT; break;
        case 'i': case 'I': fmt.datatype = SigmfFormat::DataType::SIGNED_INT; break;
        case 'u': case 'U': fmt.datatype = SigmfFormat::DataType::UNSIGNED_INT; break;
        default: return std::nullopt;
    }
    sv.remove_prefix(1);

    if (sv.empty()) return std::nullopt;

    // Remaining: bitwidth
    uint32_t bw{};
    auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), bw);
    if (ec != std::errc{} || ptr != sv.data() + sv.size()) {
        return std::nullopt;
    }
    fmt.bitwidth = bw;

    return fmt;
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

    try {
        nlohmann::json doc;
        meta_file >> doc;

        if (!doc.contains("global")) {
            logger()->warn("sigmf_source: no 'global' section in metadata");
            return;
        }

        auto& global = doc["global"];

        // Required fields
        m_sample_rate = global.at("core:sample_rate").get<double>();

        // Optional fields with defaults
        m_center_frequency = global.value("core:frequency", 0.0);
        m_description = global.value("core:description", std::string{});

        // Parse datatype for format/endianness info
        if (global.contains("core:datatype")) {
            std::string datatype = global["core:datatype"].get<std::string>();
            auto parsed = parse_datatype(datatype);
            if (parsed) {
                m_format = *parsed;
                logger()->debug("sigmf_source: parsed datatype '{}' - complex={}, big_endian={}, bits={}",
                               datatype, m_format.is_complex, m_format.is_big_endian, m_format.bitwidth);
            } else {
                logger()->warn("sigmf_source: failed to parse datatype '{}', using defaults", datatype);
            }
        }

        // Check captures array for frequency override
        if (doc.contains("captures") && doc["captures"].is_array() && !doc["captures"].empty()) {
            auto& cap0 = doc["captures"][0];
            if (cap0.contains("core:frequency")) {
                m_center_frequency = cap0["core:frequency"].get<double>();
            }
        }

    } catch (const nlohmann::json::exception& e) {
        logger()->error("sigmf_source: JSON parse error in '{}': {}", meta_path, e.what());
        throw std::runtime_error(std::format("sigmf_source: failed to parse metadata: {}", e.what()));
    }
}

template<typename T>
void sigmf_source<T>::byte_swap_16(std::vector<T>& data) {
    auto* bytes = reinterpret_cast<uint8_t*>(data.data());
    const std::size_t num_bytes = data.size() * sizeof(T);
    for (std::size_t i = 0; i + 1 < num_bytes; i += 2) {
        std::swap(bytes[i], bytes[i + 1]);
    }
}

template<typename T>
void sigmf_source<T>::byte_swap_32(std::vector<T>& data) {
    auto* bytes = reinterpret_cast<uint8_t*>(data.data());
    const std::size_t num_bytes = data.size() * sizeof(T);
    for (std::size_t i = 0; i + 3 < num_bytes; i += 4) {
        std::swap(bytes[i], bytes[i + 3]);
        std::swap(bytes[i + 1], bytes[i + 2]);
    }
}

template<typename T>
void sigmf_source<T>::apply_endianness_swap() {
    // Only swap if file is big-endian and host is little-endian (or vice versa)
    // x86/x64 is always little-endian
    constexpr bool host_is_little_endian = (std::endian::native == std::endian::little);

    if (m_format.is_big_endian == host_is_little_endian) {
        // Need to swap
        if (m_format.bitwidth == 16) {
            byte_swap_16(m_file_data);
            logger()->debug("sigmf_source: applied 16-bit byte swap for endianness");
        } else if (m_format.bitwidth == 32) {
            byte_swap_32(m_file_data);
            logger()->debug("sigmf_source: applied 32-bit byte swap for endianness");
        } else if (m_format.bitwidth == 64) {
            // For 64-bit, swap as two 32-bit values then swap them
            auto* bytes = reinterpret_cast<uint8_t*>(m_file_data.data());
            const std::size_t num_bytes = m_file_data.size() * sizeof(T);
            for (std::size_t i = 0; i + 7 < num_bytes; i += 8) {
                std::swap(bytes[i], bytes[i + 7]);
                std::swap(bytes[i + 1], bytes[i + 6]);
                std::swap(bytes[i + 2], bytes[i + 5]);
                std::swap(bytes[i + 3], bytes[i + 4]);
            }
            logger()->debug("sigmf_source: applied 64-bit byte swap for endianness");
        }
    }
}

template<typename T>
void sigmf_source<T>::calculate_timing() {
    // Determine effective sample rate
    m_effective_sample_rate = m_sample_rate;
    if (m_max_sample_rate > 0 && m_max_sample_rate < m_sample_rate) {
        m_effective_sample_rate = m_max_sample_rate;
        logger()->info("sigmf_source: limiting playback rate from {} to {} Hz",
                      m_sample_rate, m_effective_sample_rate);
    }

    if (m_effective_sample_rate <= 0 || m_chunk_samples == 0) {
        // Can't calculate timing without valid sample rate
        m_rate_control = false;
        logger()->warn("sigmf_source: invalid sample rate or chunk size, disabling rate control");
        return;
    }

    // Calculate chunks per second and ideal interval
    double chunks_per_sec = m_effective_sample_rate / static_cast<double>(m_chunk_samples);
    double interval_us = 1e6 / chunks_per_sec;

    // Minimum wakeup interval (500 microseconds) to avoid excessive thread scheduling
    constexpr double MIN_WAKE_US = 500.0;

    if (interval_us >= MIN_WAKE_US) {
        // Slow enough to send one chunk per wakeup
        m_chunks_per_wakeup = 1;
        m_chunk_interval = std::chrono::microseconds(static_cast<uint64_t>(interval_us + 0.5));
    } else {
        // Bundle multiple chunks to maintain minimum wakeup interval
        m_chunks_per_wakeup = static_cast<uint32_t>(std::ceil(MIN_WAKE_US / interval_us));
        m_chunk_interval = std::chrono::microseconds(
            static_cast<uint64_t>(interval_us * m_chunks_per_wakeup + 0.5));
    }

    logger()->info("sigmf_source: rate control enabled - {} Hz, {} samples/chunk, {} chunks/wakeup, {} us interval",
                  m_effective_sample_rate, m_chunk_samples, m_chunks_per_wakeup, m_chunk_interval.count());
}

template<typename T>
auto sigmf_source<T>::initialize() -> void {
    if (m_file_path.empty()) {
        throw std::runtime_error("sigmf_source: file_path property is required");
    }

    parse_metadata();

    // Determine data file path
    std::string data_path = m_file_path;
    if (data_path.ends_with(".sigmf-meta")) {
        data_path = data_path.substr(0, data_path.length() - 11) + ".sigmf-data";
    } else if (!data_path.ends_with(".sigmf-data")) {
        data_path = data_path + ".sigmf-data";
    }

    // Get file size
    if (!std::filesystem::exists(data_path)) {
        throw std::runtime_error(std::format("sigmf_source: data file not found: {}", data_path));
    }
    auto file_size = std::filesystem::file_size(data_path);
    m_total_samples = file_size / sizeof(T);

    // For endianness handling, we need to preload the file if byte swapping is needed
    constexpr bool host_is_little_endian = (std::endian::native == std::endian::little);
    bool needs_swap = (m_format.is_big_endian == host_is_little_endian) && (m_format.bitwidth >= 16);

    if (needs_swap) {
        // Preload entire file for byte swapping
        logger()->info("sigmf_source: preloading file for endianness conversion");
        m_file_data.resize(m_total_samples);
        std::ifstream in(data_path, std::ios::binary);
        in.read(reinterpret_cast<char*>(m_file_data.data()), file_size);
        apply_endianness_swap();
        m_data_preloaded = true;
    } else {
        // Open file for streaming reads
        m_data_file.open(data_path, std::ios::binary);
        if (!m_data_file) {
            throw std::runtime_error(std::format("sigmf_source: cannot open data file: {}", data_path));
        }
        m_data_preloaded = false;
    }

    // Calculate rate control timing
    if (m_rate_control) {
        calculate_timing();
    }

    logger()->info("sigmf_source initialized: {} samples, sr={} Hz, cf={} Hz, rate_control={}",
                  m_total_samples, m_sample_rate, m_center_frequency, m_rate_control);
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
auto sigmf_source<T>::start() -> void {
    logger()->info("sigmf_source starting playback");
    m_samples_read = 0;
    m_file_data_index = 0;
    m_eof = false;

    if (!m_data_preloaded && m_data_file.is_open()) {
        m_data_file.clear();
        m_data_file.seekg(0, std::ios::beg);
    }

    // Recalculate timing in case max_sample_rate changed
    if (m_rate_control) {
        calculate_timing();
        m_next_send_time = std::chrono::steady_clock::now();
    }

    send_metadata_to_port();
    composite::component::start();
}

template<typename T>
auto sigmf_source<T>::stop() -> void {
    if (m_data_file.is_open()) {
        m_data_file.close();
    }
    composite::component::stop();
}

template<typename T>
auto sigmf_source<T>::process_chunk() -> composite::retval {
    using enum composite::retval;

    if (m_eof && !m_loop) {
        return NOOP;
    }

    if (m_eof && m_loop) {
        // Reset to beginning for looping
        logger()->trace("sigmf_source: looping back to start");
        if (m_data_preloaded) {
            m_file_data_index = 0;
        } else {
            m_data_file.clear();
            m_data_file.seekg(0, std::ios::beg);
        }
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
    std::size_t actually_read = 0;

    if (m_data_preloaded) {
        // Copy from preloaded buffer
        std::copy_n(m_file_data.begin() + m_file_data_index, samples_to_read, data_vec->begin());
        m_file_data_index += samples_to_read;
        actually_read = samples_to_read;
    } else {
        // Read from file
        m_data_file.read(reinterpret_cast<char*>(data_vec->data()), samples_to_read * sizeof(T));
        actually_read = m_data_file.gcount() / sizeof(T);
    }

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
    if (m_effective_sample_rate > 0) {
        double total_seconds = static_cast<double>(m_samples_read) / m_effective_sample_rate;
        ts.seconds = static_cast<uint32_t>(total_seconds);
        ts.picoseconds = static_cast<uint64_t>((total_seconds - ts.seconds) * 1e12);
    }

    // Send data
    composite::immutable_buffer<T> buf(data_vec);
    m_out_port.send_data(std::move(buf), ts);

    return NORMAL;
}

template<typename T>
auto sigmf_source<T>::process() -> composite::retval {
    using enum composite::retval;

    if (!m_rate_control) {
        // Fast-as-possible mode - just process one chunk
        return process_chunk();
    }

    // Rate-controlled mode
    auto now = std::chrono::steady_clock::now();

    // Wait until it's time to send
    if (now < m_next_send_time) {
        std::this_thread::sleep_until(m_next_send_time);
    }

    // Send the bundled chunks
    for (uint32_t i = 0; i < m_chunks_per_wakeup; ++i) {
        auto result = process_chunk();
        if (result != NORMAL) {
            return result;
        }
    }

    // Schedule next wakeup
    m_next_send_time += m_chunk_interval;

    // If we fell behind schedule, catch up
    now = std::chrono::steady_clock::now();
    if (m_next_send_time < now) {
        m_next_send_time = now + m_chunk_interval;
    }

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
