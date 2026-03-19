/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * SigMF file source component with rate-controlled playback.
 * Uses mmap for zero-copy file reading at high sample rates.
 */

#include "component.hpp"
#include "blue/BlueFile.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <thread>

sigmf_source::sigmf_source(std::string_view id)
    : composite::component(id) {
    using enum composite::properties::config_type;
    add_port(&m_out_port);

    // Streaming control - starts disabled, set streaming=true after configuring file_path
    add_property("streaming", m_streaming, RUNTIME);

    // File configuration (all RUNTIME so they can be set before enabling)
    add_property("file_path", m_file_path, RUNTIME);
    add_property("chunk_samples", m_chunk_samples, RUNTIME);
    add_property("loop", m_loop, RUNTIME);
    add_property("stream_id", m_stream_id, RUNTIME);

    // Rate control
    add_property("rate_control", m_rate_control, RUNTIME);
    add_property("max_sample_rate", m_max_sample_rate, RUNTIME);

    // Timestamp mode
    add_property("wallclock_timestamps", m_wallclock_timestamps, RUNTIME);

    // Metadata overrides
    add_property("overrides", m_overrides, RUNTIME);
}

auto sigmf_source::parse_datatype(std::string_view sv) -> std::optional<SigmfFormat> {
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

auto sigmf_source::to_composite_format(const SigmfFormat& fmt) -> composite::data_format {
    composite::data_format df;
    df.is_complex = fmt.is_complex;
    df.bit_width = fmt.bitwidth;
    df.endianness = fmt.is_big_endian ? std::endian::big : std::endian::little;

    switch (fmt.datatype) {
        case SigmfFormat::DataType::FLOAT:
            df.type = composite::data_type::floating_point;
            break;
        case SigmfFormat::DataType::SIGNED_INT:
            df.type = composite::data_type::signed_integer;
            break;
        case SigmfFormat::DataType::UNSIGNED_INT:
            df.type = composite::data_type::unsigned_integer;
            break;
    }

    return df;
}

void sigmf_source::parse_metadata() {
    std::string meta_path = m_file_path;
    if (meta_path.ends_with(".sigmf-data")) {
        meta_path = meta_path.substr(0, meta_path.length() - 11) + ".sigmf-meta";
    } else if (meta_path.ends_with(".blue")) {
        // Blue file - derive meta path by replacing .blue with .sigmf-meta
        meta_path = meta_path.substr(0, meta_path.length() - 5) + ".sigmf-meta";
    } else if (!meta_path.ends_with(".sigmf-meta")) {
        // Check if the file exists as-is (raw binary file without .sigmf- extension)
        if (!std::filesystem::exists(m_file_path) || std::filesystem::is_directory(m_file_path)) {
            meta_path = meta_path + ".sigmf-meta";
        } else {
            // File exists as-is, derive meta path by appending .sigmf-meta
            meta_path = m_file_path + ".sigmf-meta";
        }
    }

    std::ifstream meta_file(meta_path);
    if (!meta_file) {
        logger()->warn("sigmf_source: no metadata file found at {}, will use overrides/defaults", meta_path);
        // Don't return - fall through to apply overrides
    } else {
        try {
            nlohmann::json doc;
            meta_file >> doc;

            if (!doc.contains("global")) {
                logger()->warn("sigmf_source: no 'global' section in metadata");
            } else {
                auto& global = doc["global"];

                // Sample rate from file (may be overridden below)
                if (global.contains("core:sample_rate")) {
                    m_sample_rate = global["core:sample_rate"].get<double>();
                }

                // Optional fields with defaults
                m_center_frequency = global.value("core:frequency", 0.0);
                m_description = global.value("core:description", std::string{});

                // Check for filetype extension (for MIDAS Blue support)
                // Supports both namespaced and non-namespaced field names
                for (const auto& key : {"filetype", "core:filetype"}) {
                    if (global.contains(key)) {
                        m_filetype = global[key].get<std::string>();
                        logger()->debug("sigmf_source: found filetype '{}' in metadata", m_filetype);
                        break;
                    }
                }

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
            }

        } catch (const nlohmann::json::exception& e) {
            logger()->error("sigmf_source: JSON parse error in '{}': {}", meta_path, e.what());
            throw std::runtime_error(std::format("sigmf_source: failed to parse metadata: {}", e.what()));
        }
    }

    // Apply property overrides (these take precedence over file metadata)
    // These are always applied, even if no metadata file was found
    if (m_overrides.sample_rate.has_value()) {
        logger()->info("sigmf_source: overriding sample_rate {} -> {}", m_sample_rate, *m_overrides.sample_rate);
        m_sample_rate = *m_overrides.sample_rate;
    }
    if (m_overrides.center_frequency.has_value()) {
        logger()->info("sigmf_source: overriding center_frequency {} -> {}", m_center_frequency, *m_overrides.center_frequency);
        m_center_frequency = *m_overrides.center_frequency;
    }
    if (m_overrides.datatype.has_value()) {
        auto parsed = parse_datatype(*m_overrides.datatype);
        if (parsed) {
            logger()->info("sigmf_source: overriding datatype -> '{}'", *m_overrides.datatype);
            m_format = *parsed;
        } else {
            logger()->warn("sigmf_source: failed to parse override datatype '{}', keeping previous", *m_overrides.datatype);
        }
    }
    if (m_overrides.filetype.has_value()) {
        logger()->info("sigmf_source: overriding filetype -> '{}'", *m_overrides.filetype);
        m_filetype = *m_overrides.filetype;
    }
    // Note: bandwidth override is applied in send_metadata_to_port()
}

void sigmf_source::apply_endianness_swap() {
    // Only swap if file is big-endian and host is little-endian (or vice versa)
    // x86/x64 is always little-endian
    constexpr bool host_is_little_endian = (std::endian::native == std::endian::little);

    if (!m_mmap || !m_mmap->valid()) {
        return;
    }

    if (m_format.is_big_endian != host_is_little_endian) {
        // Same endianness, no swap needed
        return;
    }

    auto* bytes = reinterpret_cast<uint8_t*>(m_mmap->data());
    const std::size_t num_bytes = m_mmap->file_size();

    if (m_format.bitwidth == 16) {
        for (std::size_t i = 0; i + 1 < num_bytes; i += 2) {
            std::swap(bytes[i], bytes[i + 1]);
        }
        logger()->debug("sigmf_source: applied 16-bit byte swap for endianness on mmap region");
    } else if (m_format.bitwidth == 32) {
        for (std::size_t i = 0; i + 3 < num_bytes; i += 4) {
            std::swap(bytes[i], bytes[i + 3]);
            std::swap(bytes[i + 1], bytes[i + 2]);
        }
        logger()->debug("sigmf_source: applied 32-bit byte swap for endianness on mmap region");
    } else if (m_format.bitwidth == 64) {
        for (std::size_t i = 0; i + 7 < num_bytes; i += 8) {
            std::swap(bytes[i], bytes[i + 7]);
            std::swap(bytes[i + 1], bytes[i + 6]);
            std::swap(bytes[i + 2], bytes[i + 5]);
            std::swap(bytes[i + 3], bytes[i + 4]);
        }
        logger()->debug("sigmf_source: applied 64-bit byte swap for endianness on mmap region");
    }
}

void sigmf_source::calculate_timing() {
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

void sigmf_source::send_metadata_to_port() {
    composite::metadata meta;
    meta.annotations["stream_id"] = std::to_string(m_stream_id);
    meta.sample_rate = m_sample_rate;
    meta.center_frequency = m_center_frequency;
    // Bandwidth: use override if set, otherwise default to sample rate
    meta.bandwidth = m_overrides.bandwidth.value_or(m_sample_rate);
    if (!m_description.empty()) {
        meta.annotations["description"] = m_description;
    }
    // Include format info from the SigMF file so downstream knows the data type
    meta.format = to_composite_format(m_format);
    logger()->debug("sigmf_source: sending metadata with format: complex={}, type={}, bits={}",
                   meta.format.is_complex, static_cast<int>(meta.format.type), meta.format.bit_width);
    m_out_port.send_metadata(meta);
}

auto sigmf_source::initialize() -> void {
    // Component starts with streaming disabled - configure properties then set streaming=true
    logger()->info("sigmf_source initialized (streaming disabled, set streaming=true after configuring file_path)");
}

auto sigmf_source::property_change_handler() -> void {
    logger()->info("sigmf_source: property_change_handler called - streaming={}, configured={}, file_path='{}'",
                   m_streaming, m_configured, m_file_path);

    if (m_streaming && !m_configured && !m_file_path.empty()) {
        // Becoming enabled with a file path - configure the file
        logger()->info("sigmf_source: triggering configure_file()");
        configure_file();
    } else if (!m_streaming && m_configured) {
        // Becoming disabled - release resources
        logger()->info("sigmf_source: streaming disabled, releasing file resources");
        m_mmap.reset();
        m_configured = false;
        m_total_samples = 0;
    } else {
        logger()->info("sigmf_source: no action taken (streaming={}, configured={}, file_path_empty={})",
                       m_streaming, m_configured, m_file_path.empty());
    }

    // Recalculate timing if rate control settings changed
    if (m_configured && m_rate_control) {
        calculate_timing();
    }
}

auto sigmf_source::detect_filetype(const std::string& data_path) -> std::string {
    // Priority: 1) override, 2) file extension, 3) file magic, 4) default to "raw"

    // Override already applied in parse_metadata(), check if set
    if (m_filetype != "raw") {
        logger()->debug("sigmf_source: using filetype '{}' from metadata/override", m_filetype);
        return m_filetype;
    }

    // Check file extension
    if (data_path.ends_with(".blue")) {
        logger()->info("sigmf_source: detected MIDAS Blue file by extension");
        return "bluefile-1000";
    }

    // Check file magic (first 4 bytes = "BLUE")
    if (blue::isBlueFile(data_path)) {
        logger()->info("sigmf_source: detected MIDAS Blue file by magic");
        return "bluefile-1000";
    }

    return "raw";
}

void sigmf_source::configure_blue_file() {
    logger()->info("sigmf_source: configuring MIDAS Blue file");

    // Determine the .blue file path
    std::string blue_path = m_file_path;
    if (blue_path.ends_with(".sigmf-meta")) {
        blue_path = blue_path.substr(0, blue_path.length() - 11) + ".blue";
    } else if (blue_path.ends_with(".sigmf-data")) {
        blue_path = blue_path.substr(0, blue_path.length() - 11) + ".blue";
    } else if (!blue_path.ends_with(".blue")) {
        // Check if file exists as-is, otherwise try adding .blue
        if (!std::filesystem::exists(blue_path) || !blue::isBlueFile(blue_path)) {
            blue_path = blue_path + ".blue";
        }
    }

    if (!std::filesystem::exists(blue_path)) {
        throw std::runtime_error(std::format("sigmf_source: Blue file not found: {}", blue_path));
    }

    // Parse the Blue file header
    auto blueInfo = blue::parseBlueFile(blue_path);
    if (!blueInfo || !blueInfo->valid) {
        throw std::runtime_error(std::format("sigmf_source: failed to parse Blue file: {}", blue_path));
    }

    // Convert Blue format to SigmfFormat
    m_format.is_complex = blueInfo->format.isComplex;
    m_format.bitwidth = blueInfo->format.scalarSize * 8;
    m_format.is_big_endian = blueInfo->needsDataSwap;  // If needs swap, it's opposite endianness

    // Map Blue type code to SigmfFormat::DataType
    switch (blueInfo->format.typeCode) {
        case 'F': m_format.datatype = SigmfFormat::DataType::FLOAT; break;
        case 'D': m_format.datatype = SigmfFormat::DataType::FLOAT; break;  // Double is also float
        case 'I': case 'L': case 'X':
            m_format.datatype = SigmfFormat::DataType::SIGNED_INT; break;
        case 'B': m_format.datatype = SigmfFormat::DataType::SIGNED_INT; break;
        case 'O': case 'U': case 'V':
            m_format.datatype = SigmfFormat::DataType::UNSIGNED_INT; break;
        default:
            m_format.datatype = SigmfFormat::DataType::FLOAT;
    }

    // Build datatype string for logging
    char rc = m_format.is_complex ? 'c' : 'r';
    char tc = (m_format.datatype == SigmfFormat::DataType::FLOAT) ? 'f' :
              (m_format.datatype == SigmfFormat::DataType::SIGNED_INT) ? 'i' : 'u';
    m_format.datatype_str = std::format("{}{}{}", rc, tc, m_format.bitwidth);

    // Use sample rate from Blue header if not already set (and not overridden)
    if (m_sample_rate <= 0 && blueInfo->sampleRate > 0) {
        m_sample_rate = blueInfo->sampleRate;
        logger()->info("sigmf_source: using sample rate {} Hz from Blue header", m_sample_rate);
    }

    // Calculate bytes per sample
    m_bytes_per_sample = blueInfo->format.sampleSize;
    if (m_bytes_per_sample == 0) {
        throw std::runtime_error("sigmf_source: invalid Blue format - bytes_per_sample is 0");
    }

    // Memory-map the file
    bool needs_swap = blueInfo->needsDataSwap && (m_format.bitwidth >= 16);
    logger()->info("sigmf_source: memory-mapping Blue file {} (writable={}, data_offset={}, data_size={})",
                  blue_path, needs_swap, blueInfo->dataOffset, blueInfo->dataSize);

    m_mmap = std::make_shared<MmapRegion<std::byte>>(blue_path, needs_swap);
    m_data_start_offset = blueInfo->dataOffset;
    m_total_samples = blueInfo->sampleCount;
    m_current_byte_offset = m_data_start_offset;

    // Apply endianness conversion on the data portion only
    if (needs_swap) {
        logger()->info("sigmf_source: applying endianness swap on Blue data ({} bytes)", blueInfo->dataSize);
        auto* data_start = reinterpret_cast<uint8_t*>(m_mmap->data()) + m_data_start_offset;
        blue::byteswapData(data_start, blueInfo->dataSize / (m_format.bitwidth / 8),
                          blueInfo->format.typeCode);
        m_format.is_big_endian = (std::endian::native == std::endian::big);
    }

    // Calculate rate control timing
    if (m_rate_control) {
        calculate_timing();
    }

    m_configured = true;
    m_filetype = "bluefile-1000";

    // Reset rate control epoch so the catch-up logic doesn't see the time
    // between start() and now as a deficit to fill, which causes a burst
    // that OOM-kills the pod.
    m_start_time = std::chrono::steady_clock::now();
    m_samples_sent = 0;

    // Capture UTC epoch for absolute timestamps
    auto utc_now = std::chrono::system_clock::now();
    m_utc_epoch_seconds = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(utc_now.time_since_epoch()).count());

    logger()->info("sigmf_source configured (Blue): {} samples, format={}, sr={} Hz, cf={} Hz",
                  m_total_samples, m_format.datatype_str, m_sample_rate, m_center_frequency);
}

void sigmf_source::configure_file() {
    logger()->info("sigmf_source: configure_file() called with file_path='{}'", m_file_path);

    if (m_file_path.empty()) {
        logger()->warn("sigmf_source: cannot configure - file_path is empty");
        return;
    }

    // Reset all file-specific state so detection starts fresh.
    // Without this, switching between raw and blue files leaves stale
    // values (especially m_filetype) that corrupt the next detection.
    m_filetype = "raw";
    m_data_start_offset = 0;
    m_sample_rate = 0.0;
    m_center_frequency = 0.0;
    m_description.clear();
    m_format = SigmfFormat{};
    m_bytes_per_sample = 0;
    m_total_samples = 0;

    logger()->info("sigmf_source: calling parse_metadata()");
    parse_metadata();

    // Determine data file path
    std::string data_path = m_file_path;
    if (data_path.ends_with(".sigmf-meta")) {
        // Explicit meta file path - derive data file
        data_path = data_path.substr(0, data_path.length() - 11) + ".sigmf-data";
    } else if (data_path.ends_with(".sigmf-data")) {
        // Already a data file path - use as-is
    } else if (data_path.ends_with(".blue")) {
        // Blue file - handle separately
    } else if (std::filesystem::exists(data_path) && !std::filesystem::is_directory(data_path)) {
        // File exists as-is (raw binary file without .sigmf- extension) - use directly
        logger()->info("sigmf_source: using raw data file directly: {}", data_path);
    } else {
        // Fall back to appending .sigmf-data
        data_path = data_path + ".sigmf-data";
    }

    // Detect and handle file type
    std::string detected_filetype = detect_filetype(data_path);

    if (detected_filetype == "bluefile-1000") {
        // Configure as MIDAS Blue file
        configure_blue_file();
        send_metadata_to_port();
        return;
    }

    // Continue with raw/SigMF file handling
    m_filetype = "raw";
    m_data_start_offset = 0;  // Raw files start at byte 0

    // Verify file exists
    if (!std::filesystem::exists(data_path)) {
        throw std::runtime_error(std::format("sigmf_source: data file not found: {}", data_path));
    }

    // Determine if we need to byte-swap for endianness
    constexpr bool host_is_little_endian = (std::endian::native == std::endian::little);
    bool needs_swap = (m_format.is_big_endian == host_is_little_endian) && (m_format.bitwidth >= 16);

    // Calculate bytes per sample from format
    m_bytes_per_sample = m_format.bytes_per_sample();
    if (m_bytes_per_sample == 0) {
        throw std::runtime_error("sigmf_source: invalid format - bytes_per_sample is 0");
    }

    // Memory-map the file as raw bytes
    // Use writable=true if we need to swap bytes (MAP_PRIVATE allows in-place modification)
    logger()->info("sigmf_source: memory-mapping file {} (writable={}, bytes_per_sample={})",
                  data_path, needs_swap, m_bytes_per_sample);
    m_mmap = std::make_shared<MmapRegion<std::byte>>(data_path, needs_swap);
    m_total_samples = m_mmap->file_size() / m_bytes_per_sample;
    m_current_byte_offset = 0;

    // Apply endianness conversion in-place if needed
    if (needs_swap) {
        logger()->info("sigmf_source: applying endianness swap on {} bytes", m_mmap->file_size());
        apply_endianness_swap();
        m_format.is_big_endian = (std::endian::native == std::endian::big);
    }

    // Calculate rate control timing
    if (m_rate_control) {
        calculate_timing();
    }

    m_configured = true;

    // Reset rate control epoch so the catch-up logic doesn't see the time
    // between start() and now as a deficit to fill, which causes a burst
    // that OOM-kills the pod.
    m_start_time = std::chrono::steady_clock::now();
    m_samples_sent = 0;

    // Capture UTC epoch for absolute timestamps
    auto utc_now = std::chrono::system_clock::now();
    m_utc_epoch_seconds = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(utc_now.time_since_epoch()).count());

    logger()->info("sigmf_source configured: {} samples ({} MB), format={}, sr={} Hz, cf={} Hz, rate_control={}",
                  m_total_samples, m_mmap->file_size() / (1024 * 1024),
                  m_format.datatype_str, m_sample_rate, m_center_frequency, m_rate_control);

    // Send metadata to downstream components
    send_metadata_to_port();
}

auto sigmf_source::start() -> void {
    logger()->info("sigmf_source starting (streaming={}, configured={})", m_streaming, m_configured);
    m_current_byte_offset = m_data_start_offset;  // For Blue files, this skips the header
    m_samples_sent = 0;
    m_eof = false;

    // Always track start time (needed for wallclock_timestamps and rate_control)
    auto now = std::chrono::steady_clock::now();
    m_start_time = now;

    // Capture UTC epoch for absolute timestamps
    auto utc_now = std::chrono::system_clock::now();
    m_utc_epoch_seconds = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(utc_now.time_since_epoch()).count());

    // Only do full setup if configured
    if (m_configured) {
        // Recalculate timing in case max_sample_rate changed
        if (m_rate_control) {
            calculate_timing();
            m_next_send_time = now;
        }
        send_metadata_to_port();
    }

    composite::component::start();
}

auto sigmf_source::stop() -> void {
    // mmap is cleaned up automatically via shared_ptr
    composite::component::stop();
}

auto sigmf_source::process_chunk() -> composite::retval {
    using enum composite::retval;

    if (m_eof && !m_loop) {
        return NOOP;
    }

    if (m_eof && m_loop) {
        // Reset to beginning for looping (accounting for data offset in Blue files)
        logger()->trace("sigmf_source: looping back to start");
        m_current_byte_offset = m_data_start_offset;
        m_eof = false;
        send_metadata_to_port();
    }

    // Calculate samples to read (working in sample units, converting to bytes for mmap)
    // Account for data_start_offset (non-zero for Blue files where header precedes data)
    std::size_t current_sample = (m_current_byte_offset - m_data_start_offset) / m_bytes_per_sample;
    std::size_t samples_remaining = m_total_samples - current_sample;
    std::size_t samples_to_read = std::min(m_chunk_samples, samples_remaining);

    if (samples_to_read == 0) {
        m_eof = true;
        if (!m_loop) {
            logger()->info("sigmf_source: reached end of file after {} samples", m_samples_sent);
        }
        return NOOP;
    }

    // Convert to bytes for the mmap view
    std::size_t bytes_to_read = samples_to_read * m_bytes_per_sample;

    // Create zero-copy view into the mmap region (in bytes)
    auto view = std::make_shared<MmapView<std::byte>>(m_mmap, m_current_byte_offset, bytes_to_read);
    m_current_byte_offset += bytes_to_read;
    m_samples_sent += samples_to_read;

    // Create timestamp (absolute UTC)
    // Offset from start is computed as either elapsed wall-clock or sample-based time,
    // then anchored to the UTC epoch captured at start.
    composite::timestamp ts;
    double offset_seconds;

    if (m_wallclock_timestamps) {
        // Wall-clock mode: offset based on real elapsed time since start
        auto now = std::chrono::steady_clock::now();
        offset_seconds = std::chrono::duration<double>(now - m_start_time).count();
    } else {
        // File-time mode (default): offset based on sample position
        if (m_effective_sample_rate > 0) {
            offset_seconds = static_cast<double>(m_samples_sent) / m_effective_sample_rate;
        } else {
            offset_seconds = 0.0;
        }
    }

    // Split offset into integer seconds and sub-second fractional part
    auto offset_int = static_cast<uint32_t>(offset_seconds);
    double offset_frac = offset_seconds - offset_int;

    ts.seconds = m_utc_epoch_seconds + offset_int;
    ts.picoseconds = static_cast<uint64_t>(offset_frac * 1e12);

    // Send data - zero-copy, the view keeps the mmap alive
    composite::immutable_buffer<std::byte> buf(view);
    m_out_port.send_data(std::move(buf), ts);

    return NORMAL;
}

auto sigmf_source::process() -> composite::retval {
    using enum composite::retval;

    // Don't process if not streaming or not configured
    if (!m_streaming || !m_configured) {
        return NOOP;
    }

    if (!m_rate_control) {
        // Fast-as-possible mode - just process one chunk
        return process_chunk();
    }

    // Rate-controlled mode with wall-clock tracking for long-term accuracy
    auto now = std::chrono::steady_clock::now();

    // Calculate how many samples we SHOULD have sent by now based on wall-clock time
    auto elapsed = std::chrono::duration<double>(now - m_start_time).count();
    auto expected_samples = static_cast<uint64_t>(elapsed * m_effective_sample_rate);

    // If we're ahead of schedule, sleep until we should send more
    if (m_samples_sent >= expected_samples) {
        // Calculate when we should next send based on samples already sent
        double next_send_time_sec = static_cast<double>(m_samples_sent) / m_effective_sample_rate;
        auto wake_time = m_start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(next_send_time_sec));

        // But don't sleep for less than the minimum interval
        if (wake_time > now) {
            std::this_thread::sleep_until(wake_time);
        }

        // Recalculate expected samples after sleeping
        now = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<double>(now - m_start_time).count();
        expected_samples = static_cast<uint64_t>(elapsed * m_effective_sample_rate);
    }

    // Send chunks until we've caught up to where we should be
    // This compensates for sleep jitter and processing overhead
    uint32_t chunks_sent = 0;
    while (m_samples_sent < expected_samples || chunks_sent < m_chunks_per_wakeup) {
        auto result = process_chunk();
        if (result != NORMAL) {
            return result;
        }
        chunks_sent++;

        // Safety: don't send more than 2x the normal bundle to avoid runaway
        if (chunks_sent >= m_chunks_per_wakeup * 2) {
            break;
        }
    }

    return NORMAL;
}

#ifndef UNIT_TESTS
extern "C" {
auto create(std::string_view id) -> std::shared_ptr<composite::component> {
    return std::make_shared<sigmf_source>(id);
}
}
#endif
