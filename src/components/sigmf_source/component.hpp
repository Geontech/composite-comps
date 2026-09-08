/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * SigMF file source component with rate-controlled playback.
 * Uses mmap for zero-copy file reading at high sample rates.
 */

#pragma once

#include <composite/composite.hpp>
#include <array>
#include <chrono>
#include <complex>
#include <cstdint>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

/**
 * Parsed SigMF datatype format information.
 * Supports formats like: cf32_le, ri16_be, cu8, etc.
 */
struct SigmfFormat {
    std::string datatype_str;
    bool is_complex{true};
    bool is_big_endian{false};
    enum class DataType { FLOAT, SIGNED_INT, UNSIGNED_INT } datatype{DataType::FLOAT};
    uint32_t bitwidth{32};

    [[nodiscard]] uint32_t bytes_per_sample() const {
        uint32_t bytes = bitwidth / 8;
        return is_complex ? bytes * 2 : bytes;
    }
};

namespace struct_props {

/**
 * @brief Metadata override configuration for sigmf_source
 *
 * Allows manual override of SigMF file metadata.
 * If a field has a value, it overrides what's in the .sigmf-meta file.
 */
struct sigmf_overrides {
    // Runtime-mutable: edits propagate via refresh_runtime_from_spec.
    std::optional<double> sample_rate;
    std::optional<double> center_frequency;
    std::optional<double> bandwidth;
    // Load-time-only: changes after the file is loaded are ignored (the mmap
    // was sized / byte-swapped / opened against the original values, and
    // retroactively reinterpreting bytes is unsafe). To change either field,
    // remove the file (files[i] = null) and add it back.
    std::optional<std::string> datatype;  // e.g., "cf32_le", "ri16_be", "cu8"
    std::optional<std::string> filetype;  // "raw" or "bluefile-1000"
};

// Incoming file identity
struct file_spec {
    std::string file_path; // identity of file
    int32_t stream_id{0};
    // Per-file egress destination. Assigned by the controlling service, which
    // owns address allocation; this component only forwards what it is given.
    // destination_ip empty -> emit no annotation, so udp_sink's default_dest_ip
    // applies. destination_port 0 -> fall back to the component-level
    // destination_port property. Streams are separated by group address, not by
    // port, so that receivers joining one group are not forwarded the others.
    std::string destination_ip;
    uint32_t destination_port{0};
    bool loop{false};
    bool rate_control{true};
    bool wallclock_timestamps{false};
    double max_sample_rate{-1.0};
    sigmf_overrides overrides{};
};

} // namespace struct_props

// Property traits for sigmf_overrides struct
template<>
struct composite::properties::property_traits<struct_props::sigmf_overrides> {
    static void register_fields(composite::properties::property_set& ps, struct_props::sigmf_overrides& o) {
        using enum composite::properties::config_type;
        ps.add("sample_rate", o.sample_rate, RUNTIME);
        ps.add("center_frequency", o.center_frequency, RUNTIME);
        ps.add("bandwidth", o.bandwidth, RUNTIME);
        ps.add("datatype", o.datatype, RUNTIME);
        ps.add("filetype", o.filetype, RUNTIME);
    }
};

// File template
template<>
struct composite::properties::property_traits<struct_props::file_spec> {
    static constexpr std::string_view type_name = "file_spec";
    static void register_fields(composite::properties::property_set& ps,
                    struct_props::file_spec& s) {
        using composite::properties::config_type;
        ps.add("path", s.file_path, config_type::RUNTIME);
        ps.add("stream_id", s.stream_id, config_type::RUNTIME);
        ps.add("destination_ip", s.destination_ip, config_type::RUNTIME);
        ps.add("destination_port", s.destination_port, config_type::RUNTIME);
        ps.add("loop", s.loop, config_type::RUNTIME);
        ps.add("rate_control", s.rate_control, config_type::RUNTIME);
        ps.add("wallclock_timestamps", s.wallclock_timestamps, config_type::RUNTIME);
        ps.add("max_sample_rate", s.max_sample_rate, config_type::RUNTIME);
        ps.add("overrides", s.overrides, config_type::RUNTIME);
    }
};

/**
 * RAII wrapper for memory-mapped file region.
 * Handles mmap/munmap lifecycle and provides container-like interface.
 */
template<typename T>
class MmapRegion {
public:
    using value_type = T;

    MmapRegion() = default;

    MmapRegion(const std::string& path, bool writable = false) {
        int flags = writable ? O_RDWR : O_RDONLY;
        m_fd = ::open(path.c_str(), flags);
        if (m_fd < 0) {
            throw std::runtime_error("MmapRegion: failed to open file: " + path);
        }

        // Get file size
        m_file_size = ::lseek(m_fd, 0, SEEK_END);
        ::lseek(m_fd, 0, SEEK_SET);

        if (m_file_size == 0) {
            ::close(m_fd);
            m_fd = -1;
            throw std::runtime_error("MmapRegion: file is empty: " + path);
        }

        // Map the file
        int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
        int mflags = writable ? MAP_PRIVATE : MAP_SHARED;  // MAP_PRIVATE allows in-place modification
        m_data = static_cast<T*>(::mmap(nullptr, m_file_size, prot, mflags, m_fd, 0));

        if (m_data == MAP_FAILED) {
            ::close(m_fd);
            m_fd = -1;
            m_data = nullptr;
            throw std::runtime_error("MmapRegion: mmap failed for: " + path);
        }

        m_size = m_file_size / sizeof(T);

        // Advise kernel we'll be reading sequentially
        ::madvise(m_data, m_file_size, MADV_SEQUENTIAL);
    }

    ~MmapRegion() {
        if (m_data && m_data != MAP_FAILED) {
            ::munmap(m_data, m_file_size);
        }
        if (m_fd >= 0) {
            ::close(m_fd);
        }
    }

    // Non-copyable
    MmapRegion(const MmapRegion&) = delete;
    MmapRegion& operator=(const MmapRegion&) = delete;

    // Movable
    MmapRegion(MmapRegion&& other) noexcept
        : m_data(other.m_data), m_size(other.m_size),
          m_file_size(other.m_file_size), m_fd(other.m_fd) {
        other.m_data = nullptr;
        other.m_size = 0;
        other.m_file_size = 0;
        other.m_fd = -1;
    }

    MmapRegion& operator=(MmapRegion&& other) noexcept {
        if (this != &other) {
            if (m_data && m_data != MAP_FAILED) {
                ::munmap(m_data, m_file_size);
            }
            if (m_fd >= 0) {
                ::close(m_fd);
            }
            m_data = other.m_data;
            m_size = other.m_size;
            m_file_size = other.m_file_size;
            m_fd = other.m_fd;
            other.m_data = nullptr;
            other.m_size = 0;
            other.m_file_size = 0;
            other.m_fd = -1;
        }
        return *this;
    }

    T* data() noexcept { return m_data; }
    const T* data() const noexcept { return m_data; }
    std::size_t size() const noexcept { return m_size; }
    std::size_t file_size() const noexcept { return m_file_size; }
    bool valid() const noexcept { return m_data != nullptr && m_data != MAP_FAILED; }

    T& operator[](std::size_t idx) { return m_data[idx]; }
    const T& operator[](std::size_t idx) const { return m_data[idx]; }

private:
    T* m_data{nullptr};
    std::size_t m_size{0};
    std::size_t m_file_size{0};
    int m_fd{-1};
};

/**
 * @brief SigMF file source component with dynamic datatype support
 *
 * Reads SigMF files and outputs raw bytes with format metadata.
 * The datatype is determined at runtime from the .sigmf-meta file,
 * allowing dynamic file switching via REST API without recompilation.
 *
 * Output: immutable_buffer<std::byte> with metadata.format populated
 */
class sigmf_source : public composite::component {
    using output_port_t = composite::output_port<composite::immutable_buffer<std::byte>>;

public:
    explicit sigmf_source(std::string_view id = "sigmf_source");
    ~sigmf_source() override = default;

    auto initialize() -> void override;
    auto start() -> void override;
    auto stop() -> void override;
    auto process() -> composite::retval override;

private:
    struct file_runtime {
        // File identity
        std::string file_path;
        std::string data_path;

        // File / Format 
        char blue_type_code{0};
        std::shared_ptr<MmapRegion<std::byte>> mmap;
        composite::metadata cached_meta;
        std::size_t bytes_per_sample{0};
        std::size_t total_samples{0};
        std::size_t data_start_offset{0};
        bool needs_swap{false};

        // Playback state
        std::size_t current_byte_offset{0};
        uint64_t samples_sent{0};
        std::chrono::steady_clock::time_point start_time;
        uint32_t utc_epoch_seconds{0};
        bool eof{false};

        // Timing / rate control
        double effective_sample_rate{0.0};
        std::chrono::microseconds chunk_interval{0};
        uint32_t chunks_per_wakeup{1};
        bool rate_control{true};

        // Spec mirrored flags
        bool loop{false};
        bool wallclock_timestamps{false};

        // Lifecycle
        bool load_error{false};

        // Metrics
        static constexpr std::size_t kRollingWindow = 6;
        uint64_t prev_samples_sent{0};
        std::chrono::steady_clock::time_point prev_tick_time{};
        std::array<double, kRollingWindow> recent_rates{};
        std::size_t recent_idx{0};
        std::size_t recent_count{0};
        double rolling_rate_sps{0.0};
        uint64_t loop_count{0};
        enum class health { ok, warn, err };
        health prev_health{health::ok};
        bool prev_eof{false};
    };

    // Stores files metadata
    struct parsed_metadata {
        std::string filetype{"raw"};
        double sample_rate{0.0};
        double center_frequency{0.0};
        std::string description;
        SigmfFormat format;
    };

    // Property change listeners
    bool on_files_changed();
    bool on_streaming_changed();

    // File loading
    bool load_file(const struct_props::file_spec& spec, file_runtime& rt);
    bool load_blue_file(const struct_props::file_spec& spec, parsed_metadata& meta, file_runtime& rt);
    bool parse_metadata(const struct_props::file_spec& spec, parsed_metadata& meta);

    // Runtime Helpers
    void reset_runtime_for_playback(file_runtime& rt, std::chrono::steady_clock::time_point now, uint32_t utc_epoch_seconds);
    bool remap_runtime(file_runtime& rt);
    void refresh_runtime_from_spec(const struct_props::file_spec& spec, file_runtime& rt);
    void calculate_timing(const struct_props::file_spec& spec, file_runtime& rt, parsed_metadata& meta);

    // Metrics
    void spawn_metrics_thread();
    void stop_metrics_thread();
    void log_metrics();

    // Processing
    auto process_chunk(file_runtime& rt) -> composite::retval;
    bool apply_endianness_swap_raw(file_runtime& rt);
    bool apply_endianness_swap_blue(file_runtime& rt);
    void build_metadata(const struct_props::file_spec& spec, file_runtime& rt, const parsed_metadata& meta);
    void stamp_destination(const struct_props::file_spec& spec, file_runtime& rt);
    
    // Static helpers
    static auto parse_datatype(std::string_view datatype_str) -> std::optional<SigmfFormat>;
    static auto to_composite_format(const SigmfFormat& fmt) -> composite::data_format;
    static std::string format_rate(double sps);
    static std::string stream_id_of(const file_runtime& rt); 

    // Single output port
    output_port_t m_out_port{"data_out"};

    // Vector of statuses
    std::vector<file_runtime> m_runtime;

    // mutex and thread
    std::shared_mutex m_runtime_mutex;
    std::jthread m_metrics_thread;


    // Configuration (RUNTIME) properties
    bool m_streaming{false};
    std::vector<struct_props::file_spec> m_files;
    std::size_t m_chunk_samples{8192};
    uint32_t m_destination_port{5000};

    // Metrics constants
    static constexpr auto kMetricsTickInterval = std::chrono::seconds(5);
    static constexpr double kRateDeviationThreshold = 0.05;  // 5%
};

#ifndef UNIT_TESTS
extern "C" {
auto create(std::string_view id) -> std::shared_ptr<composite::component>;
}
#endif
