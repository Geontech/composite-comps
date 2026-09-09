/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * SigMF file source component with rate-controlled playback.
 * Uses mmap for zero-copy file reading at high sample rates.
 */

#pragma once

#include <composite/composite.hpp>
// Not pulled in by the composite.hpp umbrella.
#include <composite/core/source_component.hpp>
#include <array>
#include <chrono>
#include <complex>
#include <cstdint>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
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

    // A vector-valued property is diffed by value, so the element type (and every
    // nested struct) has to be equality-comparable.
    friend auto operator==(const sigmf_overrides&, const sigmf_overrides&) -> bool = default;

    COMPOSITE_FIELDS(sigmf_overrides,
        (sample_rate, runtime, unit("sps")),
        (center_frequency, runtime, unit("Hz")),
        (bandwidth, runtime, unit("Hz")),
        (datatype, runtime),
        (filetype, runtime));
};

// Incoming file identity
struct file_spec {
    // NOTE: the reflected field name IS the wire key ("path"), so this member is
    // named for the JSON contract rather than for its C++ role. file_runtime has
    // its own file_path, which is internal and not reflected.
    std::string path; // identity of file
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

    friend auto operator==(const file_spec&, const file_spec&) -> bool = default;

    COMPOSITE_FIELDS(file_spec,
        (path, runtime),
        (stream_id, runtime),
        (destination_ip, runtime),
        (destination_port, runtime),
        (loop, runtime),
        (rate_control, runtime),
        (wallclock_timestamps, runtime),
        (max_sample_rate, runtime, unit("sps")),
        (overrides, runtime));
};

} // namespace struct_props

/**
 * RAII wrapper for memory-mapped file region.
 * Handles mmap/munmap lifecycle and provides container-like interface.
 */
template<typename T>
class MmapRegion {
public:
    using value_type = T;

    MmapRegion() = default;

    explicit MmapRegion(const std::string& path) {
        // Read-only throughout. Byte swapping happens on the per-chunk copy, not
        // on the mapping, so nothing ever writes here -- which is what keeps a
        // foreign-endian recording from faulting a private copy of the whole file
        // into RSS, and lets a read-only mount or recording play.
        m_fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (m_fd < 0) {
            throw std::runtime_error("MmapRegion: failed to open file: " + path);
        }

        // fstat rather than lseek: lseek reports -1 on failure, which as an
        // unsigned size became an enormous length handed straight to mmap, and it
        // says nothing about what kind of file this is. Anything but a regular
        // file has no meaningful mappable length (a directory, a FIFO, a socket).
        struct ::stat st{};
        if (::fstat(m_fd, &st) != 0) {
            ::close(m_fd);
            m_fd = -1;
            throw std::runtime_error("MmapRegion: fstat failed for: " + path);
        }
        if (!S_ISREG(st.st_mode)) {
            ::close(m_fd);
            m_fd = -1;
            throw std::runtime_error("MmapRegion: not a regular file: " + path);
        }
        m_file_size = static_cast<std::size_t>(st.st_size);

        if (m_file_size == 0) {
            ::close(m_fd);
            m_fd = -1;
            throw std::runtime_error("MmapRegion: file is empty: " + path);
        }

        // MAP_PRIVATE with PROT_READ only: private so nothing can write through to
        // the file, and with no writes there is no copy-on-write either.
        m_data = static_cast<T*>(::mmap(nullptr, m_file_size, PROT_READ, MAP_PRIVATE, m_fd, 0));

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
class sigmf_source : public composite::source_component<composite::immutable_buffer<std::byte>> {
    using base_t = composite::source_component<composite::immutable_buffer<std::byte>>;

public:
    explicit sigmf_source(std::string_view id = "sigmf_source");
    ~sigmf_source() override = default;

    auto initialize() -> void override;

protected:
    // The base owns the produce loop: it HOLDS a packet the downstream ring cannot
    // accept and re-sends it (no drop), and turns done() into EOS + FINISH. Both are
    // why this is a source_component rather than a plain component.
    auto produce() -> produce_result override;

    // start()/stop() are final in composite::component as of 0.5.2.
    auto on_worker_start() -> void override;
    auto on_worker_stop() -> void override;

private:
    struct file_runtime {
        // File identity
        std::string file_path;
        std::string data_path;

        // File / Format 
        char blue_type_code{0};
        std::shared_ptr<MmapRegion<std::byte>> mmap;
        composite::metadata cached_meta;
        // The published snapshot handed to every packet. Rebuilt only when
        // cached_meta changes (md_dirty), because make_metadata allocates and
        // this is the per-chunk hot path.
        composite::metadata_ptr latched_md;
        bool md_dirty{true};
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
    auto next_chunk(file_runtime& rt) -> produce_result;
    // Byte-swaps one freshly-copied chunk in place, gated by rt.needs_swap.
    // Replaces the previous pair of whole-mapping swaps: those had to map the file
    // writably and touch every page up front, and re-ran in full on every remap.
    static void byteswap_chunk(const file_runtime& rt, std::span<std::byte> chunk);
    void build_metadata(const struct_props::file_spec& spec, file_runtime& rt, const parsed_metadata& meta);
    void stamp_destination(const struct_props::file_spec& spec, file_runtime& rt);
    
    // Static helpers. parse_datatype is protected rather than private so the
    // format-validation tests can exercise it directly -- it is pure and has no
    // component state, and the alternative is asserting on it only indirectly
    // through a file load.
protected:
    static auto parse_datatype(std::string_view datatype_str) -> std::optional<SigmfFormat>;

private:
    static auto to_composite_format(const SigmfFormat& fmt) -> composite::data_format;
    static std::string format_rate(double sps);
    static std::string stream_id_of(const file_runtime& rt); 

    // Vector of statuses
    std::vector<file_runtime> m_runtime;

    // Round-robin cursor into m_runtime. produce() yields ONE packet per call, so
    // without this the scan would always restart at 0 and starve later streams.
    std::size_t m_next_file{0};

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

    // MUST be the last data member. source_component::produce() is pure, so a worker
    // still in the produce loop when the leaf destructor has run would dispatch to a
    // pure virtual and abort. A base-class member runs too late; the stop has to live
    // in the leaf. See source_component's "Destruction (IMPORTANT)" note.
    composite::component::auto_stop m_auto_stop{*this};
};

// The factory ABI is emitted by COMPOSITE_REGISTER_SIMPLE in component.cpp; it
// declares create() and the composite_abi_version handshake the loader checks.
