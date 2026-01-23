/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * SigMF file source component with rate-controlled playback.
 * Uses mmap for zero-copy file reading at high sample rates.
 */

#pragma once

#include <composite/composite.hpp>

#include <chrono>
#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
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
    std::optional<double> sample_rate;
    std::optional<double> center_frequency;
    std::optional<double> bandwidth;
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
 * Lightweight view into a portion of an MmapRegion.
 * Satisfies ValidBufferContainer for use with immutable_buffer.
 * Keeps the underlying MmapRegion alive via shared_ptr.
 */
template<typename T>
class MmapView {
public:
    using value_type = T;

    MmapView(std::shared_ptr<MmapRegion<T>> region, std::size_t offset, std::size_t count)
        : m_region(std::move(region)), m_offset(offset), m_count(count) {}

    MmapView(const MmapView&) = default;
    MmapView& operator=(const MmapView&) = default;
    MmapView(MmapView&&) = default;
    MmapView& operator=(MmapView&&) = default;

    T* data() noexcept { return m_region->data() + m_offset; }
    const T* data() const noexcept { return m_region->data() + m_offset; }
    std::size_t size() const noexcept { return m_count; }

    T* begin() noexcept { return data(); }
    T* end() noexcept { return data() + m_count; }
    const T* begin() const noexcept { return data(); }
    const T* end() const noexcept { return data() + m_count; }

private:
    std::shared_ptr<MmapRegion<T>> m_region;
    std::size_t m_offset{0};
    std::size_t m_count{0};
};

template<typename T>
class sigmf_source : public composite::component {
    using output_port_t = composite::output_port<composite::immutable_buffer<T>>;

public:
    explicit sigmf_source(std::string_view id = "sigmf_source");
    ~sigmf_source() override = default;

    auto initialize() -> void override;
    auto start() -> void override;
    auto stop() -> void override;
    auto process() -> composite::retval override;

private:
    void parse_metadata();
    void send_metadata_to_port();
    void calculate_timing();
    auto process_chunk() -> composite::retval;

    // Datatype parsing
    static auto parse_datatype(std::string_view datatype_str) -> std::optional<SigmfFormat>;

    // Endianness handling - swap bytes in mmap region (MAP_PRIVATE allows this)
    void apply_endianness_swap();

    output_port_t m_out_port{"data_out"};

    // Configuration properties
    std::string m_file_path;
    std::size_t m_chunk_samples{1024};
    bool m_loop{false};
    uint32_t m_stream_id{0};

    // Rate control properties
    bool m_rate_control{true};
    double m_max_sample_rate{-1.0};

    // Timestamp mode: false = file-time (sample-based), true = wall-clock
    bool m_wallclock_timestamps{false};

    // Metadata overrides (override values from .sigmf-meta file)
    struct_props::sigmf_overrides m_overrides;

    // Memory-mapped file
    std::shared_ptr<MmapRegion<T>> m_mmap;
    std::size_t m_current_index{0};
    std::size_t m_total_samples{0};
    bool m_eof{false};

    // Metadata from SigMF file
    double m_sample_rate{0.0};
    double m_center_frequency{0.0};
    std::string m_description;
    SigmfFormat m_format;

    // Rate control timing state
    double m_effective_sample_rate{0.0};
    std::chrono::steady_clock::time_point m_start_time;
    std::chrono::steady_clock::time_point m_next_send_time;
    std::chrono::microseconds m_chunk_interval{0};
    uint32_t m_chunks_per_wakeup{1};

    // Track samples sent for timestamp calculation
    uint64_t m_samples_sent{0};
};

// Type aliases for common sample types
using sigmf_source_cf32 = sigmf_source<std::complex<float>>;
using sigmf_source_ci16 = sigmf_source<std::complex<int16_t>>;
using sigmf_source_ci8 = sigmf_source<std::complex<int8_t>>;
using sigmf_source_f32 = sigmf_source<float>;
using sigmf_source_i16 = sigmf_source<int16_t>;

// Property traits for sigmf_overrides struct
template<>
struct composite::properties::property_traits<struct_props::sigmf_overrides> {
    static void register_fields(composite::properties::property_set& ps, struct_props::sigmf_overrides& o) {
        using enum composite::properties::config_type;
        ps.add("sample_rate", o.sample_rate, RUNTIME);
        ps.add("center_frequency", o.center_frequency, RUNTIME);
        ps.add("bandwidth", o.bandwidth, RUNTIME);
    }
};

#ifndef UNIT_TESTS
extern "C" {
auto create(std::string_view id, std::string_view type = "cf32") -> std::shared_ptr<composite::component>;
}
#endif
