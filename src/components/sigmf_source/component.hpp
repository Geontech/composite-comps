/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * SigMF file source component with rate-controlled playback.
 * Reads SigMF metadata and data files, outputs samples through a port
 * at the file's native sample rate (or a configurable max rate).
 */

#pragma once

#include <composite/composite.hpp>

#include <chrono>
#include <complex>
#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * Parsed SigMF datatype format information.
 * Supports formats like: cf32_le, ri16_be, cu8, etc.
 */
struct SigmfFormat {
    std::string datatype_str;
    bool is_complex{true};
    bool is_big_endian{false};  // true = big endian, false = little endian
    enum class DataType { FLOAT, SIGNED_INT, UNSIGNED_INT } datatype{DataType::FLOAT};
    uint32_t bitwidth{32};

    [[nodiscard]] uint32_t bytes_per_sample() const {
        uint32_t bytes = bitwidth / 8;
        return is_complex ? bytes * 2 : bytes;
    }
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

    // Endianness handling - swap bytes in loaded data buffer
    void apply_endianness_swap();
    static void byte_swap_16(std::vector<T>& data);
    static void byte_swap_32(std::vector<T>& data);

    output_port_t m_out_port{"data_out"};

    // Configuration properties
    std::string m_file_path;              // Path to .sigmf-meta or base name
    std::size_t m_chunk_samples{1024};    // Samples per output buffer
    bool m_loop{false};                   // Loop file when EOF reached
    uint32_t m_stream_id{0};              // Stream identifier

    // Rate control properties
    bool m_rate_control{true};            // Enable sample-rate pacing (default: enabled)
    double m_max_sample_rate{-1.0};       // Max rate limit, -1 = use file's sample rate

    // File state
    std::ifstream m_data_file;
    std::size_t m_samples_read{0};
    std::size_t m_total_samples{0};
    bool m_eof{false};

    // Metadata from SigMF file
    double m_sample_rate{0.0};
    double m_center_frequency{0.0};
    std::string m_description;
    SigmfFormat m_format;

    // Rate control timing state
    double m_effective_sample_rate{0.0};
    std::chrono::steady_clock::time_point m_next_send_time;
    std::chrono::microseconds m_chunk_interval{0};
    uint32_t m_chunks_per_wakeup{1};

    // Pre-loaded file data for endianness conversion
    std::vector<T> m_file_data;
    std::size_t m_file_data_index{0};
    bool m_data_preloaded{false};
};

// Type aliases for common sample types
using sigmf_source_cf32 = sigmf_source<std::complex<float>>;
using sigmf_source_ci16 = sigmf_source<std::complex<int16_t>>;
using sigmf_source_ci8 = sigmf_source<std::complex<int8_t>>;
using sigmf_source_f32 = sigmf_source<float>;
using sigmf_source_i16 = sigmf_source<int16_t>;

#ifndef UNIT_TESTS
extern "C" {
auto create(std::string_view id, std::string_view type = "cf32") -> std::shared_ptr<composite::component>;
}
#endif
