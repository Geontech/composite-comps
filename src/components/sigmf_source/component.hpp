/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * Simple SigMF file source component for testing.
 * Reads SigMF metadata and data files, outputs samples through a port.
 */

#pragma once

#include <composite/composite.hpp>

#include <complex>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>

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

    output_port_t m_out_port{"data_out"};

    // Configuration
    std::string m_file_path;           // Path to .sigmf-meta or base name
    std::size_t m_chunk_samples{4096}; // Samples per output buffer
    bool m_loop{false};                // Loop file when EOF reached

    // State
    std::ifstream m_data_file;
    std::size_t m_samples_read{0};
    std::size_t m_total_samples{0};
    bool m_eof{false};

    // Metadata from SigMF file
    double m_sample_rate{0.0};
    double m_center_frequency{0.0};
    std::string m_description;
    uint32_t m_stream_id{0};
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
