// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../component.hpp"
#include "../../pkt_debug/component.hpp"

#include <composite/composite.hpp>

#include <complex>
#include <cstring>
#include <vector>

using Catch::Matchers::WithinAbs;

// =============================================================================
// Test Fixture - Connects pkt_builder to pkt_debug for end-to-end testing
// =============================================================================

// Helper to get composite::data_format for a type
template<typename T>
auto get_data_format() -> composite::data_format {
    composite::data_format fmt{};
    if constexpr (std::is_same_v<T, std::complex<float>>) {
        fmt.is_complex = true;
        fmt.type = composite::data_type::floating_point;
        fmt.bit_width = 32;
    } else if constexpr (std::is_same_v<T, std::complex<int16_t>>) {
        fmt.is_complex = true;
        fmt.type = composite::data_type::signed_integer;
        fmt.bit_width = 16;
    } else if constexpr (std::is_same_v<T, std::complex<int8_t>>) {
        fmt.is_complex = true;
        fmt.type = composite::data_type::signed_integer;
        fmt.bit_width = 8;
    } else if constexpr (std::is_same_v<T, float>) {
        fmt.is_complex = false;
        fmt.type = composite::data_type::floating_point;
        fmt.bit_width = 32;
    } else if constexpr (std::is_same_v<T, int16_t>) {
        fmt.is_complex = false;
        fmt.type = composite::data_type::signed_integer;
        fmt.bit_width = 16;
    }
    return fmt;
}

// Templated test fixture - internally uses non-templated pkt_builder
// but provides typed interface for test convenience
template<typename T>
struct PktBuilderTestFixture {
    using sample_t = T;

    std::shared_ptr<pkt_builder> builder;
    std::shared_ptr<pkt_debug> debug;

    // Source port to feed data into pkt_builder (as std::byte)
    std::shared_ptr<composite::output_port<composite::immutable_buffer<std::byte>>> source_port;

    // Intermediate port to capture raw packets (between builder and debug)
    std::shared_ptr<composite::input_port<composite::immutable_buffer<uint8_t>>> packet_capture_port;

    PktBuilderTestFixture() {
        builder = std::make_shared<pkt_builder>("test_pkt_builder");
        debug = std::make_shared<pkt_debug>("test_pkt_debug");
        debug->set_properties({{"log_interval_sec", "1000"}});  // Disable frequent logging in tests

        source_port = std::make_shared<composite::output_port<composite::immutable_buffer<std::byte>>>("source");
        packet_capture_port = std::make_shared<composite::input_port<composite::immutable_buffer<uint8_t>>>("capture");

        // Get ports from components
        auto* builder_in = builder->get_port<composite::input_port<composite::immutable_buffer<std::byte>>>("data_in");
        auto* builder_out = builder->get_port<composite::output_port<composite::immutable_buffer<uint8_t>>>("data_out");
        auto* debug_in = debug->get_port<composite::input_port<composite::immutable_buffer<uint8_t>>>("data_in");

        REQUIRE(builder_in != nullptr);
        REQUIRE(builder_out != nullptr);
        REQUIRE(debug_in != nullptr);

        // Connect: source -> pkt_builder -> pkt_debug
        source_port->connect(builder_in);
        builder_out->connect(debug_in);

        // Also capture packets for inspection
        builder_out->connect(packet_capture_port.get());
    }

    ~PktBuilderTestFixture() {
        debug->stop();  // Ensure metrics thread is terminated
    }

    void send_data(const std::vector<sample_t>& data, composite::metadata meta = {}) {
        // Set format metadata for the sample type
        meta.format = get_data_format<sample_t>();

        // Convert typed data to bytes
        auto byte_vec = std::make_shared<std::vector<std::byte>>(data.size() * sizeof(sample_t));
        std::memcpy(byte_vec->data(), data.data(), byte_vec->size());

        composite::immutable_buffer<std::byte> buf(byte_vec);
        source_port->send_metadata(meta);
        source_port->send_data(std::move(buf), composite::timestamp{123, 456000});
    }

    void process_pipeline() {
        // Process pkt_builder (single call processes all input data)
        builder->process();
        // Process pkt_debug until no more data (with safety limit)
        for (int i = 0; i < 100 && debug->process() == composite::retval::NORMAL; ++i) {}
    }

    auto get_captured_packets() -> std::vector<std::vector<uint8_t>> {
        std::vector<std::vector<uint8_t>> packets;
        while (packet_capture_port->size() > 0) {
            auto [data, ts, meta] = packet_capture_port->get_data();
            if (data) {
                packets.emplace_back(data.begin(), data.end());
            }
        }
        return packets;
    }
};

// =============================================================================
// Helper Functions
// =============================================================================

// Read big-endian uint32 from buffer
uint32_t read_be32(const uint8_t* p) {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) |
           static_cast<uint32_t>(p[3]);
}

// Read big-endian uint64 from buffer
uint64_t read_be64(const uint8_t* p) {
    return (static_cast<uint64_t>(read_be32(p)) << 32) | read_be32(p + 4);
}

// Generate test signal
template<typename T>
auto generate_test_signal(std::size_t num_samples) -> std::vector<T> {
    std::vector<T> signal(num_samples);
    for (std::size_t i = 0; i < num_samples; ++i) {
        if constexpr (std::is_same_v<T, std::complex<float>>) {
            signal[i] = {static_cast<float>(i), static_cast<float>(-static_cast<int>(i))};
        } else if constexpr (std::is_same_v<T, std::complex<int16_t>>) {
            signal[i] = {static_cast<int16_t>(i % 1000), static_cast<int16_t>(-(i % 1000))};
        } else if constexpr (std::is_same_v<T, std::complex<int8_t>>) {
            signal[i] = {static_cast<int8_t>(i % 100), static_cast<int8_t>(-(i % 100))};
        } else if constexpr (std::is_same_v<T, float>) {
            signal[i] = static_cast<float>(i);
        } else if constexpr (std::is_same_v<T, int16_t>) {
            signal[i] = static_cast<int16_t>(i % 1000);
        }
    }
    return signal;
}

// =============================================================================
// Tests - Complex Float (default type)
// =============================================================================

TEST_CASE("pkt_builder_cf32: Basic packet generation", "[pkt_builder][cf32]") {
    PktBuilderTestFixture<std::complex<float>> fixture;

    auto signal = generate_test_signal<std::complex<float>>(100);
    fixture.send_data(signal);

    // Process the pipeline
    fixture.builder->process();

    auto packets = fixture.get_captured_packets();

    // Should have at least 2 packets: 1 context + 1 data
    REQUIRE(packets.size() >= 2);

    // First packet should be context (type 0x4)
    uint32_t header0 = read_be32(packets[0].data());
    uint8_t pkt_type0 = (header0 >> 28) & 0x0F;
    REQUIRE(pkt_type0 == 0x04);  // Context packet

    // Second packet should be data (type 0x1 = data with stream ID)
    uint32_t header1 = read_be32(packets[1].data());
    uint8_t pkt_type1 = (header1 >> 28) & 0x0F;
    REQUIRE(pkt_type1 == 0x01);  // Data packet with stream ID
}

TEST_CASE("pkt_builder_cf32: Stream ID from metadata", "[pkt_builder][cf32]") {
    PktBuilderTestFixture<std::complex<float>> fixture;

    composite::metadata meta;
    meta.annotations["stream_id"] = "42";

    auto signal = generate_test_signal<std::complex<float>>(100);
    fixture.send_data(signal, meta);
    fixture.builder->process();

    auto packets = fixture.get_captured_packets();
    REQUIRE(packets.size() >= 2);

    // Check stream ID in context packet (offset 4 after header)
    uint32_t stream_id = read_be32(packets[0].data() + 4);
    REQUIRE(stream_id == 42);
}

TEST_CASE("pkt_builder_cf32: Default stream ID when no metadata", "[pkt_builder][cf32]") {
    PktBuilderTestFixture<std::complex<float>> fixture;

    // Set a non-zero default stream ID
    fixture.builder->set_properties({{"default_stream_id", "99"}});

    auto signal = generate_test_signal<std::complex<float>>(100);
    fixture.send_data(signal);  // No metadata
    fixture.builder->process();

    auto packets = fixture.get_captured_packets();
    REQUIRE(packets.size() >= 2);

    // Check stream ID uses default
    uint32_t stream_id = read_be32(packets[0].data() + 4);
    REQUIRE(stream_id == 99);
}

TEST_CASE("pkt_builder_cf32: Sequence numbers increment", "[pkt_builder][cf32]") {
    PktBuilderTestFixture<std::complex<float>> fixture;

    // Send multiple buffers to generate multiple data packets
    auto signal = generate_test_signal<std::complex<float>>(100);

    for (int i = 0; i < 5; ++i) {
        fixture.send_data(signal);
        fixture.builder->process();
    }

    auto packets = fixture.get_captured_packets();

    // Extract sequence numbers from data packets
    std::vector<uint16_t> data_seqs;
    for (const auto& pkt : packets) {
        uint32_t header = read_be32(pkt.data());
        uint8_t pkt_type = (header >> 28) & 0x0F;
        if (pkt_type == 0x01) {  // Data packet
            uint16_t seq = (header >> 16) & 0x0F;
            data_seqs.push_back(seq);
        }
    }

    // Verify sequence numbers increment (mod 16)
    REQUIRE(data_seqs.size() >= 5);
    for (size_t i = 1; i < data_seqs.size(); ++i) {
        uint16_t expected = (data_seqs[i - 1] + 1) % 16;
        REQUIRE(data_seqs[i] == expected);
    }
}

TEST_CASE("pkt_builder_cf32: Large payload splits into multiple packets", "[pkt_builder][cf32]") {
    PktBuilderTestFixture<std::complex<float>> fixture;

    // Set small max payload to force splitting
    fixture.builder->set_properties({{"max_payload_size", "50"}});

    // Send 200 samples - should split into multiple packets
    auto signal = generate_test_signal<std::complex<float>>(200);
    fixture.send_data(signal);
    fixture.builder->process();

    auto packets = fixture.get_captured_packets();

    // Count data packets
    int data_packet_count = 0;
    for (const auto& pkt : packets) {
        uint32_t header = read_be32(pkt.data());
        uint8_t pkt_type = (header >> 28) & 0x0F;
        if (pkt_type == 0x01) {
            data_packet_count++;
        }
    }

    // With 200 samples and max 50 per packet, should have at least 4 data packets
    REQUIRE(data_packet_count >= 4);
}

// =============================================================================
// Tests - Complex Int16
// =============================================================================

TEST_CASE("pkt_builder_ci16: Generates correct format in context", "[pkt_builder][ci16]") {
    PktBuilderTestFixture<std::complex<int16_t>> fixture;

    auto signal = generate_test_signal<std::complex<int16_t>>(100);
    fixture.send_data(signal);
    fixture.builder->process();

    auto packets = fixture.get_captured_packets();
    REQUIRE(packets.size() >= 2);

    // Context packet should indicate complex signed int 16-bit
    // Parse CIF0 and data format field
    // This is a basic sanity check - detailed format parsing is done by pkt_debug
    uint32_t header = read_be32(packets[0].data());
    uint8_t pkt_type = (header >> 28) & 0x0F;
    REQUIRE(pkt_type == 0x04);  // Context packet
}

TEST_CASE("pkt_builder_ci16: Correct payload size", "[pkt_builder][ci16]") {
    PktBuilderTestFixture<std::complex<int16_t>> fixture;

    auto signal = generate_test_signal<std::complex<int16_t>>(100);
    fixture.send_data(signal);
    fixture.builder->process();

    auto packets = fixture.get_captured_packets();

    // Find data packet and verify size
    for (const auto& pkt : packets) {
        uint32_t header = read_be32(pkt.data());
        uint8_t pkt_type = (header >> 28) & 0x0F;
        if (pkt_type == 0x01) {
            // complex<int16_t> = 4 bytes per sample
            // 100 samples = 400 bytes payload
            // Plus header overhead
            REQUIRE(pkt.size() >= 400);
            break;
        }
    }
}

// =============================================================================
// Tests - Complex Int8
// =============================================================================

TEST_CASE("pkt_builder_ci8: Generates valid packets", "[pkt_builder][ci8]") {
    PktBuilderTestFixture<std::complex<int8_t>> fixture;

    auto signal = generate_test_signal<std::complex<int8_t>>(100);
    fixture.send_data(signal);
    fixture.builder->process();

    auto packets = fixture.get_captured_packets();
    REQUIRE(packets.size() >= 2);

    // Verify context and data packets are present
    bool has_context = false;
    bool has_data = false;
    for (const auto& pkt : packets) {
        uint32_t header = read_be32(pkt.data());
        uint8_t pkt_type = (header >> 28) & 0x0F;
        if (pkt_type == 0x04) has_context = true;
        if (pkt_type == 0x01) has_data = true;
    }
    REQUIRE(has_context);
    REQUIRE(has_data);
}

// =============================================================================
// Tests - Real Float
// =============================================================================

TEST_CASE("pkt_builder_f32: Generates valid packets for real data", "[pkt_builder][f32]") {
    PktBuilderTestFixture<float> fixture;

    auto signal = generate_test_signal<float>(100);
    fixture.send_data(signal);
    fixture.builder->process();

    auto packets = fixture.get_captured_packets();
    REQUIRE(packets.size() >= 2);

    // Verify we get context and data
    bool has_context = false;
    bool has_data = false;
    for (const auto& pkt : packets) {
        uint32_t header = read_be32(pkt.data());
        uint8_t pkt_type = (header >> 28) & 0x0F;
        if (pkt_type == 0x04) has_context = true;
        if (pkt_type == 0x01) has_data = true;
    }
    REQUIRE(has_context);
    REQUIRE(has_data);
}

// =============================================================================
// Tests - pkt_debug parsing
// =============================================================================

TEST_CASE("pkt_debug: Parses packets without errors", "[pkt_debug]") {
    PktBuilderTestFixture<std::complex<float>> fixture;

    // Initialize debug component
    fixture.debug->initialize();

    auto signal = generate_test_signal<std::complex<float>>(100);
    fixture.send_data(signal);

    // Process full pipeline
    fixture.process_pipeline();

    // If we get here without exceptions, parsing succeeded
    SUCCEED("pkt_debug parsed all packets without errors");

    fixture.debug->stop();
}

TEST_CASE("pkt_debug: Parses multiple data types", "[pkt_debug]") {
    // Test that pkt_debug can handle packets from different pkt_builder types

    SECTION("complex float") {
        PktBuilderTestFixture<std::complex<float>> fixture;
        fixture.debug->initialize();
        fixture.send_data(generate_test_signal<std::complex<float>>(50));
        fixture.process_pipeline();
        fixture.debug->stop();
        SUCCEED();
    }

    SECTION("complex int16") {
        PktBuilderTestFixture<std::complex<int16_t>> fixture;
        fixture.debug->initialize();
        fixture.send_data(generate_test_signal<std::complex<int16_t>>(50));
        fixture.process_pipeline();
        fixture.debug->stop();
        SUCCEED();
    }

    SECTION("complex int8") {
        PktBuilderTestFixture<std::complex<int8_t>> fixture;
        fixture.debug->initialize();
        fixture.send_data(generate_test_signal<std::complex<int8_t>>(50));
        fixture.process_pipeline();
        fixture.debug->stop();
        SUCCEED();
    }

    SECTION("real float") {
        PktBuilderTestFixture<float> fixture;
        fixture.debug->initialize();
        fixture.send_data(generate_test_signal<float>(50));
        fixture.process_pipeline();
        fixture.debug->stop();
        SUCCEED();
    }
}

// =============================================================================
// Tests - Metadata propagation
// =============================================================================

TEST_CASE("pkt_builder: Context contains RF metadata", "[pkt_builder][metadata]") {
    PktBuilderTestFixture<std::complex<float>> fixture;

    // Set default metadata via properties
    fixture.builder->set_properties({
        {"default_center_frequency", "100000000"},  // 100 MHz
        {"default_bandwidth", "10000000"},          // 10 MHz
        {"default_sample_rate", "20000000"}         // 20 MHz
    });

    auto signal = generate_test_signal<std::complex<float>>(100);
    fixture.send_data(signal);
    fixture.builder->process();

    auto packets = fixture.get_captured_packets();
    REQUIRE(packets.size() >= 1);

    // Context packet should contain the metadata
    // The actual values are encoded as fixed-point in the context fields
    // This test just verifies the packet is large enough to contain context fields
    uint32_t header = read_be32(packets[0].data());
    uint16_t pkt_size_words = header & 0xFFFF;

    // Context packet with CIF0 + bandwidth + RF freq + sample rate + data format
    // should be at least 15 words (60 bytes)
    REQUIRE(pkt_size_words >= 15);
}
