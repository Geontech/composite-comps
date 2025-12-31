#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <complex>
#include <vector>
#include <numeric>
#include <memory>

#include "component.hpp"
#include <composite/composite.hpp>
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>

// Helper to create metadata
composite::metadata create_metadata(bool is_complex, uint8_t bit_width, composite::data_type type, double sample_rate = 1.0) {
    composite::metadata meta;
    meta.format.type = type;
    meta.format.bit_width = bit_width;
    meta.format.is_complex = is_complex;
    meta.format.endianness = std::endian::native;
    meta.sample_rate = sample_rate;
    return meta;
}

// A simple test fixture to own the component and ports
struct FramerTestFixture {
    std::shared_ptr<framer<std::complex<float>>> uut;
    std::shared_ptr<composite::output_port<composite::immutable_buffer<uint8_t>>> source_port;
    std::shared_ptr<composite::input_port<composite::immutable_buffer<std::complex<float>>>> sink_port;

    FramerTestFixture() {
        uut = std::make_shared<framer<std::complex<float>>>("test");
        source_port = std::make_shared<composite::output_port<composite::immutable_buffer<uint8_t>>>("source");
        sink_port = std::make_shared<composite::input_port<composite::immutable_buffer<std::complex<float>>>>("sink");

        // Connect ports now that we have friend access
        source_port->connect(&uut->m_in_port);
        uut->m_out_port.connect(sink_port.get());
    }

    void reset() {
        // Create fresh component and reconnect ports
        uut = std::make_shared<framer<std::complex<float>>>("test");
        source_port = std::make_shared<composite::output_port<composite::immutable_buffer<uint8_t>>>("source");
        sink_port = std::make_shared<composite::input_port<composite::immutable_buffer<std::complex<float>>>>("sink");
        source_port->connect(&uut->m_in_port);
        uut->m_out_port.connect(sink_port.get());
    }

    void run_real_i8_test() {
        // 1. Set properties
        uut->m_frame_size = 16;
        uut->m_overlap = 8;
        uut->m_frame_count = 4;
        uut->property_change_handler(); // Manually trigger initialization

        // 2. Create metadata and input data
        auto meta = create_metadata(false, 8, composite::data_type::signed_integer);
        auto input_vec = std::make_shared<std::vector<uint8_t>>(32);
        // Fill with 0, 1, ..., 31
        std::iota(input_vec->begin(), input_vec->end(), 0);
        composite::immutable_buffer<uint8_t> input_buffer(input_vec);
        composite::timestamp ts{0, 0};

        // 3. Send metadata and data
        source_port->send_metadata(meta);
        source_port->send_data(input_buffer, ts);

        // 4. Process
        REQUIRE(uut->process() == composite::retval::NORMAL);

        // 5. Check output from the sink
        REQUIRE(sink_port->size() == 3); // Should have emitted 3 frames

        // Verify Frame 1
        auto [output_buf1, ts1, meta1] = sink_port->get_data();
        REQUIRE(output_buf1.size() == 16);
        for (size_t i = 0; i < 16; ++i) {
            INFO("Frame 1, Sample " << i);
            REQUIRE(output_buf1.data()[i].real() == static_cast<float>(static_cast<int8_t>(i)));
            REQUIRE(output_buf1.data()[i].imag() == 0.0f);
        }

        // Verify Frame 2
        auto [output_buf2, ts2, meta2] = sink_port->get_data();
        REQUIRE(output_buf2.size() == 16);
        for (size_t i = 0; i < 16; ++i) {
            INFO("Frame 2, Sample " << i);
            REQUIRE(output_buf2.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 8)));
            REQUIRE(output_buf2.data()[i].imag() == 0.0f);
        }

        // Verify Frame 3
        auto [output_buf3, ts3, meta3] = sink_port->get_data();
        REQUIRE(output_buf3.size() == 16);
        for (size_t i = 0; i < 16; ++i) {
            INFO("Frame 3, Sample " << i);
            REQUIRE(output_buf3.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 16)));
            REQUIRE(output_buf3.data()[i].imag() == 0.0f);
        }
    }

    void run_complex_i8_test() {
        reset();

        uut->m_frame_size = 8;
        uut->m_overlap = 4;
        uut->m_frame_count = 4;
        uut->property_change_handler();

        // Create complex i8 data: [0+0i, 1+1i, 2+2i, ..., 15+15i] = 32 bytes (16 complex samples)
        auto meta = create_metadata(true, 8, composite::data_type::signed_integer, 1e6);
        auto input_vec = std::make_shared<std::vector<uint8_t>>(32);
        for (size_t i = 0; i < 16; ++i) {
            (*input_vec)[i * 2] = static_cast<uint8_t>(i);     // Real (I)
            (*input_vec)[i * 2 + 1] = static_cast<uint8_t>(i); // Imag (Q)
        }
        composite::immutable_buffer<uint8_t> input_buffer(input_vec);
        composite::timestamp ts{1, 0};

        source_port->send_metadata(meta);
        source_port->send_data(input_buffer, ts);
        REQUIRE(uut->process() == composite::retval::NORMAL);

        // Should emit 3 frames: [0-7], [4-11], [8-15]
        REQUIRE(sink_port->size() == 3);

        // Frame 1: samples 0-7
        auto [out1, ts1, meta1] = sink_port->get_data();
        REQUIRE(out1.size() == 8);
        for (size_t i = 0; i < 8; ++i) {
            INFO("Frame 1, Sample " << i);
            REQUIRE(out1.data()[i].real() == static_cast<float>(static_cast<int8_t>(i)));
            REQUIRE(out1.data()[i].imag() == static_cast<float>(static_cast<int8_t>(i)));
        }

        // Frame 2: samples 4-11 (overlap with frame 1)
        auto [out2, ts2, meta2] = sink_port->get_data();
        REQUIRE(out2.size() == 8);
        for (size_t i = 0; i < 8; ++i) {
            INFO("Frame 2, Sample " << i);
            REQUIRE(out2.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 4)));
            REQUIRE(out2.data()[i].imag() == static_cast<float>(static_cast<int8_t>(i + 4)));
        }

        // Frame 3: samples 8-15
        auto [out3, ts3, meta3] = sink_port->get_data();
        REQUIRE(out3.size() == 8);
        for (size_t i = 0; i < 8; ++i) {
            INFO("Frame 3, Sample " << i);
            REQUIRE(out3.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 8)));
            REQUIRE(out3.data()[i].imag() == static_cast<float>(static_cast<int8_t>(i + 8)));
        }
    }

    void run_complex_i16_test() {
        reset();

        uut->m_frame_size = 4;
        uut->m_overlap = 2;
        uut->m_frame_count = 4;
        uut->property_change_handler();

        // Create complex i16 data: [1000+1000i, 2000+2000i, ..., 8000+8000i] = 32 bytes (8 complex samples)
        auto meta = create_metadata(true, 16, composite::data_type::signed_integer, 2e6);
        auto input_vec = std::make_shared<std::vector<uint8_t>>(32);
        auto* i16_data = reinterpret_cast<int16_t*>(input_vec->data());
        for (size_t i = 0; i < 8; ++i) {
            int16_t value = static_cast<int16_t>((i + 1) * 1000);
            i16_data[i * 2] = value;     // Real (I)
            i16_data[i * 2 + 1] = value; // Imag (Q)
        }
        composite::immutable_buffer<uint8_t> input_buffer(input_vec);
        composite::timestamp ts{2, 500'000'000'000}; // 2.5 seconds

        source_port->send_metadata(meta);
        source_port->send_data(input_buffer, ts);
        REQUIRE(uut->process() == composite::retval::NORMAL);

        // Should emit 3 frames: [0-3], [2-5], [4-7]
        REQUIRE(sink_port->size() == 3);

        // Frame 1: samples 0-3 (1000, 2000, 3000, 4000)
        auto [out1, ts1, meta1] = sink_port->get_data();
        REQUIRE(out1.size() == 4);
        for (size_t i = 0; i < 4; ++i) {
            float expected = static_cast<float>((i + 1) * 1000);
            INFO("Frame 1, Sample " << i);
            REQUIRE(out1.data()[i].real() == expected);
            REQUIRE(out1.data()[i].imag() == expected);
        }

        // Frame 2: samples 2-5 (3000, 4000, 5000, 6000)
        auto [out2, ts2, meta2] = sink_port->get_data();
        REQUIRE(out2.size() == 4);
        for (size_t i = 0; i < 4; ++i) {
            float expected = static_cast<float>((i + 3) * 1000);
            INFO("Frame 2, Sample " << i);
            REQUIRE(out2.data()[i].real() == expected);
            REQUIRE(out2.data()[i].imag() == expected);
        }

        // Frame 3: samples 4-7 (5000, 6000, 7000, 8000)
        auto [out3, ts3, meta3] = sink_port->get_data();
        REQUIRE(out3.size() == 4);
        for (size_t i = 0; i < 4; ++i) {
            float expected = static_cast<float>((i + 5) * 1000);
            INFO("Frame 3, Sample " << i);
            REQUIRE(out3.data()[i].real() == expected);
            REQUIRE(out3.data()[i].imag() == expected);
        }
    }

    void run_complex_cf32_test() {
        reset();

        uut->m_frame_size = 4;
        uut->m_overlap = 0; // No overlap for simplicity
        uut->m_frame_count = 4;
        uut->property_change_handler();

        // Create complex float data: [1.5+2.5i, 3.5+4.5i, 5.5+6.5i, 7.5+8.5i] = 32 bytes (4 complex samples)
        auto meta = create_metadata(true, 32, composite::data_type::floating_point, 5e6);
        auto input_vec = std::make_shared<std::vector<uint8_t>>(32);
        auto* cf32_data = reinterpret_cast<std::complex<float>*>(input_vec->data());
        cf32_data[0] = {1.5f, 2.5f};
        cf32_data[1] = {3.5f, 4.5f};
        cf32_data[2] = {5.5f, 6.5f};
        cf32_data[3] = {7.5f, 8.5f};
        composite::immutable_buffer<uint8_t> input_buffer(input_vec);
        composite::timestamp ts{5, 0};

        source_port->send_metadata(meta);
        source_port->send_data(input_buffer, ts);
        REQUIRE(uut->process() == composite::retval::NORMAL);

        // Should emit 1 frame: [0-3]
        REQUIRE(sink_port->size() == 1);

        auto [out1, ts1, meta1] = sink_port->get_data();
        REQUIRE(out1.size() == 4);
        REQUIRE(out1.data()[0] == std::complex<float>{1.5f, 2.5f});
        REQUIRE(out1.data()[1] == std::complex<float>{3.5f, 4.5f});
        REQUIRE(out1.data()[2] == std::complex<float>{5.5f, 6.5f});
        REQUIRE(out1.data()[3] == std::complex<float>{7.5f, 8.5f});
    }

    void run_multiple_batches_test() {
        reset();

        uut->m_frame_size = 8;
        uut->m_overlap = 4;
        uut->m_frame_count = 8;
        uut->property_change_handler();

        auto meta = create_metadata(false, 8, composite::data_type::signed_integer, 1e6);
        source_port->send_metadata(meta);

        // Send first batch: 0-15 (16 samples)
        // With frame_size=8, overlap=4 (hop=4), should emit frames at: 0, 4, 8
        auto input1 = std::make_shared<std::vector<uint8_t>>(16);
        std::iota(input1->begin(), input1->end(), 0);
        source_port->send_data(composite::immutable_buffer<uint8_t>(input1), {0, 0});
        REQUIRE(uut->process() == composite::retval::NORMAL);

        // Should emit 3 frames: [0-7], [4-11], [8-15]
        REQUIRE(sink_port->size() == 3);

        // Validate Frame 1: samples 0-7
        auto [out1, ts1, meta1] = sink_port->get_data();
        REQUIRE(out1.size() == 8);
        for (size_t i = 0; i < 8; ++i) {
            INFO("Batch 1, Frame 1, Sample " << i);
            REQUIRE(out1.data()[i].real() == static_cast<float>(static_cast<int8_t>(i)));
            REQUIRE(out1.data()[i].imag() == 0.0f);
        }

        // Validate Frame 2: samples 4-11
        auto [out2, ts2, meta2] = sink_port->get_data();
        REQUIRE(out2.size() == 8);
        for (size_t i = 0; i < 8; ++i) {
            INFO("Batch 1, Frame 2, Sample " << i);
            REQUIRE(out2.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 4)));
            REQUIRE(out2.data()[i].imag() == 0.0f);
        }

        // Validate Frame 3: samples 8-15
        auto [out_discard, ts_discard, meta_discard] = sink_port->get_data();
        REQUIRE(out_discard.size() == 8);
        for (size_t i = 0; i < 8; ++i) {
            INFO("Batch 1, Frame 3, Sample " << i);
            REQUIRE(out_discard.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 8)));
            REQUIRE(out_discard.data()[i].imag() == 0.0f);
        }

        // Send second batch: 16-31 (16 samples)
        // Should emit frames at: 12, 16, 20, 24 (but we only have data up to 31, so: 12, 16, 20, 24)
        auto input2 = std::make_shared<std::vector<uint8_t>>(16);
        std::iota(input2->begin(), input2->end(), 16);
        source_port->send_data(composite::immutable_buffer<uint8_t>(input2), {0, 16'000'000}); // 16 µs later
        REQUIRE(uut->process() == composite::retval::NORMAL);

        // Should emit 4 more frames: [12-19], [16-23], [20-27], [24-31]
        REQUIRE(sink_port->size() == 4);

        // Validate Frame 4: samples 12-19 (spans first and second batch)
        auto [out3, ts3, meta3] = sink_port->get_data();
        REQUIRE(out3.size() == 8);
        for (size_t i = 0; i < 8; ++i) {
            INFO("Batch 2, Frame 1 (cross-batch), Sample " << i);
            REQUIRE(out3.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 12)));
            REQUIRE(out3.data()[i].imag() == 0.0f);
        }

        // Validate Frame 5: samples 16-23
        auto [out4, ts4, meta4] = sink_port->get_data();
        REQUIRE(out4.size() == 8);
        for (size_t i = 0; i < 8; ++i) {
            INFO("Batch 2, Frame 2, Sample " << i);
            REQUIRE(out4.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 16)));
            REQUIRE(out4.data()[i].imag() == 0.0f);
        }

        // Validate Frame 6: samples 20-27
        auto [out5, ts5, meta5] = sink_port->get_data();
        REQUIRE(out5.size() == 8);
        for (size_t i = 0; i < 8; ++i) {
            INFO("Batch 2, Frame 3, Sample " << i);
            REQUIRE(out5.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 20)));
            REQUIRE(out5.data()[i].imag() == 0.0f);
        }

        // Validate Frame 7: samples 24-31
        auto [out6, ts6, meta6] = sink_port->get_data();
        REQUIRE(out6.size() == 8);
        for (size_t i = 0; i < 8; ++i) {
            INFO("Batch 2, Frame 4, Sample " << i);
            REQUIRE(out6.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 24)));
            REQUIRE(out6.data()[i].imag() == 0.0f);
        }
    }

    void run_timestamp_test() {
        reset();

        uut->m_frame_size = 4;
        uut->m_overlap = 0; // No overlap for easier timestamp calculation
        uut->m_frame_count = 4;
        uut->property_change_handler();

        // Use 1 MHz sample rate for easy picosecond calculations
        // 1 sample @ 1 MHz = 1 microsecond = 1,000,000 picoseconds
        auto meta = create_metadata(false, 8, composite::data_type::signed_integer, 1e6);
        composite::timestamp ts_origin{100, 0}; // Start at 100 seconds

        source_port->send_metadata(meta);

        // Send 12 samples (should produce 3 frames of 4 samples each)
        auto input = std::make_shared<std::vector<uint8_t>>(12);
        std::iota(input->begin(), input->end(), 0);
        source_port->send_data(composite::immutable_buffer<uint8_t>(input), ts_origin);
        REQUIRE(uut->process() == composite::retval::NORMAL);

        REQUIRE(sink_port->size() == 3);

        // Frame 1: samples 0-3, timestamp should be ts_origin
        auto [out1, ts1, meta1] = sink_port->get_data();
        REQUIRE(out1.size() == 4);
        REQUIRE(ts1.seconds == 100);
        REQUIRE(ts1.picoseconds == 0);
        // Validate frame content (ramp: 0, 1, 2, 3)
        for (size_t i = 0; i < 4; ++i) {
            INFO("Frame 1, Sample " << i);
            REQUIRE(out1.data()[i].real() == static_cast<float>(static_cast<int8_t>(i)));
            REQUIRE(out1.data()[i].imag() == 0.0f);
        }

        // Frame 2: samples 4-7, timestamp should be ts_origin + 4 samples
        // 4 samples @ 1 MHz = 4 microseconds = 4,000,000 picoseconds
        auto [out2, ts2, meta2] = sink_port->get_data();
        REQUIRE(out2.size() == 4);
        REQUIRE(ts2.seconds == 100);
        REQUIRE(ts2.picoseconds == 4'000'000);
        // Validate frame content (ramp: 4, 5, 6, 7)
        for (size_t i = 0; i < 4; ++i) {
            INFO("Frame 2, Sample " << i);
            REQUIRE(out2.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 4)));
            REQUIRE(out2.data()[i].imag() == 0.0f);
        }

        // Frame 3: samples 8-11, timestamp should be ts_origin + 8 samples
        // 8 samples @ 1 MHz = 8 microseconds = 8,000,000 picoseconds
        auto [out3, ts3, meta3] = sink_port->get_data();
        REQUIRE(out3.size() == 4);
        REQUIRE(ts3.seconds == 100);
        REQUIRE(ts3.picoseconds == 8'000'000);
        // Validate frame content (ramp: 8, 9, 10, 11)
        for (size_t i = 0; i < 4; ++i) {
            INFO("Frame 3, Sample " << i);
            REQUIRE(out3.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 8)));
            REQUIRE(out3.data()[i].imag() == 0.0f);
        }
    }

    void run_metadata_propagation_test() {
        reset();

        uut->m_frame_size = 4;
        uut->m_overlap = 0;
        uut->m_frame_count = 4;
        uut->property_change_handler();

        // Send real i8 metadata
        auto meta = create_metadata(false, 8, composite::data_type::signed_integer, 10e6);
        meta.center_frequency = 915e6; // 915 MHz

        source_port->send_metadata(meta);

        auto input = std::make_shared<std::vector<uint8_t>>(8);
        std::iota(input->begin(), input->end(), 0);
        source_port->send_data(composite::immutable_buffer<uint8_t>(input), {0, 0});
        REQUIRE(uut->process() == composite::retval::NORMAL);

        REQUIRE(sink_port->size() == 2);

        auto [out1, ts1, meta1] = sink_port->get_data();
        REQUIRE(out1.size() == 4);

        // Output should be complex cf32
        REQUIRE(meta1.has_value());
        REQUIRE(meta1->format.is_complex == true);
        REQUIRE(meta1->format.bit_width == 32);
        REQUIRE(meta1->format.type == composite::data_type::floating_point);

        // Sample rate and center frequency should be preserved
        REQUIRE(meta1->sample_rate == 10e6);
        REQUIRE(meta1->center_frequency == 915e6);

        // Validate frame content (ramp: 0, 1, 2, 3)
        for (size_t i = 0; i < 4; ++i) {
            INFO("Metadata test, Frame 1, Sample " << i);
            REQUIRE(out1.data()[i].real() == static_cast<float>(static_cast<int8_t>(i)));
            REQUIRE(out1.data()[i].imag() == 0.0f);
        }

        // Validate second frame as well (samples 4-7)
        auto [out2, ts2, meta2] = sink_port->get_data();
        REQUIRE(out2.size() == 4);
        for (size_t i = 0; i < 4; ++i) {
            INFO("Metadata test, Frame 2, Sample " << i);
            REQUIRE(out2.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 4)));
            REQUIRE(out2.data()[i].imag() == 0.0f);
        }
    }

    void run_non_aligned_buffer_test() {
        reset();

        uut->m_frame_size = 4;
        uut->m_overlap = 0;
        uut->m_frame_count = 4;
        uut->property_change_handler();

        // Use complex i16 (4 bytes per sample)
        auto meta = create_metadata(true, 16, composite::data_type::signed_integer, 1e6);
        source_port->send_metadata(meta);

        // Send 18 bytes (4.5 samples) - should process 4 complete samples, drop 2 bytes
        auto input = std::make_shared<std::vector<uint8_t>>(18);
        std::fill(input->begin(), input->end(), 0x42);
        source_port->send_data(composite::immutable_buffer<uint8_t>(input), {0, 0});

        // Should still process successfully (warning logged, partial sample dropped)
        REQUIRE(uut->process() == composite::retval::NORMAL);

        // Should emit 1 frame of 4 samples
        REQUIRE(sink_port->size() == 1);
    }

    void run_wraparound_test() {
        reset();

        // Configure small ring to force wraparound quickly
        // frame_size=8, overlap=4, hop=4, frame_count=4
        // ring_size = 4 * 4 + 4 = 20 samples
        uut->m_frame_size = 8;
        uut->m_overlap = 4;
        uut->m_frame_count = 4;
        uut->property_change_handler();

        auto meta = create_metadata(false, 8, composite::data_type::signed_integer, 1e6);
        source_port->send_metadata(meta);

        // Batch 1: Send samples 0-15 (16 samples)
        // write_head will be at 16 (no wrap yet, ring_size=20)
        auto input1 = std::make_shared<std::vector<uint8_t>>(16);
        std::iota(input1->begin(), input1->end(), 0);
        source_port->send_data(composite::immutable_buffer<uint8_t>(input1), {0, 0});
        REQUIRE(uut->process() == composite::retval::NORMAL);

        // Should emit 3 frames: [0-7], [4-11], [8-15]
        REQUIRE(sink_port->size() == 3);

        // Consume and validate frames, releasing them immediately via scoping
        {
            auto [f1, ts1, m1] = sink_port->get_data();
            REQUIRE(f1.size() == 8);
            for (size_t i = 0; i < 8; ++i) {
                INFO("Wraparound Batch 1, Frame 1, Sample " << i);
                REQUIRE(f1.data()[i].real() == static_cast<float>(static_cast<int8_t>(i)));
            }
        } // Frame 1 released here

        {
            auto [f2, ts2, m2] = sink_port->get_data();
            REQUIRE(f2.size() == 8);
            for (size_t i = 0; i < 8; ++i) {
                INFO("Wraparound Batch 1, Frame 2, Sample " << i);
                REQUIRE(f2.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 4)));
            }
        } // Frame 2 released here

        {
            auto [f3, ts3, m3] = sink_port->get_data();
            REQUIRE(f3.size() == 8);
            for (size_t i = 0; i < 8; ++i) {
                INFO("Wraparound Batch 1, Frame 3, Sample " << i);
                REQUIRE(f3.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 8)));
            }
        } // Frame 3 released here

        // Batch 2: Send samples 16-31 (16 samples)
        // write_head goes from 16 to 32
        // 32 % 20 = 12, so we've wrapped! Data now at ring positions 12-19, then 0-7
        auto input2 = std::make_shared<std::vector<uint8_t>>(16);
        std::iota(input2->begin(), input2->end(), 16);
        source_port->send_data(composite::immutable_buffer<uint8_t>(input2), {0, 16'000'000});
        REQUIRE(uut->process() == composite::retval::NORMAL);

        // Should emit 4 more frames: [12-19], [16-23], [20-27], [24-31]
        REQUIRE(sink_port->size() == 4);

        {
            auto [f4, ts4, m4] = sink_port->get_data();
            REQUIRE(f4.size() == 8);
            for (size_t i = 0; i < 8; ++i) {
                INFO("Wraparound Batch 2, Frame 1, Sample " << i);
                REQUIRE(f4.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 12)));
            }
        }

        {
            auto [f5, ts5, m5] = sink_port->get_data();
            REQUIRE(f5.size() == 8);
            for (size_t i = 0; i < 8; ++i) {
                INFO("Wraparound Batch 2, Frame 2, Sample " << i);
                REQUIRE(f5.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 16)));
            }
        }

        // Frame 6: samples 20-27
        // Ring positions: 20%20=0 to 27%20=7 (WRAPS at ring boundary!)
        {
            auto [f6, ts6, m6] = sink_port->get_data();
            REQUIRE(f6.size() == 8);
            for (size_t i = 0; i < 8; ++i) {
                INFO("Wraparound Batch 2, Frame 3 (WRAPAROUND), Sample " << i);
                REQUIRE(f6.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 20)));
            }
        }

        // Frame 7: samples 24-31
        // Ring positions: 24%20=4 to 31%20=11 (data written after wraparound)
        {
            auto [f7, ts7, m7] = sink_port->get_data();
            REQUIRE(f7.size() == 8);
            for (size_t i = 0; i < 8; ++i) {
                INFO("Wraparound Batch 2, Frame 4 (post-wrap), Sample " << i);
                REQUIRE(f7.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 24)));
            }
        }

        // Batch 3: Send samples 32-47 (16 more samples)
        // write_head goes from 32 to 48
        // 48 % 20 = 8, so we've wrapped multiple times
        auto input3 = std::make_shared<std::vector<uint8_t>>(16);
        std::iota(input3->begin(), input3->end(), 32);
        source_port->send_data(composite::immutable_buffer<uint8_t>(input3), {0, 32'000'000});
        REQUIRE(uut->process() == composite::retval::NORMAL);

        // Should emit 4 more frames: [28-35], [32-39], [36-43], [40-47]
        REQUIRE(sink_port->size() == 4);

        // Frame 8: samples 28-35
        // Ring positions: 28%20=8 to 35%20=15
        {
            auto [f8, ts8, m8] = sink_port->get_data();
            REQUIRE(f8.size() == 8);
            for (size_t i = 0; i < 8; ++i) {
                INFO("Wraparound Batch 3, Frame 1, Sample " << i);
                REQUIRE(f8.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 28)));
            }
        }

        // Frame 9: samples 32-39
        // Ring positions: 32%20=12 to 39%20=19
        {
            auto [f9, ts9, m9] = sink_port->get_data();
            REQUIRE(f9.size() == 8);
            for (size_t i = 0; i < 8; ++i) {
                INFO("Wraparound Batch 3, Frame 2, Sample " << i);
                REQUIRE(f9.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 32)));
            }
        }

        // Frame 10: samples 36-43
        // Ring positions: 36%20=16 to 43%20=3 (WRAPS!)
        {
            auto [f10, ts10, m10] = sink_port->get_data();
            REQUIRE(f10.size() == 8);
            for (size_t i = 0; i < 8; ++i) {
                INFO("Wraparound Batch 3, Frame 3 (WRAPAROUND), Sample " << i);
                REQUIRE(f10.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 36)));
            }
        }

        // Frame 11: samples 40-47
        // Ring positions: 40%20=0 to 47%20=7
        {
            auto [f11, ts11, m11] = sink_port->get_data();
            REQUIRE(f11.size() == 8);
            for (size_t i = 0; i < 8; ++i) {
                INFO("Wraparound Batch 3, Frame 4, Sample " << i);
                REQUIRE(f11.data()[i].real() == static_cast<float>(static_cast<int8_t>(i + 40)));
            }
        }
    }
};


TEST_CASE_METHOD(FramerTestFixture, "Framer processes real_i8 input correctly", "[framer][integration]") {
    run_real_i8_test();
}

TEST_CASE_METHOD(FramerTestFixture, "Framer processes complex_i8 input correctly", "[framer][integration]") {
    run_complex_i8_test();
}

TEST_CASE_METHOD(FramerTestFixture, "Framer processes complex_i16 input correctly", "[framer][integration]") {
    run_complex_i16_test();
}

TEST_CASE_METHOD(FramerTestFixture, "Framer processes complex_cf32 passthrough correctly", "[framer][integration]") {
    run_complex_cf32_test();
}

TEST_CASE_METHOD(FramerTestFixture, "Framer handles multiple data batches correctly", "[framer][integration]") {
    run_multiple_batches_test();
}

TEST_CASE_METHOD(FramerTestFixture, "Framer validates timestamp progression", "[framer][integration]") {
    run_timestamp_test();
}

TEST_CASE_METHOD(FramerTestFixture, "Framer propagates metadata correctly", "[framer][integration]") {
    run_metadata_propagation_test();
}

TEST_CASE_METHOD(FramerTestFixture, "Framer handles non-sample-aligned buffers", "[framer][integration]") {
    run_non_aligned_buffer_test();
}

TEST_CASE_METHOD(FramerTestFixture, "Framer handles ring buffer wraparound correctly", "[framer][integration]") {
    run_wraparound_test();
}
