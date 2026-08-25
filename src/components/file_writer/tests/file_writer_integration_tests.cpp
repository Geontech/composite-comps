/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

// Component-level tests for file_writer: drive the real component's process() through a
// connected port and verify the bytes on disk — exact content and ordering across a
// multi-packet writev batch, the num_bytes cap landing mid-buffer and across a batch,
// re-initialize truncation, and the loud open failure. This component previously had NO
// test coverage at all.

#include "component.hpp"

#include <catch2/catch_test_macros.hpp>

#include <composite/composite.hpp>
#include <composite/ports/output_port.hpp>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

auto make_packet(std::size_t size, uint8_t seed) -> composite::immutable_buffer<uint8_t> {
    auto v = std::make_shared<std::vector<uint8_t>>(size);
    std::iota(v->begin(), v->end(), seed);
    return composite::immutable_buffer<uint8_t>(v);
}

auto read_file(const std::filesystem::path& p) -> std::vector<uint8_t> {
    std::ifstream in(p, std::ios::binary);
    REQUIRE(in.good());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

struct writer_fixture {
    static auto next_id() -> std::string {
        static std::atomic<int> n{0};
        return "file_writer_test_" + std::to_string(n.fetch_add(1));
    }

    std::filesystem::path path{std::filesystem::temp_directory_path() /
                               ("comps_" + next_id() + ".bin")};
    std::shared_ptr<file_writer> uut{std::make_shared<file_writer>(next_id())};
    composite::output_port<composite::immutable_buffer<uint8_t>> src{"src"};

    writer_fixture() {
        src.connect(uut->get_port<composite::input_port<composite::immutable_buffer<uint8_t>>>(
            "data_in"));
    }
    ~writer_fixture() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    void configure(uint64_t num_bytes = 0) {
        uut->set_properties(
            composite::properties::json{{"filename", path.string()}, {"num_bytes", num_bytes}},
            composite::properties::config_type::INITIALIZE);
        uut->initialize();
    }
};

} // namespace

TEST_CASE_METHOD(writer_fixture, "a multi-packet batch lands on disk complete and in order",
                 "[file_writer][integration]") {
    configure();
    // All four packets are queued BEFORE process() runs, so one get_batch drains them and
    // one writev writes them — exact bytes and ordering prove the iovec gather.
    src.send_data(make_packet(64, 1), composite::timestamp{}, nullptr);
    src.send_data(make_packet(1, 100), composite::timestamp{}, nullptr);
    src.send_data(make_packet(300, 7), composite::timestamp{}, nullptr);
    src.send_data(make_packet(16, 200), composite::timestamp{}, nullptr);
    REQUIRE(uut->process() == composite::retval::NORMAL);
    REQUIRE(uut->process() == composite::retval::NOOP);

    const auto bytes = read_file(path);
    REQUIRE(bytes.size() == 64 + 1 + 300 + 16);
    CHECK(bytes[0] == 1);
    CHECK(bytes[63] == 64);
    CHECK(bytes[64] == 100);
    CHECK(bytes[65] == 7);
    CHECK(bytes[65 + 299] == static_cast<uint8_t>(7 + 299));
    CHECK(bytes[365] == 200);
}

TEST_CASE_METHOD(writer_fixture, "num_bytes cap clamps mid-buffer and finishes",
                 "[file_writer][integration]") {
    configure(100);
    src.send_data(make_packet(64, 0), composite::timestamp{}, nullptr);
    REQUIRE(uut->process() == composite::retval::NORMAL);
    // The cap lands 36 bytes into this packet; the write clamps and the component finishes.
    src.send_data(make_packet(64, 0), composite::timestamp{}, nullptr);
    REQUIRE(uut->process() == composite::retval::FINISH);

    const auto bytes = read_file(path);
    REQUIRE(bytes.size() == 100);
    CHECK(bytes[99] == 35);  // byte 36 of the second packet (seed 0 + offset 35)
}

TEST_CASE_METHOD(writer_fixture, "num_bytes cap inside one batch drops the packets past it",
                 "[file_writer][integration]") {
    configure(80);
    // Three packets in ONE batch: the cap lands inside the second; the third must not leak
    // into the capture.
    src.send_data(make_packet(64, 1), composite::timestamp{}, nullptr);
    src.send_data(make_packet(64, 1), composite::timestamp{}, nullptr);
    src.send_data(make_packet(64, 99), composite::timestamp{}, nullptr);
    REQUIRE(uut->process() == composite::retval::FINISH);

    const auto bytes = read_file(path);
    REQUIRE(bytes.size() == 80);
    CHECK(bytes[64] == 1);   // second packet's first byte
    CHECK(bytes[79] == 16);  // second packet's 16th byte — and nothing from the third
}

TEST_CASE_METHOD(writer_fixture, "re-initialize truncates and restarts the capture",
                 "[file_writer][integration]") {
    configure();
    src.send_data(make_packet(128, 1), composite::timestamp{}, nullptr);
    REQUIRE(uut->process() == composite::retval::NORMAL);

    configure();  // the documented contract: every initialize() starts a FRESH capture
    src.send_data(make_packet(8, 42), composite::timestamp{}, nullptr);
    REQUIRE(uut->process() == composite::retval::NORMAL);

    const auto bytes = read_file(path);
    REQUIRE(bytes.size() == 8);
    CHECK(bytes[0] == 42);
}

TEST_CASE_METHOD(writer_fixture, "an unopenable filename fails initialize loudly",
                 "[file_writer][integration]") {
    uut->set_properties(
        composite::properties::json{{"filename", "/nonexistent_dir_for_test/capture.bin"}},
        composite::properties::config_type::INITIALIZE);
    CHECK_THROWS_AS(uut->initialize(), std::runtime_error);
}
