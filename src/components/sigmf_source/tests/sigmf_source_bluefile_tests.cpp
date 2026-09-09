// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

// Covers the MIDAS Blue path, which had no test coverage at all.
//
// The header's data window (dataStart/dataSize) arrives as IEEE doubles from an
// untrusted file and is then used to address an mmap and to bound an IN-PLACE
// byte swap. So the cases that matter are not "does a good file parse" but "does
// a header that lies get rejected before anything indexes the mapping".

#include <catch2/catch_test_macros.hpp>

#include "../component.hpp"
#include "../blue/BlueFile.hpp"

#include <composite/composite.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace {

// Re-exposes the pieces under test that are not part of the public surface.
struct testable_sigmf_source : sigmf_source {
    using sigmf_source::sigmf_source;
    using sigmf_source::produce;
    using sigmf_source::parse_datatype;
};

auto next_dir_id() -> std::size_t {
    static std::size_t counter = 0;
    return ++counter;
}

struct temp_dir {
    std::filesystem::path path;
    temp_dir() {
        path = std::filesystem::temp_directory_path() /
               ("sigmf_blue_" + std::to_string(next_dir_id()));
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~temp_dir() {
        std::error_code ec;
        std::filesystem::permissions(path, std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::add, ec);
        std::filesystem::remove_all(path, ec);
    }
};

// Knobs are deliberately raw doubles / raw byte counts so a test can express a
// header that disagrees with the file on disk.
struct blue_opts {
    double data_start{512.0};
    std::optional<double> data_size{};   // default: match bytes actually written
    const char* format_code{"CF"};       // complex float32 -> 8 bytes/sample
    uint32_t type_code{1000};
    double x_delta{1.0 / 1000000.0};     // 1 MHz
    uint32_t data_rep{blue::IEEE};
    std::size_t data_bytes{512};         // payload written after the header
    bool corrupt_magic{false};
};

auto write_blue_file(const std::filesystem::path& dir, const std::string& stem,
                     const blue_opts& o = {}) -> std::string {
    const auto p = dir / (stem + ".blue");

    blue::HeaderControlBlock hcb{};
    std::memcpy(hcb.version, o.corrupt_magic ? "NOPE" : "BLUE", 4);
    hcb.headerEndianness = blue::IEEE;   // header itself is native
    hcb.dataEndianness = o.data_rep;
    hcb.extStart = 0;
    hcb.extSize = 0;
    hcb.dataStart = o.data_start;
    hcb.dataSize = o.data_size.value_or(static_cast<double>(o.data_bytes));
    hcb.typeCode = o.type_code;
    std::memcpy(hcb.formatCode, o.format_code, 2);
    hcb.adjunct1000.xStart = 0.0;
    hcb.adjunct1000.xDelta = o.x_delta;
    hcb.adjunct1000.xUnits = 1;

    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&hcb), sizeof(hcb));
    const std::vector<char> payload(o.data_bytes, 0x5A);
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    out.close();
    return p.string();
}

} // namespace

// ---------------------------------------------------------------------------
// Baseline: a well-formed file must keep working.
// ---------------------------------------------------------------------------

TEST_CASE("blue: a well-formed Type 1000 header parses", "[sigmf_source][blue]") {
    temp_dir d;
    const auto path = write_blue_file(d.path, "good", {.data_bytes = 8 * 100});

    REQUIRE(blue::isBlueFile(path));
    auto info = blue::parseBlueFile(path);
    REQUIRE(info.has_value());
    CHECK(info->valid);
    CHECK(info->dataOffset == 512);
    CHECK(info->dataSize == 8 * 100);
    CHECK(info->format.sampleSize == 8);     // complex float32
    CHECK(info->sampleCount == 100);
    CHECK(info->sampleRate == 1000000.0);    // 1/xDelta
}

TEST_CASE("blue: a non-Blue file is not mistaken for one", "[sigmf_source][blue]") {
    temp_dir d;
    const auto path = write_blue_file(d.path, "notblue", {.corrupt_magic = true});
    CHECK_FALSE(blue::isBlueFile(path));
    CHECK_FALSE(blue::parseBlueFile(path).has_value());
}

TEST_CASE("blue: an unknown scalar type code is rejected", "[sigmf_source][blue]") {
    temp_dir d;
    // 'A' (ASCII) has no scalar size. This is also what stands between
    // parseBlueFile and a divide-by-zero on sampleSize.
    const auto path = write_blue_file(d.path, "badcode", {.format_code = "CA"});
    CHECK_FALSE(blue::parseBlueFile(path).has_value());
}

// ---------------------------------------------------------------------------
// #2: the data window is untrusted. Each of these headers disagrees with the
// file on disk, and each one currently reaches an mmap index / in-place swap.
// ---------------------------------------------------------------------------

TEST_CASE("blue: a negative dataStart is rejected", "[sigmf_source][blue][validation]") {
    temp_dir d;
    // Casting a negative double to size_t is UB before any bounds check runs.
    const auto path = write_blue_file(d.path, "negstart", {.data_start = -512.0});
    CHECK_FALSE(blue::parseBlueFile(path).has_value());
}

TEST_CASE("blue: a negative dataSize is rejected", "[sigmf_source][blue][validation]") {
    temp_dir d;
    const auto path = write_blue_file(d.path, "negsize", {.data_size = -1024.0});
    CHECK_FALSE(blue::parseBlueFile(path).has_value());
}

TEST_CASE("blue: a non-finite dataStart is rejected", "[sigmf_source][blue][validation]") {
    temp_dir d;
    const auto nan_path = write_blue_file(d.path, "nanstart",
        {.data_start = std::numeric_limits<double>::quiet_NaN()});
    CHECK_FALSE(blue::parseBlueFile(nan_path).has_value());

    const auto inf_path = write_blue_file(d.path, "infstart",
        {.data_start = std::numeric_limits<double>::infinity()});
    CHECK_FALSE(blue::parseBlueFile(inf_path).has_value());
}

TEST_CASE("blue: a non-finite dataSize is rejected", "[sigmf_source][blue][validation]") {
    temp_dir d;
    const auto nan_path = write_blue_file(d.path, "nansize",
        {.data_size = std::numeric_limits<double>::quiet_NaN()});
    CHECK_FALSE(blue::parseBlueFile(nan_path).has_value());

    const auto inf_path = write_blue_file(d.path, "infsize",
        {.data_size = std::numeric_limits<double>::infinity()});
    CHECK_FALSE(blue::parseBlueFile(inf_path).has_value());
}

TEST_CASE("blue: a dataSize past the end of the file is rejected", "[sigmf_source][blue][validation]") {
    temp_dir d;
    // 512 bytes of payload on disk, header claims 1 MB. This is the case that
    // walks the in-place byte swap off the end of the mapping.
    const auto path = write_blue_file(d.path, "oversize",
        {.data_size = 1024.0 * 1024.0, .data_bytes = 512});
    CHECK_FALSE(blue::parseBlueFile(path).has_value());
}

TEST_CASE("blue: a dataStart past the end of the file is rejected", "[sigmf_source][blue][validation]") {
    temp_dir d;
    const auto path = write_blue_file(d.path, "farstart",
        {.data_start = 1024.0 * 1024.0, .data_bytes = 512});
    CHECK_FALSE(blue::parseBlueFile(path).has_value());
}

TEST_CASE("blue: a dataStart before the header block is rejected", "[sigmf_source][blue][validation]") {
    temp_dir d;
    // Inside the 512-byte HCB: the payload would overlap the header it came from.
    const auto path = write_blue_file(d.path, "instart", {.data_start = 64.0});
    CHECK_FALSE(blue::parseBlueFile(path).has_value());
}

TEST_CASE("blue: a dataStart+dataSize overflow is rejected", "[sigmf_source][blue][validation]") {
    temp_dir d;
    const auto path = write_blue_file(d.path, "ovf",
        {.data_start = 1e18, .data_size = 1e18, .data_bytes = 512});
    CHECK_FALSE(blue::parseBlueFile(path).has_value());
}

// ---------------------------------------------------------------------------
// #9: the SigMF datatype width is used as `bytes / (bitwidth / 8)` to stride the
// byte swap, so a width that is not a whole number of bytes strides wrongly.
// ---------------------------------------------------------------------------

TEST_CASE("datatype: supported widths parse", "[sigmf_source][datatype]") {
    for (const auto* dt : {"cf32_le", "cf64_le", "ri16_be", "cu8", "ci8", "ci16_le", "ci64_le"}) {
        auto f = testable_sigmf_source::parse_datatype(dt);
        CHECKED_IF(f.has_value()) {
            CHECK(f->bitwidth % 8 == 0);
            CHECK(f->bytes_per_sample() > 0);
        }
        CHECK(f.has_value());
    }
}

TEST_CASE("datatype: a width that is not 8/16/32/64 is rejected", "[sigmf_source][datatype]") {
    // ci12 is a real SigMF spelling for packed 12-bit, which this component
    // cannot address: bitwidth/8 truncates to 1 and the swap strides by a byte.
    CHECK_FALSE(testable_sigmf_source::parse_datatype("ci12_le").has_value());
    CHECK_FALSE(testable_sigmf_source::parse_datatype("cf24_le").has_value());
    CHECK_FALSE(testable_sigmf_source::parse_datatype("ri4").has_value());
    CHECK_FALSE(testable_sigmf_source::parse_datatype("cf0").has_value());
}

TEST_CASE("datatype: malformed spellings are rejected", "[sigmf_source][datatype]") {
    CHECK_FALSE(testable_sigmf_source::parse_datatype("").has_value());
    CHECK_FALSE(testable_sigmf_source::parse_datatype("xf32").has_value());
    CHECK_FALSE(testable_sigmf_source::parse_datatype("cz32").has_value());
    CHECK_FALSE(testable_sigmf_source::parse_datatype("cf32_xx").has_value());
}

// ---------------------------------------------------------------------------
// #11 / #12 / #13: component-level, driving the real load path.
// ---------------------------------------------------------------------------

namespace {

// Loads one spec and reports whether the component accepted it. The metadata of
// the first emitted packet is returned so a test can assert what was advertised.
struct load_result {
    bool accepted{false};
    composite::metadata_ptr md{};
};

auto load_one(testable_sigmf_source& src, const composite::properties::json& spec) -> load_result {
    composite::properties::json files = composite::properties::json::array();
    files.push_back(spec);
    src.set_properties({{"chunk_samples", 16}, {"files", files}, {"streaming", true}});

    load_result r;
    for (int i = 0; i < 50; ++i) {
        auto p = src.produce();
        if (p.status == testable_sigmf_source::produce_status::data) {
            r.accepted = true;
            r.md = p.md;
            break;
        }
    }
    return r;
}

} // namespace

TEST_CASE("blue: an explicit datatype override is not silently discarded", "[sigmf_source][blue][override]") {
    temp_dir d;
    // A CF (complex float32) Blue file, told to read as complex int16 instead.
    // The override exists precisely for headers that lie, so it must survive the
    // Blue header rather than being overwritten by it.
    const auto path = write_blue_file(d.path, "override", {.format_code = "CF", .data_bytes = 8 * 64});

    testable_sigmf_source src("blue_override");
    auto r = load_one(src, {
        {"path", path},
        {"stream_id", 1},
        {"rate_control", false},
        {"loop", true},
        {"overrides", {{"filetype", "bluefile-1000"}, {"datatype", "ci16_le"}}},
    });

    REQUIRE(r.accepted);
    REQUIRE(r.md != nullptr);
    // ci16 -> 16-bit signed, complex. If the Blue header won, this reads 32/float.
    CHECK(r.md->format.bit_width == 16);
    CHECK(r.md->format.is_complex == true);
}

TEST_CASE("blue: byte-swapped data plays back from a read-only file", "[sigmf_source][blue][readonly]") {
    temp_dir d;
    // dataRep EEEI forces the swap path. The swap is done on a MAP_PRIVATE
    // mapping, which needs PROT_WRITE but NOT an O_RDWR descriptor -- so a
    // recording on a read-only mount must still play.
    const auto path = write_blue_file(d.path, "readonly",
        {.data_rep = blue::EEEI, .data_bytes = 8 * 64});
    std::filesystem::permissions(path, std::filesystem::perms::owner_read,
                                 std::filesystem::perm_options::replace);

    testable_sigmf_source src("blue_readonly");
    auto r = load_one(src, {
        {"path", path},
        {"stream_id", 1},
        {"rate_control", false},
        {"loop", true},
        {"overrides", {{"filetype", "bluefile-1000"}}},
    });

    CHECK(r.accepted);
}

TEST_CASE("source: a non-regular file is rejected without crashing", "[sigmf_source][validation]") {
    temp_dir d;
    // A directory reports a size via lseek that is meaningless, and an unchecked
    // lseek failure becomes a huge size_t handed straight to mmap.
    const auto dir_as_file = (d.path / "iam_a_dir").string();
    std::filesystem::create_directories(dir_as_file);

    testable_sigmf_source src("nonregular");
    auto r = load_one(src, {
        {"path", dir_as_file},
        {"stream_id", 1},
        {"rate_control", false},
        {"overrides", {{"datatype", "cf32_le"}}},
    });

    CHECK_FALSE(r.accepted);
}

TEST_CASE("blue: a header that lies about its data window is refused at load", "[sigmf_source][blue][validation]") {
    temp_dir d;
    const auto path = write_blue_file(d.path, "liar",
        {.data_size = 1024.0 * 1024.0, .data_bytes = 512});

    testable_sigmf_source src("blue_liar");
    auto r = load_one(src, {
        {"path", path},
        {"stream_id", 1},
        {"rate_control", false},
        {"overrides", {{"filetype", "bluefile-1000"}}},
    });

    // Must be refused rather than mapped and byte-swapped past the end.
    CHECK_FALSE(r.accepted);
}

// ---------------------------------------------------------------------------
// Byte-swap correctness. These pin the OBSERVABLE result -- the bytes a
// consumer receives -- rather than how the swap is staged, so they hold whether
// the swap happens up front on the mapping or per chunk on the copy.
// ---------------------------------------------------------------------------

namespace {

// Writes a raw SigMF pair with caller-supplied payload bytes and datatype.
auto write_raw_sigmf(const std::filesystem::path& dir, const std::string& stem,
                     const std::string& datatype,
                     const std::vector<unsigned char>& payload) -> std::string {
    const auto data_path = dir / (stem + ".sigmf-data");
    {
        std::ofstream out(data_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(payload.data()),
                  static_cast<std::streamsize>(payload.size()));
    }
    std::ofstream meta(dir / (stem + ".sigmf-meta"));
    meta << R"({"global":{"core:datatype":")" << datatype
         << R"(","core:sample_rate":1000000.0,"core:version":"1.0.0"},)"
         << R"("captures":[{"core:sample_start":0}],"annotations":[]})";
    return data_path.string();
}

// Pulls the first emitted chunk's bytes.
auto first_chunk_bytes(testable_sigmf_source& src, const composite::properties::json& spec)
    -> std::vector<unsigned char> {
    composite::properties::json files = composite::properties::json::array();
    files.push_back(spec);
    src.set_properties({{"chunk_samples", 4}, {"files", files}, {"streaming", true}});
    for (int i = 0; i < 50; ++i) {
        auto p = src.produce();
        if (p.status == testable_sigmf_source::produce_status::data) {
            std::vector<unsigned char> out(p.buffer.size());
            std::memcpy(out.data(), p.buffer.data(), p.buffer.size());
            return out;
        }
    }
    return {};
}

} // namespace

TEST_CASE("swap: big-endian 16-bit input is delivered native", "[sigmf_source][endianness]") {
    temp_dir d;
    // Four real int16 samples, big-endian on disk.
    const std::vector<unsigned char> be{0x01,0x02, 0x03,0x04, 0x05,0x06, 0x07,0x08};
    const auto path = write_raw_sigmf(d.path, "be16", "ri16_be", be);

    testable_sigmf_source src("swap16");
    auto got = first_chunk_bytes(src, {
        {"path", path}, {"stream_id", 1}, {"rate_control", false},
    });

    REQUIRE(got.size() == be.size());
    // Each 2-byte element reversed.
    const std::vector<unsigned char> want{0x02,0x01, 0x04,0x03, 0x06,0x05, 0x08,0x07};
    CHECK(got == want);
}

TEST_CASE("swap: big-endian 32-bit input is delivered native", "[sigmf_source][endianness]") {
    temp_dir d;
    // Two real float32 samples' worth of bytes, big-endian on disk.
    const std::vector<unsigned char> be{0x01,0x02,0x03,0x04, 0x11,0x12,0x13,0x14,
                                        0x21,0x22,0x23,0x24, 0x31,0x32,0x33,0x34};
    const auto path = write_raw_sigmf(d.path, "be32", "rf32_be", be);

    testable_sigmf_source src("swap32");
    auto got = first_chunk_bytes(src, {
        {"path", path}, {"stream_id", 1}, {"rate_control", false},
    });

    REQUIRE(got.size() == be.size());
    const std::vector<unsigned char> want{0x04,0x03,0x02,0x01, 0x14,0x13,0x12,0x11,
                                          0x24,0x23,0x22,0x21, 0x34,0x33,0x32,0x31};
    CHECK(got == want);
}

TEST_CASE("swap: little-endian input is delivered unchanged", "[sigmf_source][endianness]") {
    temp_dir d;
    const std::vector<unsigned char> le{0x01,0x02, 0x03,0x04, 0x05,0x06, 0x07,0x08};
    const auto path = write_raw_sigmf(d.path, "le16", "ri16_le", le);

    testable_sigmf_source src("noswap");
    auto got = first_chunk_bytes(src, {
        {"path", path}, {"stream_id", 1}, {"rate_control", false},
    });

    REQUIRE(got.size() == le.size());
    CHECK(got == le);
}

TEST_CASE("swap: a looped read re-delivers the same native bytes", "[sigmf_source][endianness]") {
    temp_dir d;
    // Guards a real hazard of swapping the source once rather than per read: a
    // second pass over the same region must not double-swap back to foreign order.
    const std::vector<unsigned char> be{0x01,0x02, 0x03,0x04, 0x05,0x06, 0x07,0x08};
    const auto path = write_raw_sigmf(d.path, "loop16", "ri16_be", be);
    const std::vector<unsigned char> want{0x02,0x01, 0x04,0x03, 0x06,0x05, 0x08,0x07};

    testable_sigmf_source src("swaploop");
    composite::properties::json files = composite::properties::json::array();
    files.push_back({{"path", path}, {"stream_id", 1}, {"rate_control", false}, {"loop", true}});
    src.set_properties({{"chunk_samples", 4}, {"files", files}, {"streaming", true}});

    // chunk_samples covers the whole file, so successive packets are repeat passes.
    for (int pass = 0; pass < 3; ++pass) {
        std::vector<unsigned char> got;
        for (int i = 0; i < 50 && got.empty(); ++i) {
            auto p = src.produce();
            if (p.status == testable_sigmf_source::produce_status::data) {
                got.resize(p.buffer.size());
                std::memcpy(got.data(), p.buffer.data(), p.buffer.size());
            }
        }
        REQUIRE(got.size() == want.size());
        CHECK(got == want);
    }
}
