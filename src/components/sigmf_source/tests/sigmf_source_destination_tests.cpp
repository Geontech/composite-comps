// SPDX-License-Identifier: LGPL-3.0-or-later
// SPDX-FileCopyrightText: 2025 Geon Technologies, LLC

// Covers the egress destination that sigmf_source stamps into output metadata
// for udp_sink to route on. The invariant under test: concurrent streams are
// separated by group address with the port held flat, because IGMP membership
// is per group and a shared group forwards every stream to every subscriber.

#include <catch2/catch_test_macros.hpp>

#include "../component.hpp"

#include <composite/composite.hpp>

#include <complex>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

// produce() is protected on the component. Re-exposing it here lets a test read the
// emitted packet's metadata directly, with no worker thread and no connected sink --
// the annotations under test are stamped into that metadata.
struct testable_sigmf_source : sigmf_source {
    using sigmf_source::sigmf_source;
    using sigmf_source::produce;
};

constexpr std::size_t kTestSamples = 64;
constexpr double kSampleRate = 1000000.0;
constexpr auto kComponentDefaultPort = "5000";

using annotation_map = std::map<std::string, std::string>;

auto next_dir_id() -> std::size_t {
    static std::size_t counter = 0;
    return ++counter;
}

// Writes a minimal cf32_le SigMF pair, returning the .sigmf-data path.
auto write_sigmf_pair(const std::filesystem::path& dir, const std::string& stem) -> std::string {
    const auto data_path = dir / (stem + ".sigmf-data");
    const auto meta_path = dir / (stem + ".sigmf-meta");

    std::vector<std::complex<float>> samples(kTestSamples);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        samples[i] = {static_cast<float>(i), -static_cast<float>(i)};
    }

    std::ofstream data(data_path, std::ios::binary);
    data.write(reinterpret_cast<const char*>(samples.data()),
               static_cast<std::streamsize>(samples.size() * sizeof(std::complex<float>)));
    data.close();

    std::ofstream meta(meta_path);
    meta << R"({"global":{"core:datatype":"cf32_le","core:sample_rate":)" << kSampleRate
         << R"(},"captures":[{"core:sample_start":0,"core:frequency":100000000.0}]})";
    meta.close();

    return data_path.string();
}

struct destination_fixture {
    using byte_buffer = composite::immutable_buffer<std::byte>;

    std::filesystem::path dir;
    std::shared_ptr<testable_sigmf_source> source;
    // Mirrors the component's "files" array. 0.5.2 writes properties as whole JSON
    // values, so an edit re-sends the full array rather than patching one element.
    composite::properties::json files_json = composite::properties::json::array();

    destination_fixture() {
        dir = std::filesystem::temp_directory_path() /
              ("sigmf_source_dest_" + std::to_string(next_dir_id()));
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);

        source = std::make_shared<testable_sigmf_source>("test_sigmf_source");

        // The port still has to exist under its documented name even though these
        // tests read the packet from produce() instead of across a connection.
        REQUIRE(source->get_port<composite::output_port<byte_buffer>>("data_out") != nullptr);

        source->set_properties({{"chunk_samples", kTestSamples}});
    }

    ~destination_fixture() {
        source->set_properties({{"streaming", false}});
        source->stop();
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
    }

    // Appends a file spec. Empty dest_ip / zero dest_port are omitted from the
    // spec entirely, exercising the fall-through-to-default paths.
    auto add_file(const std::string& stem,
                  int stream_id,
                  const std::string& dest_ip = "",
                  uint32_t dest_port = 0) -> std::size_t {
        composite::properties::json spec{
            {"path", write_sigmf_pair(dir, stem)},
            {"stream_id", stream_id},
            {"rate_control", false},  // produce on demand, unthrottled
            {"loop", true},
        };
        if (!dest_ip.empty()) {
            spec["destination_ip"] = dest_ip;
        }
        if (dest_port != 0) {
            spec["destination_port"] = dest_port;
        }
        files_json.push_back(std::move(spec));
        source->set_properties({{"files", files_json}});
        return files_json.size() - 1;
    }

    // Patches one element and re-sends the array, which is the live-edit path
    // (refresh_runtime_from_spec) rather than the initial load.
    void update_file(std::size_t index, const composite::properties::json& patch) {
        for (auto it = patch.begin(); it != patch.end(); ++it) {
            files_json[index][it.key()] = it.value();
        }
        source->set_properties({{"files", files_json}});
    }

    void start_streaming() {
        source->set_properties({{"streaming", true}});
    }

    // Pumps the component until every expected stream has emitted, returning the
    // annotations each one carried, keyed by stream_id.
    auto collect_annotations(std::size_t expected_streams) -> std::map<std::string, annotation_map> {
        std::map<std::string, annotation_map> by_stream;
        for (int attempt = 0; attempt < 200 && by_stream.size() < expected_streams; ++attempt) {
            auto r = source->produce();
            if (r.status != testable_sigmf_source::produce_status::data || !r.md) {
                continue;
            }
            auto id = r.md->annotations.find("stream_id");
            if (id == r.md->annotations.end()) {
                continue;
            }
            // annotation_value is a variant in 0.5.2; flatten to strings for the
            // comparisons these tests make.
            annotation_map flat;
            for (const auto& [key, value] : r.md->annotations) {
                flat[key] = value.to_string();
            }
            by_stream[id->second.to_string()] = std::move(flat);
        }
        return by_stream;
    }
};

} // namespace

TEST_CASE("sigmf_source: concurrent streams get distinct groups on a flat port", "[sigmf_source][destination]") {
    destination_fixture fx;

    fx.add_file("stream_a", 1, "239.0.10.3");
    fx.add_file("stream_b", 2, "239.0.10.4");
    fx.add_file("stream_c", 3, "239.0.10.5");
    fx.start_streaming();

    auto streams = fx.collect_annotations(3);
    REQUIRE(streams.size() == 3);

    CHECK(streams["1"]["destination_ip"] == "239.0.10.3");
    CHECK(streams["2"]["destination_ip"] == "239.0.10.4");
    CHECK(streams["3"]["destination_ip"] == "239.0.10.5");

    // The regression this component's addressing exists to prevent: the port
    // must NOT be offset per stream, or receivers joining one group are
    // forwarded all of them.
    CHECK(streams["1"]["destination_port"] == kComponentDefaultPort);
    CHECK(streams["2"]["destination_port"] == kComponentDefaultPort);
    CHECK(streams["3"]["destination_port"] == kComponentDefaultPort);
}

TEST_CASE("sigmf_source: absent destination_ip emits no annotation", "[sigmf_source][destination]") {
    destination_fixture fx;

    fx.add_file("no_dest", 7);
    fx.start_streaming();

    auto streams = fx.collect_annotations(1);
    REQUIRE(streams.size() == 1);

    // No annotation at all, so udp_sink applies its default_dest_ip rather than
    // routing to a half-configured destination.
    CHECK_FALSE(streams["7"].contains("destination_ip"));
    CHECK(streams["7"]["destination_port"] == kComponentDefaultPort);
}

TEST_CASE("sigmf_source: destination_port falls from per-file to component", "[sigmf_source][destination]") {
    destination_fixture fx;

    // Both arms of the port fallback at once, and against a component port that
    // is NOT the built-in 5000 -- otherwise the inheriting stream cannot tell the
    // component property from a hardcoded default. The built-in itself is covered
    // by the tests above, which never set the property.
    fx.source->set_properties({{"destination_port", 7000}});
    fx.add_file("explicit_port", 1, "239.0.10.3", 6100);
    fx.add_file("inherits_port", 2, "239.0.10.4");
    fx.start_streaming();

    auto streams = fx.collect_annotations(2);
    REQUIRE(streams.size() == 2);

    CHECK(streams["1"]["destination_port"] == "6100");
    CHECK(streams["2"]["destination_port"] == "7000");
}

TEST_CASE("sigmf_source: a live edit re-stamps, then clears, the destination", "[sigmf_source][destination]") {
    destination_fixture fx;

    auto index = fx.add_file("movable", 1, "239.0.10.3");
    fx.start_streaming();

    auto before = fx.collect_annotations(1);
    REQUIRE(before.size() == 1);
    REQUIRE(before["1"]["destination_ip"] == "239.0.10.3");

    // refresh_runtime_from_spec and the initial load share one stamp_destination,
    // so what a live edit has to prove is not that the values map correctly --
    // that is already covered -- but that an edit re-runs the stamp at all and
    // the new values reach an emitted packet.
    fx.update_file(index, {{"destination_ip", "239.0.10.9"}, {"destination_port", 6200}});

    auto moved = fx.collect_annotations(1);
    REQUIRE(moved.size() == 1);
    CHECK(moved["1"]["destination_ip"] == "239.0.10.9");
    CHECK(moved["1"]["destination_port"] == "6200");

    // Clearing exercises the erase arm, which only a live edit can reach: a spec
    // that never carried an IP never populates the key in the first place. A
    // stale value here would keep transmitting to the group the controller has
    // already released.
    fx.update_file(index, {{"destination_ip", ""}});

    auto cleared = fx.collect_annotations(1);
    REQUIRE(cleared.size() == 1);
    CHECK_FALSE(cleared["1"].contains("destination_ip"));
    // Dropping the group must not also drop the port.
    CHECK(cleared["1"]["destination_port"] == "6200");
}

TEST_CASE("sigmf_source: duplicate destinations do not both stream", "[sigmf_source][destination]") {
    destination_fixture fx;

    fx.add_file("first", 1, "239.0.10.3");
    fx.add_file("collides", 2, "239.0.10.3");  // same group and port as `first`
    fx.start_streaming();

    // Interleaving two files on one destination is unrecoverable for a receiver,
    // so the later spec is errored rather than transmitted.
    auto streams = fx.collect_annotations(2);
    CHECK(streams.size() == 1);
    CHECK(streams.contains("1"));
}

TEST_CASE("sigmf_source: only one spec may omit destination_ip", "[sigmf_source][destination]") {
    destination_fixture fx;

    fx.add_file("implicit_default", 1);
    fx.add_file("also_implicit_default", 2);  // both resolve to udp_sink's default group
    fx.start_streaming();

    auto streams = fx.collect_annotations(2);
    CHECK(streams.size() == 1);
    CHECK(streams.contains("1"));
}
