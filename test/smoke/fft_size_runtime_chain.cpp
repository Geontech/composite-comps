// Runtime FFT-size transition smoke. Loads the shipped modules (rather than linking their
// colliding create() entry points), connects framer -> fft -> psd, and requires recovery at
// each new size. Packets may be dropped while the sequential control-plane writes take effect.
#include <composite/composite.hpp>
#include <composite/core/register.hpp>

#include <dlfcn.h>

#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

using create_fn = std::shared_ptr<composite::component> (*)(
    std::string_view, const composite::create_args&);
using cf = std::complex<float>;
using json = composite::properties::json;
using composite::properties::config_type;

struct loaded_component {
    void* handle{};
    std::shared_ptr<composite::component> component;
};

static auto load_component(const char* path, std::string_view id, std::string_view type)
    -> loaded_component {
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        throw std::runtime_error(dlerror());
    }
    auto create = reinterpret_cast<create_fn>(dlsym(handle, "create"));
    if (create == nullptr) {
        dlclose(handle);
        throw std::runtime_error("module has no create symbol");
    }
    composite::create_args args;
    args.values = json{{"type", type}};
    return {handle, create(id, args)};
}

static auto send_frame(composite::output_port<composite::immutable_buffer<uint8_t>>& source,
                       std::size_t frame_size, uint32_t& sequence) -> void {
    composite::metadata md;
    md.format.type = composite::data_type::signed_integer;
    md.format.bit_width = 8;
    md.format.is_complex = true;
    md.format.endianness = std::endian::native;
    md.sample_rate = 1.0e6;

    const auto tag = sequence++;
    auto data = composite::make_mutable<uint8_t>(frame_size * 2);
    for (std::size_t i = 0; i < frame_size; ++i) {
        data[i * 2] = static_cast<uint8_t>((i + tag) & 0x7f);
        data[i * 2 + 1] = static_cast<uint8_t>((2 * i + tag) & 0x7f);
    }
    source.send_data(std::move(data).to_immutable(), composite::timestamp{tag, 0}, md);
}

static auto wait_for_size(composite::output_port<composite::immutable_buffer<uint8_t>>& source,
                          composite::input_port<composite::mutable_buffer<float>>& sink,
                          std::size_t expected, uint32_t& sequence) -> bool {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    auto next_send = std::chrono::steady_clock::time_point::min();
    while (std::chrono::steady_clock::now() < deadline) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_send) {
            // Keep producing until recovery is observed. A fixed burst can be entirely queued
            // before the framer's staged on_apply reaction runs and then cleared on a loaded CI
            // host, leaving the remainder of the wait with no data in flight.
            send_frame(source, expected, sequence);
            next_send = now + std::chrono::milliseconds(2);
        }
        auto [data, ts, md] = sink.get_data();
        if (data.size() == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        if (data.size() != expected || md == nullptr) {
            continue;  // an in-flight frame from the previous generation is allowed
        }
        const auto it = md->annotations.find("fft_size");
        if (it == md->annotations.end() || it->second.to_string() != std::to_string(expected)) {
            continue;
        }
        bool finite = true;
        for (const auto value : data) {
            finite = finite && std::isfinite(value);
        }
        if (finite) {
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::printf("usage: fft_size_runtime_chain <framer.so> <fft.so> <psd.so>\n");
        return 2;
    }

    loaded_component framer_module;
    loaded_component fft_module;
    loaded_component psd_module;
    int failures = 0;
    try {
        framer_module = load_component(argv[1], "runtime_framer", "cf32");
        fft_module = load_component(argv[2], "runtime_fft", "cf32");
        psd_module = load_component(argv[3], "runtime_psd", "f32");

        auto& framer = framer_module.component;
        auto& fft = fft_module.component;
        auto& psd = psd_module.component;
        framer->set_properties(json{{"frame_size", 8}, {"overlap", 0}, {"frame_count", 16}},
                               config_type::INITIALIZE);
        fft->set_properties(json{{"fft_size", 8}, {"window", ""}, {"shift", false}},
                            config_type::INITIALIZE);
        psd->set_properties(json{{"power_based_normalization", true}}, config_type::INITIALIZE);

        auto* framer_in = framer->get_port<composite::input_port<composite::immutable_buffer<uint8_t>>>("data_in");
        auto* framer_out = framer->get_port<composite::output_port<composite::immutable_buffer<cf>>>("data_out");
        auto* fft_in = fft->get_port<composite::input_port<composite::immutable_buffer<cf>>>("data_in");
        auto* fft_out = fft->get_port<composite::output_port<composite::immutable_buffer<cf>>>("data_out");
        auto* psd_in = psd->get_port<composite::input_port<composite::immutable_buffer<cf>>>("data_in");
        auto* psd_out = psd->get_port<composite::output_port<composite::mutable_buffer<float>>>("data_out");
        if (!framer_in || !framer_out || !fft_in || !fft_out || !psd_in || !psd_out) {
            throw std::runtime_error("required port not found");
        }

        composite::output_port<composite::immutable_buffer<uint8_t>> source{"source"};
        composite::input_port<composite::mutable_buffer<float>> sink{"sink"};
        source.connect(framer_in);
        framer_out->connect(fft_in);
        fft_out->connect(psd_in);
        psd_out->connect(&sink);

        psd->start();
        fft->start();
        framer->start();

        uint32_t sequence = 1;
        if (!wait_for_size(source, sink, 8, sequence)) {
            std::printf("FAIL: chain did not produce an 8-bin PSD\n");
            ++failures;
        }

        // Configure downstream first. Sequential REST writes have the same ordering; any old
        // packets encountered between writes are permitted to complete or drop.
        fft->set_properties(json{{"fft_size", 16}}, config_type::RUNTIME);
        framer->set_properties(json{{"frame_size", 16}, {"overlap", 0}}, config_type::RUNTIME);
        if (!wait_for_size(source, sink, 16, sequence)) {
            std::printf("FAIL: chain did not recover at 16 bins\n");
            ++failures;
        }

        fft->set_properties(json{{"fft_size", 8}}, config_type::RUNTIME);
        framer->set_properties(json{{"frame_size", 8}, {"overlap", 0}}, config_type::RUNTIME);
        if (!wait_for_size(source, sink, 8, sequence)) {
            std::printf("FAIL: chain did not recover after returning to 8 bins\n");
            ++failures;
        }

        framer->stop();
        fft->stop();
        psd->stop();
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        ++failures;
    }

    // Destroy component instances before unloading the DSOs that own their vtables.
    framer_module.component.reset();
    fft_module.component.reset();
    psd_module.component.reset();
    if (framer_module.handle) dlclose(framer_module.handle);
    if (fft_module.handle) dlclose(fft_module.handle);
    if (psd_module.handle) dlclose(psd_module.handle);

    std::printf(failures ? "%d FAILURE(S)\n" : "RUNTIME FFT-SIZE CHAIN PASSED\n", failures);
    return failures ? 1 : 0;
}
