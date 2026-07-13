// TSan stress for #39 (psd config-vs-private-thread UAF fix).
//
// psd runs a private input thread that solely owns m_window/m_sample_rate and rebuilds m_window
// from per-packet metadata (fft_size/fft_window annotations). Pre-fix, property_change_handler
// (writer thread, under park) called calculate_norm_const() which READ m_window/m_sample_rate
// while the unparked input thread was rebuilding them -> data race / UAF. Post-fix, PCH only
// publishes an atomic snapshot (m_pbn) + raises m_norm_dirty; the input thread does the recompute.
//
// This test hammers set_properties(power_based_normalization=...) (-> PCH) on one thread while
// another feeds data carrying VARYING fft_size/fft_window metadata (-> the input thread keeps
// rebuilding m_window). A clean TSan run (no race report) == pass.
#include "component.hpp"   // psd<T>

#include <composite/buffers/buffer.hpp>
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>
#include <composite/core/metadata.hpp>

#include <atomic>
#include <chrono>
#include <complex>
#include <cstdio>
#include <string>
#include <thread>

#include <spdlog/spdlog.h>

using namespace composite;
using composite::properties::config_type;
using json = composite::properties::json;

int main() {
    spdlog::set_level(spdlog::level::off);

    auto comp = std::make_shared<psd<float>>("psd_race");
    comp->set_properties(json{{"num_workers", 2}, {"power_based_normalization", true}},
                         config_type::INITIALIZE);

    using in_buf  = immutable_buffer<std::complex<float>>;
    using out_buf = mutable_buffer<float>;

    auto* in  = comp->get_port<input_port<in_buf>>("data_in");
    auto* out = comp->get_port<output_port<out_buf>>("data_out");
    if (in == nullptr || out == nullptr) { std::printf("FAIL: ports not found\n"); return 1; }

    output_port<in_buf> source("src");
    input_port<out_buf> sink("snk");
    source.connect(in);
    out->connect(&sink);

    comp->start();

    std::atomic<bool> stop{false};

    // Feeder: vary fft_size/fft_window metadata so the input thread keeps rebuilding m_window.
    std::thread feeder([&] {
        std::uint64_t n = 0;
        const char* wins[] = {"BLACKMAN_HARRIS", "HAMMING", "NONE"};
        while (!stop.load(std::memory_order_acquire)) {
            auto buf = make_mutable<std::complex<float>>(256);
            metadata md;
            md.sample_rate = 1.0e6 + static_cast<double>(n % 7);
            md.annotations["fft_size"]   = static_cast<std::int64_t>(128 + 64 * (n % 4));
            md.annotations["fft_window"] = std::string(wins[n % 3]);
            source.send_data(std::move(buf).to_immutable(), timestamp{0, 0}, md);
            ++n;
            std::this_thread::yield();
        }
    });

    // Drainer: keep the output ring from filling so the framework worker keeps draining futures.
    std::thread drainer([&] {
        while (!stop.load(std::memory_order_acquire)) {
            (void)sink.get_data();
            std::this_thread::yield();
        }
    });

    // Hammer: toggle the RUNTIME config -> property_change_handler publishes snapshot + dirty flag.
    std::thread hammer([&] {
        bool b = false;
        while (!stop.load(std::memory_order_acquire)) {
            try {
                comp->set_properties(json{{"power_based_normalization", b}}, config_type::RUNTIME);
            } catch (...) {
            }
            b = !b;
            std::this_thread::yield();
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    stop.store(true, std::memory_order_release);
    feeder.join();
    drainer.join();
    hammer.join();
    comp->stop();

    std::printf("PSD RACE STRESS COMPLETED (pass == no TSan report above)\n");
    return 0;
}
