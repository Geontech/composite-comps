// Functional smoke for the psd pipeline_component migration (#43). Like fft_smoke, but psd's
// window is DATA-DRIVEN: each input carries fft_size/fft_window metadata, and the window is built
// per pool worker. The feeder VARIES the window across packets to exercise the per-worker rebuild,
// and the test still requires submission-ORDER, lossless output (the slot ring's guarantee) plus
// the right output size + the psd_power_based_normalization annotation.
#include "component.hpp"  // psd<T>

#include <composite/buffers/buffer.hpp>
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>
#include <composite/core/metadata.hpp>

#include <atomic>
#include <chrono>
#include <complex>
#include <cstdio>
#include <thread>
#include <vector>

using namespace composite;
using cf = std::complex<float>;
using json = composite::properties::json;
using composite::properties::config_type;

int main() {
    constexpr std::size_t N_SAMP = 256;
    constexpr int N = 64;

    auto comp = std::make_shared<psd<float>>("psd_smoke");
    comp->set_properties(json{{"power_based_normalization", true}, {"num_workers", 4}},
                         config_type::INITIALIZE);

    auto* in  = comp->get_port<input_port<mutable_buffer<cf>>>("data_in");
    auto* out = comp->get_port<output_port<mutable_buffer<float>>>("data_out");
    if (in == nullptr || out == nullptr) { std::printf("FAIL: ports not found\n"); return 1; }

    output_port<mutable_buffer<cf>> src("src");
    input_port<mutable_buffer<float>> sink("snk");
    src.connect(in);
    out->connect(&sink);

    comp->start();

    std::atomic<bool> done{false};
    std::thread feeder([&] {
        const char* wins[] = {"BLACKMAN_HARRIS", "HAMMING", "NONE"};
        for (int i = 0; i < N; ++i) {
            auto buf = make_mutable<cf>(N_SAMP);
            for (std::size_t k = 0; k < N_SAMP; ++k) { buf[k] = cf(1.0F + static_cast<float>(k % 8), 0.5F); }
            metadata md;
            md.sample_rate = 1.0e6;
            md.annotations["fft_size"]   = static_cast<std::int64_t>(N_SAMP);
            md.annotations["fft_window"] = std::string(wins[(i / 16) % 3]);  // vary -> per-worker window rebuild
            src.send_data(std::move(buf), timestamp{static_cast<uint64_t>(i), 0}, md);
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        done.store(true, std::memory_order_release);
    });

    std::vector<uint64_t> order;
    bool size_ok = true;
    bool annot_ok = true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (static_cast<int>(order.size()) < N && std::chrono::steady_clock::now() < deadline) {
        auto [data, ts, md] = sink.get_data();
        if (data.size() != 0) {
            if (data.size() != N_SAMP) { size_ok = false; }
            if (!md.has_value() || md->annotations.find("psd_power_based_normalization") == md->annotations.end()) { annot_ok = false; }
            order.push_back(ts.seconds);
        } else {
            std::this_thread::yield();
        }
    }

    feeder.join();
    comp->stop();

    int failures = 0;
    if (static_cast<int>(order.size()) != N) { std::printf("FAIL: got %zu/%d outputs (lossy)\n", order.size(), N); ++failures; }
    if (!size_ok) { std::printf("FAIL: an output had the wrong size (expected %zu)\n", N_SAMP); ++failures; }
    if (!annot_ok) { std::printf("FAIL: an output lacked the psd_power_based_normalization annotation\n"); ++failures; }
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (order[i] != i) { std::printf("FAIL: output %zu out of order (tag=%llu)\n", i, static_cast<unsigned long long>(order[i])); ++failures; break; }
    }

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nPSD PIPELINE SMOKE PASSED (%d ordered, lossless outputs)\n",
                failures ? failures : N);
    return failures ? 1 : 0;
}
