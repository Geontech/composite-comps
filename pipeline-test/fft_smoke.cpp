// Functional smoke for the fft pipeline_component migration (#43). Feeds N tagged buffers and
// verifies the output comes back in SUBMISSION ORDER (the slot-ring's job), losslessly, with the
// right size + the fft_size/fft_window metadata annotation. Exercises work()/prepare() across a
// multi-worker pool. (fft had no functional test before; pipeline_component's generic ordering is
// covered by the framework's test_pipeline_component — this verifies fft's hooks end to end.)
#include "component.hpp"  // fft<T>

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
    constexpr std::size_t FFT_SIZE = 8;
    constexpr int N = 64;

    auto comp = std::make_shared<fft<cf>>("fft_smoke");
    comp->set_properties(json{{"fft_size", FFT_SIZE}, {"window", "BLACKMAN_HARRIS"},
                              {"num_workers", 4}, {"shift", true}},
                         config_type::INITIALIZE);

    auto* in  = comp->get_port<input_port<immutable_buffer<cf>>>("data_in");
    auto* out = comp->get_port<output_port<immutable_buffer<cf>>>("data_out");
    if (in == nullptr || out == nullptr) { std::printf("FAIL: ports not found\n"); return 1; }

    output_port<immutable_buffer<cf>> src("src");
    input_port<immutable_buffer<cf>> sink("snk");
    src.connect(in);
    out->connect(&sink);

    comp->start();

    std::atomic<bool> done{false};
    // Feeder: N buffers of size FFT_SIZE, each tagged with ts.seconds = its index. Paced so the
    // input ring never overflows (an 8-point FFT on a 4-worker pool keeps up trivially), so the
    // test is lossless + deterministic.
    std::thread feeder([&] {
        for (int i = 0; i < N; ++i) {
            auto buf = make_mutable<cf>(FFT_SIZE);
            for (std::size_t k = 0; k < FFT_SIZE; ++k) { buf[k] = cf(static_cast<float>(i), static_cast<float>(k)); }
            metadata md;
            md.sample_rate = 1.0e6;
            src.send_data(std::move(buf).to_immutable(), timestamp{static_cast<uint64_t>(i), 0}, md);
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        done.store(true, std::memory_order_release);
    });

    // Drain: collect outputs; record the tag (ts.seconds) order + check size/annotation.
    std::vector<uint64_t> order;
    bool size_ok = true;
    bool annot_ok = true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (static_cast<int>(order.size()) < N && std::chrono::steady_clock::now() < deadline) {
        auto [data, ts, md] = sink.get_data();
        if (data.size() != 0) {
            if (data.size() != FFT_SIZE) { size_ok = false; }
            if (!md.has_value() || md->annotations.find("fft_size") == md->annotations.end()) { annot_ok = false; }
            order.push_back(ts.seconds);
        } else {
            std::this_thread::yield();
        }
    }

    feeder.join();
    comp->stop();

    int failures = 0;
    if (static_cast<int>(order.size()) != N) { std::printf("FAIL: got %zu/%d outputs (lossy)\n", order.size(), N); ++failures; }
    if (!size_ok) { std::printf("FAIL: an output had the wrong size (expected %zu)\n", FFT_SIZE); ++failures; }
    if (!annot_ok) { std::printf("FAIL: an output lacked the fft_size annotation\n"); ++failures; }
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (order[i] != i) { std::printf("FAIL: output %zu out of order (tag=%llu, expected %zu)\n",
                                         i, static_cast<unsigned long long>(order[i]), i); ++failures; break; }
    }

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nFFT PIPELINE SMOKE PASSED (%d ordered, lossless outputs)\n",
                failures ? failures : N);
    return failures ? 1 : 0;
}
