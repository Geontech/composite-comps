// Numeric correctness for the psd component against a double-precision reference, across the
// normalization contracts and hardening paths:
//   - name-keyed window fallback (fft_window annotation, window rebuilt locally)
//   - DECLARED sum(w^2) contract (fft_window_sum_sq): normalization must follow the declared
//     value even for a window name this component has never heard of — the drift-bug case
//   - no window (norm = 1/fs), zero-power bins (exactly -inf), SIMD tail (odd sizes)
//   - malformed fft_size metadata: counted, degrades to unnormalized-by-window, never NaN
//   - pbn annotation/content coherence across RUNTIME flips with packets in flight
//   - pooled outputs: holding more frames than the per-worker pool forces the heap fallback
#include "component.hpp"  // psd<T>
#include "windows.hpp"

#include <composite/buffers/buffer.hpp>
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>
#include <composite/core/metadata.hpp>
#include <composite/metrics/registry.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace composite;
using cf = std::complex<float>;
using json = composite::properties::json;
using composite::properties::config_type;

namespace {

int failures = 0;
auto check(bool ok, const char* what) -> void {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

// Reference: 10*log10(norm * |x|^2) in double; exact -inf for zero power.
auto ref_psd(const std::vector<cf>& x, double norm) -> std::vector<double> {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        const auto power = static_cast<double>(x[i].real()) * x[i].real() +
                           static_cast<double>(x[i].imag()) * x[i].imag();
        out[i] = 10.0 * std::log10(norm * power);
    }
    return out;
}

auto window_sum_sq(const char* type, std::size_t n) -> double {
    std::unique_ptr<composite::aligned_mem<double>> w;
    if (std::string_view{type} == "BLACKMAN_HARRIS") {
        w = windows::blackman_harris<double>(n, false);
    } else if (std::string_view{type} == "HAMMING") {
        w = windows::hamming<double>(n, false);
    } else {
        return 0.0;
    }
    double sum = 0.0;
    for (std::size_t k = 0; k < n; ++k) { sum += w->at(k) * w->at(k); }
    return sum;
}

struct harness {
    std::shared_ptr<psd<float>> comp;
    output_port<immutable_buffer<cf>> src{"src"};
    input_port<mutable_buffer<float>> sink{"snk"};

    static auto next_id() -> std::string {
        static int n = 0;
        return "psd_numeric_" + std::to_string(n++);
    }
    const std::string id{next_id()};

    explicit harness(bool pbn, int workers = 2) {
        comp = std::make_shared<psd<float>>(id);
        comp->set_properties(json{{"power_based_normalization", pbn}, {"num_workers", workers}},
                             config_type::INITIALIZE);
        auto* in = comp->get_port<input_port<immutable_buffer<cf>>>("data_in");
        auto* out = comp->get_port<output_port<mutable_buffer<float>>>("data_out");
        src.connect(in);
        out->connect(&sink);
        comp->start();
    }
    ~harness() { comp->stop(); }

    auto send(const std::vector<cf>& x, metadata md, uint32_t tag = 0) -> void {
        auto buf = make_mutable<cf>(x.size());
        std::copy(x.begin(), x.end(), buf.begin());
        src.send_data(std::move(buf).to_immutable(), timestamp{tag, 0}, md);
    }
    auto recv() -> std::tuple<mutable_buffer<float>, timestamp, metadata_ptr> {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            auto pkt = sink.get_data();
            if (std::get<0>(pkt).size() != 0) {
                return pkt;
            }
            std::this_thread::yield();
        }
        return {};
    }
};

auto make_input(std::size_t n, unsigned seed) -> std::vector<cf> {
    std::vector<cf> x(n);
    for (std::size_t k = 0; k < n; ++k) {
        x[k] = cf(std::sin(0.53f * static_cast<float>(k + seed)) + 1.5f,
                  std::cos(0.29f * static_cast<float>(k) + static_cast<float>(seed)) - 0.25f);
    }
    return x;
}

auto compare(const mutable_buffer<float>& got, const std::vector<double>& ref, const char* what) -> void {
    check(got.size() == ref.size(), what);
    if (got.size() != ref.size()) { return; }
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double g = got.data()[i];
        if (std::isinf(ref[i])) {
            if (!(std::isinf(g) && g < 0)) {
                std::printf("FAIL: %s: bin %zu expected -inf, got %g\n", what, i, g);
                ++failures;
                return;
            }
            continue;
        }
        if (std::abs(g - ref[i]) > 1e-3) {
            std::printf("FAIL: %s: bin %zu err %.4g dB (got %g, want %g)\n", what, i,
                        std::abs(g - ref[i]), g, ref[i]);
            ++failures;
            return;
        }
    }
}

auto base_md(std::size_t fft_size, const char* wtype, double fs = 1.0e6) -> metadata {
    metadata md;
    md.sample_rate = fs;
    md.annotations["fft_size"] = std::to_string(fft_size);
    if (wtype != nullptr && *wtype != '\0') {
        md.annotations["fft_window"] = std::string{wtype};
    }
    return md;
}

} // namespace

int main() {
    constexpr std::size_t N = 37;  // odd: exercises the SIMD tail on every path
    constexpr double FS = 1.0e6;

    // 1. Name-keyed window fallback, power-based normalization.
    {
        harness h(/*pbn=*/true);
        const auto x = make_input(N, 1);
        h.send(x, base_md(N, "BLACKMAN_HARRIS"));
        auto [out, ts, md] = h.recv();
        const double norm = 1.0 / (FS * (window_sum_sq("BLACKMAN_HARRIS", N) / N));
        compare(out, ref_psd(x, norm), "name-keyed window, power-based");
    }

    // 2. Name-keyed window, energy normalization (pbn off).
    {
        harness h(/*pbn=*/false);
        const auto x = make_input(N, 2);
        h.send(x, base_md(N, "HAMMING"));
        auto [out, ts, md] = h.recv();
        const double norm = 1.0 / (FS * window_sum_sq("HAMMING", N));
        compare(out, ref_psd(x, norm), "name-keyed window, energy");
    }

    // 3. DECLARED sum(w^2) contract: an unknown window name must still normalize correctly
    //    from the declared value (the name-only fallback would silently use norm = 1/fs).
    {
        harness h(/*pbn=*/true);
        const auto x = make_input(N, 3);
        const double declared = 7.25;
        auto md_in = base_md(N, "KAISER_BESSEL_7");  // psd has never heard of this window
        md_in.annotations["fft_window_sum_sq"] = std::to_string(declared);
        h.send(x, md_in);
        auto [out, ts, md] = h.recv();
        const double norm = 1.0 / (FS * (declared / N));
        compare(out, ref_psd(x, norm), "declared sum(w^2) with unknown window name");
    }

    // 4. No window: norm = 1/fs; a zero sample must come out exactly -inf.
    {
        harness h(/*pbn=*/true);
        auto x = make_input(N, 4);
        x[5] = cf{0.0f, 0.0f};
        metadata md_in;
        md_in.sample_rate = FS;
        h.send(x, md_in);
        auto [out, ts, md] = h.recv();
        compare(out, ref_psd(x, 1.0 / FS), "no window, zero-power bin");
    }

    // 4b. No window, ENERGY mode: the implicit rectangular window has sum(w^2) = N, so the
    //     norm is 1/(fs*N) — the historical unit factor left these outputs high by 10*log10(N).
    {
        harness h(/*pbn=*/false);
        const auto x = make_input(N, 40);
        metadata md_in;
        md_in.sample_rate = FS;
        h.send(x, md_in);
        auto [out, ts, md] = h.recv();
        compare(out, ref_psd(x, 1.0 / (FS * N)), "no window, energy mode uses sum(w^2) = N");
    }

    // 4c. A declared fft_size that contradicts the delivered spectrum length (possible via
    //     the parser annotation-override route) is malformed: counted, normalization
    //     degrades to no-window instead of dividing by the false declaration.
    {
        harness h(/*pbn=*/true);
        const auto x = make_input(N, 41);
        auto md_in = base_md(2 * N, "HAMMING");  // declares twice the actual length
        md_in.annotations["fft_window_sum_sq"] = std::to_string(7.25);
        h.send(x, md_in);
        auto [out, ts, md] = h.recv();
        compare(out, ref_psd(x, 1.0 / FS), "contradictory fft_size degrades to no-window norm");
        auto& registry = composite::metrics::registry::instance();
        check(registry.get_or_create_counter("psd.bad_metadata", "", "1",
                                             {{"component_id", h.id}}).value() >= 1,
              "contradictory fft_size is counted");
    }

    // 4d. A declared sum(w^2) that overflows this precision (1e300 is a positive finite
    //     double but +inf as float) is malformed, not an inf norm.
    {
        harness h(/*pbn=*/true);
        const auto x = make_input(N, 42);
        auto md_in = base_md(N, "SOME_REMOTE_WINDOW");
        md_in.annotations["fft_window_sum_sq"] = std::string{"1e300"};
        h.send(x, md_in);
        auto [out, ts, md] = h.recv();
        compare(out, ref_psd(x, 1.0 / FS), "float-overflowing sum(w^2) degrades cleanly");
        auto& registry = composite::metrics::registry::instance();
        check(registry.get_or_create_counter("psd.bad_metadata", "", "1",
                                             {{"component_id", h.id}}).value() >= 1,
              "overflowing sum(w^2) is counted");
    }

    // 4e. Mutable fan-out isolation with pooled outputs: two mutable consumers must receive
    //     INDEPENDENT frames (the type-erased deep copy runs the container's copy
    //     constructor — a bare external_buffer container would alias the slab).
    {
        auto comp = std::make_shared<psd<float>>("psd_numeric_fanout");
        comp->set_properties(json{{"power_based_normalization", true}, {"num_workers", 1}},
                             config_type::INITIALIZE);
        output_port<immutable_buffer<cf>> src{"src"};
        input_port<mutable_buffer<float>> sink_a{"snk_a"};
        input_port<mutable_buffer<float>> sink_b{"snk_b"};
        src.connect(comp->get_port<input_port<immutable_buffer<cf>>>("data_in"));
        auto* out_port = comp->get_port<output_port<mutable_buffer<float>>>("data_out");
        out_port->connect(&sink_a);
        out_port->connect(&sink_b);
        comp->start();

        const auto x = make_input(N, 50);
        metadata md_in;
        md_in.sample_rate = FS;
        auto buf = make_mutable<cf>(x.size());
        std::copy(x.begin(), x.end(), buf.begin());
        src.send_data(std::move(buf).to_immutable(), timestamp{0, 0}, md_in);

        auto recv_from = [](input_port<mutable_buffer<float>>& sink) -> mutable_buffer<float> {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                auto [data, ts, md] = sink.get_data();
                if (data.size() != 0) { return std::move(data); }
                std::this_thread::yield();
            }
            return {};
        };
        auto a = recv_from(sink_a);
        auto b = recv_from(sink_b);
        comp->stop();

        const auto ref = ref_psd(x, 1.0 / FS);
        compare(a, ref, "fan-out copy A is numerically correct");
        compare(b, ref, "fan-out copy B is numerically correct");
        if (a.size() == N && b.size() == N) {
            a.data()[0] = 12345.0f;  // writing one consumer's frame must not touch the other's
            check(b.data()[0] != 12345.0f, "fan-out frames are independent (no slab aliasing)");
        }
    }

    // 5. Malformed fft_size with a window name: counted, and the output degrades to the
    //    no-window norm — never NaN (the size-0 window used to make the norm 0/0).
    {
        harness h(/*pbn=*/true);
        const auto x = make_input(N, 5);
        metadata md_in;
        md_in.sample_rate = FS;
        md_in.annotations["fft_size"] = std::string{"garbage"};
        md_in.annotations["fft_window"] = std::string{"BLACKMAN_HARRIS"};
        h.send(x, md_in);
        auto [out, ts, md] = h.recv();
        compare(out, ref_psd(x, 1.0 / FS), "malformed fft_size degrades cleanly");
        auto& registry = composite::metrics::registry::instance();
        check(registry.get_or_create_counter("psd.bad_metadata", "", "1",
                                             {{"component_id", h.id}}).value() >= 1,
              "malformed fft_size is counted");
    }

    // 6. pbn annotation/content coherence across RUNTIME flips with bursts in flight: every
    //    output must match the normalization mode its OWN metadata declares.
    {
        harness h(/*pbn=*/true, /*workers=*/4);
        const double sum_sq = window_sum_sq("HAMMING", N);
        uint32_t sent = 0;
        uint32_t received = 0;
        bool coherent = true;
        bool pbn_now = true;

        auto drain = [&](bool wait_all) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (received < sent && std::chrono::steady_clock::now() < deadline) {
                auto [out, ts, md] = h.sink.get_data();
                if (out.size() == 0) {
                    if (!wait_all) { return; }
                    std::this_thread::yield();
                    continue;
                }
                const auto ann = md ? md->annotations.find("psd_power_based_normalization")
                                    : decltype(md->annotations.find(""))();
                if (!md || ann == md->annotations.end()) { coherent = false; ++received; continue; }
                const bool ann_pbn = ann->second.to_string() == "1";
                const double wnc = ann_pbn ? sum_sq / N : sum_sq;
                const auto x = make_input(N, ts.seconds);
                const auto ref = ref_psd(x, 1.0 / (FS * wnc));
                for (std::size_t b = 0; b < N; ++b) {
                    if (std::abs(out.data()[b] - ref[b]) > 1e-3) {
                        std::printf("FAIL: packet %u does not match its own pbn annotation (%d)\n",
                                    ts.seconds, ann_pbn ? 1 : 0);
                        coherent = false;
                        break;
                    }
                }
                ++received;
            }
        };

        for (int round = 0; round < 20; ++round) {
            for (int b = 0; b < 12; ++b) {
                h.send(make_input(N, sent), base_md(N, "HAMMING"), sent);
                ++sent;
            }
            pbn_now = !pbn_now;
            h.comp->set_properties(json{{"power_based_normalization", pbn_now}}, config_type::RUNTIME);
            drain(false);
        }
        drain(true);
        check(coherent, "every output matches its own pbn annotation");
        check(received == sent, "lossless across runtime pbn flips");
    }

    // 7. Pooled outputs: hold every frame so the per-worker pool (64 slots at this size)
    //    exhausts and the heap fallback engages — all frames still numerically correct.
    {
        harness h(/*pbn=*/true, /*workers=*/1);
        std::vector<mutable_buffer<float>> held;
        bool all_ok = true;
        for (int i = 0; i < 96; ++i) {
            const auto x = make_input(N, static_cast<unsigned>(i));
            metadata md_in;
            md_in.sample_rate = FS;
            h.send(x, md_in, static_cast<uint32_t>(i));
            auto [out, ts, md] = h.recv();
            if (out.size() != N) { all_ok = false; break; }
            const auto ref = ref_psd(x, 1.0 / FS);
            for (std::size_t b = 0; b < N; ++b) {
                if (std::abs(out.data()[b] - ref[b]) > 1e-3) { all_ok = false; }
            }
            held.push_back(std::move(out));
        }
        check(all_ok, "pooled + heap-fallback outputs all numerically correct");
        check(held.size() == 96, "no frame lost across pool exhaustion");
    }

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nPSD NUMERIC SMOKE PASSED\n", failures);
    return failures ? 1 : 0;
}
