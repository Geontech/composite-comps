// Numeric correctness for the fft component against a naive reference DFT, across every
// work() path:
//   - window + shift: the fftshift is FOLDED into the window (odd-k sign flip) — the output
//     must still equal reference windowed-DFT + explicit fftshift.
//   - no window + shift: the post-FFT rotate path.
//   - no window, UNALIGNED input: the copy-through-scratch path.
//   - pooled output frames: holding more outputs than the pool's capacity forces the heap
//     fallback mid-stream; every frame (pooled and fallback) must be numerically right.
// Plus the degenerate length-1 window guard (0/0 = NaN regression).
#include "component.hpp"  // fft<T>
#include "windows.hpp"

#include <composite/buffers/buffer.hpp>
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>
#include <composite/core/metadata.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>
#include <thread>
#include <vector>

using namespace composite;
using cf = std::complex<float>;
using cd = std::complex<double>;
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

// Naive reference DFT in double precision; optional per-sample window, optional fftshift.
auto ref_dft(const std::vector<cf>& x, const std::vector<double>& win, bool shift)
    -> std::vector<cd> {
    const std::size_t n = x.size();
    std::vector<cd> spectrum(n);
    for (std::size_t m = 0; m < n; ++m) {
        cd acc{};
        for (std::size_t k = 0; k < n; ++k) {
            const auto xv = cd{x[k]} * (win.empty() ? 1.0 : win[k]);
            const auto phase = -2.0 * std::numbers::pi * static_cast<double>(k * m) / static_cast<double>(n);
            acc += xv * cd{std::cos(phase), std::sin(phase)};
        }
        spectrum[m] = acc;
    }
    if (shift) {
        std::vector<cd> shifted(n);
        for (std::size_t i = 0; i < n; ++i) {
            shifted[i] = spectrum[(i + n / 2) % n];
        }
        return shifted;
    }
    return spectrum;
}

// Drive one fft component configuration over one input; return the single output frame.
struct harness {
    std::shared_ptr<fft<cf>> comp;
    output_port<immutable_buffer<cf>> src{"src"};
    input_port<immutable_buffer<cf>> sink{"snk"};

    static auto next_id() -> std::string {
        static int n = 0;
        return "fft_numeric_" + std::to_string(n++);
    }

    explicit harness(std::size_t fft_size, const char* window, bool shift, int workers = 2) {
        comp = std::make_shared<fft<cf>>(next_id());
        comp->set_properties(json{{"fft_size", fft_size}, {"window", window},
                                  {"num_workers", workers}, {"shift", shift}},
                             config_type::INITIALIZE);
        auto* in = comp->get_port<input_port<immutable_buffer<cf>>>("data_in");
        auto* out = comp->get_port<output_port<immutable_buffer<cf>>>("data_out");
        src.connect(in);
        out->connect(&sink);
        comp->start();
    }
    ~harness() { comp->stop(); }

    auto run_one(immutable_buffer<cf> in) -> immutable_buffer<cf> {
        metadata md;
        md.sample_rate = 1.0e6;
        src.send_data(std::move(in), timestamp{0, 0}, md);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            auto [data, ts, m] = sink.get_data();
            if (data.size() != 0) {
                return data;
            }
            std::this_thread::yield();
        }
        return {};
    }
};

auto make_test_input(std::size_t n, unsigned seed) -> std::vector<cf> {
    std::vector<cf> x(n);
    for (std::size_t k = 0; k < n; ++k) {
        // Deterministic, non-symmetric pattern with both components exercised.
        x[k] = cf(std::sin(0.37f * static_cast<float>(k + seed)) + 0.25f,
                  std::cos(0.71f * static_cast<float>(k) + static_cast<float>(seed)));
    }
    return x;
}

auto compare(const immutable_buffer<cf>& got, const std::vector<cd>& ref, const char* what) -> void {
    check(got.size() == ref.size(), what);
    if (got.size() != ref.size()) { return; }
    double max_mag = 1.0;
    for (const auto& r : ref) { max_mag = std::max(max_mag, std::abs(r)); }
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const auto err = std::abs(cd{got.data()[i]} - ref[i]);
        if (err > 1e-4 * max_mag) {
            std::printf("FAIL: %s: bin %zu err %.3g (got %g%+gi, want %g%+gi)\n", what, i, err,
                        got.data()[i].real(), got.data()[i].imag(), ref[i].real(), ref[i].imag());
            ++failures;
            return;
        }
    }
}

// Per-sample window values matching what the component builds (pre-fold).
auto window_values(const char* type, std::size_t n) -> std::vector<double> {
    std::vector<double> w;
    if (std::string_view{type} == "BLACKMAN_HARRIS") {
        auto win = windows::blackman_harris<double>(n);
        w.resize(n);
        for (std::size_t k = 0; k < n; ++k) { w[k] = win->at(2 * k); }
    } else if (std::string_view{type} == "HAMMING") {
        auto win = windows::hamming<double>(n);
        w.resize(n);
        for (std::size_t k = 0; k < n; ++k) { w[k] = win->at(2 * k); }
    }
    return w;
}

} // namespace

int main() {
    constexpr std::size_t N = 16;
    const auto x = make_test_input(N, 3);

    auto to_aligned_immutable = [](const std::vector<cf>& v) {
        auto buf = make_aligned_buffer_uninitialized<cf>(64, v.size());
        std::copy(v.begin(), v.end(), buf.begin());
        return std::move(buf).to_immutable();
    };

    // 1. Window + shift: the folded-shift path must equal reference windowed DFT + fftshift.
    {
        harness h(N, "BLACKMAN_HARRIS", /*shift=*/true);
        auto out = h.run_one(to_aligned_immutable(x));
        compare(out, ref_dft(x, window_values("BLACKMAN_HARRIS", N), true),
                "windowed + shift (folded fftshift)");
    }

    // 2. Window, no shift: plain windowed transform.
    {
        harness h(N, "HAMMING", /*shift=*/false);
        auto out = h.run_one(to_aligned_immutable(x));
        compare(out, ref_dft(x, window_values("HAMMING", N), false), "windowed, no shift");
    }

    // 3. No window + shift: the post-FFT rotate path on an aligned direct input.
    {
        harness h(N, "", /*shift=*/true);
        auto out = h.run_one(to_aligned_immutable(x));
        compare(out, ref_dft(x, {}, true), "no window + shift (rotate)");
    }

    // 4. No window, UNALIGNED input: slicing one element off a 64-aligned buffer guarantees an
    //    8-byte-offset (under-aligned) data pointer, forcing the copy-through-scratch path.
    {
        harness h(N, "", /*shift=*/false);
        std::vector<cf> padded(N + 1, cf{});
        for (std::size_t k = 0; k < N; ++k) { padded[k + 1] = x[k]; }
        auto out = h.run_one(to_aligned_immutable(padded).slice(1, N));
        compare(out, ref_dft(x, {}, false), "no window, unaligned input (copy path)");
    }

    // 5. Pool exhaustion: hold every output. The 16-point pool caps at 64 buffers, so frames
    //    65..96 must come from the heap fallback — all of them still numerically correct.
    {
        harness h(N, "", /*shift=*/true, /*workers=*/1);
        std::vector<immutable_buffer<cf>> held;
        bool all_ok = true;
        for (int i = 0; i < 96; ++i) {
            const auto xi = make_test_input(N, static_cast<unsigned>(i));
            auto out = h.run_one(to_aligned_immutable(xi));
            if (out.size() != N) { all_ok = false; break; }
            const auto ref = ref_dft(xi, {}, true);
            double max_mag = 1.0;
            for (const auto& r : ref) { max_mag = std::max(max_mag, std::abs(r)); }
            for (std::size_t b = 0; b < N; ++b) {
                if (std::abs(cd{out.data()[b]} - ref[b]) > 1e-4 * max_mag) { all_ok = false; }
            }
            held.push_back(std::move(out));  // keep the slab; exhausts the pool mid-run
        }
        check(all_ok, "pooled + heap-fallback outputs all numerically correct");
        check(held.size() == 96, "no frame lost across pool exhaustion");
    }

    // 6. Annotation/content coherence across RUNTIME config changes with packets in flight:
    //    every output must be transformed with exactly the config its OWN metadata describes
    //    (work() binds each packet to its stamped generation), and nothing may be lost or
    //    spuriously dropped across the changes.
    {
        harness h(N, "", /*shift=*/true, /*workers=*/4);
        uint32_t sent = 0;
        uint32_t received = 0;
        bool coherent = true;
        bool shift_now = true;
        std::string window_now{};

        auto verify_one = [&](const immutable_buffer<cf>& data, uint32_t idx, const metadata_ptr& md) {
            if (!md) { coherent = false; return; }
            const auto sh = md->annotations.find("fft_shift");
            const auto wn = md->annotations.find("fft_window");
            if (sh == md->annotations.end() || wn == md->annotations.end()) { coherent = false; return; }
            const auto ann_window = wn->second.to_string();
            const auto ann_shift = sh->second.to_string();
            const auto x = make_test_input(N, idx);
            const auto ref = ref_dft(x, window_values(ann_window.c_str(), N), ann_shift == "1");
            double max_mag = 1.0;
            for (const auto& r : ref) { max_mag = std::max(max_mag, std::abs(r)); }
            for (std::size_t b = 0; b < N; ++b) {
                if (data.size() != N || std::abs(cd{data.data()[b]} - ref[b]) > 1e-4 * max_mag) {
                    std::printf("FAIL: packet %u does not match its own annotations (window='%s' shift=%s)\n",
                                idx, ann_window.c_str(), ann_shift.c_str());
                    coherent = false;
                    return;
                }
            }
        };
        auto drain = [&](bool wait_all) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (received < sent && std::chrono::steady_clock::now() < deadline) {
                auto [data, ts, md] = h.sink.get_data();
                if (data.size() == 0) {
                    if (!wait_all) { return; }
                    std::this_thread::yield();
                    continue;
                }
                verify_one(data, ts.seconds, md);
                ++received;
            }
        };

        // 40 rounds = 40 config publications, far beyond the retained-history depth (16):
        // the history deduplicates by (size, window, shift) key, so churn between the four
        // keys exercised here must never evict a generation a queued packet still needs.
        for (int round = 0; round < 40; ++round) {
            for (int b = 0; b < 12; ++b) {
                metadata md;
                md.sample_rate = 1.0e6;
                h.src.send_data(to_aligned_immutable(make_test_input(N, sent)),
                                timestamp{sent, 0}, md);
                ++sent;
            }
            // Flip config while the burst may still be in flight.
            if (round % 2 == 0) {
                shift_now = !shift_now;
                h.comp->set_properties(json{{"shift", shift_now}}, config_type::RUNTIME);
            } else {
                window_now = window_now.empty() ? "HAMMING" : "";
                h.comp->set_properties(json{{"window", window_now}}, config_type::RUNTIME);
            }
            drain(false);
        }
        drain(true);
        check(coherent, "every output matches its own fft_window/fft_shift annotations");
        check(received == sent, "lossless across runtime config changes");
    }

    // 7. Degenerate length-1 windows are unit-valued, not 0/0 = NaN.
    {
        auto bh = windows::blackman_harris<float>(1);
        auto hm = windows::hamming<float>(1);
        check(bh->at(0) == 1.0f && bh->at(1) == 1.0f, "blackman_harris(1) is a unit window");
        check(hm->at(0) == 1.0f && hm->at(1) == 1.0f, "hamming(1) is a unit window");
        check(!std::isnan(bh->at(0)) && !std::isnan(hm->at(0)), "length-1 windows are not NaN");
    }

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nFFT NUMERIC SMOKE PASSED\n", failures);
    return failures ? 1 : 0;
}
