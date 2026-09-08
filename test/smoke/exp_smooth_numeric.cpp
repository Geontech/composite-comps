// Numeric correctness for exp_smooth against a double-precision EWMA reference (the component
// had no functional test):
//   - the smoothing recursion and its one-frame output delay, including the EOS flush of the
//     held frame
//   - NaN/Inf sanitization of the CURRENT frame (bad sample -> previous average holds)
//   - the prev-poisoning regression: a non-finite bin in the BASELINE frame (e.g. a -inf PSD
//     bin from zero power) must re-seed from the next finite sample, not latch forever
//   - size-change re-baselining (flush + counter + restart)
//   - pass-through mode (num_averages = 0)
// Odd frame size exercises the SIMD tails; 64-aligned frames engage the AVX paths.
#include "component.hpp"  // exp_smooth<T>

#include <composite/buffers/buffer.hpp>
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>
#include <composite/core/metadata.hpp>
#include <composite/metrics/registry.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace composite;
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

template <typename S>
struct harness {
    std::shared_ptr<exp_smooth<S>> comp;
    output_port<mutable_buffer<S>> src{"src"};
    input_port<mutable_buffer<S>> sink{"snk"};

    static auto next_id() -> std::string {
        static int n = 0;
        return "exp_smooth_numeric_" + std::to_string(n++);
    }
    const std::string id{next_id()};

    explicit harness(uint32_t num_averages) {
        comp = std::make_shared<exp_smooth<S>>(id);
        comp->set_properties(json{{"num_averages", num_averages}}, config_type::INITIALIZE);
        src.connect(comp->template get_port<input_port<mutable_buffer<S>>>("data_in"));
        comp->template get_port<output_port<mutable_buffer<S>>>("data_out")->connect(&sink);
        comp->start();
    }

    auto send(const std::vector<S>& x, uint32_t tag, double sample_rate = 1.0e6) -> void {
        auto buf = make_aligned_buffer_uninitialized<S>(64, x.size());
        std::copy(x.begin(), x.end(), buf.begin());
        metadata md;
        md.sample_rate = sample_rate;
        src.send_data(std::move(buf), timestamp{tag, 0}, md);
    }
    auto send_bare(const std::vector<S>& x, uint32_t tag) -> void {
        auto buf = make_aligned_buffer_uninitialized<S>(64, x.size());
        std::copy(x.begin(), x.end(), buf.begin());
        src.send_data(std::move(buf), timestamp{tag, 0});  // no metadata attached
    }
    auto recv() -> std::tuple<mutable_buffer<S>, timestamp, metadata_ptr> {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline) {
            auto [data, ts, md] = sink.get_data();
            if (data.size() != 0) {
                return {std::move(data), ts, md};
            }
            std::this_thread::yield();
        }
        return {};
    }
    // Quiesce the worker, then flush the held frame the way end-of-stream does
    // (on_end_of_stream is public; after stop() the worker no longer touches the state).
    auto finish() -> void {
        comp->stop();
        comp->on_end_of_stream();
    }
};

template <typename S>
auto alpha_for(uint32_t num_averages) -> double {
    // Same formula as the component: -expm1(ln(0.02)/N) in double, then narrowed to the
    // component's precision, then widened for the reference recursion.
    return static_cast<double>(static_cast<S>(-std::expm1(std::log(1.0 - 0.98) / num_averages)));
}

template <typename S>
auto compare(const mutable_buffer<S>& got, const std::vector<double>& ref, const char* what) -> void {
    check(got.size() == ref.size(), what);
    if (got.size() != ref.size()) { return; }
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double g = got.data()[i];
        if (std::isinf(ref[i]) && std::isinf(g) && std::signbit(ref[i]) == std::signbit(g)) { continue; }
        if (std::abs(g - ref[i]) > 1e-4 * std::max(1.0, std::abs(ref[i]))) {
            std::printf("FAIL: %s: bin %zu (got %g, want %g)\n", what, i, g, ref[i]);
            ++failures;
            return;
        }
    }
}

template <typename S = float>
auto make_frame(std::size_t n, unsigned seed) -> std::vector<S> {
    std::vector<S> x(n);
    for (std::size_t k = 0; k < n; ++k) {
        x[k] = static_cast<S>(std::sin(0.41 * static_cast<double>(k + seed)) * 30.0 - 60.0);
    }
    return x;
}

} // namespace

int main() {
    constexpr std::size_t N = 37;  // odd: SIMD strides + scalar tail on every tier
    constexpr uint32_t AVG = 10;

    // 1. EWMA recursion + one-frame delay + EOS flush, with NaN/Inf sanitization of curr and
    //    the prev-poisoning regression folded into the same stream.
    {
        harness<float> h(AVG);
        const auto alpha = alpha_for<float>(AVG);

        auto f1 = make_frame(N, 1);
        f1[3] = -std::numeric_limits<float>::infinity();  // a silent PSD bin in the BASELINE
        auto f2 = make_frame(N, 2);
        f2[7] = std::numeric_limits<float>::quiet_NaN();  // a corrupt sample mid-stream
        const auto f3 = make_frame(N, 3);

        // Reference accumulators with the component's semantics.
        std::vector<double> a1(N), a2(N), a3(N);
        for (std::size_t i = 0; i < N; ++i) { a1[i] = f1[i]; }
        for (std::size_t i = 0; i < N; ++i) {
            const bool c_ok = std::isfinite(f2[i]);
            const bool p_ok = std::isfinite(a1[i]);
            const double seed = c_ok ? f2[i] : a1[i];
            a2[i] = p_ok ? alpha * seed + (1.0 - alpha) * a1[i] : seed;
        }
        for (std::size_t i = 0; i < N; ++i) {
            const bool c_ok = std::isfinite(f3[i]);
            const bool p_ok = std::isfinite(a2[i]);
            const double seed = c_ok ? f3[i] : a2[i];
            a3[i] = p_ok ? alpha * seed + (1.0 - alpha) * a2[i] : seed;
        }

        h.send(f1, 1);
        h.send(f2, 2);
        h.send(f3, 3);
        auto [o1, t1, md_o1] = h.recv();
        auto [o2, t2, md_o2] = h.recv();
        h.finish();
        auto [o3, t3, md_o3] = h.recv();

        compare(o1, a1, "first output is the raw baseline frame");
        check(t1.seconds == 1, "one-frame delay pairs output 1 with input 1's timestamp");
        compare(o2, a2, "second output follows the EWMA recursion");
        // The regression the kernel fix exists for: bin 3 was -inf in the baseline, and the
        // next finite sample must RE-SEED it (the old kernels left it -inf forever).
        if (o2.size() == N) {
            check(std::isfinite(o2.data()[3]) &&
                      std::abs(o2.data()[3] - f2[3]) < 1e-4 * std::abs(f2[3]),
                  "a non-finite baseline bin re-seeds from the next finite sample");
            check(std::abs(o2.data()[7] - a1[7]) < 1e-4 * std::abs(a1[7]),
                  "a NaN current sample holds the previous average");
        }
        compare(o3, a3, "end-of-stream flushes the held final average");
    }

    // 2. Size-change re-baseline: flush the old-size average, restart at the new size, count it.
    {
        harness<float> h(AVG);
        const auto f1 = make_frame(N, 4);
        const auto g1 = make_frame(2 * N, 5);  // new size mid-stream
        h.send(f1, 1);
        h.send(g1, 2);
        auto [o1, t1, md_o1] = h.recv();
        h.finish();
        auto [o2, t2, md_o2] = h.recv();

        check(o1.size() == N && o2.size() == 2 * N, "size change flushes old size then adopts new");
        auto& registry = composite::metrics::registry::instance();
        check(registry.get_or_create_counter("exp_smooth.size_mismatch", "", "1",
                                             {{"component_id", h.id}}).value() == 1,
              "size change is counted");
    }

    // 3. Pass-through mode (num_averages = 0): frames flow unchanged with no delay or hold.
    {
        harness<float> h(0);
        const auto f1 = make_frame(N, 6);
        h.send(f1, 1);
        auto [o1, t1, md_o1] = h.recv();
        std::vector<double> ref(f1.begin(), f1.end());
        compare(o1, ref, "pass-through mode forwards frames unchanged");
        check(t1.seconds == 1, "pass-through keeps the frame's own timestamp");
        h.finish();  // nothing held: must not emit
        auto [flush, tf] = std::pair{h.sink.get_data(), 0};
        check(std::get<0>(flush).size() == 0, "pass-through holds nothing at end-of-stream");
    }

    // 4. Runtime num_averages changes: a smoothing->smoothing change PRESERVES the running
    //    average (only re-weights future samples), and a smoothing->pass-through switch
    //    FLUSHES the held frame instead of silently dropping it.
    {
        harness<float> h(AVG);
        const auto a1f = alpha_for<float>(AVG);
        const auto f1 = make_frame(N, 7);
        const auto f2 = make_frame(N, 8);
        const auto f3 = make_frame(N, 9);

        h.send(f1, 1);
        h.send(f2, 2);
        auto [o1, t1, md_o1] = h.recv();  // = f1 (baseline)

        constexpr uint32_t AVG2 = 100;
        h.comp->set_properties(json{{"num_averages", AVG2}}, config_type::RUNTIME);
        const auto a2f = alpha_for<float>(AVG2);

        h.send(f3, 3);
        auto [o2, t2, md_o2] = h.recv();  // = avg after f2 (alpha1)
        h.finish();
        auto [o3, t3, md_o3] = h.recv();  // = avg after f3 (alpha2 over PRESERVED history)

        std::vector<double> acc2(N), acc3(N);
        for (std::size_t i = 0; i < N; ++i) {
            acc2[i] = a1f * f2[i] + (1.0 - a1f) * f1[i];
            acc3[i] = a2f * f3[i] + (1.0 - a2f) * acc2[i];
        }
        compare(o2, acc2, "output before the alpha change uses the old alpha");
        compare(o3, acc3, "an alpha change re-weights future samples over PRESERVED history");
    }
    {
        harness<float> h(AVG);
        const auto f1 = make_frame(N, 10);
        const auto f2 = make_frame(N, 11);
        h.send(f1, 1);  // held (baseline)
        h.comp->set_properties(json{{"num_averages", 0}}, config_type::RUNTIME);
        h.send(f2, 2);
        auto [oflush, tf, md_oflush] = h.recv();
        auto [opass, tp, md_opass] = h.recv();
        std::vector<double> r1(f1.begin(), f1.end());
        std::vector<double> r2(f2.begin(), f2.end());
        compare(oflush, r1, "switching to pass-through flushes the held frame (not dropped)");
        check(tf.seconds == 1, "the flushed frame keeps its own timestamp");
        compare(opass, r2, "pass-through then forwards the current frame");
    }

    // 5. Metadata discontinuity (same size): a VALUE change re-baselines (flush + adopt +
    //    counter); value-equal fresh instances (every send here) never trigger it.
    {
        harness<float> h(AVG);
        const auto f1 = make_frame(N, 12);
        const auto f2 = make_frame(N, 13);
        const auto f3 = make_frame(N, 14);
        h.send(f1, 1);            // 1 MHz
        h.send(f2, 2);            // 1 MHz, fresh-but-equal metadata instance: no re-baseline
        h.send(f3, 3, 2.0e6);     // retune: metadata VALUE changes -> re-baseline
        auto [o1, t1, md_o1] = h.recv();
        auto [o2, t2, md_o2] = h.recv();
        h.finish();
        auto [o3, t3, md_o3] = h.recv();

        const auto a = alpha_for<float>(AVG);
        std::vector<double> acc2(N);
        for (std::size_t i = 0; i < N; ++i) { acc2[i] = a * f2[i] + (1.0 - a) * f1[i]; }
        std::vector<double> r3(f3.begin(), f3.end());
        compare(o2, acc2, "value-equal metadata instances do not re-baseline");
        compare(o3, r3, "a metadata value change re-baselines (old average not smoothed in)");
        auto& registry = composite::metrics::registry::instance();
        check(registry.get_or_create_counter("exp_smooth.metadata_rebaselines", "", "1",
                                             {{"component_id", h.id}}).value() == 1,
              "exactly the one metadata change is counted");
    }

    // 5b. Null <-> non-null metadata transitions. Per the port contract ("nullptr = none",
    //     no latch), metadata is a per-packet fact: a transition in EITHER direction is a
    //     semantic discontinuity and re-baselines, and a bare frame's output stays bare.
    {
        harness<float> h(AVG);
        const auto f1 = make_frame(N, 20);
        const auto f2 = make_frame(N, 21);
        h.send_bare(f1, 1);   // never-declared semantics
        h.send(f2, 2);        // metadata appears -> re-baseline (flush f1 raw, adopt f2)
        auto [o1, t1, m1] = h.recv();
        h.finish();
        auto [o2, t2, m2] = h.recv();

        std::vector<double> r1(f1.begin(), f1.end());
        std::vector<double> r2(f2.begin(), f2.end());
        compare(o1, r1, "metadata appearing flushes the undeclared baseline raw");
        compare(o2, r2, "metadata appearing adopts the declared frame as the new baseline");
        auto& registry = composite::metrics::registry::instance();
        check(registry.get_or_create_counter("exp_smooth.metadata_rebaselines", "", "1",
                                             {{"component_id", h.id}}).value() == 1,
              "metadata appearance is counted as a re-baseline");
    }
    {
        harness<float> h(AVG);
        const auto f1 = make_frame(N, 22);
        const auto f2 = make_frame(N, 23);
        h.send(f1, 1);        // declared
        h.send_bare(f2, 2);   // metadata VANISHES -> re-baseline (flush f1 raw, adopt f2)
        auto [o1, t1, m1] = h.recv();
        h.finish();
        auto [o2, t2, m2] = h.recv();

        std::vector<double> r1(f1.begin(), f1.end());
        std::vector<double> r2(f2.begin(), f2.end());
        compare(o1, r1, "declared -> bare: the declared baseline flushes raw");
        check(m1 != nullptr, "the flushed frame keeps the metadata it arrived with");
        compare(o2, r2, "declared -> bare: the bare frame re-baselines (not smoothed in)");
        check(m2 == nullptr, "a bare frame's output stays bare (nullptr = none, no latch)");
        auto& registry = composite::metrics::registry::instance();
        check(registry.get_or_create_counter("exp_smooth.metadata_rebaselines", "", "1",
                                             {{"component_id", h.id}}).value() == 1,
              "metadata disappearance is counted as a re-baseline");
    }

    // 6. Very large num_averages must smooth very slowly, not FREEZE: the naive
    //    1 - pow(10, ...) alpha cancelled to exactly 0.0f at this magnitude.
    {
        harness<float> h(1'000'000'000u);
        const std::vector<float> zeros(N, 0.0f);
        const std::vector<float> hundreds(N, 100.0f);
        h.send(zeros, 1);
        h.send(hundreds, 2);
        auto [o1, t1, md_o1] = h.recv();
        h.finish();
        auto [o2, t2, md_o2] = h.recv();
        check(o2.size() == N && o2.data()[0] > 1e-8f,
              "alpha stays nonzero at num_averages = 1e9 (filter not frozen)");
        check(o2.size() == N && o2.data()[0] < 1e-4f,
              "alpha stays tiny at num_averages = 1e9 (still heavy smoothing)");
    }

    // 7. The f64 kernels (modified alongside f32): recursion + the -inf baseline re-seed.
    {
        harness<double> h(AVG);
        const auto alpha = alpha_for<double>(AVG);
        auto f1 = make_frame<double>(N, 15);
        f1[5] = -std::numeric_limits<double>::infinity();
        const auto f2 = make_frame<double>(N, 16);

        std::vector<double> a1(f1.begin(), f1.end());
        std::vector<double> a2(N);
        for (std::size_t i = 0; i < N; ++i) {
            const bool p_ok = std::isfinite(a1[i]);
            a2[i] = p_ok ? alpha * f2[i] + (1.0 - alpha) * a1[i] : f2[i];
        }

        h.send(f1, 1);
        h.send(f2, 2);
        auto [o1, t1, md_o1] = h.recv();
        h.finish();
        auto [o2, t2, md_o2] = h.recv();
        compare(o1, a1, "f64: first output is the raw baseline");
        compare(o2, a2, "f64: EWMA recursion + -inf baseline re-seed");
    }

    std::printf(failures ? "\n%d FAILURE(S)\n" : "\nEXP_SMOOTH NUMERIC SMOKE PASSED\n", failures);
    return failures ? 1 : 0;
}
