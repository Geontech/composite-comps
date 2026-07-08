/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * This file is part of composite-comps.
 *
 * composite-comps is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * composite-comps is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see http://www.gnu.org/licenses/.
 */

// Integration test for pkt_parser's protocol auto-re-detection. Drives the real
// component's process() through connected ports and asserts on the private
// detection state (via the redetect_harness friend). It exercises:
//   1. first-packet protocol lock-in,
//   2. that a single good packet resets the consecutive-failure counter (isolated
//      corruption on a valid stream must NOT re-detect),
//   3. that a sustained run of parse failures (a framing change, e.g. a warm-pool
//      re-steer to a different digitizer) un-locks the parser after
//      REDETECT_AFTER_FAILURES and the next valid packet re-detects and re-locks.
//
// SDDS is used as the concrete protocol because a valid packet is trivial to craft
// (size 1080; only the first two header bytes gate detection). Detection itself is
// protocol-agnostic — it just locks the first parser whose can_parse() matches — so
// re-locking SDDS after an un-lock exercises the identical path a genuine SDDS->VITA49
// switch would take.

#include "component.hpp"

#include <composite/composite.hpp>
#include <composite/ports/input_port.hpp>
#include <composite/ports/output_port.hpp>

#include <vrtgen/vrtgen.hpp>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

constexpr std::size_t SDDS_PACKET_SIZE = 1080;

// A minimal valid SDDS packet: data_mode=2 (16-bit signed) in byte 0, bps=16 in
// byte 1 — the (data_mode, bps) pair sdds_parser::can_parse accepts. Everything else
// zero, which parses cleanly (seq 0, ttag 0).
auto make_sdds() -> composite::immutable_buffer<uint8_t> {
    auto v = std::make_shared<std::vector<uint8_t>>(SDDS_PACKET_SIZE, 0);
    (*v)[0] = 0x02;  // data_mode = 2 (DM_16BIT_SIGNED)
    (*v)[1] = 0x10;  // bps = 16
    return composite::immutable_buffer<uint8_t>(v);
}

// A packet the locked-in SDDS parser rejects: any size != 1080 makes sdds_parser::parse
// throw, which the component turns into a counted drop (a failure for the counter).
auto make_garbage() -> composite::immutable_buffer<uint8_t> {
    auto v = std::make_shared<std::vector<uint8_t>>(1000, 0);
    return composite::immutable_buffer<uint8_t>(v);
}

// --- VITA 49.1 packet builders (VRL header wrapping a big-endian inner V49 packet) ---
// Built with the same vrtgen packing classes the overlay parses with, so the bit layout
// cannot drift from the parser's expectations.

auto wrap_vrl(const std::vector<uint8_t>& inner) -> composite::immutable_buffer<uint8_t> {
    auto v = std::make_shared<std::vector<uint8_t>>();
    v->reserve(8 + inner.size());
    v->insert(v->end(), {'V', 'R', 'L', 'P'});   // big-endian VRLP magic on the wire
    v->insert(v->end(), {0, 0, 0, 0});           // frame count/size word (unread by the parser)
    v->insert(v->end(), inner.begin(), inner.end());
    return composite::immutable_buffer<uint8_t>(v);
}

// Minimal big-endian V49 signal-data packet: header + stream id + 8 payload bytes.
auto make_v491_data() -> composite::immutable_buffer<uint8_t> {
    vrtgen::packing::Header hdr;
    hdr.packet_type(vrtgen::packing::PacketType::SIGNAL_DATA_STREAM_ID);
    std::vector<uint8_t> buf(hdr.size() + sizeof(uint32_t) + 8, 0);
    hdr.packet_size(static_cast<uint16_t>(buf.size() / sizeof(uint32_t)));
    hdr.pack_into(buf.data());
    return wrap_vrl(buf);
}

// Big-endian V49 context packet carrying a sample rate and a signal data format
// (complex, 16-bit signed) — the fields the vita49 parser folds into metadata.
auto make_v491_context(double sample_rate) -> composite::immutable_buffer<uint8_t> {
    vrtgen::packing::Header hdr;
    hdr.packet_type(vrtgen::packing::PacketType::CONTEXT);
    vrtgen::packing::CIF0 cif;
    cif.sample_rate(true);
    cif.signal_data_format(true);
    vrtgen::packing::PayloadFormat pf;
    pf.real_complex_type(vrtgen::packing::DataSampleType::COMPLEX_CARTESIAN);
    pf.data_item_format(vrtgen::packing::DataItemFormat::SIGNED_FIXED);
    pf.data_item_size(16);

    // Layout the overlay parses: header | stream id | CIF0 | sample rate (fixed 44.20,
    // big-endian u64) | payload format.
    std::vector<uint8_t> buf(hdr.size() + sizeof(uint32_t) + cif.size() + sizeof(uint64_t) + pf.size(), 0);
    hdr.packet_size(static_cast<uint16_t>(buf.size() / sizeof(uint32_t)));
    std::size_t off = 0;
    hdr.pack_into(buf.data());
    off += hdr.size();
    off += sizeof(uint32_t);  // stream id (zeros)
    cif.pack_into(buf.data() + off);
    off += cif.size();
    auto sr_fixed = static_cast<uint64_t>(vrtgen::fixed::to_int<44, 20>(sample_rate));
    if constexpr (std::endian::native == std::endian::little) {
        sr_fixed = std::byteswap(sr_fixed);
    }
    std::memcpy(buf.data() + off, &sr_fixed, sizeof(sr_fixed));
    off += sizeof(uint64_t);
    pf.pack_into(buf.data() + off);
    return wrap_vrl(buf);
}

} // namespace (helpers)

// Global scope so the `friend struct redetect_harness;` in pkt_parser grants access.
struct redetect_harness {
    // Unique id per harness: component ids must be unique while instances overlap (the
    // metric series are labeled by id, and creating a duplicate series throws).
    static auto next_id() -> std::string {
        static int n = 0;
        return "pkt_parser_redetect_test_" + std::to_string(n++);
    }
    std::shared_ptr<pkt_parser> uut{std::make_shared<pkt_parser>(next_id())};
    composite::output_port<composite::immutable_buffer<uint8_t>> src{"src"};
    composite::input_port<composite::immutable_buffer<uint8_t>> sink{"sink"};

    redetect_harness() {
        // Register the parsers. In production this is driven by the initial property
        // batch; here we invoke the reaction hook directly (default overrides =>
        // empty transport => all three parsers registered, auto-detect).
        uut->property_change_handler(composite::properties::json::object());
        src.connect(&uut->m_in_port);
        uut->m_out_port.connect(&sink);
    }

    composite::metadata_ptr last_md;  ///< metadata that rode the most recent emitted packet

    auto active_name() const -> std::string_view {
        return uut->m_active_parser ? uut->m_active_parser->name() : std::string_view{"<none>"};
    }
    auto failures() const -> uint32_t { return uut->m_consecutive_parse_failures; }
    auto threshold() const -> uint32_t { return pkt_parser::REDETECT_AFTER_FAILURES; }

    // Send one packet, run one process() cycle, and return how many packets the
    // component emitted downstream (drained from the sink). Records the metadata that
    // rode the last emitted packet in last_md.
    auto feed(const composite::immutable_buffer<uint8_t>& pkt) -> std::size_t {
        src.send_data(pkt, composite::timestamp{0, 0});
        uut->process();
        std::size_t emitted = 0;
        while (sink.size() > 0) {
            auto [buf, ts, md] = sink.get_data();
            last_md = md;
            ++emitted;
        }
        return emitted;
    }
};

namespace {

int failures_seen = 0;
auto check(bool ok, const char* what) -> void {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures_seen;
    }
}

} // namespace

int main() {
    // --- 1. First-packet lock-in --------------------------------------------------
    redetect_harness h;
    check(h.active_name() == "<none>", "starts with no locked protocol");

    auto emitted = h.feed(make_sdds());
    check(h.active_name() == "sdds", "locks SDDS on first valid packet");
    check(emitted == 1, "first SDDS packet is parsed and forwarded");
    check(h.failures() == 0, "no failures after a clean parse");

    // --- 1b. Metadata contract: present on every packet, stable across unchanged ones ---
    // Every emitted packet must carry metadata (fft/psd substitute a default on a null
    // pointer), and the metadata must reflect the SDDS packet. The changed-flag design must
    // NOT drop metadata on steady-state packets.
    check(h.last_md != nullptr, "emitted packet carries metadata");
    check(h.last_md && h.last_md->format.bit_width == 16, "metadata bit_width = SDDS bps (16)");
    check(h.last_md && !h.last_md->format.is_complex, "metadata is_complex = false");
    check(h.last_md && h.last_md->annotations.count("protocol") == 1 &&
              h.last_md->annotations.at("protocol").to_string() == "sdds",
          "metadata carries protocol=sdds annotation");
    // Feed identical SDDS packets: metadata is unchanged, so the component must reuse the SAME
    // shared instance (pointer identity) — proving no per-packet rebuild/copy.
    auto* first_ptr = h.last_md.get();
    for (int i = 0; i < 5; ++i) {
        auto e = h.feed(make_sdds());
        check(e == 1, "steady-state SDDS packet still forwarded with metadata");
        check(h.last_md != nullptr, "steady-state packet still carries metadata");
        check(h.last_md.get() == first_ptr, "unchanged metadata reuses the same shared instance (no rebuild)");
    }

    // --- 2. Isolated corruption must NOT re-detect --------------------------------
    // Alternate one bad packet with one good one, many more times than the threshold.
    // The good packet resets the counter each round, so it never reaches the limit and
    // the parser stays locked to SDDS the whole time.
    for (int i = 0; i < static_cast<int>(h.threshold()) * 3; ++i) {
        auto dropped = h.feed(make_garbage());
        check(dropped == 0, "garbage packet is dropped (no output)");
        check(h.active_name() == "sdds", "stays locked through isolated corruption");
        auto ok = h.feed(make_sdds());
        check(ok == 1, "interleaved good SDDS packet still forwarded");
        check(h.failures() == 0, "one good packet resets the failure counter");
    }

    // --- 3. Sustained failure run un-locks, then re-detects -----------------------
    // Feed threshold-1 consecutive bad packets: still locked, counter climbing.
    for (uint32_t i = 1; i < h.threshold(); ++i) {
        (void)h.feed(make_garbage());
        check(h.active_name() == "sdds", "still locked before threshold");
        check(h.failures() == i, "consecutive failure counter increments");
    }
    // The threshold-th consecutive failure un-locks the parser for re-detection.
    (void)h.feed(make_garbage());
    check(h.active_name() == "<none>", "un-locks after REDETECT_AFTER_FAILURES failures");
    check(h.failures() == 0, "failure counter reset on un-lock");

    // The next valid packet re-runs detection and re-locks (self-heal, no reconfig).
    emitted = h.feed(make_sdds());
    check(h.active_name() == "sdds", "re-detects and re-locks on the next valid packet");
    check(emitted == 1, "output resumes after re-detection");
    check(h.failures() == 0, "clean state after re-lock");

    // --- 4. VITA 49.1: metadata stays shared across data packets AND context repeats ----
    // V49.1 delegates to the inner V49 parser, which stamps the transport annotation itself.
    // A repeated identical context packet must NOT look "changed" (the annotation the change
    // detection compares must be the one the component publishes), while a genuinely changed
    // context packet must republish.
    {
        redetect_harness h2;
        check(h2.feed(make_v491_context(1e6)) == 0, "v49.1 context packet emits no data");
        check(h2.active_name() == "vita49.1", "locks v49.1 on the context packet");

        auto e = h2.feed(make_v491_data());
        check(e == 1, "v49.1 data packet forwarded");
        check(h2.last_md != nullptr, "v49.1 data packet carries metadata");
        check(h2.last_md && h2.last_md->annotations.count("protocol") == 1 &&
                  h2.last_md->annotations.at("protocol").to_string() == "v49.1",
              "metadata carries protocol=v49.1 annotation");
        check(h2.last_md && h2.last_md->sample_rate == 1e6, "sample rate from context packet");
        check(h2.last_md && h2.last_md->format.is_complex, "complex format from context packet");
        check(h2.last_md && h2.last_md->format.bit_width == 16, "bit width from context packet");

        // Steady-state data packets share ONE metadata instance (pointer identity).
        auto* v491_ptr = h2.last_md.get();
        for (int i = 0; i < 3; ++i) {
            (void)h2.feed(make_v491_data());
            check(h2.last_md.get() == v491_ptr, "steady-state v49.1 data reuses the shared instance");
        }

        // An IDENTICAL context repeat (periodic context re-send) must not rebuild it either.
        check(h2.feed(make_v491_context(1e6)) == 0, "repeated identical context emits no data");
        (void)h2.feed(make_v491_data());
        check(h2.last_md.get() == v491_ptr,
              "identical context repeat does not rebuild the shared metadata");

        // A context packet with a CHANGED field must republish (change detection still works).
        check(h2.feed(make_v491_context(2e6)) == 0, "changed context emits no data");
        (void)h2.feed(make_v491_data());
        check(h2.last_md.get() != v491_ptr, "changed context rebuilds the shared metadata");
        check(h2.last_md && h2.last_md->sample_rate == 2e6, "new sample rate published");
        check(h2.last_md && h2.last_md->annotations.count("protocol") == 1 &&
                  h2.last_md->annotations.at("protocol").to_string() == "v49.1",
              "annotation still v49.1 after the rebuild");
    }

    if (failures_seen != 0) {
        std::fprintf(stderr, "pkt_parser re-detect: %d check(s) failed\n", failures_seen);
        return 1;
    }
    std::printf("pkt_parser re-detect: all checks passed\n");
    return 0;
}
