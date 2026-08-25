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

// Real-world SDDS sets the standard-format flag (0x80): its high nibble reads as an
// UNDEFINED VITA packet type (8..15) when misinterpreted as a V49 header — the exact bytes
// a vita49-locked parser sees when the pipeline is re-steered to an SDDS receiver.
auto make_sdds_standard_format() -> composite::immutable_buffer<uint8_t> {
    auto v = std::make_shared<std::vector<uint8_t>>(SDDS_PACKET_SIZE, 0);
    (*v)[0] = 0x82;  // standard_format | data_mode = 2 (DM_16BIT_SIGNED)
    (*v)[1] = 0x10;  // bps = 16
    return composite::immutable_buffer<uint8_t>(v);
}

// A BARE (un-wrapped) big-endian V49 signal-data packet, so detection locks plain vita49
// rather than vita49.1.
auto make_v49_data() -> composite::immutable_buffer<uint8_t> {
    vrtgen::packing::Header hdr;
    hdr.packet_type(vrtgen::packing::PacketType::SIGNAL_DATA_STREAM_ID);
    auto v = std::make_shared<std::vector<uint8_t>>(hdr.size() + sizeof(uint32_t) + 8, 0);
    hdr.packet_size(static_cast<uint16_t>(v->size() / sizeof(uint32_t)));
    hdr.pack_into(v->data());
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
    auto feed(const composite::immutable_buffer<uint8_t>& pkt,
              composite::metadata_ptr md = nullptr) -> std::size_t {
        src.send_data(pkt, composite::timestamp{0, 0}, std::move(md));
        uut->process();
        std::size_t emitted = 0;
        while (sink.size() > 0) {
            auto [buf, ts, md] = sink.get_data();
            last_md = md;
            ++emitted;
        }
        return emitted;
    }

    // Enqueue a producer batch, invoke process() exactly once, and verify that
    // the component's bounded get_batch drain handles the complete ordered run.
    auto feed_batch(std::vector<composite::immutable_buffer<uint8_t>> packets) -> std::size_t {
        src.send_batch(std::span{packets}, composite::timestamp{0, 0});
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

    // --- 3a. Mid-stream PROTOCOL FLIP, both directions ------------------------------
    // Re-steering a pipeline from one receiver to another (SDDS <-> V49) must self-heal via
    // re-detection. The V49 -> SDDS direction used to be permanently broken: an SDDS flags
    // byte (standard_format 0x80) reads as an UNDEFINED V49 packet type, and the vita49
    // parser treated that as a SUCCESSFUL "unsupported type" parse — resetting the failure
    // counter every packet, so re-detection never fired and the pipeline went silent forever.
    {
        redetect_harness flip;

        // Lock plain vita49 with a bare data packet and confirm data flows.
        check(flip.feed(make_v49_data()) == 1, "flip: bare V49 data packet locks and forwards");
        check(flip.active_name() == "vita49", "flip: locked on vita49");

        // The stream becomes SDDS. Every packet must now COUNT as a failure...
        for (uint32_t i = 1; i < flip.threshold(); ++i) {
            check(flip.feed(make_sdds_standard_format()) == 0, "flip: SDDS packet dropped while locked on vita49");
            check(flip.failures() == i, "flip: SDDS packets count toward re-detection");
        }
        // ...until the threshold un-locks, and the NEXT packet re-detects as SDDS.
        (void)flip.feed(make_sdds_standard_format());
        check(flip.active_name() == "<none>", "flip: un-locks after sustained SDDS traffic");
        check(flip.feed(make_sdds_standard_format()) == 1, "flip: re-detects SDDS and output resumes");
        check(flip.active_name() == "sdds", "flip: locked on sdds after the flip");

        // And back: SDDS -> V49. Wrong-size packets throw in sdds_parser, so this direction
        // counts failures and re-detects the same way.
        for (uint32_t i = 0; i < flip.threshold(); ++i) {
            (void)flip.feed(make_v49_data());
        }
        check(flip.active_name() == "<none>", "flip back: un-locks after sustained V49 traffic");
        check(flip.feed(make_v49_data()) == 1, "flip back: re-detects V49 and output resumes");
        check(flip.active_name() == "vita49", "flip back: locked on vita49 again");
    }

    // A GENUINE V49 extension packet on a V49 stream must remain a quiet drop, never a
    // re-detection trigger (the fix distinguishes undefined type codes from defined ones).
    {
        redetect_harness ext;
        check(ext.feed(make_v49_data()) == 1, "ext: vita49 locked");
        vrtgen::packing::Header hdr;
        hdr.packet_type(vrtgen::packing::PacketType::EXTENSION_DATA);
        auto v = std::make_shared<std::vector<uint8_t>>(hdr.size() + 8, 0);
        hdr.packet_size(static_cast<uint16_t>(v->size() / sizeof(uint32_t)));
        hdr.pack_into(v->data());
        const auto ext_pkt = composite::immutable_buffer<uint8_t>(v);
        for (uint32_t i = 0; i < ext.threshold() * 2; ++i) {
            check(ext.feed(ext_pkt) == 0, "ext: extension packet dropped quietly");
        }
        check(ext.active_name() == "vita49", "ext: genuine extension packets never trigger re-detection");
        check(ext.failures() == 0, "ext: extension packets are not parse failures");
    }

    // --- 3a2. IN-BAND stream boundary (the primary path) ------------------------------
    // udp_source stamps a monotonic stream_session annotation, bumped per receiver
    // (re)construction. A session change resets detection CAUSALLY with the first packet of
    // the new stream: the very first packet of the new protocol is detected and forwarded —
    // zero packets spent on failure counting, no orchestration race.
    {
        redetect_harness inband;
        auto session = [](std::int64_t n) {
            composite::metadata md;
            md.annotations["stream_session"] = n;
            return composite::make_metadata(std::move(md));
        };
        const auto s1 = session(1);
        const auto s2 = session(2);

        check(inband.feed(make_v49_data(), s1) == 1, "in-band: vita49 locked and forwarding");
        check(inband.active_name() == "vita49", "in-band: locked on vita49");
        check(inband.feed(make_v49_data(), s1) == 1, "in-band: same session stays locked");
        check(inband.failures() == 0, "in-band: same session is not a failure");

        // The receiver was rebuilt (new stream): the FIRST packet re-detects and forwards.
        check(inband.feed(make_sdds_standard_format(), s2) == 1,
              "in-band: first packet of the new session is forwarded (zero loss)");
        check(inband.active_name() == "sdds", "in-band: locked on sdds at the session boundary");

        // The published metadata carries the session for downstream stream-state consumers —
        // with its ORIGINAL type preserved (an int64 stays an int64, never coerced to string) —
        // and the previous stream's carried metadata did NOT leak across the boundary.
        check(inband.last_md != nullptr &&
                  inband.last_md->annotations.count("stream_session") == 1 &&
                  inband.last_md->annotations.at("stream_session").holds<std::int64_t>() &&
                  inband.last_md->annotations.at("stream_session").get<std::int64_t>() == 2,
              "in-band: published metadata carries the new stream_session, type preserved");
    }

    // FIRST-OBSERVED session against pre-existing state: a parser locked from an UNANNOTATED
    // source, then re-pointed at an annotated one, must treat the first session it ever sees
    // as a boundary — not parse the new stream's first packets as the old protocol.
    {
        redetect_harness first;
        check(first.feed(make_v49_data()) == 1, "first-session: locked from an unannotated source");
        check(first.active_name() == "vita49", "first-session: vita49 active");

        composite::metadata md;
        md.annotations["stream_session"] = std::string{"udp0:7"};
        check(first.feed(make_sdds_standard_format(), composite::make_metadata(std::move(md))) == 1,
              "first-session: first annotated packet re-detects and forwards (zero loss)");
        check(first.active_name() == "sdds", "first-session: locked on sdds");
    }

    // Source REPLACEMENT: a new udp_source instance restarts its counter, so the token must
    // embed source identity ("<id>:<generation>") — two sources both at generation 1 are
    // still DIFFERENT sessions, and the parser must reset between them.
    {
        redetect_harness swap;
        auto session_of = [](const char* token) {
            composite::metadata md;
            md.annotations["stream_session"] = std::string{token};
            return composite::make_metadata(std::move(md));
        };
        check(swap.feed(make_v49_data(), session_of("source_a:1")) == 1, "swap: locked via source A");
        check(swap.active_name() == "vita49", "swap: vita49 active");
        check(swap.feed(make_sdds_standard_format(), session_of("source_b:1")) == 1,
              "swap: replacement source at the SAME generation still resets (identity in the token)");
        check(swap.active_name() == "sdds", "swap: re-detected across the source replacement");
    }

    // --- 3b. One process cycle drains a complete producer batch -------------------
    {
        redetect_harness hb;
        std::vector<composite::immutable_buffer<uint8_t>> packets;
        for (std::size_t i = 0; i < 32; ++i) {
            packets.push_back(make_sdds());
        }
        check(hb.feed_batch(std::move(packets)) == 32,
              "one process cycle parses and forwards a 32-packet input batch");
        check(hb.active_name() == "sdds", "batch preserves ordered protocol lock-in");
        check(hb.last_md != nullptr, "last packet in parser batch carries metadata");
    }

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

    // --- 5. SDDS unsigned data modes publish UNSIGNED metadata --------------------
    // can_parse() accepts DM_8BIT_UNSIGNED/DM_16BIT_UNSIGNED, but the parser used to publish
    // format.type = signed_integer unconditionally — every downstream sample interpretation
    // (framer conversion, histogram pivot) off by half full-scale. The type must follow the
    // packet's data mode, and a mid-stream signed->unsigned flip must republish metadata.
    {
        redetect_harness h3;
        auto make_sdds_unsigned16 = [] {
            auto v = std::make_shared<std::vector<uint8_t>>(SDDS_PACKET_SIZE, 0);
            (*v)[0] = 0x06; // data_mode = 6 (DM_16BIT_UNSIGNED)
            (*v)[1] = 0x10; // bps = 16
            return composite::immutable_buffer<uint8_t>(v);
        };
        check(h3.feed(make_sdds_unsigned16()) == 1, "unsigned-mode SDDS packet is parsed and forwarded");
        check(h3.last_md != nullptr, "unsigned-mode packet carries metadata");
        check(h3.last_md && h3.last_md->format.type == composite::data_type::unsigned_integer,
              "DM_16BIT_UNSIGNED publishes format.type = unsigned_integer");

        // Flip to the signed 16-bit mode: the type change must be detected and republished.
        auto* unsigned_ptr = h3.last_md.get();
        check(h3.feed(make_sdds()) == 1, "signed-mode SDDS packet still forwarded after the flip");
        check(h3.last_md.get() != unsigned_ptr, "signed<->unsigned flip rebuilds the shared metadata");
        check(h3.last_md && h3.last_md->format.type == composite::data_type::signed_integer,
              "DM_16BIT_SIGNED publishes format.type = signed_integer");
    }

    // --- annotation overrides: the ingest-boundary hook for stream facts the wire protocol
    // cannot carry (e.g. a stream that is already FFT data from a remote producer, where a
    // downstream psd needs fft_size / fft_window_sum_sq to normalize) ----------------------
    {
        redetect_harness h4;
        h4.uut->set_properties(
            composite::properties::json{{"signal_overrides",
                {{"annotations", {"fft_size=1024", "fft_window_sum_sq=7.25", "protocol=remote-fft"}}}}},
            composite::properties::config_type::INITIALIZE);

        check(h4.feed(make_sdds()) == 1, "packet forwarded with annotation overrides configured");
        check(h4.last_md != nullptr, "annotated packet carries metadata");
        if (h4.last_md != nullptr) {
            const auto& ann = h4.last_md->annotations;
            const auto size_it = ann.find("fft_size");
            const auto sq_it = ann.find("fft_window_sum_sq");
            const auto proto_it = ann.find("protocol");
            check(size_it != ann.end() && size_it->second.to_string() == "1024",
                  "declared fft_size annotation rides the published metadata");
            check(sq_it != ann.end() && sq_it->second.to_string() == "7.25",
                  "declared fft_window_sum_sq annotation rides the published metadata");
            check(proto_it != ann.end() && proto_it->second.to_string() == "remote-fft",
                  "an operator annotation overrides the parser-set key");
        }

        // Steady state must keep the SAME shared instance (overrides are merged only on
        // rebuild, not per packet).
        auto* first = h4.last_md.get();
        check(h4.feed(make_sdds()) == 1, "second annotated packet forwarded");
        check(h4.last_md.get() == first, "annotation overrides do not retrigger metadata republish");
    }

    if (failures_seen != 0) {
        std::fprintf(stderr, "pkt_parser re-detect: %d check(s) failed\n", failures_seen);
        return 1;
    }
    std::printf("pkt_parser re-detect: all checks passed\n");
    return 0;
}
