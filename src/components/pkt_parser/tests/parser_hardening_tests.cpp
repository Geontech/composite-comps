/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * Correctness/hardening regressions for the protocol parsers:
 *  - VITA 49 CIF0 context decoding must honor the 64-bit widths of Timestamp Adjustment and
 *    Device Identifier (mis-sizing either shifted every later field, corrupting the decoded
 *    payload format for the whole stream).
 *  - A data packet whose claimed geometry yields no payload is malformed (throws -> counted
 *    drop), never forwarded as an empty buffer.
 *  - Sequence-gap detection flags every gap (seq_gap) but warns one-shot per activation.
 *  - SDDS parity packets (pp_id, seq % 32 == 31) carry FEC parity, not samples: they must
 *    consume their sequence slot without being forwarded and without a spurious gap.
 *  - PLRV (little-endian VRL) byteswap round-trips exactly through the pooled-slab path,
 *    including when the pool is exhausted and the heap fallback engages.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "parsers/sdds_parser.hpp"
#include "parsers/vita49_parser.hpp"
#include "parsers/vita49dot1_parser.hpp"
#include "config.hpp"

#include <composite/buffers/buffer.hpp>
#include <composite/core/metadata.hpp>

#include <vrtgen/vrtgen.hpp>

#include <bit>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using Catch::Matchers::WithinRel;

namespace {

// --- VITA 49 builders (vrtgen packing classes, so the layout cannot drift) -----------

auto make_v49_data(uint8_t packet_count, std::size_t payload_words = 2) -> composite::immutable_buffer<uint8_t> {
    vrtgen::packing::Header hdr;
    hdr.packet_type(vrtgen::packing::PacketType::SIGNAL_DATA);
    hdr.packet_count(packet_count);
    auto storage = std::make_shared<std::vector<uint8_t>>(hdr.size() + payload_words * 4, 0);
    hdr.packet_size(static_cast<uint16_t>(storage->size() / 4));
    hdr.pack_into(storage->data());
    return composite::immutable_buffer<uint8_t>{storage};
}

// Context packet: header | stream id | CIF0 | sample rate (64) | timestamp adjustment (64)
// | device id (64) | payload format (64). The two middle fields are exactly the ones whose
// widths were mis-sized; with the bug, the payload format is read 8 bytes early.
auto make_v49_context_with_64bit_skips(double sample_rate) -> composite::immutable_buffer<uint8_t> {
    vrtgen::packing::Header hdr;
    hdr.packet_type(vrtgen::packing::PacketType::CONTEXT);
    vrtgen::packing::CIF0 cif;
    cif.sample_rate(true);
    cif.timestamp_adjustment(true);
    cif.device_id(true);
    cif.signal_data_format(true);
    vrtgen::packing::PayloadFormat pf;
    pf.real_complex_type(vrtgen::packing::DataSampleType::COMPLEX_CARTESIAN);
    pf.data_item_format(vrtgen::packing::DataItemFormat::SIGNED_FIXED);
    pf.data_item_size(16);

    auto storage = std::make_shared<std::vector<uint8_t>>(
        hdr.size() + sizeof(uint32_t) + cif.size()
        + sizeof(uint64_t)   // sample rate
        + sizeof(uint64_t)   // timestamp adjustment (zeros)
        + sizeof(uint64_t)   // device id (zeros)
        + pf.size(), 0);
    hdr.packet_size(static_cast<uint16_t>(storage->size() / 4));

    std::size_t off = 0;
    hdr.pack_into(storage->data());
    off += hdr.size();
    off += sizeof(uint32_t);  // stream id (zeros)
    cif.pack_into(storage->data() + off);
    off += cif.size();
    auto sr_fixed = static_cast<uint64_t>(vrtgen::fixed::to_int<44, 20>(sample_rate));
    if constexpr (std::endian::native == std::endian::little) {
        sr_fixed = std::byteswap(sr_fixed);
    }
    std::memcpy(storage->data() + off, &sr_fixed, sizeof(sr_fixed));
    off += sizeof(uint64_t);
    off += sizeof(uint64_t);  // timestamp adjustment: zeros
    off += sizeof(uint64_t);  // device id: zeros
    pf.pack_into(storage->data() + off);
    return composite::immutable_buffer<uint8_t>{storage};
}

// --- SDDS builder --------------------------------------------------------------------

constexpr std::size_t SDDS_PACKET_SIZE = 1080;

auto make_sdds_packet(uint16_t seq, bool pp_id) -> composite::immutable_buffer<uint8_t> {
    auto storage = std::make_shared<std::vector<uint8_t>>(SDDS_PACKET_SIZE, 0);
    (*storage)[0] = static_cast<uint8_t>(0x02 | (pp_id ? 0x20 : 0x00));  // dm=2 (16-bit signed)
    (*storage)[1] = 0x10;                                                // bps=16
    const uint16_t be_seq = std::byteswap(seq);
    std::memcpy(storage->data() + 2, &be_seq, sizeof(be_seq));
    return composite::immutable_buffer<uint8_t>{storage};
}

auto make_v49_parser() -> parsers::vita49_parser {
    return parsers::vita49_parser{struct_props::signal_overrides{}, {}};
}

// --- VITA 49.1 (VRL) builders ---------------------------------------------------------

// Little-endian VRL frame: 'PLRV' magic + frame word + the inner packet with every 32-bit
// word byteswapped (hand-rolled here — the test must not build its input with the
// code under test).
auto wrap_plrv(const std::vector<uint8_t>& inner_be) -> composite::immutable_buffer<uint8_t> {
    auto storage = std::make_shared<std::vector<uint8_t>>();
    storage->reserve(8 + inner_be.size());
    storage->insert(storage->end(), {'P', 'L', 'R', 'V'});
    storage->insert(storage->end(), {0, 0, 0, 0});
    for (std::size_t i = 0; i + 4 <= inner_be.size(); i += 4) {
        storage->push_back(inner_be[i + 3]);
        storage->push_back(inner_be[i + 2]);
        storage->push_back(inner_be[i + 1]);
        storage->push_back(inner_be[i + 0]);
    }
    return composite::immutable_buffer<uint8_t>{storage};
}

// Big-endian inner V49 data packet with a recognizable payload pattern.
auto make_inner_be_data(uint8_t packet_count, uint8_t seed) -> std::vector<uint8_t> {
    vrtgen::packing::Header hdr;
    hdr.packet_type(vrtgen::packing::PacketType::SIGNAL_DATA);
    hdr.packet_count(packet_count);
    std::vector<uint8_t> buf(hdr.size() + 16, 0);
    hdr.packet_size(static_cast<uint16_t>(buf.size() / 4));
    hdr.pack_into(buf.data());
    for (std::size_t i = hdr.size(); i < buf.size(); ++i) {
        buf[i] = static_cast<uint8_t>(seed + i);
    }
    return buf;
}

} // namespace

// ============================================================================
// VITA 49 CIF0 64-bit field widths
// ============================================================================

TEST_CASE("context decoding honors 64-bit timestamp-adjustment and device-id fields",
          "[pkt_parser][hardening][cif0]") {
    auto parser = make_v49_parser();
    const composite::metadata current{};

    auto result = parser.parse(make_v49_context_with_64bit_skips(5e6), current);

    REQUIRE(result.metadata_changed);
    CHECK_THAT(result.metadata.sample_rate, WithinRel(5e6, 1e-9));
    // With the old 32-bit skips the payload format was read 8 bytes early — garbage here.
    CHECK(result.metadata.format.is_complex == true);
    CHECK(result.metadata.format.type == composite::data_type::signed_integer);
    CHECK(result.metadata.format.bit_width == 16);
    CHECK_FALSE(result.should_send);  // context packets carry no data
}

// ============================================================================
// Empty-payload data packets are malformed, not forwarded
// ============================================================================

TEST_CASE("a data packet with no payload is rejected, not forwarded empty",
          "[pkt_parser][hardening][payload]") {
    auto parser = make_v49_parser();
    const composite::metadata current{};

    // Header-only data packet: claimed size covers no payload words.
    auto pkt = make_v49_data(1, /*payload_words=*/0);
    CHECK_THROWS_AS(parser.parse(pkt, current), std::out_of_range);
}

// ============================================================================
// VITA 49 sequence gaps: per-gap flag, one-shot warning
// ============================================================================

TEST_CASE("vita49 sequence gaps set seq_gap per gap but warn one-shot",
          "[pkt_parser][hardening][seq]") {
    auto parser = make_v49_parser();
    const composite::metadata current{};

    (void)parser.parse(make_v49_data(1), current);              // first packet: no gap check
    auto ok = parser.parse(make_v49_data(2), current);          // in sequence
    CHECK_FALSE(ok.seq_gap);

    auto gap1 = parser.parse(make_v49_data(5), current);        // expected 3
    CHECK(gap1.seq_gap);
    CHECK(gap1.warning.has_value());

    auto gap2 = parser.parse(make_v49_data(9), current);        // expected 6
    CHECK(gap2.seq_gap);
    CHECK_FALSE(gap2.warning.has_value());                      // warning already latched

    parser.on_activated();                                      // new stream re-arms it
    (void)parser.parse(make_v49_data(1), current);
    auto gap3 = parser.parse(make_v49_data(4), current);
    CHECK(gap3.seq_gap);
    CHECK(gap3.warning.has_value());
}

// ============================================================================
// SDDS parity packets
// ============================================================================

TEST_CASE("SDDS parity packets are consumed, not forwarded as sample data",
          "[pkt_parser][hardening][sdds][parity]") {
    parsers::sdds_parser parser{struct_props::signal_overrides{}};
    const composite::metadata current{};

    auto d30 = parser.parse(make_sdds_packet(30, false), current);
    CHECK(d30.should_send);

    // Parity slot: pp_id set, seq % 32 == 31. FEC bytes, not samples.
    auto parity = parser.parse(make_sdds_packet(31, true), current);
    CHECK_FALSE(parity.should_send);
    CHECK_FALSE(parity.seq_gap);
    CHECK_FALSE(parity.warning.has_value());   // a healthy parity stream is not an anomaly

    // The parity packet consumed its sequence slot: the next data packet is in sequence.
    auto d32 = parser.parse(make_sdds_packet(32, false), current);
    CHECK(d32.should_send);
    CHECK_FALSE(d32.seq_gap);
    CHECK_FALSE(d32.warning.has_value());
}

TEST_CASE("SDDS sequence gaps set seq_gap per gap but warn one-shot",
          "[pkt_parser][hardening][sdds][seq]") {
    parsers::sdds_parser parser{struct_props::signal_overrides{}};
    const composite::metadata current{};

    (void)parser.parse(make_sdds_packet(1, false), current);   // first packet: no gap check
    auto gap1 = parser.parse(make_sdds_packet(5, false), current);
    CHECK(gap1.seq_gap);
    CHECK(gap1.warning.has_value());

    auto gap2 = parser.parse(make_sdds_packet(10, false), current);
    CHECK(gap2.seq_gap);
    CHECK_FALSE(gap2.warning.has_value());

    parser.on_activated();
    (void)parser.parse(make_sdds_packet(1, false), current);
    auto gap3 = parser.parse(make_sdds_packet(9, false), current);
    CHECK(gap3.seq_gap);
    CHECK(gap3.warning.has_value());
}

TEST_CASE("SDDS parity-flag anomalies warn one-shot", "[pkt_parser][hardening][sdds][parity]") {
    parsers::sdds_parser parser{struct_props::signal_overrides{}};
    const composite::metadata current{};

    // pp_id set on a non-parity slot: anomalous, warned once. Consecutive sequence numbers
    // keep the gap detector quiet so the anomaly warning is the only candidate.
    auto first = parser.parse(make_sdds_packet(1, true), current);
    CHECK(first.warning.has_value());
    auto second = parser.parse(make_sdds_packet(2, true), current);
    CHECK_FALSE(second.warning.has_value());
}

// ============================================================================
// Sequence tracking around counter wrap and zero-valued sequence numbers
// ============================================================================

TEST_CASE("vita49 gap detection stays armed after a packet with count 0",
          "[pkt_parser][hardening][seq][wrap]") {
    auto parser = make_v49_parser();
    const composite::metadata current{};

    (void)parser.parse(make_v49_data(15), current);              // seed
    auto wrap = parser.parse(make_v49_data(0), current);         // 15 -> 0 wrap, in sequence
    CHECK_FALSE(wrap.seq_gap);

    // The old `m_pkt_count != 0` sentinel suppressed this check after every count-0 packet
    // (1 in 16 packets with a 4-bit counter).
    auto gap = parser.parse(make_v49_data(2), current);          // expected 1
    CHECK(gap.seq_gap);
}

TEST_CASE("SDDS gap detection stays armed across the 65535 wrap and parity slots",
          "[pkt_parser][hardening][sdds][seq][wrap]") {
    parsers::sdds_parser parser{struct_props::signal_overrides{}};
    const composite::metadata current{};

    (void)parser.parse(make_sdds_packet(65534, false), current);       // seed
    // 65535 % 32 == 31: the wrap slot is a parity slot on a parity-enabled stream.
    auto parity = parser.parse(make_sdds_packet(65535, true), current);
    CHECK_FALSE(parity.seq_gap);
    CHECK_FALSE(parity.should_send);
    auto wrap = parser.parse(make_sdds_packet(0, false), current);     // in sequence after wrap
    CHECK_FALSE(wrap.seq_gap);

    // The old `m_pkt_count != 0` sentinel suppressed this check right after the wrap.
    auto gap = parser.parse(make_sdds_packet(2, false), current);      // expected 1
    CHECK(gap.seq_gap);
}

TEST_CASE("a parity packet arriving early reveals the loss of the data packets before it",
          "[pkt_parser][hardening][sdds][parity][seq]") {
    parsers::sdds_parser parser{struct_props::signal_overrides{}};
    const composite::metadata current{};

    (void)parser.parse(make_sdds_packet(1, false), current);           // seed
    // Parity for the slot-31 boundary arrives next: data packets 2..30 were lost. Checking
    // the sequence after the parity early-return used to hide this entirely.
    auto parity = parser.parse(make_sdds_packet(31, true), current);
    CHECK(parity.seq_gap);
    CHECK_FALSE(parity.should_send);
}

// ============================================================================
// Locked-in re-validation
// ============================================================================

TEST_CASE("a locked SDDS parser rejects an invalid data-mode/bps combination",
          "[pkt_parser][hardening][sdds][revalidate]") {
    parsers::sdds_parser parser{struct_props::signal_overrides{}};
    const composite::metadata current{};

    // Right size, garbage mode fields: data_mode=3 (undefined), bps=16.
    auto storage = std::make_shared<std::vector<uint8_t>>(SDDS_PACKET_SIZE, 0);
    (*storage)[0] = 0x03;
    (*storage)[1] = 0x10;
    composite::immutable_buffer<uint8_t> pkt{storage};

    CHECK_FALSE(parser.can_parse(pkt));
    CHECK_THROWS_AS(parser.parse(pkt, current), std::out_of_range);
}

TEST_CASE("a locked VITA 49.1 parser rejects packets without the VRL magic word",
          "[pkt_parser][hardening][plrv][revalidate]") {
    parsers::vita49dot1_parser parser{struct_props::signal_overrides{}};
    const composite::metadata current{};

    // A perfectly valid BARE V49 data packet, padded in front by 8 non-VRL bytes: without
    // magic re-validation this parsed bytes 8+ as an inner packet instead of rejecting.
    auto inner = make_inner_be_data(1, 0x11);
    auto storage = std::make_shared<std::vector<uint8_t>>();
    storage->insert(storage->end(), {'X', 'X', 'X', 'X', 0, 0, 0, 0});
    storage->insert(storage->end(), inner.begin(), inner.end());
    composite::immutable_buffer<uint8_t> pkt{storage};

    CHECK_FALSE(parser.can_parse(pkt));
    CHECK_THROWS_AS(parser.parse(pkt, current), std::out_of_range);
}

TEST_CASE("a PLRV inner packet whose size claim does not fit the datagram is rejected",
          "[pkt_parser][hardening][plrv][revalidate]") {
    parsers::vita49dot1_parser parser{struct_props::signal_overrides{}};
    const composite::metadata current{};

    // Claimed packet_size = 0 words.
    auto zero_claim = make_inner_be_data(1, 0x22);
    {
        vrtgen::packing::Header hdr;
        hdr.packet_type(vrtgen::packing::PacketType::SIGNAL_DATA);
        hdr.packet_size(0);
        hdr.pack_into(zero_claim.data());
    }
    CHECK_THROWS_AS(parser.parse(wrap_plrv(zero_claim), current), std::out_of_range);

    // Claimed packet_size larger than the datagram carries.
    auto over_claim = make_inner_be_data(1, 0x33);
    {
        vrtgen::packing::Header hdr;
        hdr.packet_type(vrtgen::packing::PacketType::SIGNAL_DATA);
        hdr.packet_size(static_cast<uint16_t>(over_claim.size() / 4 + 1));
        hdr.pack_into(over_claim.data());
    }
    CHECK_THROWS_AS(parser.parse(wrap_plrv(over_claim), current), std::out_of_range);
}

// ============================================================================
// Timestamp invariant (picoseconds < 1e12)
// ============================================================================

namespace {

auto make_v49_data_with_tsf(vrtgen::packing::TSF tsf, uint64_t frac,
                            std::size_t payload_words = 2) -> composite::immutable_buffer<uint8_t> {
    vrtgen::packing::Header hdr;
    hdr.packet_type(vrtgen::packing::PacketType::SIGNAL_DATA);
    hdr.packet_count(1);
    hdr.tsf(tsf);
    auto storage = std::make_shared<std::vector<uint8_t>>(
        hdr.size() + sizeof(uint64_t) + payload_words * 4, 0);
    hdr.packet_size(static_cast<uint16_t>(storage->size() / 4));
    hdr.pack_into(storage->data());
    auto be_frac = frac;
    if constexpr (std::endian::native == std::endian::little) {
        be_frac = std::byteswap(be_frac);
    }
    std::memcpy(storage->data() + hdr.size(), &be_frac, sizeof(be_frac));
    return composite::immutable_buffer<uint8_t>{storage};
}

} // namespace

TEST_CASE("fractional timestamps are normalized to the picoseconds-per-second invariant",
          "[pkt_parser][hardening][timestamp]") {
    const composite::metadata no_rate{};

    SECTION("REAL_TIME: an out-of-range wire value carries into seconds") {
        auto parser = make_v49_parser();
        // 3.5 seconds expressed in picoseconds — invalid on the wire, normalized here.
        auto result = parser.parse(
            make_v49_data_with_tsf(vrtgen::packing::TSF::REAL_TIME, 3'500'000'000'000ULL), no_rate);
        CHECK(result.timestamp.seconds == 3);
        CHECK(result.timestamp.picoseconds == 500'000'000'000ULL);
    }

    SECTION("SAMPLE_COUNT: a count spanning more than a second carries into seconds") {
        auto parser = make_v49_parser();
        composite::metadata with_rate{};
        with_rate.sample_rate = 1e6;
        // 2,500,000 samples at 1 MHz = 2.5 s.
        auto result = parser.parse(
            make_v49_data_with_tsf(vrtgen::packing::TSF::SAMPLE_COUNT, 2'500'000ULL), with_rate);
        CHECK(result.timestamp.seconds == 2);
        CHECK(result.timestamp.picoseconds == 500'000'000'000ULL);
    }
}

// ============================================================================
// PLRV byteswap round-trip through the pooled-slab path
// ============================================================================

TEST_CASE("PLRV packets round-trip exactly through the pooled byteswap, including pool exhaustion",
          "[pkt_parser][hardening][plrv]") {
    parsers::vita49dot1_parser parser{struct_props::signal_overrides{}};
    const composite::metadata current{};

    // Hold every payload alive so the slab pool (128 buffers) exhausts partway through and
    // the heap fallback engages — both paths must produce identical, correct bytes.
    std::vector<composite::immutable_buffer<uint8_t>> held;
    constexpr int PACKETS = 200;

    for (int n = 0; n < PACKETS; ++n) {
        const auto count = static_cast<uint8_t>((n + 1) % 16);
        const auto seed = static_cast<uint8_t>(n);
        auto inner_be = make_inner_be_data(count, seed);

        auto result = parser.parse(wrap_plrv(inner_be), current);
        REQUIRE(result.should_send);

        // The parsed payload must equal the big-endian inner packet's payload bytes.
        const std::size_t header_bytes = 4;
        REQUIRE(result.payload.size() == inner_be.size() - header_bytes);
        REQUIRE(std::memcmp(result.payload.data(), inner_be.data() + header_bytes,
                            result.payload.size()) == 0);

        held.push_back(std::move(result.payload));
    }

    // Release and run more packets: the pool recycles and stays correct.
    held.clear();
    for (int n = 0; n < 10; ++n) {
        auto inner_be = make_inner_be_data(static_cast<uint8_t>(n % 16), static_cast<uint8_t>(0x40 + n));
        auto result = parser.parse(wrap_plrv(inner_be), current);
        REQUIRE(result.should_send);
        REQUIRE(std::memcmp(result.payload.data(), inner_be.data() + 4, result.payload.size()) == 0);
    }
}
