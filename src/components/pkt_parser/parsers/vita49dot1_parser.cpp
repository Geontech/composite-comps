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

#include "overlay/vita49.hpp"
#include "vita49dot1_parser.hpp"
#include "vita49_parser.hpp"

#include <bit>
#include <cstring>
#include <format>
#include <stdexcept>

namespace parsers {

// VITA 49.1 VRL Framing Constants
constexpr uint32_t VRLP_MAGIC = 0x56524C50;      // "VRLP" magic word (big-endian)
constexpr std::size_t VRL_HEADER_SIZE = 8;       // VRLP word + frame word
constexpr std::size_t MIN_V491_PACKET_SIZE = 12; // VRL header + minimal V49 header

vita49dot1_parser::vita49dot1_parser(const struct_props::signal_overrides& overrides) :
  m_overrides(overrides),
  // The inner parser stamps the transport annotation itself, so its metadata change
  // detection compares like-for-like with what the component publishes. Rewriting the
  // annotation here after the fact would make every context packet's candidate ("v49")
  // differ from the published value ("v49.1") — a spurious republish per context packet.
  m_vita49_parser(overrides, "v49.1") {}

auto vita49dot1_parser::can_parse(const composite::immutable_buffer<uint8_t>& data) const -> bool {
    // Need at least VRL header + minimal V49 header
    if (data.size() < MIN_V491_PACKET_SIZE) {
        return false;
    }

    // Check for VRLP magic word (big or little endian). memcpy avoids the
    // misaligned-load / strict-aliasing UB of a reinterpret_cast.
    uint32_t first_word{};
    std::memcpy(&first_word, data.data(), sizeof(first_word));
    return (first_word == VRLP_MAGIC) || (first_word == std::byteswap(VRLP_MAGIC));
}

auto vita49dot1_parser::parse(
    const composite::immutable_buffer<uint8_t>& data,
    const composite::metadata& current_metadata
) -> parse_result {
    // Untrusted input: protocol lock-in does NOT trust later packets. Re-validate
    // the minimum size before reading the VRL magic word and stripping the 8-byte
    // VRL header — otherwise data.size() - VRL_HEADER_SIZE underflows to ~SIZE_MAX
    // and is fed to slice() as a giant count. Caught by the component -> drop.
    if (data.size() < MIN_V491_PACKET_SIZE) {
        throw std::out_of_range("vita49dot1_parser: packet smaller than minimum V49.1 size");
    }

    // Check for PLRV (little-endian, needs byteswap). memcpy, not reinterpret_cast,
    // to avoid a misaligned-load / strict-aliasing UB.
    uint32_t first_word{};
    std::memcpy(&first_word, data.data(), sizeof(first_word));
    bool is_little_endian = (first_word == VRLP_MAGIC);  // VRLP in memory = PLRV on wire

    // Slice off VRL framing (size >= MIN_V491_PACKET_SIZE guarantees no underflow)
    auto inner_v49_packet = data.slice(VRL_HEADER_SIZE, data.size() - VRL_HEADER_SIZE);

    // If PLRV (little-endian), byteswap entire inner packet to big-endian
    // so overlay can parse it correctly
    composite::immutable_buffer<uint8_t> packet_to_parse;
    std::vector<uint8_t> byteswapped_packet;

    if (is_little_endian) {
        // Byteswap entire packet at 32-bit word boundaries using SIMD (load-swap-store)
        byteswapped_packet.resize(inner_v49_packet.size());
        overlay::v49::byteswap_u32_words(
            std::span{inner_v49_packet.data(), inner_v49_packet.size()},
            std::span{byteswapped_packet.data(), byteswapped_packet.size()}
        );

        // move (not copy) the byteswapped bytes into the shared buffer — the prior
        // make_shared(byteswapped_packet) copied the whole vector per PLRV packet.
        packet_to_parse = composite::immutable_buffer<uint8_t>(std::make_shared<std::vector<uint8_t>>(std::move(byteswapped_packet)));
    } else {
        packet_to_parse = inner_v49_packet;
    }

    // Delegate to vita49_parser for parsing the inner V49 packet (now big-endian). No
    // metadata fix-up is needed here: the inner parser stamps the "v49.1" transport
    // annotation itself (constructor parameter), and it only ever sees big-endian bytes
    // (PLRV packets were byteswapped above), so the endianness it extracts is already big.
    return m_vita49_parser.parse(packet_to_parse, current_metadata);
}

} // namespace parsers
