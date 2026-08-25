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
#include <span>
#include <stdexcept>
#include <vector>

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
    const bool is_little_endian = (first_word == VRLP_MAGIC);  // VRLP in memory = PLRV on wire
    const bool is_big_endian = (first_word == std::byteswap(VRLP_MAGIC));

    // Protocol lock-in does NOT trust later packets: re-validate the VRL magic word, exactly
    // as can_parse() does. Without this, any packet without VRL framing fell into the
    // big-endian path and was parsed as though bytes 8+ were a V49 packet.
    if (!is_little_endian && !is_big_endian) {
        throw std::out_of_range("vita49dot1_parser: missing VRL magic word");
    }

    // Slice off VRL framing (size >= MIN_V491_PACKET_SIZE guarantees no underflow)
    auto inner_v49_packet = data.slice(VRL_HEADER_SIZE, data.size() - VRL_HEADER_SIZE);

    // If PLRV (little-endian), byteswap entire inner packet to big-endian
    // so overlay can parse it correctly. The swapped bytes need a NEW buffer whose
    // lifetime escapes downstream; steady state recycles pooled slabs (zero heap
    // allocations per packet), falling back to the heap when the pool is exhausted
    // or a larger packet forces a pool rebuild mid-drain.
    composite::immutable_buffer<uint8_t> packet_to_parse;

    if (is_little_endian) {
        // Validate the inner packet's claimed geometry BEFORE any allocation or copy: the
        // V49 header word is the first inner word (little-endian on a PLRV wire), and its
        // low 16 bits claim the packet length in words. A claim that does not fit the
        // datagram (or claims nothing) is malformed — and honoring the claim also bounds
        // the byteswap/copy to the actual packet instead of the whole datagram.
        uint32_t inner_word0{};
        std::memcpy(&inner_word0, inner_v49_packet.data(), sizeof(inner_word0));
        if constexpr (std::endian::native == std::endian::big) {
            inner_word0 = std::byteswap(inner_word0);
        }
        const auto claimed_bytes = static_cast<std::size_t>(inner_word0 & 0xFFFFu) * sizeof(uint32_t);
        if (claimed_bytes < sizeof(uint32_t) || claimed_bytes > inner_v49_packet.size()) {
            throw std::out_of_range("vita49dot1_parser: inner packet size claim does not fit the datagram");
        }
        const auto src = std::span{inner_v49_packet.data(), claimed_bytes};

        if (m_swap_pool == nullptr || m_swap_pool_size < claimed_bytes) {
            // Lazy create / grow-by-recreate, with power-of-two slab sizes so untrusted
            // packet sizes cannot force a pool reallocation per packet: growth happens at
            // most log2(max datagram) times over the parser's lifetime. Outstanding slabs
            // hold shared ownership of their pool, so buffers already sent downstream stay
            // valid until released.
            const auto slab_size = std::bit_ceil(claimed_bytes);
            m_swap_pool = composite::slab_pool<uint8_t>::create(slab_size, SWAP_POOL_BUFFERS);
            m_swap_pool_size = slab_size;
        }

        if (auto slab = m_swap_pool->acquire()) {
            // claimed_bytes is a whole number of 32-bit words, so the byteswap covers the
            // slice exactly — no unswapped tail to scrub.
            overlay::v49::byteswap_u32_words(src, std::span{slab->data(), claimed_bytes});
            packet_to_parse = composite::immutable_buffer<uint8_t>(std::move(*slab)).slice(0, claimed_bytes);
        } else {
            // Pool exhausted: heap fallback keeps the stream flowing instead of stalling.
            std::vector<uint8_t> byteswapped_packet(claimed_bytes);
            overlay::v49::byteswap_u32_words(
                src, std::span{byteswapped_packet.data(), byteswapped_packet.size()});
            packet_to_parse = composite::immutable_buffer<uint8_t>(
                std::make_shared<std::vector<uint8_t>>(std::move(byteswapped_packet)));
        }
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
