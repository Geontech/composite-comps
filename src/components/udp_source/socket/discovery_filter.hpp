/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#pragma once

#include <cstdint>
#include <sys/types.h>

namespace udp {

/**
 * Frame-size autodiscovery classifier.
 *
 * When msg_size is not configured, the receiver sizes its pool from the first datagram on the
 * wire that looks like payload rather than signalling. This is the pure predicate for that
 * decision: no I/O, no state, so the classification is unit-testable without a socket.
 */
struct discovery_filter {
    // Minimum data packet size, in an attempt to exclude context packets. Deliberately signed
    // to match recvfrom's ssize_t return: as std::size_t the comparison below would be
    // unsigned, promoting a -1 error return to SIZE_MAX and passing the size check.
    static constexpr ssize_t MIN_DATA_PACKET_SIZE = 512;

    // SDDS is a fixed-size protocol: 56-byte header + 1024-byte payload. sdds_parser rejects
    // anything else outright, so this size is exact, not a minimum.
    static constexpr ssize_t SDDS_PACKET_SIZE = 1080;

    // VITA 49 packet type identifiers, in the high nibble of byte 0.
    static constexpr uint8_t V49_CONTEXT_PACKET = 0x40;
    static constexpr uint8_t V49_EXT_CONTEXT_PACKET = 0x50;

    /**
     * True if this datagram may be used to size the receive pool.
     *
     * V49 context packets are skipped so the pool is sized from data packets: in VITA 49 the
     * high nibble of byte 0 is the packet type. In SDDS that same byte is a flags field --
     * standard_format (0x80), pp_id (0x20), data_mode (0x07) -- and nothing requires
     * standard_format to be set, so a legitimate SDDS packet can present ANY high nibble, 0x40
     * and 0x50 included. Applying the nibble test to it discarded every packet of a pure-SDDS
     * stream, exhausting the discovery window and throwing -- rejecting at this layer exactly
     * what sdds_parser::can_parse accepts downstream. Size disambiguates: at exactly the SDDS
     * size the bytes are not a V49 header, so the type test does not apply.
     */
    [[nodiscard]] static constexpr auto is_sizing_candidate(ssize_t recvd, uint8_t first_byte) noexcept -> bool {
        if (recvd <= MIN_DATA_PACKET_SIZE) {
            return false;
        }
        if (recvd == SDDS_PACKET_SIZE) {
            return true;
        }
        const auto pkt_type = static_cast<uint8_t>(first_byte & 0xF0);
        return (pkt_type != V49_CONTEXT_PACKET) && (pkt_type != V49_EXT_CONTEXT_PACKET);
    }
};

} // namespace udp
