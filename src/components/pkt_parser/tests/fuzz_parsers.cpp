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

// Seeded-random fuzz of the protocol parsers on UNTRUSTED packet bytes. Pins the
// pkt_parser bounds/geometry guards: feeding arbitrary/truncated buffers through
// can_parse + parse must never read out of bounds or crash — a packet that does
// not fit its claimed geometry is rejected (throws, which the component turns into
// a counted drop). Run under ASan/UBSan to actually catch an OOB read.
//
// Build/run is gated on BUILD_TESTING; see tests/CMakeLists.txt. For the strongest
// signal, build the enclosing project with -fsanitize=address,undefined.

#include "parsers/sdds_parser.hpp"
#include "parsers/vita49_parser.hpp"
#include "parsers/vita49dot1_parser.hpp"
#include "config.hpp"

#include <composite/buffers/buffer.hpp>
#include <composite/core/metadata.hpp>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

int main() {
    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int> byte(0, 255);

    struct_props::signal_overrides ov{};
    parsers::sdds_parser sdds(ov);
    parsers::vita49_parser v49(ov);
    parsers::vita49dot1_parser v491(ov);
    parsers::protocol_parser* parsers_under_test[] = {&sdds, &v49, &v491};

    const composite::metadata md;
    std::size_t handled = 0;
    std::size_t rejected = 0;

    // Sizes 0..1199 cover sub-minimum, the exact 1080-byte SDDS size, and beyond.
    for (int iter = 0; iter < 120000; ++iter) {
        const auto n = static_cast<std::size_t>(iter % 1200);
        auto storage = std::make_shared<std::vector<uint8_t>>(n);
        for (auto& b : *storage) { b = static_cast<uint8_t>(byte(rng)); }
        composite::immutable_buffer<uint8_t> data(storage);

        for (auto* p : parsers_under_test) {
            try {
                if (p->can_parse(data)) { (void)p->parse(data, md); }
                ++handled;
            } catch (const std::exception&) {
                ++rejected;  // bounds/geometry guard fired — correct rejection
            }
        }
    }

    std::printf("pkt_parser fuzz: %zu handled, %zu rejected, no OOB/crash\n", handled, rejected);
    // Sanity: the guards must actually engage on this corpus (truncated packets are
    // present), otherwise the test is not exercising the rejection paths.
    if (rejected == 0) {
        std::fprintf(stderr, "FAIL: no malformed packet was rejected — guards not exercised\n");
        return 1;
    }
    return 0;
}
