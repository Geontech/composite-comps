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

#pragma once

#include "protocol_parser.hpp"
#include "vita49_parser.hpp"
#include "config.hpp"

#include <composite/buffers/slab_pool.hpp>

#include <cstdint>
#include <memory>

namespace parsers {

/**
 * @brief VITA 49.1 protocol parser
 */
class vita49dot1_parser : public protocol_parser {
public:
    explicit vita49dot1_parser(const struct_props::signal_overrides& overrides);
    ~vita49dot1_parser() override = default;

    auto can_parse(const composite::immutable_buffer<uint8_t>& data) const -> bool override;
    auto parse(
        const composite::immutable_buffer<uint8_t>& data,
        const composite::metadata& current_metadata
    ) -> parse_result override;
    auto name() const -> std::string_view override { return "vita49.1"; }
    // Delegates parsing (and thus metadata change-tracking) to the inner V49 parser, so the
    // (re)activation latch must be forwarded to it.
    auto on_activated() -> void override { m_vita49_parser.on_activated(); }

private:
    struct_props::signal_overrides m_overrides;
    vita49_parser m_vita49_parser;  // Delegate to standard V49 parser

    // PLRV (little-endian VRL) packets must be byteswapped into a NEW buffer whose lifetime
    // escapes downstream, which used to cost two heap allocations per packet. Steady state
    // now recycles slabs from this pool (created lazily at the first PLRV packet, resized by
    // recreation if a larger packet arrives; outstanding slabs keep the old pool alive until
    // released). Exhaustion (downstream holding > capacity packets) falls back to the heap.
    static constexpr std::size_t SWAP_POOL_BUFFERS{128};
    std::shared_ptr<composite::slab_pool<uint8_t>> m_swap_pool;
    std::size_t m_swap_pool_size{0};  ///< slab size of m_swap_pool (no pool accessor for it)

}; // class vita49dot1_parser

} // namespace parsers
