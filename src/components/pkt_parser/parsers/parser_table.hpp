/*
 * Copyright (C) 2026 Geon Technologies, LLC
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 * The one place the set of protocol parsers is declared.
 *
 * Why this file exists: adding a parser used to mean editing component.cpp in two places (an
 * include and an if-block in the middle of configure()), plus the component's source list. For a
 * downstream tree that carries a custom parser as a patch, that is three patch sites in files full
 * of unrelated logic — the kind of patch that conflicts on every upstream release.
 *
 * Two ways to add a parser, in order of preference:
 *
 *  1. NO PATCH. Drop a `parsers/local_parsers.inc` into the include path. It is picked up by
 *     __has_include below, at file scope, and must define:
 *
 *         #include "parsers/my_ext_parser.hpp"
 *         namespace parsers {
 *         inline auto register_local_parsers(std::vector<parser_entry>& t) -> void {
 *             t.push_back(make_parser_entry<my_ext_parser>("vita49-ext", 250));
 *         }
 *         } // namespace parsers
 *
 *     Nothing in this repository provides that file, and nothing here needs to change for it to
 *     work: the downstream tree adds its parser's .hpp/.cpp plus the .inc, adds the .cpp to its
 *     own source list, and patches zero lines.
 *
 *  2. Edit the table in parser_table() below — one line. Fine for an in-tree parser; for a
 *     downstream one it is a one-line patch to a file that contains nothing but this list, so a
 *     rebase conflict is both unlikely and trivial to resolve.
 *
 * SPECIFICITY, and why it is not insertion order: a more specific parser must be probed before a
 * more general one that would also claim the packet — the reason V49.1 is tried before V49. If
 * that ordering came from insertion order it would depend on the order translation units run
 * their initializers, which is unspecified and can differ between a static link and a DSO. The
 * failure mode is silent: the general parser claims the packet and the specific one never runs.
 * So order is an explicit number, sorted descending, and gaps of 100 leave room to insert.
 *
 *   300  sdds
 *   250  <- room for a downstream V49 extension that must beat both V49.1 and V49
 *   200  vita49.1   (more specific than vita49)
 *   100  vita49
 */
#pragma once

#include "config.hpp"
#include "protocol_parser.hpp"
#include "sdds_parser.hpp"
#include "vita49_parser.hpp"
#include "vita49dot1_parser.hpp"

#include <algorithm>
#include <memory>
#include <string_view>
#include <vector>

namespace parsers {

/// One row of the parser table.
struct parser_entry {
    std::string_view transport;   ///< matches the `transport` signal override; "" selects all
    int specificity;              ///< probed in DESCENDING order; see the header comment
    auto (*make)(const struct_props::signal_overrides&) -> std::unique_ptr<protocol_parser>;
};

template <typename Parser>
auto make_parser_entry(std::string_view transport, int specificity) -> parser_entry {
    return parser_entry{
        transport, specificity,
        [](const struct_props::signal_overrides& overrides) -> std::unique_ptr<protocol_parser> {
            return std::make_unique<Parser>(overrides);
        }};
}

} // namespace parsers

// Downstream hook, file scope so the .inc can #include its parser's header.
#if __has_include("parsers/local_parsers.inc")
#  include "parsers/local_parsers.inc"
#  define COMPS_HAS_LOCAL_PARSERS 1
#endif

namespace parsers {

#if !defined(COMPS_HAS_LOCAL_PARSERS)
/// No local parsers present; the hook is a no-op.
inline auto register_local_parsers(std::vector<parser_entry>&) -> void {}
#endif

/// The parser table, ordered most-specific-first. Built per call; called once per configure().
inline auto parser_table() -> std::vector<parser_entry> {
    std::vector<parser_entry> table;
    table.push_back(make_parser_entry<sdds_parser>("sdds", 300));
    table.push_back(make_parser_entry<vita49dot1_parser>("vita49.1", 200));
    table.push_back(make_parser_entry<vita49_parser>("vita49", 100));

    register_local_parsers(table);

    // stable_sort so equal specificities keep table order rather than becoming arbitrary.
    std::stable_sort(table.begin(), table.end(),
                     [](const parser_entry& a, const parser_entry& b) {
                         return a.specificity > b.specificity;
                     });
    return table;
}

} // namespace parsers
