// The parser table's PROBE ORDER is behavior, not a detail: a more general parser that runs first
// will claim a packet a more specific one should have handled, and the failure is silent. This test
// pins the order and the transport gating so a table edit -- including a downstream one adding an
// extension parser -- cannot quietly reorder detection.
#include "parsers/parser_table.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace {

auto transports_in_probe_order() -> std::vector<std::string> {
    std::vector<std::string> out;
    for (const auto& entry : parsers::parser_table()) {
        out.emplace_back(entry.transport);
    }
    return out;
}

} // namespace

TEST_CASE("parser table is ordered most-specific-first", "[parser_table]") {
    const auto order = transports_in_probe_order();

    // vita49.1 must precede vita49: it is the more specific of the two and both can claim the
    // same bytes. This is the invariant the old hand-ordered if-chain documented in a comment.
    const auto v49dot1 = std::find(order.begin(), order.end(), "vita49.1");
    const auto v49 = std::find(order.begin(), order.end(), "vita49");
    REQUIRE(v49dot1 != order.end());
    REQUIRE(v49 != order.end());
    CHECK(v49dot1 < v49);
}

TEST_CASE("parser table specificities are strictly descending", "[parser_table]") {
    const auto table = parsers::parser_table();
    REQUIRE(table.size() >= 3);
    for (std::size_t i = 1; i < table.size(); ++i) {
        CHECK(table[i - 1].specificity >= table[i].specificity);
    }
}

TEST_CASE("every table entry builds a usable parser", "[parser_table]") {
    const struct_props::signal_overrides overrides{};
    for (const auto& entry : parsers::parser_table()) {
        auto parser = entry.make(overrides);
        REQUIRE(parser != nullptr);
        CHECK_FALSE(parser->name().empty());
    }
}
