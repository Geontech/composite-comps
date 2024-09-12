#include "component.hpp"

#include <iostream>

basic::basic() : caddie::component("basic") {
    add_port(m_in_port.get());
    add_port(m_out_port.get());
}

auto basic::process() -> caddie::retval {
    using enum caddie::retval;
    if (auto data = m_in_port->get_data()) {
        std::cout << "GOT DATA: " << data->size() << std::endl;
        return NORMAL;
    }
    return NOOP;
}

extern "C" {
    auto create() -> std::shared_ptr<caddie::component> {
        return std::make_shared<basic>();
    }
}