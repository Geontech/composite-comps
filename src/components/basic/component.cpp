#include <caddie/component.hpp>
#include <iostream>

class basic : public caddie::component {
    using input_port_t = caddie::input_port<std::vector<uint8_t>>;
    using output_port_t = caddie::output_port<std::vector<uint8_t>>;
public:
    basic() :
      caddie::component("basic") {
        add_port(m_in_port.get());
        add_port(m_out_port.get());
    }

    auto process() -> caddie::retval override {
        using enum caddie::retval;
        if (auto data = m_in_port->get_data()) {
            std::cout << "GOT DATA: " << data->size() << std::endl;
            return NORMAL;
        }
        return NOOP;
    }

private:
    std::unique_ptr<input_port_t> m_in_port{std::make_unique<input_port_t>("data_in")};
    std::unique_ptr<output_port_t> m_out_port{std::make_unique<output_port_t>("data_out")};

}; // class component

extern "C" {
    auto create() -> std::shared_ptr<caddie::component> {
        return std::make_shared<basic>();
    }
}