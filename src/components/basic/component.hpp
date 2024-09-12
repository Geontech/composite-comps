#include <caddie/component.hpp>
#include <iostream>

class basic : public caddie::component {
    using input_port_t = caddie::input_port<std::vector<uint8_t>>;
    using output_port_t = caddie::output_port<std::vector<uint8_t>>;
public:
    basic();
    auto process() -> caddie::retval override;

private:
    // Ports
    std::unique_ptr<input_port_t> m_in_port{std::make_unique<input_port_t>("data_in")};
    std::unique_ptr<output_port_t> m_out_port{std::make_unique<output_port_t>("data_out")};

}; // class basic
