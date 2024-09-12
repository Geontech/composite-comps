#include <caddie/component.hpp>
#include <vector>

class file_writer : public caddie::component {
    using input_port_t = caddie::input_port<std::vector<std::vector<uint8_t>>>;
public:
    file_writer();
    ~file_writer() override;
    auto initialize() -> void override;
    auto process() -> caddie::retval override;

private:
    // Ports
    std::unique_ptr<input_port_t> m_in_port{std::make_unique<input_port_t>("data_in")};

    // Properties
    std::string m_filename;
    uint64_t m_num_bytes{};

    // Members
    int m_file{-1};
    uint64_t m_total_bytes{};

}; // class file_writer
