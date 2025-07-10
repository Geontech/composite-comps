#include "component.hpp"

#include <complex>
#include <string_view>

extern "C" {
auto create(std::string_view type) -> std::shared_ptr<composite::component> {
    return std::make_shared<simple_file_writer<std::complex<float>>>("bin");
}
};
