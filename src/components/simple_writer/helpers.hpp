#pragma once
#include <nlohmann/json.hpp>
#include <composite/metadata.hpp>
#include <fstream>
#include <string>

namespace sigmf {
    void write_sigmf_meta(const composite::timestamp& ts);
    inline std::string encode_datatype(const composite::metadata& meta);
    nlohmann::json build_sigmf_json(const composite::metadata& meta);
}