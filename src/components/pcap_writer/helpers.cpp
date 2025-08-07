#include <nlohmann/json.hpp>
#include <composite/metadata.hpp>
#include <composite/timestamp.hpp>
#include <helpers.hpp>
#include <fstream>
#include <string>
#include <unordered_set>

namespace sigmf {
    nlohmann::json build_sigmf_json(const composite::metadata& meta) {
        nlohmann::json j;

        j["global"] = {
            {"core:version", "1.0.0"},
            {"core:datatype", encode_datatype(meta)}, 
            {"core:sample_rate", meta.sample_rate},
            {"core:center_frequency", meta.center_frequency},
            {"core:author", "RFNEXT"},
            {"core:extensions", {
                    {"name", "rfnext"},
                    {"version", "0.0.1"},
                    {"optional", "true"}
                },
            },
            {"rfnext:filetype", "pcap"},
            {"rfnext:timestamp", {}}
        };

        if (meta.bandwidth > 0) {
            j["global"]["core:bandwidth"] = meta.bandwidth;
        }

        return j;
    }

    inline std::string encode_datatype(const composite::metadata& meta) {
        const auto& format = meta.format;
        std::string dtype;
    
        if (format.is_complex) {
            dtype += "c";
        } else {
            dtype += "r";
        }
    
        switch (format.type) {
            case composite::data_type::signed_integer:
                dtype += "i";
                break;
            case composite::data_type::unsigned_integer:
                dtype += "u";
                break;
            case composite::data_type::floating_point:
                dtype += "f";
                break;
            default:
                throw std::invalid_argument("Unknown data_type in metadata");
        }
    
        dtype += std::to_string(format.bit_width);
    
        if (format.bit_width > 8) {
            switch (format.endianness) {
                case std::endian::little:
                    dtype += "le";
                    break;
                case std::endian::big:
                    dtype += "be";
                    break;
                default:
                    // std::endian::native or unrecognized — assume host-native (do nothing or log?)
                    break;
            }
        }
    
        return dtype;
    }
    
} //namespace sigmf