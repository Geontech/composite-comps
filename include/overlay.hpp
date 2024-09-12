#include <map>
#include <span>
#include <vrtgen/vrtgen.hpp>

auto is_data(const vrtgen::packing::Header& header) -> bool {
    using enum vrtgen::packing::PacketType;
    return (header.packet_type() == SIGNAL_DATA) || (header.packet_type() == SIGNAL_DATA_STREAM_ID);
}

auto is_context(const vrtgen::packing::Header& header) -> bool {
    using enum vrtgen::packing::PacketType;
    return (header.packet_type() == CONTEXT);
}

class v49_overlay {
public:
    explicit v49_overlay(std::span<const uint8_t> data) :
      m_data(data) {
        m_header.unpack_from(m_data.data());
        auto curr_idx = m_header.size();
        if (m_header.packet_type() != vrtgen::packing::PacketType::SIGNAL_DATA) {
            m_positions["stream_id"] = curr_idx;
            curr_idx += sizeof(uint32_t); // stream_id
        }
        if (m_header.class_id_enable()) {
            m_positions["class_id"] = curr_idx;
            curr_idx += 8; // bytes, class_id
        }
        if (m_header.tsi() != vrtgen::packing::TSI::NONE) {
            m_positions["integer_timestamp"] = curr_idx;
            curr_idx += sizeof(uint32_t); // integer_timestamp
        }
        if (m_header.tsf() != vrtgen::packing::TSF::NONE) {
            m_positions["fractional_timestamp"] = curr_idx;
            curr_idx += sizeof(uint64_t); // fractional_timestamp
        }
        if (is_data(m_header)) {
            m_positions["payload"] = curr_idx;
            auto data_header = vrtgen::packing::DataHeader{};
            data_header.unpack_from(m_data.data());
            if (data_header.trailer_included()) {
                m_positions["trailer"] = (m_header.packet_size() - 1) * sizeof(uint32_t)/*word size*/;
            }
        }
    }

    auto header() const -> const vrtgen::packing::Header& {
        return m_header;
    }

    auto stream_id() const -> std::optional<uint32_t> {
        if (!m_positions.contains("stream_id")) {
            return {};
        }
        auto pos = m_positions.at("stream_id");
        return vrtgen::swap::from_be(*reinterpret_cast<const uint32_t*>(m_data.data() + pos));
    }

    auto class_id() const -> std::optional<vrtgen::packing::ClassIdentifier> {
        if (!m_positions.contains("class_id")) {
            return {};
        }
        auto pos = m_positions.at("class_id");
        auto class_id = vrtgen::packing::ClassIdentifier{};
        class_id.unpack_from(m_data.data() + pos);
        return class_id;
    }

    auto integer_timestamp() const -> std::optional<uint32_t> {
        if (!m_positions.contains("integer_timestamp")) {
            return {};
        }
        auto pos = m_positions.at("integer_timestamp");
        return vrtgen::swap::from_be(*reinterpret_cast<const uint32_t*>(m_data.data() + pos));
    }

    auto fractional_timestamp() const -> std::optional<uint64_t> {
        if (!m_positions.contains("fractional_timestamp")) {
            return {};
        }
        auto pos = m_positions.at("fractional_timestamp");
        return vrtgen::swap::from_be(*reinterpret_cast<const uint64_t*>(m_data.data() + pos));
    }

    template<typename T>
    auto payload() const -> std::span<const T> {
        if (!m_positions.contains("payload")) {
            return {};
        }
        auto pos = m_positions.at("payload");
        auto data = reinterpret_cast<const T*>(m_data.data() + pos);
        return std::span<const T>(data, payload_size() / sizeof(T));
    }

    auto payload_size() const -> size_t {
        if (!m_positions.contains("payload")) {
            return {};
        }
        auto size = (m_header.packet_size() * sizeof(uint32_t)/*word size*/) - m_positions.at("payload");
        if (m_positions.contains("trailer")) {
            size -= sizeof(uint32_t);
        }
        return size;
    }

private:
    std::span<const uint8_t> m_data;
    vrtgen::packing::Header m_header;
    std::map<std::string, std::size_t> m_positions;

}; // class v49_overlay