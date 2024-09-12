#include "component.hpp"
#include "overlay.hpp"

#include <fcntl.h>
#include <fstream>
#include <sys/uio.h>

file_writer::file_writer() : caddie::component("file_writer") {
    add_port(m_in_port.get());
    add_property("filename", &m_filename);
    add_property("num_bytes", &m_num_bytes);
}

file_writer::~file_writer() {
    close(m_file);
}

auto file_writer::initialize() -> void {
    m_file = open(m_filename.c_str(), O_CREAT|O_TRUNC|O_WRONLY, 0644);
}

auto file_writer::process() -> caddie::retval {
    auto data = m_in_port->get_data();
    if (data == nullptr) {
        return caddie::retval::NOOP;
    }
    auto curr_total = m_total_bytes;
    using iovec_t = struct iovec;
    auto iovecs = std::vector<iovec_t>{};
    for (auto& v : *data) {
        auto packet = v49_overlay(v);
        auto& header = packet.header();
        if (!is_data(header)) {
            continue;
        }
        auto payload = packet.payload<uint8_t>();
        iovecs.emplace_back(const_cast<uint8_t*>(payload.data()), payload.size_bytes());
        curr_total += payload.size_bytes();
        if (curr_total >= m_num_bytes) {
            iovecs.back().iov_len -= (curr_total - m_num_bytes);
            break;
        }
    }
    if (auto num_written = writev(m_file, iovecs.data(), iovecs.size()); num_written != -1) {
        m_total_bytes += num_written;
        if (m_total_bytes >= m_num_bytes) {
            m_in_port->clear();
            return caddie::retval::FINISH;
        }
    }
    return caddie::retval::NORMAL;
}

extern "C" {
    auto create() -> std::shared_ptr<caddie::component> {
        return std::make_shared<file_writer>();
    }
}