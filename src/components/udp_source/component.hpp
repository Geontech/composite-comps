#include <array>
#include <caddie/component.hpp>
#include <mutex>
#include <poll.h>
#include <queue>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/uio.h>
#include <thread>
#include <unistd.h>

namespace udpsrc::net {

class mmsgs {
public:
    using buffer_type = std::vector<std::vector<uint8_t>>;

    mmsgs(size_t num_msgs, size_t msg_size);

    std::vector<struct mmsghdr> msgs;
    std::vector<struct iovec> iovecs;
    std::unique_ptr<buffer_type> buffers = std::make_unique<buffer_type>();

}; // class mmsgs

auto get_interface_ip(int fd, std::string_view interface) -> std::string;

} // namespace udpsrc::net

class udp_source : public caddie::component {
    using output_port_t = caddie::output_port<std::vector<std::vector<uint8_t>>>;
    static constexpr std::uint32_t RECV_BUF_SIZE{0xFFFF};
public:
    udp_source();
    ~udp_source() override;
    auto initialize() -> void override;
    auto start() -> void override;
    auto stop() -> void override;
    auto process() -> caddie::retval override;

private:
    // Ports
    std::unique_ptr<output_port_t> m_out_port{std::make_unique<output_port_t>("data_out")};
    
    // Properties
    std::string m_interface;
    std::string m_ip_addr;
    uint32_t m_port{};
    uint32_t m_recv_buf_size{RECV_BUF_SIZE};
    uint32_t m_msg_size{};
    uint32_t m_num_msgs{};

    // Members
    int m_socket{-1};
    std::array<struct pollfd, 1> m_pfds;
    size_t m_queue_size{m_num_msgs / 2};
    std::queue<std::unique_ptr<udpsrc::net::mmsgs>> m_queue;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    std::jthread m_filler;

    auto keep_full(std::stop_token token) -> void;

}; // class udp_source
