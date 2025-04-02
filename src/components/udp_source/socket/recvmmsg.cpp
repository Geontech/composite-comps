#include "helpers.hpp"
#include "recvmmsg.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <format>
#include <netinet/in.h>
#include <net/if.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#include <iostream>

namespace udp {

recvmmsg::recvmmsg(const config& config) :
  m_queue(std::make_unique<queue_t>(32768)) {
    m_id = config.id;

    // Create socket
    m_socket = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (m_socket < 0) {
        throw std::runtime_error(std::format("failed to create socket: {}", std::string{strerror(errno)}));
    }

    // Bind the socket
    auto bind_addr = htonl(inet_addr(config.ip_addr.data()));
    auto is_multicast = net::is_ipv4_multicast(config.ip_addr);
    if (is_multicast) {
        bind_addr = INADDR_ANY;
    }
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = ntohl(bind_addr);
    addr.sin_port = htons(config.port);
    if (::bind(m_socket, (const struct sockaddr*)&addr, sizeof(addr)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to bind socket: {}", std::string{strerror(errno)}));
    }

    if (is_multicast) {
        // Join multicast group
        auto ip_mreq = net::create_ip_mreq(m_socket, config.interface, config.ip_addr);
        if (::setsockopt(m_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &ip_mreq, sizeof(ip_mreq)) < 0) {
            ::close(m_socket);
            throw std::runtime_error(std::format("failed to join multicast group: {}", std::string{strerror(errno)}));
        }
    }

    // Set receive buffer size
    if (config.recv_buf_size > 0) {
        if (::setsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, (char*)&config.recv_buf_size, sizeof(config.recv_buf_size)) < 0) {
            ::close(m_socket);
            throw std::runtime_error(std::format("failed to set receive buffer size: {}", std::string{strerror(errno)}));
        }
        int optval;
        socklen_t optlen = sizeof(optval);
        if (getsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, &optval, &optlen) != -1) {
            std::cout << std::format("set socket receive buffer size to {}\n", optval);
        }
    }
}

recvmmsg::~recvmmsg() {
    m_recv_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
    m_queue.reset();
    if (m_socket != -1) {
        ::close(m_socket); 
    }
}

auto recvmmsg::start_recv() -> void {
    m_recv_thread = std::jthread(&recvmmsg::receive, this);
    pthread_setname_np(m_recv_thread.native_handle(), std::format("{}:packet_mmap", m_id).c_str());
}

auto recvmmsg::stop_recv() -> void {
    m_recv_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
}

auto recvmmsg::get_data(buffer_ptr_t& data) -> bool {
    return m_queue->try_dequeue(data);
}

auto recvmmsg::get_stats() -> statistics {
    auto stats = statistics{};
    auto tp_stats = tpacket_stats{};
    socklen_t len = sizeof(tp_stats);
    if (getsockopt(m_socket, SOL_PACKET, PACKET_STATISTICS, &tp_stats, &len) == 0) {
        stats.pkts_recvd_kernel = tp_stats.tp_packets;
        stats.pkts_dropped_kernel = tp_stats.tp_drops;
    }
    stats.pkts_recvd_user = m_pkts_recvd.exchange(0);
    return stats;
}

auto recvmmsg::receive(std::stop_token token) -> void {
    auto empty_blocks = uint64_t{};
    auto ready_blocks = uint64_t{};
    auto total_delay = uint64_t{};
    std::size_t frame_idx = 0;
    auto msg_size = 1080;
    std::array<struct mmsghdr, 128> msgs;
    std::array<struct iovec, 128> iovecs;
    std::array<buffer_ptr_t, 128> buffers;

    while (!token.stop_requested()) {
        for (auto i=0u; i<msgs.size(); ++i) {
            // Create a pmr vector for the iovec
            buffers.at(i) = std::make_shared<buffer_t>(msg_size, 0, &m_pool_resource);
            iovecs.at(i).iov_base = buffers.at(i)->data();
            iovecs.at(i).iov_len = msg_size;
            msgs.at(i).msg_hdr.msg_iov = &iovecs.at(i);
            msgs.at(i).msg_hdr.msg_iovlen = 1;
        }

        // Call recvmmsg
        struct timespec ts{.tv_sec=0, .tv_nsec=1000};
        auto recvd = ::recvmmsg(m_socket, msgs.data(), msgs.size(), 0, &ts);
                    

        // Place onto queue
        for (auto& buffer : buffers) {
            while (!m_queue->try_enqueue(std::move(buffer))) {
                std::this_thread::yield();
            }
        }
    }

    // std::cout << std::format(
    //     "-----------------------\nTOTAL LOOPS: {}\nPOLL READY: {}\nREADY BLOCKS: {}\nEMPTY BLOCKS: {}\n-----------------------\n",
    //     total_loops, poll_ready, ((float)ready_blocks/total_loops)*100, ((float)empty_blocks/total_loops)*100
    // );
}

} // namespace udp