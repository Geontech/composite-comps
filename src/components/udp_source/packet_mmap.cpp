#include "helpers.hpp"
#include "packet_mmap.hpp"
#include "overlay.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <format>
#include <immintrin.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <poll.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#include <iostream>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <sys/ioctl.h>


#define FRAME_SIZE 2048
#define BLOCK_SIZE (1 << 20)
#define BLOCK_NR 256
#define FRAME_NR (BLOCK_SIZE * BLOCK_NR) / FRAME_SIZE
#define RETIRE_TOV 1 // ms

namespace udp {

packet_mmap::packet_mmap(std::string_view interface, std::string_view ip_addr, uint16_t port) :
  m_upstream_alloc(64),
  m_pool_resource({}, &m_upstream_alloc),
  m_queue(std::make_unique<moodycamel::ReaderWriterQueue<std::shared_ptr<std::pmr::vector<uint8_t>>>>(32768)) {
    // Create socket
    m_socket = ::socket(AF_PACKET, SOCK_DGRAM, htons(ETH_P_IP));
    if (m_socket < 0) {
        throw std::runtime_error(std::format("failed to create socket: {}", std::string{strerror(errno)}));
    }

    // Set packet version to v3
    // int version = TPACKET_V3;
    int version = TPACKET_V2;
    if (::setsockopt(m_socket, SOL_PACKET, PACKET_VERSION, &version, sizeof(version)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to set packet version to v3: {}", std::string{strerror(errno)}));
    }

    // Bind the socket
    auto sll = sockaddr_ll{};
    sll.sll_family = AF_PACKET;
    sll.sll_ifindex = if_nametoindex(interface.data());
    sll.sll_protocol = htons(ETH_P_IP);
    if (sll.sll_ifindex == 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format(
            "failed to get ifindex for interface {}: {}",
            interface.data(), std::string{strerror(errno)}
        ));
    }
    if (::bind(m_socket, reinterpret_cast<struct sockaddr*>(&sll), sizeof(sll)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to bind socket: {}", std::string{strerror(errno)}));
    }

    if (net::is_ipv4_multicast(ip_addr)) {
        // Enable multicast mode on the interface
        auto pkt_mreq = net::create_packet_mreq(interface, ip_addr);
        if (::setsockopt(m_socket, SOL_PACKET, PACKET_ADD_MEMBERSHIP, &pkt_mreq, sizeof(pkt_mreq)) < 0) {
            ::close(m_socket);
            throw std::runtime_error(std::format("failed to add multicast membership: {}", std::string{strerror(errno)}));
        }

        // Join multicast group
        // Create join socket
        m_join_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (m_join_socket < 0) {
            throw std::runtime_error(std::format("failed to create join socket: {}", std::string{strerror(errno)}));
        }
        auto ip_mreq = net::create_ip_mreq(m_join_socket, interface, ip_addr);
        if (::setsockopt(m_join_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &ip_mreq, sizeof(ip_mreq)) < 0) {
            ::close(m_join_socket);
            ::close(m_socket);
            throw std::runtime_error(std::format("failed to join multicast group: {}", std::string{strerror(errno)}));
        }
        std::cout << "MULTICAST JOINED\n";
    }

    // Set socket ring properties
    // auto req3 = tpacket_req3{};
    // req3.tp_block_size = BLOCK_SIZE;
    // req3.tp_block_nr = BLOCK_NR;
    // req3.tp_frame_size = FRAME_SIZE;
    // req3.tp_frame_nr = (BLOCK_SIZE * BLOCK_NR) / FRAME_SIZE;
    // req3.tp_retire_blk_tov = RETIRE_TOV;
    struct tpacket_req req = {};
    req.tp_block_size = BLOCK_SIZE;   // e.g., 1 << 16 for 64KB blocks
    req.tp_block_nr   = BLOCK_NR;     // total number of blocks in the ring
    req.tp_frame_size = FRAME_SIZE;   // typically 2048 for MTU-sized frames
    req.tp_frame_nr   = FRAME_NR;

    if (::setsockopt(m_socket, SOL_PACKET, PACKET_RX_RING, &req, sizeof(req)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to set packet_rx_ring: {}", std::string{strerror(errno)}));
    }

    // Memory-map the ring buffer
    auto ring_size = req.tp_block_size * req.tp_block_nr;
    m_ring = mmap(0, ring_size,  PROT_READ | PROT_WRITE, MAP_SHARED, m_socket, 0);
    if (m_ring == MAP_FAILED) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to create mmap buffer: {}", std::string{strerror(errno)}));
    }

    int m_recv_buf_size = 256 * 1024 * 1024; // Example: 256 MB.  Adjust this!

    // Set receive buffer size
    // if (m_recv_buf_size > 0) {
        // logger()->trace("setting socket receive buffer size to {}", m_recv_buf_size);
        setsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, (char*)&m_recv_buf_size, sizeof(m_recv_buf_size));
        int optval;
        socklen_t optlen = sizeof(optval);
        if (getsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, &optval, &optlen) != -1) {
            std::cout << std::format("set socket receive buffer size to {}\n", optval);
        }
    // }

    // Request Transparent Huge Pages
    ::madvise(m_ring, ring_size, MADV_HUGEPAGE);
}

packet_mmap::~packet_mmap() {
    m_recv_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
    m_queue.reset();
    if (m_join_socket != -1) {
        ::close(m_join_socket); 
    }
    if (m_socket != -1) {
        ::close(m_socket); 
    }
    ::munmap(m_ring, BLOCK_SIZE * BLOCK_NR);
}

auto packet_mmap::start() -> void {
    m_recv_thread = std::jthread(&packet_mmap::receive, this);
    pthread_setname_np(m_recv_thread.native_handle(), "packet_mmap");
    m_stat_thread = std::jthread([this](std::stop_token stoken) {
        while (!stoken.stop_requested()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            log_stats();
        }
    });
}

auto packet_mmap::stop() -> void {
    m_recv_thread.request_stop();
    m_stat_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
    if (m_stat_thread.joinable()) {
        m_stat_thread.join();
    }
}

auto packet_mmap::get_data(std::shared_ptr<std::pmr::vector<uint8_t>>& data) -> bool {
    return m_queue->try_dequeue(data);
}

auto packet_mmap::receive(std::stop_token token) -> void {
    auto block_idx = uint64_t{};
    auto total_loops = uint64_t{};
    auto poll_ready = uint64_t{};
    auto empty_blocks = uint64_t{};
    auto ready_blocks = uint64_t{};
    auto total_delay = uint64_t{};
    uint32_t sleep_ns = 1;
    constexpr uint32_t max_sleep_ns = 1000000;
    const size_t frame_count = BLOCK_NR * (BLOCK_SIZE / FRAME_SIZE);
    std::size_t frame_idx = 0;

    while (!token.stop_requested()) {
        ++total_loops;

        // Get pointer to current frame
        auto* hdr = (struct tpacket2_hdr*)((uint8_t*)m_ring + (frame_idx * FRAME_SIZE));

        // Work through all the ready frames
        if (hdr->tp_status & TP_STATUS_USER) [[likely]] {
            ++ready_blocks;

            // Validate protocol
            auto ip_hdr = (struct iphdr*)((uint8_t*)hdr + hdr->tp_mac);
            if (ip_hdr->protocol == IPPROTO_UDP) [[likely]] {
                // Extract UDP payload
                auto* udp_hdr = (struct udphdr*)((uint8_t*)(ip_hdr) + ip_hdr->ihl * 4);
                auto* payload = (uint8_t*)(udp_hdr) + sizeof(struct udphdr);
                size_t payload_len = ntohs(udp_hdr->len) - sizeof(struct udphdr);

                // Create a pmr vector and copy udp payload into it
                auto vec = std::make_shared<std::pmr::vector<uint8_t>>(payload_len, 0, &m_pool_resource);
                std::memcpy(vec->data(), payload, payload_len);

                // Place onto queue
                while (!m_queue->try_enqueue(std::move(vec))) {
                    std::this_thread::yield();
                }
            }

            // Release the frame
            hdr->tp_status = TP_STATUS_KERNEL;
            frame_idx = (frame_idx + 1) % frame_count;
            hdr = (struct tpacket2_hdr*)((uint8_t*)m_ring + (frame_idx * FRAME_SIZE));
        } else {
            ++empty_blocks;
            if (((float)ready_blocks / total_loops) > 0.5f) {
                std::this_thread::yield();
            } else {
                struct timespec ts{.tv_sec=0, .tv_nsec=1};
                nanosleep(&ts, nullptr);
            }
        }
    }

    std::cout << std::format(
        "-----------------------\nTOTAL LOOPS: {}\nPOLL READY: {}\nREADY BLOCKS: {}\nEMPTY BLOCKS: {}\n-----------------------\n",
        total_loops, poll_ready, ((float)ready_blocks/total_loops)*100, ((float)empty_blocks/total_loops)*100
    );
}

void packet_mmap::log_stats() {
    tpacket_stats stats{};
    socklen_t len = sizeof(stats);
    if (getsockopt(m_socket, SOL_PACKET, PACKET_STATISTICS, &stats, &len) == 0) {
        std::cout << "[packet_stats] received=" << stats.tp_packets << " dropped=" << stats.tp_drops << "\n";
    } else {
        perror("getsockopt(PACKET_STATISTICS)");
    }
}

} // namespace udp