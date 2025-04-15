#include "helpers.hpp"
#include "dpdk.hpp"
#include "overlay.hpp"
#include "pmr/ring_resource.hpp"

#include <arpa/inet.h>
#include <array>
#include <cstring>
#include <fcntl.h>
#include <linux/if_ether.h>
#include <netinet/in.h>
#include <net/if.h>
#include <mutex>
#include <poll.h>
#include <queue>
#include <ranges>
#include <source_location>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include <stdexcept>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>


#include <iostream>
#include <cstdint>
#include <signal.h>
#include <atomic>
#include <csignal>
#include <pthread.h>
#include <vector>
#include <complex>
#include <future>
#include <sys/time.h>

extern "C" {
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_ring.h>
#include <rte_udp.h>
#include <rte_errno.h>
#include <rte_malloc.h>
#include <rte_thread.h>
#include <rte_lcore.h>
}

static constexpr uint8_t  NUM_RX_QUEUES = 1; //Need to update receive_loop() if changing
static constexpr uint8_t NUM_TX_QUEUES = 0;
static constexpr uint16_t RX_RING_SIZE = 4096; //2048 drops some at startup
static constexpr uint16_t RING_SIZE = 32768; // Was 32768
static constexpr uint16_t NUM_MBUFS    = (32768 * NUM_RX_QUEUES);
static constexpr uint16_t MBUF_CACHE_SIZE = 512; 
static constexpr uint16_t BURST_SIZE  = 64;
#define JUMBO_FRAME_SIZE 9000

struct DpdkConfig {
    uint16_t selected_port;
    int numa_node;
    unsigned int rx_core;
};

static DpdkConfig dpdk_config;
struct rte_eth_conf port_conf;
struct rte_ring* packet_ring;
struct rte_mempool* mbuf_pool;
bool eth_dev_configured = false;
struct rte_eth_dev_info dev_info;
struct rte_eth_stats dpdk_stats;

namespace udp {

dpdk_udp::dpdk_udp(const config& config) :
  interface(config.logger),
  m_frame_size(std::bit_ceil(config.msg_size)),
  m_frame_count(config.frame_count),
  m_resource({.frame_size=m_frame_size, .frame_count=m_frame_count, .alignment=64}) {
    const char* pci_env = std::getenv("PCIDEVICE_INTEL_COM_INTEL_SRIOV_VFIO");
    if (!pci_env) {
        throw std::runtime_error("Environment variable PCIDEVICE_INTEL_COM_INTEL_SRIOV_VFIO is not set!");
    }
    std::string pci_addr(pci_env);
    m_logger->trace("Using PCI device: {}", pci_addr);

    // Detect NUMA node of the device
    std::string node_path = "/sys/bus/pci/devices/" + pci_addr + "/numa_node";
    std::ifstream node_file(node_path);
    int dev_numa = -1;
    if (node_file.is_open()) {
        node_file >> dev_numa;
        node_file.close();
    }

    if (dev_numa < 0) {
        m_logger->warn("Could not determine NUMA node for device. Falling back to single-node config.");
        dev_numa = 0;
    }
    m_logger->trace("Device is on NUMA node: {}",dev_numa);

    // Discover all NUMA nodes
    std::vector<int> nodes;
    for (const auto& entry : std::filesystem::directory_iterator("/sys/devices/system/node")) {
        std::string name = entry.path().filename();
        if (name.rfind("node", 0) == 0) {
            try {
                int id = std::stoi(name.substr(4));
                nodes.push_back(id);
            } catch (...) {
                continue;
            }
        }
    }
    std::sort(nodes.begin(), nodes.end());

    int max_node = nodes.empty() ? 0 : nodes.back();

    // Build the --socket-mem string
    std::ostringstream socket_mem;
    for (int i = 0; i <= max_node; ++i) {
        if (i == dev_numa) {
            socket_mem << "4096";
        } else {
            socket_mem << "0";
        }
        if (i < max_node) socket_mem << ",";
    }

    std::string socket_mem_arg = "--socket-mem=" + socket_mem.str();
    
    uint16_t nb_ports;
    // Init EAL
    std::vector<std::string> arg_strings = {
        "udp_dpdk_src",
        socket_mem_arg,
        "--huge-dir=/mnt/huge_1GB/",
        "--legacy-mem",
        "--log-level=1"
    };
    
    std::vector<char*> arg_ptrs;
    for (auto& s : arg_strings) {
        arg_ptrs.push_back(s.data());  // or &s[0]
    }
    
    int ret = rte_eal_init(arg_ptrs.size(), arg_ptrs.data());
    
    // int ret = rte_eal_init(5, dpdk_app_argv);
    if (ret < 0) {
        if (rte_errno == EALREADY) {
            m_logger->trace("EAL already initialized, continuing...");
        } else {   
            rte_eal_cleanup();
            throw std::runtime_error("Failed to initialize dpdk.cpp");
        }
    }

    // Get available DPDK ports
    nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) {
        // rte_exit(EXIT_FAILURE, "No DPDK-enabled NICs found!\n");
        rte_eal_cleanup();
    }

    

    // Find NUMA node for the first available port
    m_logger->trace("Detected {} DPDK ports", nb_ports);
    dpdk_config.numa_node = -1;
    dpdk_config.selected_port = 0;

    for (uint16_t port_id = 0; port_id < nb_ports; port_id++) {
        int numa_node = rte_eth_dev_socket_id(port_id);
        m_logger->trace("Port {} is on NUMA node {}", port_id, numa_node);

        if (dpdk_config.numa_node == -1 || numa_node != SOCKET_ID_ANY) {
            dpdk_config.numa_node = (numa_node == SOCKET_ID_ANY) ? 0 : numa_node;
            dpdk_config.selected_port = port_id;
        }
    }

    if (dpdk_config.numa_node == -1) {
        rte_eal_cleanup();
        rte_exit(EXIT_FAILURE, "Failed to determine NUMA node for NICs!\n");
    }

    m_logger->trace("Selected Port: {} on NUMA node: {}", dpdk_config.selected_port, dpdk_config.numa_node);


    // Find best CPU cores from this NUMA node
    dpdk_config.rx_core = RTE_MAX_LCORE;
    unsigned int lcore_id;

    RTE_LCORE_FOREACH_WORKER(lcore_id) {
        if (rte_lcore_to_socket_id(lcore_id) == dpdk_config.numa_node) {
            if (dpdk_config.rx_core == RTE_MAX_LCORE) {
                dpdk_config.rx_core = lcore_id;
            } 
        }
    }

    // END DETECT DPDK CONFIG

    // Create mempool
    if (mbuf_pool == nullptr){
        mbuf_pool = rte_pktmbuf_pool_create(
            "MBUF_POOL",
            NUM_MBUFS,
            MBUF_CACHE_SIZE,
            0,
            9000,
            rte_socket_id()
        );
        if (mbuf_pool == nullptr) {
            rte_exit(EXIT_FAILURE, "Cannot create mbuf pool: %s\n", rte_strerror(rte_errno));
        }
    }
    // Configure device
    if (!eth_dev_configured){
        std::memset(&port_conf, 0, sizeof(port_conf));
        port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_NONE;
        port_conf.rxmode.mtu = JUMBO_FRAME_SIZE;

    }
    ret = rte_eth_dev_configure(dpdk_config.selected_port, NUM_RX_QUEUES, NUM_TX_QUEUES, &port_conf);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "rte_eth_dev_configure: err=%d, port=%u\n", ret, dpdk_config.selected_port);
    }
    eth_dev_configured = true;

    // Setup RX queue
    for (uint16_t q = 0; q < NUM_RX_QUEUES; q++){
        ret = rte_eth_rx_queue_setup(dpdk_config.selected_port, q, RX_RING_SIZE, 
                                    rte_socket_id(), nullptr, mbuf_pool);
        if (ret < 0) {
            rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup: err=%d, port=%u\n", ret, dpdk_config.selected_port);
        }
    }

    // Start the port
    ret = rte_eth_dev_start(dpdk_config.selected_port);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "rte_eth_dev_start: err=%d, port=%u\n", ret, dpdk_config.selected_port);
    }
    m_logger->trace("Port {} started", dpdk_config.selected_port);

    // START ENABLE MULTICAST FILTER
    in_addr ipv4_addr{};
    auto ip_addr = config.ip_addr;
    if (inet_pton(AF_INET, std::string(ip_addr).c_str(), &ipv4_addr) != 1) {
        m_logger->error("Not a valid IP");

    }

    uint32_t ip = ntohl(ipv4_addr.s_addr);  // convert to host byte order

    rte_ether_addr mc_mac{};
    mc_mac.addr_bytes[0] = 0x01;
    mc_mac.addr_bytes[1] = 0x00;
    mc_mac.addr_bytes[2] = 0x5e;
    mc_mac.addr_bytes[3] = static_cast<uint8_t>((ip >> 16) & 0x7F);  // only lower 7 bits
    mc_mac.addr_bytes[4] = static_cast<uint8_t>((ip >> 8) & 0xFF);
    mc_mac.addr_bytes[5] = static_cast<uint8_t>(ip & 0xFF);

    ret = rte_eth_dev_set_mc_addr_list(dpdk_config.selected_port, &mc_mac, 1);
    std::vector<uint32_t> ptypes = {RTE_PTYPE_L2_ETHER, RTE_PTYPE_L3_IPV4, RTE_PTYPE_L4_UDP};
    rte_eth_dev_set_ptypes(dpdk_config.selected_port, RTE_PTYPE_UNKNOWN, ptypes.data(), ptypes.size());
    // END ENABLE MULTICAST FILTER

    if (packet_ring == NULL){
        packet_ring = rte_ring_create("packet_ring", RING_SIZE, rte_socket_id(), RING_F_SP_ENQ | RING_F_SC_DEQ);
        if (!packet_ring) {
            rte_exit(EXIT_FAILURE, "Failed to create ring buffer\n");
        }
    }

    m_logger->trace("Starting Rx loop...");

    // IGMP JOIN 
    // Create a normal UDP socket
    m_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (m_socket < 0) {
        throw std::runtime_error(std::format("failed to create socket: {}", std::string{strerror(errno)}));
    }

    // Enable SO_REUSEADDR so multiple sockets can bind to the same address (useful for multicast)
    int reuse = 1;
    if (setsockopt(m_socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to set SO_REUSEADDR: {}", std::string{strerror(errno)}));
    }

    // Bind to the multicast port
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(12345); // Change to your multicast port
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(m_socket, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to bind socket: {}", std::string{strerror(errno)}));
    }

    // Join the multicast group
    ip_mreqn mreq{};
    mreq.imr_multiaddr.s_addr = inet_addr(std::string(ip_addr).c_str());
    mreq.imr_address.s_addr = INADDR_ANY;
    mreq.imr_ifindex = if_nametoindex(config.interface.c_str());
    if (mreq.imr_ifindex == 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to get ifindex for interface {}: {}", config.interface, std::string{strerror(errno)}));
    }

    if (setsockopt(m_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to join multicast group: {}", std::string{strerror(errno)}));
    }

    // Done — IGMP join sent by kernel
    m_logger->trace("Joined multicast group {} on interface {}", ip_addr, config.interface);
}

dpdk_udp::~dpdk_udp() {
    m_recv_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
    m_queue.clear();
    if (m_join_socket != -1) {
        ::close(m_join_socket); 
    }
    if (m_socket != -1) {
        ::close(m_socket); 
    }
    // TODO DPDK SHUTDOWN STUFF
    m_logger->trace("Cleaning up DPDK...");
    rte_eth_dev_stop(dpdk_config.selected_port);
    rte_eth_dev_close(dpdk_config.selected_port);
    if (mbuf_pool){
        rte_mempool_free(mbuf_pool);
    }
    rte_eal_cleanup();
    m_logger->trace("Clean up complete");
}

auto dpdk_udp::start_recv() -> void {
    m_recv_thread = std::jthread(&dpdk_udp::receive, this);
    pthread_setname_np(m_recv_thread.native_handle(), "dpdk_udp");
}

auto dpdk_udp::stop_recv() -> void {
    m_recv_thread.request_stop();
    if (m_recv_thread.joinable()) {
        m_recv_thread.join();
    }
    m_queue.clear();
}

auto dpdk_udp::get_data(std::shared_ptr<buffer_t>& data) -> bool {
    if (auto pop_res = m_queue.pop()) {
        data.reset(pop_res.release());
        return true;
    }
    return false;
}

auto dpdk_udp::get_stats() -> statistics {
    rte_eth_stats_get(dpdk_config.selected_port, &dpdk_stats);
    auto stats = statistics{};
    stats.pkts_recvd_user = m_pkts_recvd.exchange(0);
    stats.cycles = m_cycles.exchange(0);
    if (stats.cycles != 0 && stats.pkts_recvd_user != 0){
        stats.cycles_per_packet = stats.cycles/stats.pkts_recvd_user;
    }
    stats.pkts_recvd_nic = dpdk_stats.ipackets;
    stats.pkts_dropped_nic = dpdk_stats.imissed;
    stats.rx_nombuf = dpdk_stats.rx_nombuf;
    stats.total_nb_rx = m_pkts_in_burst.exchange(0);
    stats.iterations = m_iterations.exchange(0);
    if (stats.iterations != 0 && stats.total_nb_rx != 0){
        stats.avg_pkts_per_burst = stats.total_nb_rx/stats.iterations;
    }
    stats.no_queue = m_no_queue.exchange(0);

    rte_eth_stats_reset(dpdk_config.selected_port);

    return stats;
}

auto dpdk_udp::receive(std::stop_token token) -> void {

    // rte_cpuset_t cpuset;
    // CPU_ZERO(&cpuset);
    // CPU_SET(dpdk_config.rx_core,&cpuset);
    // if (rte_thread_set_affinity(&cpuset) < 0) {
    //     m_logger->error("Error setting RX thread affinity to core ",dpdk_config.rx_core);
    // }
    // m_logger->trace("RX thread started on core {} for port {}", dpdk_config.rx_core, dpdk_config.selected_port);

    uint8_t queue_id = 0;
    rte_mbuf* pkts[BURST_SIZE];
    rte_mbuf* valid_pkts[BURST_SIZE];
    uint16_t nb_rx{};
    uint16_t offset{};
    uint16_t pkt_len{};
    uint16_t payload_len{};
    auto allocator = std::pmr::polymorphic_allocator<std::uint8_t>(&m_resource);


    while (!token.stop_requested()) {
        uint64_t tsc_start = rte_get_tsc_cycles();
        nb_rx = rte_eth_rx_burst(dpdk_config.selected_port, queue_id, pkts, BURST_SIZE);
        if (nb_rx > 0){
            for (uint16_t i = 0; i < nb_rx; i++) {
                m_pkts_recvd.fetch_add(1, std::memory_order_relaxed);
                pkt_len = rte_pktmbuf_pkt_len(pkts[i]);


                if (pkt_len < 1000) { //Filter out ARPs and IGMP Messages
                    continue;
                }
                
                if (offset > 0) [[likely]]{ // offset and payload_len has been determined
                    uint8_t* pkt_data = rte_pktmbuf_mtod(pkts[i], uint8_t*);
                    auto payload = pkt_data + offset;
                    auto vec = std::make_unique<buffer_t>(allocator);
                    vec->resize(payload_len);
                    std::memcpy(vec->data(), payload, payload_len);

                    m_queue.push(std::move(vec));

                } else { //Determine Packet offset
                    if (pkt_len < sizeof(struct rte_ether_hdr)) {
                        continue;
                    }
                    uint8_t* pkt_data = rte_pktmbuf_mtod(pkts[i], uint8_t*);
                
                    struct rte_ether_hdr* eth_hdr = (struct rte_ether_hdr*)pkt_data;
                    if (rte_is_broadcast_ether_addr(&eth_hdr->dst_addr)) {
                        continue;
                    }
                
                    size_t l3_offset = sizeof(struct rte_ether_hdr);
                    uint16_t ether_type = rte_be_to_cpu_16(eth_hdr->ether_type);
                
                    while (ether_type == RTE_ETHER_TYPE_VLAN) {
                        if (pkt_len < l3_offset + sizeof(struct rte_vlan_hdr)) continue;;
                        struct rte_vlan_hdr* vlan_hdr = (struct rte_vlan_hdr*)(pkt_data + l3_offset);
                        ether_type = rte_be_to_cpu_16(vlan_hdr->eth_proto);
                        l3_offset += sizeof(struct rte_vlan_hdr);
                    }
                
                    if (ether_type != RTE_ETHER_TYPE_IPV4) {
                        continue;
                    }
                
                    if (pkt_len < l3_offset + sizeof(struct rte_ipv4_hdr)) continue;;
                
                    struct rte_ipv4_hdr* ip_hdr = (struct rte_ipv4_hdr*)(pkt_data + l3_offset);
                    if (ip_hdr->next_proto_id != IPPROTO_UDP) {
                        continue;
                    }
                
                    size_t ip_header_len = rte_ipv4_hdr_len(ip_hdr);
                    size_t l4_offset = l3_offset + ip_header_len;
                
                    if (pkt_len < l4_offset + sizeof(struct rte_udp_hdr)) continue;
                
                    offset = l4_offset + sizeof(struct rte_udp_hdr); 
                    if (pkt_len <= offset) continue;
                    payload_len = pkt_len - offset;
                    
                    auto payload = pkt_data + offset;
                    auto vec = std::make_unique<buffer_t>(allocator);
                    vec->resize(payload_len);
                    std::memcpy(vec->data(), payload, payload_len);

                    m_queue.push(std::move(vec));
                }

            }
            rte_pktmbuf_free_bulk(pkts, nb_rx);
            uint64_t tsc_end = rte_get_tsc_cycles();
            m_cycles.fetch_add(tsc_end-tsc_start, std::memory_order_relaxed);
            m_pkts_in_burst.fetch_add(nb_rx, std::memory_order_relaxed);
            m_iterations.fetch_add(1, std::memory_order_relaxed);
        } 
        else {
            struct timespec ts{.tv_sec=0, .tv_nsec=1};
            nanosleep(&ts, nullptr);
        }
    }
}


} //namespace udp