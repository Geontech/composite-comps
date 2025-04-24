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

namespace udp {

dpdk_udp::dpdk_udp(const config& config) :
  interface(config.logger),
  m_frame_size(std::bit_ceil(config.msg_size)),
  m_frame_count(config.frame_count),
  m_resource({.frame_size=m_frame_size, .frame_count=m_frame_count, .alignment=64}) {
    // set defaults if not provided 
    m_rx_ring_size = config.rx_ring_size;
    m_num_mbufs = config.num_mbufs;
    m_mbuf_cache_size = config.mbuf_cache_size;
    m_burst_size  = config.burst_size;
    m_socket_mem = config.socket_mem;
    m_interface = config.interface;
    m_ip_addr = config.ip_addr;

    m_logger->trace("m_rx_ring_size={}, m_num_mbufs={}, m_mbuf_cache_size={}, m_burst_size={}, m_socket_mem={}",
                    m_rx_ring_size, m_num_mbufs, m_mbuf_cache_size, m_burst_size, m_socket_mem);
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
    m_logger->trace("Cleaning up DPDK...");
    rte_eth_dev_stop(m_selected_port);
    rte_eth_dev_close(m_selected_port);
    if (m_mbuf_pool){
        rte_mempool_free(m_mbuf_pool);
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
    rte_eth_stats_get(m_selected_port, &m_dpdk_stats);
    auto stats = statistics{};
    stats.pkts_recvd_user = m_pkts_recvd.exchange(0);
    stats.cycles = m_cycles.exchange(0);
    if (stats.cycles != 0 && stats.pkts_recvd_user != 0){
        stats.cycles_per_packet = stats.cycles/stats.pkts_recvd_user;
    }
    stats.pkts_recvd_nic = m_dpdk_stats.ipackets;
    stats.pkts_dropped_nic = m_dpdk_stats.imissed;
    stats.rx_nombuf = m_dpdk_stats.rx_nombuf;
    stats.total_nb_rx = m_pkts_in_burst.exchange(0);
    stats.iterations = m_iterations.exchange(0);
    if (stats.iterations != 0 && stats.total_nb_rx != 0){
        stats.avg_pkts_per_burst = stats.total_nb_rx/stats.iterations;
    }
    stats.no_queue = m_no_queue.exchange(0);
    uint32_t free_count = rte_mempool_avail_count(m_mbuf_pool);
    m_logger->trace("MBUF_POOL free mbufs: {}", free_count);

    rte_eth_stats_reset(m_selected_port);

    return stats;
}

auto dpdk_udp::receive(std::stop_token token) -> void {
    const char* pci_env = std::getenv("PCIDEVICE_INTEL_COM_INTEL_SRIOV_VFIO");
    const char* hostname = std::getenv("HOSTNAME");
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
            socket_mem << m_socket_mem;
        } else {
            socket_mem << "0";
        }
        if (i < max_node) socket_mem << ",";
    }

    std::string socket_mem_arg = "--socket-mem=" + socket_mem.str();
    std::string file_prefix_arg = "--file-prefix=" + std::string(hostname);
    uint16_t nb_ports;
    // Init EAL
    std::vector<std::string> arg_strings = {
        "udp_dpdk_src",
        socket_mem_arg,
        file_prefix_arg,
        "--huge-dir=/mnt/huge_1GB/",
        "--legacy-mem",
        "--log-level=1"
    };

    std::vector<char*> arg_ptrs;
    for (auto& s : arg_strings) {
        arg_ptrs.push_back(s.data());  // or &s[0]
    }

    int ret = rte_eal_init(arg_ptrs.size(), arg_ptrs.data());

    if (ret < 0) {
        if (rte_errno == EALREADY) {
            m_logger->trace("EAL already initialized, continuing...");
        } else {   
            rte_eal_cleanup();
            throw std::runtime_error("Failed to initialize dpdk.cpp");
        }
    }
    rte_delay_us_sleep(500000);


    // Get available DPDK ports
    nb_ports = rte_eth_dev_count_avail();
    if (nb_ports == 0) {
        rte_eal_cleanup();
    }

    // Find NUMA node for the first available port
    m_logger->trace("Detected {} DPDK ports", nb_ports);
    int numa_node = -1;
    m_selected_port = 0;

    for (uint16_t port_id = 0; port_id < nb_ports; port_id++) {
        int temp_numa_node = rte_eth_dev_socket_id(port_id);
        m_logger->trace("Port {} is on NUMA node {}", port_id, temp_numa_node);

        if (numa_node == -1 || temp_numa_node != SOCKET_ID_ANY) {
            numa_node = (temp_numa_node == SOCKET_ID_ANY) ? 0 : temp_numa_node;
            m_selected_port = port_id;
        }
    }

    if (numa_node == -1) {
        rte_eal_cleanup();
        m_logger->error("Failed to determine NUMA node for NICs!");
        rte_exit(EXIT_FAILURE, "Failed to determine NUMA node for NICs!\n");
    }

    m_logger->trace("Selected Port: {} on NUMA node: {}", m_selected_port, numa_node);
    bool port_valid = rte_eth_dev_is_valid_port(m_selected_port);
    m_logger->trace("Port {} is valid: {}", m_selected_port, port_valid);
    if (!port_valid) {
        m_logger->error("Selected port {} is not valid!", m_selected_port);
        std::abort();
    }

    // END DETECT DPDK CONFIG

    // Create mempool
    // Unique mempool name
    std::string full_hostname(hostname);

    std::string suffix = full_hostname.length() > 10
        ? full_hostname.substr(full_hostname.length() - 10)
        : full_hostname;

    std::string pool_name = "MBUF_" + suffix;

    if (pool_name.length() >= 31)
        pool_name = pool_name.substr(0, 30);
    m_logger->info("Creating mbuf pool: name=\"{}\" (len={}), num_mbufs={}, cache={}, priv=0, data_room=9000, socket={}",
        pool_name,
        pool_name.length(),
        m_num_mbufs,
        m_mbuf_cache_size,
        rte_socket_id()
    );

    rte_errno = 0;
    struct rte_mempool* new_pool = rte_pktmbuf_pool_create(
        pool_name.c_str(),
        m_num_mbufs,
        m_mbuf_cache_size,
        0,
        9000,
        rte_socket_id()
    );

    int saved_errno = rte_errno;

    if (new_pool == nullptr) {
        m_logger->error(
            "rte_pktmbuf_pool_create() returned NULL, rte_errno={} ({})",
            saved_errno,
            rte_strerror(saved_errno)
        );
        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool: %s\n", rte_strerror(saved_errno));
    }

    m_mbuf_pool = new_pool;

    m_logger->trace("Memory pool configured");
    // Configure device
    if (!m_eth_dev_configured){
        std::memset(&m_port_conf, 0, sizeof(m_port_conf));
        m_port_conf.rxmode.mq_mode = RTE_ETH_MQ_RX_NONE;
        m_port_conf.rxmode.mtu = 9000;

    }
    m_logger->trace("Device configured");
    m_port_conf.intr_conf.rxq = 0;

    static constexpr uint8_t  NUM_RX_QUEUES = 1;
    ret = rte_eth_dev_configure(m_selected_port, NUM_RX_QUEUES, 0, &m_port_conf);
    if (ret < 0) {
        m_logger->error("rte_eth_dev_configure: err={}, port={}", rte_strerror(rte_errno), m_selected_port);
        m_logger->error("rte_eth_dev_configure: err={}, port={}", rte_strerror(rte_errno), m_selected_port);
        rte_exit(EXIT_FAILURE, "rte_eth_dev_configure: err=%d, port=%u\n", ret, m_selected_port);
    }
    m_eth_dev_configured = true;
    m_logger->trace("Device configured");
    m_logger->trace("m_selected_port={}, m_rx_ring_size={}, rte_socket_id={}",
                    m_selected_port, m_rx_ring_size, rte_socket_id());


    rte_eth_dev_info dev_info;
    rte_eth_dev_info_get(m_selected_port, &dev_info);

    if (1 >= dev_info.max_rx_queues) {
        m_logger->error("Queue index {} exceeds max_rx_queues {}", 1, dev_info.max_rx_queues);
        std::abort();
    }

    if (m_rx_ring_size % dev_info.rx_desc_lim.nb_align != 0) {
        m_logger->error("RX ring size {} is not aligned to {}", m_rx_ring_size, dev_info.rx_desc_lim.nb_align);
        std::abort();
    }

    if (m_rx_ring_size < dev_info.rx_desc_lim.nb_min || m_rx_ring_size > dev_info.rx_desc_lim.nb_max) {
        m_logger->error("RX ring size {} is out of valid range [{} - {}]",
                        m_rx_ring_size, dev_info.rx_desc_lim.nb_min, dev_info.rx_desc_lim.nb_max);
        std::abort();
    }

    if (m_mbuf_pool == nullptr) {
        m_logger->error("m_mbuf_pool is null before queue setup");
        std::abort();
    }

    uint32_t avail_mbufs = rte_mempool_avail_count(m_mbuf_pool);
    if (avail_mbufs == 0) {
        m_logger->error("mbuf pool has no available buffers");
        std::abort();
    }

    m_logger->trace("Mbuf pool name={}, size={}, available={}",
                    m_mbuf_pool->name,
                    m_mbuf_pool->size,
                    avail_mbufs);



    // Setup RX queue with retry on ENOSPC and ENOENT
    for (uint16_t q = 0; q < NUM_RX_QUEUES; q++) {
        constexpr int max_attempts = 3;
        int attempt = 0;
        do {
            ret = rte_eth_rx_queue_setup(m_selected_port, q, m_rx_ring_size,
                                        rte_socket_id(), nullptr, m_mbuf_pool);
            if (ret == 0) break;

            if ((rte_errno == ENOSPC || rte_errno == ENOENT) && attempt < max_attempts - 1) {
                m_logger->warn("rte_eth_rx_queue_setup: transient failure (errno={}, msg={}), retrying {}/{}",
                            rte_errno, rte_strerror(rte_errno), attempt + 1, max_attempts);
                std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            } else {
                m_logger->error("rte_eth_rx_queue_setup: ret={}, errno={}, msg={}, port={}",
                                ret, rte_errno, rte_strerror(rte_errno), m_selected_port);
                rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup: err=%d, port=%u\n", ret, m_selected_port);
            }

            ++attempt;
        } while (attempt < max_attempts);
    // Setup RX queue with retry on ENOSPC and ENOENT
    for (uint16_t q = 0; q < NUM_RX_QUEUES; q++) {
        constexpr int max_attempts = 3;
        int attempt = 0;
        do {
            ret = rte_eth_rx_queue_setup(m_selected_port, q, m_rx_ring_size,
                                        rte_socket_id(), nullptr, m_mbuf_pool);
            if (ret == 0) break;

            if ((rte_errno == ENOSPC || rte_errno == ENOENT) && attempt < max_attempts - 1) {
                m_logger->warn("rte_eth_rx_queue_setup: transient failure (errno={}, msg={}), retrying {}/{}",
                            rte_errno, rte_strerror(rte_errno), attempt + 1, max_attempts);
                std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            } else {
                m_logger->error("rte_eth_rx_queue_setup: ret={}, errno={}, msg={}, port={}",
                                ret, rte_errno, rte_strerror(rte_errno), m_selected_port);
                rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup: err=%d, port=%u\n", ret, m_selected_port);
            }

            ++attempt;
        } while (attempt < max_attempts);
    }
    m_logger->trace("Queue configured");


    m_logger->trace("Queue configured");



    // Start the port
    ret = rte_eth_dev_start(m_selected_port);
    if (ret < 0) {
        m_logger->error("rte_eth_dev_start: err={}, port={}", ret, m_selected_port);
        rte_exit(EXIT_FAILURE, "rte_eth_dev_start: err=%d, port=%u\n", ret, m_selected_port);
    }
    m_logger->trace("Port {} started", m_selected_port);

    // START ENABLE MULTICAST FILTER
    in_addr ipv4_addr{};
    auto ip_addr = m_ip_addr;
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

    ret = rte_eth_dev_set_mc_addr_list(m_selected_port, &mc_mac, 1);
    std::vector<uint32_t> ptypes = {RTE_PTYPE_L2_ETHER, RTE_PTYPE_L3_IPV4, RTE_PTYPE_L4_UDP};
    rte_eth_dev_set_ptypes(m_selected_port, RTE_PTYPE_UNKNOWN, ptypes.data(), ptypes.size());
    // END ENABLE MULTICAST FILTER

    m_logger->trace("Starting Rx loop...");

    // IGMP JOIN 
    // Create a normal UDP socket
    m_socket = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (m_socket < 0) {
        throw std::runtime_error(std::format("failed to create socket: {}", std::string{strerror(errno)}));
    }

    int reuse = 1;
    if (setsockopt(m_socket, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to set SO_REUSEADDR: {}", std::string{strerror(errno)}));
    }

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(12345);
    addr.sin_addr.s_addr = INADDR_ANY;
    if (bind(m_socket, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to bind socket: {}", std::string{strerror(errno)}));
    }

    // Join the multicast group
    ip_mreqn mreq{};
    mreq.imr_multiaddr.s_addr = inet_addr(std::string(ip_addr).c_str());
    mreq.imr_address.s_addr = INADDR_ANY;
    mreq.imr_ifindex = if_nametoindex(m_interface.c_str());
    if (mreq.imr_ifindex == 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to get ifindex for interface {}: {}", m_interface, std::string{strerror(errno)}));
    }

    if (setsockopt(m_socket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) < 0) {
        ::close(m_socket);
        throw std::runtime_error(std::format("failed to join multicast group: {}", std::string{strerror(errno)}));
    }

    m_logger->trace("Joined multicast group {} on interface {}", ip_addr, m_interface);

    uint8_t queue_id = 0;
    rte_mbuf* pkts[m_burst_size];
    uint16_t nb_rx{};
    uint16_t offset{};
    uint16_t pkt_len{};
    uint16_t payload_len{};
    auto allocator = std::pmr::polymorphic_allocator<std::uint8_t>(&m_resource);
    static float avg_fill = m_burst_size;

    while (!token.stop_requested()) {
        uint64_t tsc_start = rte_get_tsc_cycles();
        nb_rx = rte_eth_rx_burst(m_selected_port, queue_id, pkts, m_burst_size);
        avg_fill = 0.9 * avg_fill + 0.1 * nb_rx;
        if (nb_rx > 0){
            for (uint16_t i = 0; i < nb_rx; i++) {
                m_pkts_recvd.fetch_add(1, std::memory_order_relaxed);
                pkt_len = rte_pktmbuf_pkt_len(pkts[i]);

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
                rte_memcpy(vec->data(), payload, payload_len);

                m_queue.push(std::move(vec));
            }
            rte_pktmbuf_free_bulk(pkts, nb_rx);
            uint64_t tsc_end = rte_get_tsc_cycles();
            m_cycles.fetch_add(tsc_end-tsc_start, std::memory_order_relaxed);
            m_pkts_in_burst.fetch_add(nb_rx, std::memory_order_relaxed);
            m_iterations.fetch_add(1, std::memory_order_relaxed);
        } 
        else {
            int sleep_us = std::min(100 + (int)((1.0 - (avg_fill / m_burst_size)) * 400), 500);
            rte_delay_us_block(sleep_us);
        }
    }
}


} //namespace udp