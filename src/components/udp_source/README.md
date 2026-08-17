# udp_source

High-performance UDP packet reception component optimized for continuous high-throughput data streams.

## Overview

The `udp_source` component receives UDP packets and outputs them as immutable byte buffers.
It is designed for high-bandwidth applications such as Software Defined Radio (SDR) data ingestion,
supporting demanding data rates with minimal latency.

Key features:
- Multiple socket backend implementations for different performance characteristics
- Zero-copy buffer management using external buffers with custom deleters
- Multicast and IGMP support
- Configurable batching for throughput optimization
- Automatic packet size discovery (recvmmsg variant)
- Frame pool with pre-allocated aligned memory (recvmmsg variant)

## Socket Backends

The component supports three socket backend implementations, selectable via the `socket_type` property:

### recvmmsg (default)

Standard Linux socket API with batched message reception.

**When to use:**
- General-purpose UDP reception
- Good balance of performance and compatibility
- Works on any Linux system (kernel 2.6.33+)

**Characteristics:**
- Uses `recvmmsg()` syscall for batch reception
- Supports automatic packet size discovery
- Efficient for moderate to high data rates
- No special kernel configuration required

### packet_mmap

AF_PACKET socket with memory-mapped ring buffer (TPACKET_V2).

**When to use:**
- Maximum performance on standard Linux
- Need to minimize kernel-to-userspace copies
- Processing multicast at very high rates

**Characteristics:**
- Zero-copy ring buffer in shared memory
- Lower CPU overhead than recvmmsg
- Requires explicit `msg_size` configuration (no auto-discovery)
- Bypasses socket buffer, reads directly from network interface
- Provides kernel packet statistics (drops, etc.)

### dpdk (optional)

Intel DPDK (Data Plane Development Kit) userspace networking.

**When to use:**
- Maximum performance with DPDK-compatible NICs
- Need hardware offload capabilities
- Dedicated network interface for data reception

**Characteristics:**
- Kernel-bypass networking
- Poll-mode drivers (PMD)
- Requires DPDK-compatible NIC and configuration
- Must be compiled with `-DCOMPOSITE_HAS_DPDK=ON`

## Configuration Properties

### Basic Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `socket_type` | string | `"recvmmsg"` | Socket backend: `"recvmmsg"`, `"packet_mmap"`, or `"dpdk"` |
| `interface` | string | - | Network interface name (e.g., `"eth0"`, `"ens1f0"`) |
| `ip_addr` | string | - | IP address to bind to (for multicast, use the multicast group address) |
| `port` | uint16 | - | UDP port to listen on |
| `recv_buf_size` | uint32 | 0 | Socket receive buffer size in bytes (0 = system default) |
| `num_msgs` | uint32 | 128 | Batch size (number of messages to receive per syscall) |
| `frame_count` | uint32 | 8192 | Number of pre-allocated frame buffers in the pool |
| `autodiscovery_timeout` | uint32 | 10 | Timeout in seconds for packet size auto-discovery (recvmmsg only) |

### recvmmsg Adaptive Coalescing

The standard UDP backend waits for the socket to become readable, then uses the measured packet
rate to briefly coalesce datagrams before draining the socket with nonblocking `recvmmsg()` calls.
It drains until the socket is empty and publishes each received group downstream as a batch. This
keeps the receiver immediately stoppable while amortizing both syscall and downstream queue costs.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `recvmmsg.target_batch` | uint32 | 0 | Desired packets per first syscall; 0 selects 75% of `num_msgs` |
| `recvmmsg.min_coalesce_us` | uint32 | 0 | Lower bound for the adaptive coalescing interval |
| `recvmmsg.max_coalesce_us` | uint32 | 0 | Upper bound for the adaptive interval; 0 disables coalescing |
| `recvmmsg.adaptation_interval_ms` | uint32 | 250 | Packet-rate measurement/update interval |

Adaptive coalescing is deliberately opt-in: both an explicit `recv_buf_size` and a nonzero
`max_coalesce_us` are required. The backend reads back the effective `SO_RCVBUF` after the kernel
applies its limits, conservatively estimates per-packet socket-memory cost, and permits intentional
coalescing to consume at most 25% of that effective buffer. It clamps both `target_batch` and the
live interval to that budget.

The controller estimates packets per second with an EWMA and computes the nominal delay needed to
reach `target_batch`. `SO_MEMINFO` supplies live receive-memory occupancy and kernel-drop feedback.
A full receive vector, pool stall, high socket occupancy, or kernel drop immediately disables
coalescing for eight clean drain cycles. The EWMA cannot raise the delay
while this congestion latch is active. After one second without traffic, the old rate estimate is
discarded. All properties in the `recvmmsg` object are runtime reconfigurable (the receiver is
reconstructed).

Example:
```json
{
  "num_msgs": 256,
  "recv_buf_size": 16777216,
  "recvmmsg": {
    "target_batch": 192,
    "min_coalesce_us": 0,
    "max_coalesce_us": 10000,
    "adaptation_interval_ms": 250
  }
}
```

**Runtime reconfigurable:** `interface`, `ip_addr`, `port`, `num_msgs`, `frame_count`, `overrides`,
and `recvmmsg`

### Overrides

Optional overrides for advanced configuration:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `overrides.msg_size` | uint32 | (auto) | Maximum UDP payload size in bytes. If not set, recvmmsg will auto-discover. Required for packet_mmap. |

### DPDK Configuration

Properties under the `dpdk` struct (only applicable when `socket_type = "dpdk"`):

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `dpdk.port_id` | uint16 | (auto) | DPDK port ID (auto-resolved from interface name if not specified) |
| `dpdk.queue_id` | uint16 | (auto) | DPDK queue ID (auto-assigned if not specified) |
| `dpdk.mempool_name` | string | `"mbuf_pool"` | Name of DPDK memory pool for packet buffers |
| `dpdk.burst_size` | uint16 | 32 | Number of packets to receive per burst |
| `dpdk.src_ip` | string | - | Source IP address for IGMP (enables IGMP if destination is multicast) |
| `dpdk.igmp_respond_to_queries` | bool | true | Whether to respond to IGMP membership queries |

## Performance Tuning

### Frame Pool Sizing

The frame pool (`frame_count`) pre-allocates memory buffers for received packets. Size it based on:
- Packet rate and downstream processing latency
- Memory constraints
- Burst tolerance

**Rule of thumb:** `frame_count >= num_msgs * 2` to handle backpressure

For high-throughput continuous streams:
```json
{
  "frame_count": 32768,
  "num_msgs": 128
}
```

For bursty low-rate traffic:
```json
{
  "frame_count": 1024,
  "num_msgs": 32
}
```

### Batch Size

The `num_msgs` property controls batching, which reduces syscall overhead:
- **Larger batches** (128-256): Better throughput, slightly higher latency
- **Smaller batches** (32-64): Lower latency, more CPU overhead

For continuous high-rate streams, larger batches are recommended.

### Receive Buffer Size

Increase `recv_buf_size` to buffer more packets in the kernel, reducing drops during processing spikes:

```json
{
  "recv_buf_size": 16777216  // 16 MB
}
```

Check kernel limits: `sysctl net.core.rmem_max`

### Auto-Discovery

When `overrides.msg_size` is not set, recvmmsg will auto-discover packet size by:
1. Polling the socket for up to `autodiscovery_timeout` seconds
2. Receiving a packet and measuring its size
3. Rounding up to the next power of 2 for frame allocation

**For production:** Set `overrides.msg_size` explicitly to:
- Avoid startup delays
- Ensure deterministic behavior
- Prevent startup failure if no data is flowing

```json
{
  "overrides": {
    "msg_size": 1472  // Standard UDP payload (1500 MTU - 20 IP - 8 UDP)
  }
}
```

## IGMP and Multicast

### Standard Linux (recvmmsg, packet_mmap)

For multicast reception:
1. Set `ip_addr` to the multicast group address (e.g., `"239.1.1.1"`)
2. Specify the receiving `interface`
3. The component automatically joins the multicast group via `IP_ADD_MEMBERSHIP`

For packet_mmap, it also sets `PACKET_ADD_MEMBERSHIP` at the link layer.

Example:
```json
{
  "socket_type": "recvmmsg",
  "interface": "eth0",
  "ip_addr": "239.1.1.1",
  "port": 4991
}
```

### DPDK

For DPDK multicast with IGMP:
1. Set `ip_addr` to the multicast group
2. Set `dpdk.src_ip` to the source IP address for IGMP messages
3. The component will automatically respond to IGMP queries if `dpdk.igmp_respond_to_queries` is true

Example:
```json
{
  "socket_type": "dpdk",
  "interface": "0000:01:00.0",
  "ip_addr": "239.1.1.1",
  "port": 4991,
  "dpdk": {
    "src_ip": "192.168.1.100",
    "igmp_respond_to_queries": true
  }
}
```

## Requirements

### Core Requirements
- Linux kernel 2.6.33+ (for recvmmsg)
- Linux kernel 3.0+ (for packet_mmap TPACKET_V2)
- C++20 compiler

### Optional Requirements
- DPDK 20.11+ (for dpdk socket type)
- DPDK-compatible network interface card

### Permissions

For packet_mmap and DPDK, the process needs appropriate permissions:
- `CAP_NET_RAW` capability for packet_mmap
- Root or appropriate DPDK setup for dpdk backend

## Example Configurations

### Basic UDP Reception
```json
{
  "name": "udp_source",
  "id": "udp_rx",
  "properties": {
    "interface": "eth0",
    "ip_addr": "0.0.0.0",
    "port": 12345,
    "overrides": {
      "msg_size": 1472
    }
  }
}
```

### High-Performance Multicast (packet_mmap)
```json
{
  "name": "udp_source",
  "id": "sdr_ingest",
  "properties": {
    "socket_type": "packet_mmap",
    "interface": "eth1",
    "ip_addr": "239.255.0.1",
    "port": 4991,
    "recv_buf_size": 33554432,
    "num_msgs": 256,
    "frame_count": 16384,
    "overrides": {
      "msg_size": 8192
    }
  }
}
```

### DPDK with IGMP
```json
{
  "name": "udp_source",
  "id": "dpdk_rx",
  "properties": {
    "socket_type": "dpdk",
    "interface": "0000:81:00.0",
    "ip_addr": "239.10.10.10",
    "port": 50000,
    "dpdk": {
      "burst_size": 64,
      "mempool_name": "rx_pool",
      "src_ip": "10.10.10.100",
      "igmp_respond_to_queries": true
    }
  }
}
```

### Auto-Discovery for Development
```json
{
  "name": "udp_source",
  "id": "udp_rx",
  "properties": {
    "interface": "lo",
    "ip_addr": "127.0.0.1",
    "port": 5000,
    "autodiscovery_timeout": 10
  }
}
```

## Output

The component provides a single output port:

**Port:** `data_out`

**Type:** `immutable_buffer<uint8_t>`

**Content:** Raw UDP payload (does not include IP or UDP headers)

Connect this port to packet parsers (e.g., VITA 49, custom protocols) or other processing components.

## Statistics

Runtime statistics are logged every 5 seconds at debug level:

### recvmmsg
- `pkts_recvd`: Total packets received since component start
- `recv_syscalls`: Total `recvmmsg()` calls, including the final `EAGAIN` drain calls
- `estimated_pps`: Current EWMA packet-rate estimate
- `coalesce_us`: Current adaptive coalescing interval
- `coalesce_target_batch`: Effective target after applying the socket-capacity safety limit
- `effective_recv_buf`: Effective kernel `SO_RCVBUF` accounting limit after clamping
- `estimated_packet_charge`: Conservative socket-memory charge used for safety calculations
- `socket_rmem_bytes`: Most recently observed receive-memory allocation
- `kernel_drops`: Packets dropped at this UDP socket according to `SO_MEMINFO`

Kernel receive-queue drops are also exported separately as the
`udp_source.kernel_drops` counter; they are not mixed into `udp_source.packets_dropped`, which
remains reserved for filtering and receiver-internal errors.

### packet_mmap
- `pkts_recvd`: Total packets received by userspace since component start
- `pkts_recvd_kernel`: Total packets received by kernel (from socket statistics)
- `pkts_dropped_kernel`: Total packets dropped by kernel due to buffer full

## Troubleshooting

### Auto-discovery timeout
**Problem:** Component fails to start with "failed to discover packet size after N seconds"

**Solutions:**
- Ensure data is flowing on the network
- Set `overrides.msg_size` explicitly
- Increase `autodiscovery_timeout`

### Packet drops
**Problem:** `pkts_dropped_kernel` increasing (packet_mmap)

**Solutions:**
- Increase `frame_count` to buffer more packets
- Increase `recv_buf_size` for more kernel buffering
- Optimize downstream processing to reduce backpressure
- Consider switching to DPDK for kernel-bypass

### High CPU usage
**Problem:** Component consuming excessive CPU when idle

The recvmmsg backend sleeps indefinitely when the socket is empty and is woken by either data or an
explicit stop event. Its adaptive coalescing wait also sleeps interruptibly. Persistent CPU usage
therefore indicates an active stream, downstream backpressure, or repeated pool-acquisition stalls.

For more syscall amortization, raise `recvmmsg.target_batch` or `recvmmsg.max_coalesce_us`.

### DPDK initialization failure
**Problem:** "DPDK support not compiled in" or DPDK port initialization errors

**Solutions:**
- Ensure composite-comps was built with `-DCOMPOSITE_HAS_DPDK=ON`
- Verify DPDK is properly installed and configured
- Check DPDK port binding (`dpdk-devbind.py --status`)
- Ensure hugepages are configured (`cat /proc/meminfo | grep Huge`)
