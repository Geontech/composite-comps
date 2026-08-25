# WebSocket Sink Component

WebSocket sink component for streaming data and histograms to multiple clients with per-client FPS control and adaptive backpressure handling.

## Features

- **Dual Input Ports**: Receives streaming data and histogram data
- **Multi-Client Support**: Serves WebSocket clients simultaneously
- **Per-Client FPS**: Each client can request different frame rates (1-60 FPS)
- **Per-Client Adaptive Backpressure**: Automatically reduces a clients FPS when it falls behind
- **Zero-Copy Architecture**: Uses `shared_ptr` for efficient memory handling
- **Dynamic Upstream Control**: Disables input ports when no clients are connected
- **Real-Time Metrics**: Broadcasts performance metrics every second

## Input Ports

| Port Name | Type | Description |
|-----------|------|-------------|
| `data_in` | `immutable_buffer<T>` | Spectrum data (PSD/FFT output) |
| `histogram_in` | `immutable_buffer<uint64_t>` | Histogram bin counts |

## Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `port` | uint16 | 8080 | WebSocket server TCP port |
| `bind_address` | string | "0.0.0.0" | Network interface to bind to |
| `default_stream_fps` | float | 12.0 | Default FPS for new clients (min 1.0) |
| `max_stream_fps` | float | 60.0 | Maximum allowed FPS (min 1.0) |
| `stream_port_depth` | uint32 | 1 | Input port buffer depth (min 1) |
| `send_queue_hwm` | uint32 | 5 | High water mark for backpressure (min 2) |
| `max_clients` | uint32 | 100 | Maximum concurrent clients (min 1) |
| `compression` | bool | false | Enable per-message deflate compression |

The component sets `finish_at_end=false`: upstream end-of-stream does not finish the
worker or tear down the WebSocket server — connected clients stay attached and the
listen port stays open across finite captures. Override via the standard
`finish_at_end` property if a run-to-completion pipeline should stop the sink.

## Template Types

Supports real and complex float and double precision, selected via the create-args
`type` discriminator in the application config:

```json
{
  "id": "ws1",
  "library": "libws_sink.so",
  "args": { "type": "cf32" }
}
```

- `"f32"` / `"cf32"` - Single precision (real / complex float)
- `"f64"` / `"cf64"` - Double precision (real / complex double)

(The legacy scalar form `"create_arg": "cf32"` is also accepted by the loader.)

## Message Protocol

### Stream Data (FPS-Throttled)

**Message 1: JSON Header**
```json
{
  "event": "header",
  "payload": {
    "format": "CD",
    "sampleRate": 25000000.0,
    "bandwidth": 20000000.0,
    "frequency": 1090000000.0,
    "fftsize": 65536,
    "ape": 65536,
    "bpa": 8,
    "bpe": 524288,
    "timecode_int": 1234567890,
    "timecode_frac": 123456789012
  }
}
```

**Message 2: Binary Data**
- Raw binary data (512 KB for 65,536 complex doubles)

### Histogram Data

- Sent on receipt

```json
{
  "event": "histogram",
  "payload": [10, 20, 30, 40, ...]
}
```

### Metrics (1 Hz)

```json
{
  "event": "metrics",
  "payload": {
    "activeConnections": 5,
    "droppingFrames": false,
    "framesPerSecond": "11.8",
    "framesSent": 43200,
    "framesDropped": 120,
    "dropRatio": "0.3"
  }
}
```

## Client Messages

### Request FPS Change

```json
{
  "type": "set_stream_fps",
  "fps": 30.0
}
```

**Response:**
```json
{
  "type": "stream_fps_updated",
  "fps": 30.0,
  "requested_fps": 30.0
}
```

## Adaptive Backpressure

The component automatically adjusts client FPS based on send queue size:

| State | Condition | Action |
|-------|-----------|--------|
| **Healthy** | Queue < 25% HWM | Maintain current FPS |
| **Backpressure** | Queue ≥ HWM | Reduce FPS by 50% |
| **Recovering** | Queue < 50% HWM | Increase FPS by 25% |

- Minimum FPS: 1.0 Hz
- Frames are skipped when client can't keep up
- Client stays connected, just misses frames

## Port Depth Control

| Condition | Stream Depth | Histogram Depth | Effect |
|-----------|--------------|-----------------|--------|
| No clients | 0 | 0 | Disables upstream processing |
| 1+ clients | Configurable (default: 1) | 5 | Enables upstream processing |

## Performance

**Optimized for high-throughput:**
- Zero-copy serialization with `shared_ptr`
- Lock-free client list (RCU pattern)
- Per-client worker threads - slow clients don't block fast clients
- Non-blocking send operations (~1μs latency)

## Build Requirements

- C++23 compiler
- IXWebSocket
- nlohmann_json
- composite framework (0.5)
- pthread (for thread naming)
- zlib

## Usage

```bash
# Build
cmake -B build
cmake --build build

# Run an application config that includes a ws_sink component
composite-cli <app-config>.json
```

Connect WebSocket client to `ws://localhost:8080`

## Client Example (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8080');
ws.binaryType = 'arraybuffer';

ws.onmessage = (event) => {
  if (typeof event.data === 'string') {
    const msg = JSON.parse(event.data);

    if (msg.event === 'header') {
      console.log('Spectrum header:', msg.payload);
      // Next message will be binary data
    } else if (msg.event === 'histogram') {
      console.log('Histogram:', msg.payload);
    } else if (msg.event === 'metrics') {
      console.log('Metrics:', msg);
    }
  } else {
    // Binary spectrum data
    const samples = new Float32Array(event.data);
    console.log('Received', samples.length / 2, 'complex samples');
  }
};

// Request higher FPS
ws.send(JSON.stringify({
  type: 'set_stream_fps',
  fps: 30.0
}));
```

## Test Client (C++)

A C++ test client is available for testing the WebSocket server from within a Docker container or pod.

### Building the Test Client

Enable the test client during CMake configuration:

```bash
cmake -B build -DBUILD_WS_SINK_TEST_CLIENT=ON
cmake --build build
```

The executable will be installed to `bin/ws_sink_test_client`.

### Usage

```bash
# Connect to localhost:8081 for 30 seconds
ws_sink_test_client

# Connect to specific server/port
ws_sink_test_client -s 10.0.0.5 -p 8081

# Request 30 FPS and run for 60 seconds
ws_sink_test_client -f 30 -d 60

# Show help
ws_sink_test_client -h
```

### Options

- `-s, --server HOST` - WebSocket server hostname (default: localhost)
- `-p, --port PORT` - WebSocket server port (default: 8081)
- `-d, --duration SEC` - Test duration in seconds (default: 30)
- `-f, --fps FPS` - Request specific FPS from server
- `-h, --help` - Show help message

### Output

The test client displays:
- Connection status
- Real-time header, histogram, and metrics messages
- Binary data statistics
- Summary statistics at the end (frames received, throughput, actual FPS)

This is particularly useful for testing within Kubernetes pods where you can:
1. Shell into the pod: `kubectl exec -it <pod-name> -- /bin/bash`
2. Run the test client: `ws_sink_test_client -s localhost -p 8081`
