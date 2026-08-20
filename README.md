# Composite Comps

This repo contains a collection of signal processing components built for the [**composite**](https://github.com/geontech/composite) framework.
It also comes with headers that are used across the various components in this repo.

## Components

The following components are available:

- **[exp_smooth](src/components/exp_smooth/README.md)** - Exponential smoothing filter with AVX2/AVX-512 acceleration
- **[framer](src/components/framer/)** - Overlapped framing with SIMD type conversion
- **[fft](src/components/fft/README.md)** - Fast Fourier Transform with framing, windowing, and parallel processing
- **[file_writer](src/components/file_writer/)** - General-purpose file writer for data streams
- **[histogram](src/components/histogram/README.md)** - Real-time histogram accumulator for ADC sample distribution analysis
- **[pkt_builder](src/components/pkt_builder/)** - VITA 49 packet builder (egress)
- **[pkt_parser](src/components/pkt_parser/)** - VITA 49 packet parser
- **[psd](src/components/psd/)** - Power spectral density calculator
- **[udp_sink](src/components/udp_sink/)** - UDP packet transmitter (egress)
- **[udp_source](src/components/udp_source/README.md)** - High-performance UDP packet receiver with multiple backends
- **[ws_sink](src/components/ws_sink/README.md)** - WebSocket sink for live data, histogram and metrics streaming
(recvmmsg, packet_mmap, DPDK) optimized for continuous high-throughput streams

Each component directory may contain a detailed README with usage examples, configuration options, and performance characteristics.

## Getting started

### Prerequisites

Ensure you have the following installed:

- [CMake](https://cmake.org/) (version 3.15 or higher)
- A compatible C++ compiler (e.g., GCC, Clang) with C++20 support

### Build and Install

```cmake
cmake -B build
cmake --build build [--parallel N]
cmake --install build
```