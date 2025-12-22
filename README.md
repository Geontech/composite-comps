# Composite Comps

This repo contains a collection of signal processing components built for the [**composite**](https://github.com/geontech/composite) framework.
It also comes with headers that are used across the various components in this repo.

## Components

The following components are available:

- **[aligned_mem_writer](src/components/aligned_mem_writer/)** - Write SIMD-aligned memory buffers to files
- **[exp_smooth](src/components/exp_smooth/README.md)** - Exponential smoothing filter with AVX2/AVX-512 acceleration
- **[fft](src/components/fft/README.md)** - Fast Fourier Transform with framing, windowing, and parallel processing
- **[file_writer](src/components/file_writer/)** - General-purpose file writer for data streams
- **[halfrate](src/components/halfrate/README.md)** - High-performance polyphase half-band decimating filter with AVX-512/AVX2/scalar SIMD support
- **[histogram](src/components/histogram/README.md)** - Real-time histogram accumulator for ADC sample distribution analysis
- **[pkt_parser](src/components/pkt_parser/)** - VITA 49 packet parser
- **[psd](src/components/psd/)** - Power spectral density calculator
- **[stov](src/components/stov/)** - Sample type converter (short to various types)
- **[udp_source](src/components/udp_source/README.md)** - High-performance UDP packet receiver with multiple backends
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
