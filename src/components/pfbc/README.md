# Polyphase Filter Bank Channelizer (PFBC)

A high-performance polyphase filter bank channelizer component for the composite framework. Splits a wideband input signal into M narrowband channel outputs using an efficient polyphase filterbank decomposition.

## Overview

The PFBC implements a critically-sampled analysis filterbank using:
- Polyphase decomposition of the prototype lowpass filter
- Interleaved SIMD polyphase FIR filtering (single contiguous stream)
- FFT-based channelization via FFTW single-precision

## Properties

### Configuration (INITIALIZE)

| Property | Type | Description |
|----------|------|-------------|
| `num_channels` | uint32_t | Number of output channels M (must be power of 2, ≥8 for AVX-512 kernels) |
| `taps_per_phase` | uint32_t | FIR taps per polyphase branch K |
| `frame_size` | uint32_t | Output samples per channel per frame N |
| `prototype_filter` | vector\<float\> | Custom prototype filter (M*K taps). If empty, uses default sinc-windowed lowpass |

## Ports

- **data_in**: `input_port<immutable_buffer<complex<float>>>` - Wideband input samples (read-only)
- **data_out**: `output_port<immutable_buffer<complex<float>>>` - Per-channel output frames (pool-backed)

## Processing Pipeline

Each M-sample input block flows through a 2-stage pipeline:

1. **Interleaved polyphase filter**: Apply K-tap FIR filter across all phases in one pass
   - Input stays time-major interleaved (`[t0: p0..pM-1][t1: ...]`)
   - Coefficients are interleaved to match complex layout for SIMD FMAs

2. **FFT**: M-point FFT transforms to frequency domain, output accumulates
   - FFT output writes directly into per-channel frame accumulators
   - When N samples accumulate per channel, frames are emitted

When N samples have accumulated per channel, frames are emitted on `data_out`.

## Architecture

```
Input (M samples)
      ↓
+---------------------------+
| Interleaved Polyphase FIR |  M phases in one pass
| (time-major, SIMD)        |  coefficients interleaved for FMAs
+---------------------------+
      ↓
+------------------+
| M-point FFT      |  FFTW single-precision
+------------------+  Output directly to accumulators
      ↓
+--------------------+
| Frame Accumulators | Gather N samples per channel
+--------------------+
      ↓
Output (M channels × N samples per frame)
```

## Performance Optimizations

### Interleaved SIMD Kernel (kernels_interleaved.hpp)

Kernels are pure functions with no side effects, optimized for AVX-512:

1. **Interleaved polyphase filter** (`filter_interleaved`)
   - Processes all M phases in one sequential memory stream
   - Specialized kernels for M = 8/16/32/64/128 with time unrolling
   - Generic AVX-512 path for other channel counts

### Zero-Allocation Hot Path
- **Output buffer pool**: `slab_pool` pre-allocates output buffers, eliminating malloc/free from the processing loop
- **Pool-backed immutable_buffer**: Output buffers automatically return to pool when downstream releases them

### Cache-Friendly Memory Layout
- **Time-major interleaved input**: Single contiguous stream for filter stage
- **Frame-major accumulator**: FFT outputs written contiguously for all channels
  - Better cache locality than M separate per-channel buffers
- **Direct FFT-to-accumulator**: FFT output goes directly into frame accumulator
  - Eliminates intermediate buffer and copy

## Dependencies

- **FFTW3** (single-precision): `libfftw3f`, `libfftw3f_threads`
- **AVX-512**: Required for current SIMD kernels (no runtime dispatch in component)
- **composite**: Framework ports, buffers, properties

## Build

```bash
cmake -B build
cmake --build build --parallel
```

## Tests

```bash
cd build
ctest -R pfbc

# Or run individually:
./src/components/pfbc/tests/pfbc_kernel_tests        # Kernel unit tests (1338 assertions)
./src/components/pfbc/tests/pfbc_integration_tests   # Component integration tests (2480 assertions)
```

## Benchmarks

Performance benchmarks using Google Benchmark:

```bash
# Run all kernel benchmarks
./build/src/components/pfbc/tests/pfbc_kernel_benchmarks

# Run specific benchmarks
./build/src/components/pfbc/tests/pfbc_kernel_benchmarks --benchmark_filter="BM_Commutator"
./build/src/components/pfbc/tests/pfbc_kernel_benchmarks --benchmark_filter="BM_Filter"
./build/src/components/pfbc/tests/pfbc_kernel_benchmarks --benchmark_filter="BM_Full_Pipeline"
```

## Example Configuration

```yaml
components:
  - id: channelizer
    type: pfbc
    properties:
      num_channels: 64       # Must be power of 2, ≥8 recommended
      taps_per_phase: 12     # Filter length per phase
      frame_size: 1024       # Samples per channel per output frame
```

## Files

- `component.hpp/cpp` - Main component implementation
- `coefficients.hpp` - Prototype and interleaved coefficient helpers
- `kernels_interleaved.hpp` - Interleaved AVX-512 polyphase filter kernels
- `fft_plan.hpp` - FFTW wrapper for M-point FFT (single/batch support)
- `tests/pfbc_kernel_tests.cpp` - Kernel unit tests (Catch2)
- `tests/pfbc_integration_tests.cpp` - Component integration tests (Catch2)
- `tests/kernel_benchmarks.cpp` - Performance benchmarks (Google Benchmark)

## License

Copyright (C) 2025 Geon Technologies, LLC

This component is part of composite-comps, licensed under the GNU Lesser General Public License v3.0 or later. See LICENSE file for details.
