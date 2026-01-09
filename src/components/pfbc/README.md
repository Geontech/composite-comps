# Polyphase Filter Bank Channelizer (PFBC)

A high-performance polyphase filter bank channelizer component for the composite framework. Splits a wideband input signal into M narrowband channel outputs using an efficient polyphase filterbank decomposition with AVX-512 optimized kernels.

## Overview

The PFBC implements a critically-sampled analysis filterbank using:
- Polyphase decomposition of the prototype lowpass filter
- Blocked SIMD polyphase FIR filtering optimized for AVX-512
- Custom in-register FFT kernels for M=8, 16, 32 (no FFTW overhead)
- FFTW fallback for larger channel counts

## Properties

### Configuration (INITIALIZE)

| Property | Type | Description |
|----------|------|-------------|
| `num_channels` | uint32_t | Number of output channels M (must be power of 2, ≥8) |
| `taps_per_phase` | uint32_t | FIR taps per polyphase branch K |
| `frame_size` | uint32_t | Output samples per channel per frame N |
| `prototype_filter` | vector\<float\> | Custom prototype filter (M*K taps). If empty, uses default sinc-windowed lowpass |

## Ports

- **data_in**: `input_port<immutable_buffer<complex<float>>>` - Wideband input samples (read-only)
- **data_out**: `output_port<immutable_buffer<complex<float>>>` - Per-channel output frames (pool-backed)

## Engine Policies

The PFBC uses compile-time policy selection based on channel count M:

| M | Policy | Description | Throughput |
|---|--------|-------------|------------|
| 8 | `fused_m8_vertical` | In-register vertical FFT-8, fully fused pipeline | ~458 Msps |
| 16 | `fused_m16_vertical` | In-register vertical FFT-16, fully fused pipeline | ~417 Msps |
| 32 | `hybrid_m32` | In-register FFT-32, L1 scratch transpose | ~218 Msps |
| ≥64 | `staged` | Filter → FFTW → transpose via L1 scratch | ~150-207 Msps |

## Processing Pipeline

### Fused Pipeline (M=8, 16)

For small channel counts, the entire pipeline runs in AVX-512 registers:

```
Input (tile of 8 rows × M channels)
      ↓
+----------------------------------+
| Blocked Polyphase FIR            |  8 rows accumulated in registers
| (8 ZMM accumulators)             |
+----------------------------------+
      ↓
+----------------------------------+
| 8×M Transpose (in registers)     |  Row-major → column-major
+----------------------------------+
      ↓
+----------------------------------+
| Vertical FFT-M                   |  8 parallel FFTs, no shuffles
| (M separate FFTs per ZMM lane)   |
+----------------------------------+
      ↓
Direct scatter to M channel buffers
```

### Hybrid Pipeline (M=32)

For M=32, filter+FFT runs in registers, transpose uses L1 scratch:

```
Input (tile of 4 rows × 32 channels)
      ↓
+----------------------------------+
| Blocked Polyphase FIR            |  16 ZMM accumulators (4 rows × 4 regs)
+----------------------------------+
      ↓
+----------------------------------+
| In-Register FFT-32               |  4 ZMMs per row, sequential
+----------------------------------+
      ↓
+----------------------------------+
| L1 Scratch (4×32 samples)        |  Store FFT output
+----------------------------------+
      ↓
+----------------------------------+
| 4×32 Transpose                   |  Two 4×16 transpose passes
+----------------------------------+
      ↓
Scatter to 32 channel buffers
```

### Staged Pipeline (M≥64)

For larger channel counts, uses FFTW:

```
Input (tile of 8 rows × M channels)
      ↓
+----------------------------------+
| Blocked Polyphase FIR            |  Output to L1 scratch
+----------------------------------+
      ↓
+----------------------------------+
| FFTW M-point FFT (per row)       |  In-place on scratch
+----------------------------------+
      ↓
+----------------------------------+
| SIMD Transpose                   |  Scratch → channel buffers
+----------------------------------+
```

## Performance Optimizations

### Vertical FFT (M=8, 16)

The vertical FFT approach processes 8 parallel FFTs simultaneously:
- Each ZMM lane holds one sample from each of 8 different FFTs
- No cross-lane shuffles during FFT computation
- Direct scatter-store to channel buffers (no post-FFT transpose)
- ~4.5x faster than FFTW-based pipeline

### Blocked Coefficient Layout

Coefficients are arranged for optimal cache access:
```
Block 0 (channels 0-7):   [tap0: h0,h0,h1,h1,...,h7,h7][tap1: ...][...]
Block 1 (channels 8-15):  [tap0: h8,h8,h9,h9,...][tap1: ...][...]
...
```
- Taps contiguous within each block (prefetcher optimal)
- Each coefficient duplicated for complex multiply: h × (re,im) = (h×re, h×im)

### Zero-Allocation Hot Path

- **Output buffer pool**: `slab_pool` pre-allocates output buffers
- **Pool-backed immutable_buffer**: Automatic return to pool on release
- **L1-resident scratch**: Tile processing keeps working set in cache

## Architecture

```
src/components/pfbc/
├── component.hpp/cpp      # Main component implementation
├── pfbc_engine.hpp        # Compile-time policy selection
├── kernels.hpp            # AVX-512 kernels (filter, FFT, transpose)
├── coefficients.hpp       # Blocked coefficient preparation
├── fft_plan.hpp           # FFTW wrapper for staged pipeline
└── tests/
    ├── kernel_tests.cpp           # Kernel unit tests
    ├── kernel_benchmarks.cpp      # Performance benchmarks
    └── pfbc_integration_tests.cpp # Component integration tests
```

## Dependencies

- **FFTW3** (single-precision): For M≥64 staged pipeline
- **AVX-512F, AVX-512DQ, AVX-512VL**: Required for SIMD kernels
- **composite**: Framework ports, buffers, properties

## Build

```bash
cmake -B build
cmake --build build --parallel
```

## Tests

```bash
cd build
ctest -L pfbc

# Or run individually:
./src/components/pfbc/tests/pfbc_kernel_tests        # Kernel unit tests
./src/components/pfbc/tests/pfbc_integration_tests   # Component integration tests
```

## Benchmarks

```bash
# Run all kernel benchmarks
./build/src/components/pfbc/tests/pfbc_kernel_benchmarks

# Run specific benchmarks
./build/src/components/pfbc/tests/pfbc_kernel_benchmarks --benchmark_filter="BM_Fused"
./build/src/components/pfbc/tests/pfbc_kernel_benchmarks --benchmark_filter="BM_Hybrid"
./build/src/components/pfbc/tests/pfbc_kernel_benchmarks --benchmark_filter="BM_Pipeline_Comprehensive"
```

## Example Configuration

```yaml
components:
  - id: channelizer
    type: pfbc
    properties:
      num_channels: 32       # Uses hybrid_m32 policy
      taps_per_phase: 16     # Filter length per phase
      frame_size: 1024       # Samples per channel per output frame
```

## License

Copyright (C) 2025 Geon Technologies, LLC

This component is part of composite-comps, licensed under the GNU Lesser General Public License v3.0 or later. See LICENSE file for details.
