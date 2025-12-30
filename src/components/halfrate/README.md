# Halfrate Decimating Filter Component

High-performance polyphase half-band decimating filter optimized for single-core throughput using SIMD vectorization.

## Overview

The `halfrate` component implements a polyphase half-band FIR filter that decimates complex input samples by a factor of 2. It features:

- **Multi-platform SIMD**: AVX-512, AVX2, and scalar implementations via Multi-Function Versioning (MFV)
- **Cache-optimized fused kernel**: Deinterleave + filter in tile chunks
- **Configurable filter design**: Adjustable filter length (1..128 semi-length) and window function
- **Zero-copy architecture**: Efficient buffer management using composite framework

## Signal Processing Details

### Half-Band Filter Theory

A half-band filter has:
- **Cutoff frequency**: Fs/4 (quarter of sample rate)
- **Decimation factor**: 2 (output rate = input rate / 2)
- **Symmetric impulse response**: Every other coefficient is zero (except center tap)

This component uses a **polyphase decomposition** that exploits the zero-valued coefficients:
- **Even branch**: Center tap (typically 0.5) applied to even-indexed input samples with group delay compensation
- **Odd branch**: Non-zero FIR taps applied to odd-indexed input samples

The output combines both branches: `output[n] = FIR(odd[n]) + center_tap * even[n - delay]`

### Filter Design Parameters

Configured via properties:
- `filter_semi_length`: Number of non-zero taps on one side (e.g., 3, 7, 15). Total filter length = `2 × filter_semi_length + 1`
- `window`: Window function for coefficient generation
  - `"HAMMING"` (default) - Good passband flatness, moderate rejection (~53 dB)
  - `"BLACKMAN_HARRIS"` - Better stopband rejection (~92 dB)

The component automatically:
1. Generates coefficients using windowed-sinc method: `h[k] = sinc(0.5π·k) · window[k]`
2. Extracts polyphase branches (odd-offset taps become FIR coefficients, center tap becomes scalar)
3. Computes group delay offset for even branch alignment

## Architecture

### Processing Pipeline

```
Input (interleaved I/Q pairs)
    ↓
┌─────────────────────────────────────────┐
│     Fused Deinterleave + Filter         │
│     (Cache-tiled SIMD kernel)           │
│                                         │
│  For each L1-sized tile:                │
│    1. Copy history overlap to tile      │
│    2. Deinterleave input → tile buffers │
│    3. Filter while data is L1-hot       │
│    4. Write outputs                     │
│                                         │
│  Finally: Update history with tail      │
└─────────────────────────────────────────┘
    ↓
Output (decimated by 2)
```

### Fused SIMD Kernel

The `halfband_filter_fused` kernel combines deinterleaving and filtering in a single pass, processing in L1-cache-sized tiles (512 outputs per tile) to minimize memory traffic.

**Constraint**: `filter_semi_length` is capped at 128 (`history_len <= 256`) to keep tile buffers bounded and preserve in-place output safety.

| ISA     | Deinterleave | Filter | Notes |
|---------|--------------|--------|-------|
| AVX-512 | 16 pairs/iter | 32 outputs/iter | `permutex2var` + FMA |
| Scalar  | 1 pair/iter | 1 output/iter | Fallback |

**Multi-Function Versioning**: The compiler generates multiple ISA versions. At runtime, the CPU automatically dispatches to the best available implementation based on feature flags.

### Performance

The fused kernel achieves speedup over separate deinterleave + filter on memory-bound workloads (256K samples) by:
- Processing in 512-output tiles that fit in L1 cache
- Filtering immediately after deinterleaving while data is hot
- Only writing tail samples to history buffers (not full working set)

## Usage

### Ports

- **Input**: `data_in` - `mutable_buffer<std::complex<float>>`
- **Output**: `data_out` - `immutable_buffer<std::complex<float>>`

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `filter_semi_length` | `uint32_t` | `3` | Half-band filter semi-length (number of non-zero taps on one side, 1..128). Total filter length = `2 × filter_semi_length + 1`. Example: semi_length=3 → 7 taps total |
| `window` | `string` | `"HAMMING"` | Window function for coefficient generation. Valid values: `"HAMMING"` or `"BLACKMAN_HARRIS"` |

## Building

### Component Library

```bash
cmake -B build
cmake --build build --parallel $(nproc)
```

The component builds as `build/src/components/halfrate/libhalfrate.so`

### Tests

```bash
# Build tests
cmake --build build --target kernel_tests_avx512 halfrate_integration_tests kernel_benchmarks

# Run unit tests
./build/src/components/halfrate/tests/kernel_tests_avx512

# Run integration tests
./build/src/components/halfrate/tests/halfrate_integration_tests

# Run performance benchmarks (Google Benchmark)
./build/src/components/halfrate/tests/kernel_benchmarks
```

## Testing

### Test Coverage

Two comprehensive test suites plus benchmarks:

1. **`kernel_tests.cpp`**
   - Validates fused SIMD kernel against simple scalar reference
   - Tests various block sizes, tile boundaries, tap counts
   - Tests history update correctness
   - Impulse response validation

2. **`halfrate_integration_tests.cpp`**
   - End-to-end component behavior
   - Frequency response validation (passband, stopband)
   - Streaming tests with history management
   - Edge cases (DC, impulse, Nyquist)

3. **`kernel_benchmarks.cpp`** (Google Benchmark)
   - Throughput measurements at various block sizes
   - Parameterized size/tap count sweeps
   - Cache behavior analysis (L1/L2/L3/DRAM)

### Running Tests

```bash
# All tests via CTest
cd build && ctest

# Individual test suites (AVX-512, AVX2, or scalar variants)
./build/src/components/halfrate/tests/kernel_tests_avx512
./build/src/components/halfrate/tests/halfrate_integration_tests

# Performance benchmarks
./build/src/components/halfrate/tests/kernel_benchmarks

# Run specific benchmarks with filter
./build/src/components/halfrate/tests/kernel_benchmarks --benchmark_filter="256K"

# Export results to JSON
./build/src/components/halfrate/tests/kernel_benchmarks --benchmark_format=json --benchmark_out=results.json
```

## Implementation Details

### File Structure

```
halfrate/
├── component.hpp          # Component class definition
├── component.cpp          # Port setup, filter design, process() logic
├── kernels.hpp            # Fused SIMD kernel with MFV (header-only)
├── CMakeLists.txt         # Build configuration
└── tests/
    ├── kernel_tests.cpp               # Catch2 unit tests for fused kernel
    ├── halfrate_integration_tests.cpp # Catch2 integration tests
    ├── kernel_benchmarks.cpp          # Google Benchmark performance tests
    └── CMakeLists.txt
```

### Key Design Decisions

1. **Polyphase decomposition**: Exploits half-band symmetry for 2× efficiency
2. **Fused kernel**: Single kernel for deinterleave + filter improves cache utilization
3. **L1 tiling**: Process in 512-output tiles to keep working set in L1 cache
4. **Minimal history**: Only store `history_len` samples (not full working set)

### History Management

The component maintains compact history buffers for filter state continuity:
- **History size**: `max(num_taps, filter_semi_length)` samples per lane
- **Memory footprint**: `2 * history_len * sizeof(cf32_t)` bytes for N-tap filter

The fused kernel handles history internally:
1. For first tile: reads from persistent history buffers
2. For subsequent tiles: re-deinterleaves overlap from input
3. After processing: writes only tail samples back to history

## License

Copyright (C) 2025 Geon Technologies, LLC

This component is part of composite-comps, licensed under the GNU Lesser General Public License v3.0 or later. See LICENSE file for details.
