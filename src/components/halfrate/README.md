# Halfrate Decimating Filter Component

High-performance polyphase half-band decimating filter optimized for single-core throughput using SIMD vectorization.

## Overview

The `halfrate` component implements a polyphase half-band FIR filter that decimates complex input samples by a factor of 2. It features:

- **Multi-platform SIMD**: AVX-512, AVX2, and scalar implementations via Multi-Function Versioning (MFV)
- **Configurable filter design**: Adjustable filter length and window function
- **Zero-copy architecture**: Efficient buffer management using composite framework

## Signal Processing Details

### Half-Band Filter Theory

A half-band filter has:
- **Cutoff frequency**: Fs/4 (quarter of sample rate)
- **Decimation factor**: 2 (output rate = input rate / 2)
- **Symmetric impulse response**: Every other coefficient is zero (except center tap)

This component uses a **polyphase decomposition** that exploits the zero-valued coefficients:
- **Even branch**: Non-zero FIR taps applied to even-indexed input samples
- **Odd branch**: Center tap (typically 0.5) applied to odd-indexed samples with group delay compensation

The output combines both branches: `output[n] = FIR(even[n]) + center_tap * odd[n - delay]`

### Filter Design Parameters

Configured via properties:
- `filter_semi_length`: Number of non-zero taps on one side (e.g., 3, 7, 15). Total filter length = `2 × filter_semi_length + 1`
- `window`: Window function for coefficient generation
  - `"HAMMING"` (default) - Good passband flatness, moderate rejection (~53 dB)
  - `"BLACKMAN_HARRIS"` - Better stopband rejection (~92 dB)

The component automatically:
1. Generates coefficients using windowed-sinc method: `h[k] = sinc(0.5π·k) · window[k]`
2. Extracts polyphase branches (even taps become FIR coefficients, center tap becomes scalar)
3. Computes group delay offset for odd branch alignment

## Architecture

### Processing Pipeline

```
Input (interleaved I/Q)
    ↓
┌─────────────────────┐
│  De-interleave      │  Split interleaved complex samples
│  (SIMD kernel)      │  → even stream, odd stream
└─────────────────────┘
    ↓           ↓
┌─────────┐ ┌─────────┐
│  Even   │ │   Odd   │
│ History │ │ History │  Maintain filter state
│ Buffer  │ │ Buffer  │
└─────────┘ └─────────┘
    ↓           ↓
┌─────────────────────┐
│ Vertical Half-band  │  FIR on even + delayed odd
│ Filter (SIMD)       │  Single fused kernel
└─────────────────────┘
    ↓
Output (decimated by 2)
```

### SIMD Kernels

Two high-performance kernels with MFV support:

#### 1. `deinterleave_block`
Separates interleaved complex samples into contiguous buffers.

| ISA     | Width |
|---------|-------|
| AVX-512 | 16 pairs/iter |
| AVX2    | 4 pairs/iter |
| Scalar  | 1 pair/iter  |

#### 2. `halfband_filter_vertical`
Computes FIR filter on even branch + delayed odd branch in a single pass.

| ISA     | Width |
|---------|-------|
| AVX-512 | 32 outputs/iter |
| AVX2    | 16 outputs/iter |
| Scalar  | 1 output/iter   |

**Multi-Function Versioning**: The compiler generates all three versions. At runtime, the CPU automatically dispatches to the best available implementation based on feature flags.

## Usage

### Ports

- **Input**: `data_in` - `mutable_buffer<std::complex<float>>`
- **Output**: `data_out` - `immutable_buffer<std::complex<float>>`

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `filter_semi_length` | `uint32_t` | `3` | Half-band filter semi-length (number of non-zero taps on one side). Total filter length = `2 × filter_semi_length + 1`. Example: semi_length=3 → 7 taps total |
| `window` | `string` | `"HAMMING"` | Window function for coefficient generation. Valid values: `"HAMMING"` or `"BLACKMAN_HARRIS"` |

## Performance Characteristics

### Scaling Considerations

**Best performance when**:
- Block size ≥ 16K samples (amortizes overhead)
- Data fits in L3 cache (reduces memory bandwidth pressure)
- Input rate allows sustained processing

**Bottlenecks**:
- Memory bandwidth (large blocks)
- Buffer allocation overhead (small blocks)
- History management (memcpy ~10-15% overhead)

### Optimization Tips

1. **Use larger block sizes**: 64K-256K samples optimal for throughput
2. **Pre-allocate buffers**: Avoid malloc in hot path
3. **Pin thread to core**: Reduces context switch overhead

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
cmake --build build --target kernel_tests halfrate_integration_tests kernel_benchmarks

# Run unit tests
./build/src/components/halfrate/tests/kernel_tests

# Run integration tests
./build/src/components/halfrate/tests/halfrate_integration_tests

# Run performance benchmarks (Google Benchmark)
./build/src/components/halfrate/tests/kernel_benchmarks
```

## Testing

### Test Coverage

Three comprehensive test suites:

1. **`kernel_tests.cpp`** (6542 assertions)
   - Unit tests for SIMD kernels
   - Validates against scalar reference implementations
   - Tests boundary conditions, alignment, large blocks

2. **`halfrate_integration_tests.cpp`** (2619 assertions)
   - End-to-end component behavior
   - Frequency response validation (passband, stopband)
   - Streaming tests with history management
   - Edge cases (DC, impulse, Nyquist)

3. **`kernel_benchmarks.cpp`** (Google Benchmark)
   - Throughput measurements (deinterleave and filter kernels)
   - Parameterized size/tap count sweeps
   - Cache behavior analysis (L1/L2/L3/DRAM)
   - Statistical reporting with repetitions

### Running Tests

```bash
# All tests via CTest
cd build && ctest

# Individual test suites
./build/src/components/halfrate/tests/kernel_tests
./build/src/components/halfrate/tests/halfrate_integration_tests

# Performance benchmarks (Google Benchmark)
./build/src/components/halfrate/tests/kernel_benchmarks

# Run specific benchmarks with repetitions
./build/src/components/halfrate/tests/kernel_benchmarks --benchmark_filter="Deinterleave" --benchmark_repetitions=10

# Export results to JSON
./build/src/components/halfrate/tests/kernel_benchmarks --benchmark_format=json --benchmark_out=results.json
```

## Implementation Details

### File Structure

```
halfrate/
├── component.hpp          # Component class definition
├── component.cpp          # Port setup, filter design, process() logic
├── kernels.hpp            # SIMD kernels with MFV (header-only)
├── CMakeLists.txt         # Build configuration
└── tests/
    ├── kernel_tests.cpp               # Catch2 unit tests for SIMD kernels
    ├── halfrate_integration_tests.cpp # Catch2 integration tests
    ├── kernel_benchmarks.cpp          # Google Benchmark performance tests
    └── CMakeLists.txt
```

### Key Design Decisions

1. **Polyphase decomposition**: Exploits half-band symmetry for 2× efficiency
2. **Vertical filtering**: Processes both branches in single kernel (better cache locality)
3. **In-place de-interleave**: Minimizes memory allocations
4. **Pre-sized buffers**: History buffers allocated once during init
5. **MFV over runtime dispatch**: Zero overhead, compiler-optimized

### History Management

The component maintains circular history buffers:
- **Even history**: Size = `num_outputs + num_taps`
- **Odd history**: Size = `num_outputs + delay_offset`

On each `process()` call:
1. Copy tail of previous block to head of history buffer (maintains FIR state)
2. Append new samples
3. Run filter kernel over extended buffer
4. Output is always `input_size / 2` samples

## License

Copyright (C) 2025 Geon Technologies, LLC

This component is part of composite-comps, licensed under the GNU Lesser General Public License v3.0 or later. See LICENSE file for details.
