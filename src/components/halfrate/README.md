# Halfrate Decimating Filter Component

High-performance polyphase half-band decimating filter optimized for single-core throughput using SIMD vectorization.

## Overview

The `halfrate` component implements a polyphase half-band FIR filter that decimates complex input samples by a factor of 2. It features:

- **High throughput**: 130+ MSPS sustained on a single core
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
- `filter_length`: Total FIR taps (must be odd, e.g., 15, 31, 63)
- `window_type`: Window function for coefficient generation
  - `"blackman_harris"` (default) - Superior stopband rejection (~92 dB)
  - `"hamming"` - Good passband flatness, moderate rejection (~53 dB)

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

| ISA     | Width | Throughput       |
|---------|-------|------------------|
| AVX-512 | 8 pairs/iter | 872-5,403 MSPS |
| AVX2    | 4 pairs/iter | ~400-2,500 MSPS |
| Scalar  | 1 pair/iter  | ~100-200 MSPS |

#### 2. `halfband_filter_vertical`
Computes FIR filter on even branch + delayed odd branch in a single pass.

| ISA     | Width | Throughput       |
|---------|-------|------------------|
| AVX-512 | 32 outputs/iter | 537-877 MSPS |
| AVX2    | 16 outputs/iter | ~250-400 MSPS |
| Scalar  | 1 output/iter   | ~50-100 MSPS |

**Multi-Function Versioning**: The compiler generates all three versions. At runtime, the CPU automatically dispatches to the best available implementation based on feature flags.

## Usage

### Ports

- **Input**: `data_in` - `mutable_buffer<std::complex<float>>`
- **Output**: `data_out` - `immutable_buffer<std::complex<float>>`

### Properties

```cpp
// Filter configuration
filter_length: unsigned int = 15     // Total FIR taps (odd number)
window_type: string = "blackman_harris"  // Window function

// Runtime info (read-only)
num_coefficients: unsigned int       // Polyphase FIR taps
center_tap: float                    // Polyphase center tap value
delay_offset: unsigned int           // Group delay samples
```

### Example Configuration

```cpp
auto decimator = std::make_shared<halfrate>("my_decimator");

// Configure filter
decimator->set_property("filter_length", 31);
decimator->set_property("window_type", "blackman_harris");

// Apply configuration
decimator->property_change_handler();

// Connect ports and process...
```

### Typical Filter Lengths

| Length | Polyphase Taps | Stopband (BH) | Use Case |
|--------|----------------|---------------|----------|
| 15     | 7              | ~60 dB        | Low latency, moderate rejection |
| 31     | 15             | ~80 dB        | Balanced performance |
| 63     | 31             | ~100 dB       | High rejection, more compute |
| 127    | 63             | ~120 dB       | Extreme specs |

## Performance Characteristics

### Throughput Benchmarks

Measured on single core (Intel Xeon with AVX-512):

| Block Size | End-to-End Throughput |
|------------|-----------------------|
| 1K         | 168 MSPS              |
| 16K        | 142 MSPS              |
| 256K       | 131 MSPS (sustained)  |
| 1M         | 134 MSPS              |

### Scaling Considerations

**Best performance when**:
- Block size ≥ 16K samples (amortizes overhead)
- Data fits in L3 cache (reduces memory bandwidth pressure)
- Input rate allows sustained processing (avoid bursty traffic)

**Bottlenecks**:
- Memory bandwidth (large blocks)
- Buffer allocation overhead (small blocks)
- History management (memcpy ~10-15% overhead)

### Optimization Tips

1. **Use larger block sizes**: 64K-256K samples optimal for throughput
2. **Pre-allocate buffers**: Avoid malloc in hot path
3. **Enable AVX-512**: Ensure `-mavx512f` compile flag
4. **Pin thread to core**: Reduces context switch overhead

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
cmake --build build --target kernel_tests halfrate_integration_tests performance_benchmarks

# Run unit tests
./build/src/components/halfrate/tests/kernel_tests

# Run integration tests
./build/src/components/halfrate/tests/halfrate_integration_tests

# Run performance benchmarks
./build/src/components/halfrate/tests/performance_benchmarks
```

### Compile Requirements

- **C++23**: Uses `std::expected`, trailing return types
- **SIMD flags**: `-mavx512f -mavx512bw -mavx512vl -mavx512dq` (for AVX-512 support)
- **Optimization**: `-O3 -march=native` recommended for benchmarks

The component will work without AVX-512 (falls back to AVX2 or scalar), but compile with SIMD flags for best performance.

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

3. **`performance_benchmarks.cpp`**
   - Throughput measurements (kernel and component level)
   - Cache behavior analysis
   - Scaling tests (1K → 1M samples)

### Running Tests

```bash
# All tests via CTest
cd build && ctest

# Individual test suites
./build/src/components/halfrate/tests/kernel_tests
./build/src/components/halfrate/tests/halfrate_integration_tests

# Benchmarks (run with high sample count for stable results)
./build/src/components/halfrate/tests/performance_benchmarks --benchmark-samples 100
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
    ├── kernel_tests.cpp
    ├── halfrate_integration_tests.cpp
    ├── performance_benchmarks.cpp
    ├── mfv_test.cpp       # MFV verification utility
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

## Correctness Validation

### Frequency Response

The filter achieves:
- **Passband** (0 to Fs/8): <1 dB ripple, >90% energy preservation
- **Transition** (Fs/8 to 3Fs/8): Smooth rolloff
- **Stopband** (3Fs/8 to Fs/2): >50 dB rejection (Hamming), >80 dB (Blackman-Harris)

Verified via tone injection tests in `halfrate_integration_tests.cpp`:
- Low-frequency tone (Fs/8): Energy preserved
- High-frequency tone (0.4·Fs): Attenuated to <5%

### Edge Cases Tested

- DC signal (f=0): Passes through with correct gain
- Impulse response: Verifies FIR coefficients
- Nyquist tone (f=0.5·Fs): Heavily attenuated
- Zero input: Produces zero output
- Streaming: No discontinuities across block boundaries

## References

- **Half-band filters**: Multirate Signal Processing, Crochiere & Rabiner
- **Polyphase decomposition**: Chapter 11, "Multirate Digital Signal Processing"
- **SIMD optimization**: Intel Intrinsics Guide (software.intel.com/intrinsics)
- **Window functions**: Harris, F.J. "On the Use of Windows for Harmonic Analysis with the DFT"

## License

Copyright (C) 2025 Geon Technologies, LLC

This component is part of composite-comps, licensed under the GNU Lesser General Public License v3.0 or later. See LICENSE file for details.
