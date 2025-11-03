# Exponential Smoothing Component

A high-performance exponential smoothing filter for real-time signal processing, optimized with SIMD acceleration (AVX2/AVX-512).

## Overview

The `exp_smooth` component applies exponential smoothing to streaming data, commonly used for:
- Power spectral density (PSD) averaging
- Noise reduction in time-series data
- Trend estimation in signal processing

The filter equation is:
```
y[n] = α × x[n] + (1 - α) × y[n-1]
```

Where `α` (alpha) controls the smoothing factor.

## Features

- **SIMD Acceleration**: Runtime function multiversioning automatically selects:
  - AVX-512 (512-bit, 16 floats / 8 doubles per operation)
  - AVX2 (256-bit, 8 floats / 4 doubles per operation)
  - Scalar fallback (portable, works on any CPU)
- **Alignment-Aware**: Checks buffer alignment and uses optimal instruction set
- **NaN/Inf Handling**: Replaces non-finite values with previous valid sample
- **Runtime Configurable**: Adjust smoothing parameters without restarting

## Configuration

### Properties

| Property | Type | Configurability | Description |
|----------|------|-----------------|-------------|
| `num_averages` | `uint32` | RUNTIME | Number of averages to reach 98% of steady-state value. Set to 0 to disable smoothing (pass-through mode). |

### Alpha Calculation

When `num_averages` is set, alpha is computed as:
```cpp
α = 1 - 10^(log₁₀(1 - 0.98) / num_averages)
```

This ensures the filter reaches 98% of its steady-state value after `num_averages` samples.

**Example values:**
- `num_averages = 10`: α ≈ 0.316 (fast response)
- `num_averages = 100`: α ≈ 0.039 (heavy smoothing)
- `num_averages = 0`: Smoothing disabled (pass-through)

### Ports

| Port | Direction | Type | Description |
|------|-----------|------|-------------|
| `data_in` | Input | `mutable_buffer<T>` | Input data stream (float or double) |
| `data_out` | Output | `mutable_buffer<T>` | Smoothed output stream |

**Note**: The first sample is used as the initial filter state and is not emitted. Smoothed output begins with the second sample.

## JSON Configuration Example

```json
{
    "components": [
        {
            "name": "exp_smooth",
            "id": "smoother_f32",
            "create_arg": "f32",
            "properties": {
                "num_averages": 50
            }
        }
    ]
}
```

## Factory Types

The component supports two data types via the factory `create()` function:

- `"f32"`: Single-precision floating-point (`float`)
- `"f64"`: Double-precision floating-point (`double`)

> Specify the type when configuring the component via `"create_arg"`: The framework calls `create("f32")` or `create("f64")`.


## Performance Characteristics

### SIMD Selection Logic

The component automatically selects the best SIMD implementation based on:

1. **CPU Capabilities** (via GCC function multiversioning):
   - AVX-512 > AVX2 > Scalar

2. **Memory Alignment**:
   - AVX-512: Requires 64-byte alignment
   - AVX2: Requires 32-byte alignment
   - Scalar: No alignment requirement

3. **Fallback Strategy**:
   - AVX-512 target: If not 64-byte aligned → try AVX2 → scalar
   - AVX2 target: If not 32-byte aligned → scalar
   - Scalar target: Always works


## Implementation Details

### Non-Finite Value Handling

The filter sanitizes non-finite values (NaN, +Inf, -Inf) by replacing them with the previous valid sample before applying smoothing. This prevents contamination of the filter state.

### Remainder Handling

SIMD loops process data in chunks (8/16 elements for AVX2/AVX-512). Remainder elements that don't fit in a full SIMD vector are processed using scalar code, ensuring all data is processed correctly.

### Memory Alignment

For optimal performance:
- Use `composite::make_aligned_buffer<T>(32, size)` for AVX2
- Use `composite::make_aligned_buffer<T>(64, size)` for AVX-512

The framework's buffer system supports aligned allocations. If buffers are not aligned, the component automatically falls back to slower (but safe) implementations.

## Build Requirements

- C++20 compiler with GCC function multiversioning support
- No additional dependencies beyond the composite framework

## Related Components

- **psd**: Generates power spectral density data that often feeds into exp_smooth
- **fft**: Computes FFTs whose magnitude can be smoothed by this component

## References

- [Exponential Smoothing (Wikipedia)](https://en.wikipedia.org/wiki/Exponential_smoothing)
