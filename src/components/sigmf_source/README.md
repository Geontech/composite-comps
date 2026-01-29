# sigmf_source

A high-performance file source component that reads SigMF files or raw binary data files with rate-controlled playback. Uses memory-mapped I/O for zero-copy streaming at high sample rates.

## Features

- **SigMF Format Support**: Automatically parses `.sigmf-meta` JSON files for sample rate, center frequency, and data format
- **Raw Binary File Support**: Works with arbitrary binary files when metadata is provided via overrides
- **Zero-Copy Streaming**: Uses `mmap()` for efficient file reading without memory copies
- **Rate Control**: Accurate sample-rate-based playback timing with jitter compensation
- **Automatic Endianness Conversion**: In-place byte swapping for big-endian files on little-endian hosts
- **Looping**: Seamless file looping for continuous playback
- **Flexible Timestamps**: Sample-based or wall-clock timestamp modes

## Output Port

| Port | Type | Description |
|------|------|-------------|
| `data_out` | `immutable_buffer<std::byte>` | Raw sample data with format metadata |

The output metadata includes:
- `sample_rate` - Sample rate in Hz
- `center_frequency` - Center frequency in Hz
- `bandwidth` - Signal bandwidth in Hz
- `format` - Data format (complex/real, type, bit width, endianness)
- `stream_id` - User-configurable stream identifier

## Properties

### Core Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `file_path` | string | `""` | Path to the data file (see [File Path Handling](#file-path-handling)) |
| `streaming` | bool | `false` | Enable/disable streaming. Set to `true` after configuring other properties to start playback |
| `chunk_samples` | size_t | `8192` | Number of samples per output buffer |
| `loop` | bool | `false` | Loop back to file start when EOF is reached |
| `stream_id` | uint32_t | `0` | Identifier included in output metadata annotations |

### Rate Control Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `rate_control` | bool | `true` | Enable sample-rate-based timing. When `false`, streams as fast as possible |
| `max_sample_rate` | double | `-1.0` | Maximum playback rate in Hz. Set to `-1` for no limit (use file's native rate) |

### Timestamp Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `wallclock_timestamps` | bool | `false` | Timestamp mode: `false` = sample-based (timestamps derived from sample count), `true` = wall-clock (timestamps from real elapsed time) |

### Metadata Overrides

The `overrides` struct allows manual specification or override of metadata values. These take precedence over values read from the `.sigmf-meta` file.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `overrides.sample_rate` | double (optional) | unset | Override sample rate in Hz |
| `overrides.center_frequency` | double (optional) | unset | Override center frequency in Hz |
| `overrides.bandwidth` | double (optional) | unset | Override bandwidth in Hz (defaults to sample_rate if unset) |
| `overrides.datatype` | string (optional) | unset | Override data format string (see [Datatype Format](#datatype-format)) |

## File Path Handling

The `file_path` property accepts several formats:

| Input | Meta File Searched | Data File Used |
|-------|-------------------|----------------|
| `/path/to/file` | `/path/to/file.sigmf-meta` | `/path/to/file.sigmf-data` |
| `/path/to/file.sigmf-meta` | `/path/to/file.sigmf-meta` | `/path/to/file.sigmf-data` |
| `/path/to/file.sigmf-data` | `/path/to/file.sigmf-meta` | `/path/to/file.sigmf-data` |
| `/path/to/recording.bin` (exists) | `/path/to/recording.bin.sigmf-meta` | `/path/to/recording.bin` |

When the file exists as-is (without `.sigmf-` extension), it's used directly as the data file. This allows streaming arbitrary binary files when combined with the `overrides` properties.

## Datatype Format

The datatype string follows the SigMF specification:

```
[r|c][f|i|u]{bitwidth}[_le|_be]
```

| Component | Values | Description |
|-----------|--------|-------------|
| Complex/Real | `c`, `r` | `c` = complex (I/Q pairs), `r` = real |
| Data Type | `f`, `i`, `u` | `f` = float, `i` = signed int, `u` = unsigned int |
| Bit Width | `8`, `16`, `32`, `64` | Bits per component |
| Endianness | `_le`, `_be` | Little-endian or big-endian (defaults to `_le`) |

**Examples:**
- `cf32_le` - Complex float32, little-endian (default)
- `ci16_le` - Complex signed int16, little-endian
- `cu8` - Complex unsigned int8 (endianness irrelevant for 8-bit)
- `rf32_be` - Real float32, big-endian

## Usage Examples

### Basic SigMF File Playback

```json
{
    "id": "source",
    "library": "libsigmf_source.so",
    "properties": {
        "file_path": "/data/recording.sigmf-data",
        "streaming": true
    }
}
```

### Rate-Limited Playback with Looping

```json
{
    "id": "source",
    "library": "libsigmf_source.so",
    "properties": {
        "file_path": "/data/recording",
        "chunk_samples": 4096,
        "loop": true,
        "rate_control": true,
        "streaming": true
    }
}
```

### Throttled Playback (Lower Than Native Rate)

```json
{
    "id": "source",
    "library": "libsigmf_source.so",
    "properties": {
        "file_path": "/data/recording",
        "rate_control": true,
        "max_sample_rate": 1000000.0,
        "streaming": true
    }
}
```

### Raw Binary File (No Metadata File)

When you have a raw binary file without a `.sigmf-meta` file, use overrides to specify the format:

```json
{
    "id": "source",
    "library": "libsigmf_source.so",
    "properties": {
        "file_path": "/data/capture.bin",
        "overrides": {
            "sample_rate": 10000000.0,
            "center_frequency": 915000000.0,
            "datatype": "ci16_le"
        },
        "streaming": true
    }
}
```

### Override Metadata from SigMF File

Override specific values while using the rest from the metadata file:

```json
{
    "id": "source",
    "library": "libsigmf_source.so",
    "properties": {
        "file_path": "/data/recording.sigmf-data",
        "overrides": {
            "center_frequency": 100000000.0
        },
        "streaming": true
    }
}
```

### Fast-as-Possible Streaming (No Rate Control)

```json
{
    "id": "source",
    "library": "libsigmf_source.so",
    "properties": {
        "file_path": "/data/recording",
        "rate_control": false,
        "streaming": true
    }
}
```

## Runtime Control

All properties are runtime-configurable. Common patterns:

**Start/Stop Streaming:**
```bash
# Start
curl -X PUT http://localhost:8080/components/source/properties/streaming -d 'true'

# Stop
curl -X PUT http://localhost:8080/components/source/properties/streaming -d 'false'
```

**Switch Files:**
```bash
# Stop, change file, restart
curl -X PUT http://localhost:8080/components/source/properties/streaming -d 'false'
curl -X PUT http://localhost:8080/components/source/properties/file_path -d '"/data/new_recording"'
curl -X PUT http://localhost:8080/components/source/properties/streaming -d 'true'
```

## Implementation Notes

- **Memory Mapping**: Files are memory-mapped with `MAP_PRIVATE` when endianness conversion is needed, allowing in-place byte swapping without modifying the original file
- **Rate Control Accuracy**: Uses wall-clock tracking with catch-up logic to maintain long-term sample rate accuracy despite sleep jitter
- **Chunk Bundling**: At very high sample rates, multiple chunks are bundled per wakeup to avoid excessive thread scheduling overhead (minimum 500μs wakeup interval)
- **Metadata Events**: Metadata is sent to downstream components at startup and when looping back to file start
