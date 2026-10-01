 sigmf_source

A high-performance file source component that reads SigMF files, MIDAS Blue files, or raw binary data files with rate-controlled playback. Uses memory-mapped I/O for zero-copy streaming at high sample rates.

## Features

- **SigMF Format Support**: Automatically parses `.sigmf-meta` JSON files for sample rate, center frequency, and data format
- **MIDAS Blue File Support**: Reads `.blue` files with automatic header parsing for format and sample rate extraction
- **Raw Binary File Support**: Works with arbitrary binary files when metadata is provided via overrides
- **Rate Control**: Accurate sample-rate-based playback timing with jitter compensation
- **Automatic Endianness Conversion**: In-place byte swapping for big-endian files on little-endian hosts
- **Looping**: Seamless file looping for continuous playback
- **Flexible Timestamps**: Sample-based or wall-clock timestamp modes
- **Auto-Detection**: Automatically detects file type from extension or magic bytes
- **Multi-File Streaming**: Supports the ability to stream multiple files concurrently

## Output Port

| Port | Type | Description |
|------|------|-------------|
| `data_out` | `immutable_buffer<std::byte>` | Raw sample data with format metadata |

The output metadata for each file includes:
- `sample_rate` - Sample rate in Hz
- `center_frequency` - Center frequency in Hz
- `bandwidth` - Signal bandwidth in Hz
- `format` - Data format (complex/real, type, bit width, endianness)
- `stream_id` - User-configurable stream identifier
- `destination_ip` - Egress group/host address for this stream, from the spec's `destination_ip`. Omitted entirely when the spec does not set one, so that `udp_sink` falls back to its `default_dest_ip`
- `destination_port` - Egress port for this stream. Shared by all streams (see [Stream Addressing](#stream-addressing))

## Properties

### Core Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `files` | array of file_spec | `[]` | List of files to play. Each entry is independently configurable. See [File Spec Fields](#file-spec-fields) below |
| `streaming` | bool | `false` | Master enable/disable. When false, files can be configured but no data is emitted; mmaps are released. When true, per-file playback begins |
| `chunk_samples` | size_t | `8192` | Number of samples per output buffer. Applies to all files |
| `destination_port` | uint32_t | `5000` | Default egress port, used for any file whose spec does not set its own `destination_port`. Not offset per stream — see [Stream Addressing](#stream-addressing) |

### File Spec Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | `""` | Path to streamed data file. Must be unique within files. See [File Path Handling](#file-path-handling) below. |
| `stream_id` | int32_t | `0` | Identifier included in output metadata. Must be unique within `files`. |
| `destination_ip` | string | `""` | Egress group/host address for this file. Empty emits no annotation, so `udp_sink`'s `default_dest_ip` applies. The resolved `(destination_ip, destination_port)` pair must be unique within `files`. |
| `destination_port` | uint32_t | `0` | Egress port for this file. `0` falls back to the component-level `destination_port`. |
| `loop` | bool | `false` | If `true`, wrap back to file start when EOF is reached. If `false`, the stream stops emitting after EOF (other files in `files` are unaffected). |
| `rate_control` | bool | `true` | Enable sample-rate-based timing. When false, streams as fast as possible. |
| `max_sample_rate` | double | `-1` | Maximum playback rate in Hz. Set to -1 to use file's native rate. |
| `wallclock_timestamps` | bool | `false` | Timestamp mode: false = sample-based, true = wall-clock from real elapsed time. |
| `overrides` | sigmf_overrides | `{}` | Per-file metadata overrides. See [Metadata Overrides](#metadata-overrides) below. |

### Metadata Overrides

The `overrides` struct allows manual specification or override of metadata values. These take precedence over values read from the `.sigmf-meta` file.

**Runtime-mutable** fields can be edited via PATCH on a live file; playback continues without interruption.
**Load-time-only** fields take effect only when the file is first loaded. Editing them on a live spec is silently ignored; to apply a change, remove the file (`files[i] = null`) and re-add with the new value.

| Property | Type | Default | Runtime Mutable | Description |
|----------|------|---------|-----------------|------------|
| `overrides.sample_rate` | double (optional) | unset | yes | Override sample rate in Hz |
| `overrides.center_frequency` | double (optional) | unset | yes | Override center frequency in Hz |
| `overrides.bandwidth` | double (optional) | unset | yes | Override bandwidth in Hz (defaults to sample_rate if unset) |
| `overrides.datatype` | string (optional) | unset | no | Override data format string (see [Datatype Format](#datatype-format)) |
| `overrides.filetype` | string (optional) | unset | no | Force file type: `"raw"` or `"bluefile-1000"` (auto-detected if unset) |

## Stream Addressing

Each file's egress destination is taken **verbatim** from its spec. The component
performs no address arithmetic — it does not derive a destination from
`stream_id`. Address allocation belongs to whatever service drives the `files`
property, which is also the component that has to report the destination to its
own clients; computing it here as well would mean two implementations of one rule
that can silently diverge.

Concurrent streams are separated by **group address, with the port held
constant**:

| stream_id | destination_ip | destination_port |
|---|---|---|
| 1 | `239.0.10.3` | 5000 |
| 2 | `239.0.10.4` | 5000 |
| 3 | `239.0.10.5` | 5000 |

Do **not** separate streams by port on a shared group. IGMP membership is per
group and has no notion of port, so a receiver that joins the shared group to get
one stream is forwarded *all* of them and discards the rest after they have
already consumed fabric and NIC bandwidth. One group per stream lets the switch
filter, so a subscriber receives only what it joined.

Receivers should `bind()` to the group address rather than `INADDR_ANY`. Linux
delivers a multicast datagram to any socket bound to `0.0.0.0:<port>` that has
joined *any* group on that port, so two receivers co-located on one host, in
different groups but both bound to the wildcard address, will each see both
streams. This does not affect the fabric-level filtering above, only same-host
separation.

## Uniqueness Requirements

**Paths must be unique.** The component uses each spec's `path` as its runtime identity.
Duplicate paths are marked `load_error` and do not stream.

**Stream IDs must be unique.** Duplicate `stream_id` values are also marked `load_error`.
`stream_id` is the VITA-49 stream identity that downstream components key on, so it
must be unique regardless of where the stream is addressed.

**Destinations must be unique.** Two specs resolving to the same
`(destination_ip, destination_port)` pair interleave on the wire, which no
receiver can separate, so duplicates are marked `load_error`. Specs that leave
`destination_ip` empty all share `udp_sink`'s single default group, so at most
one spec may omit it.

Conflict resolution is "first-wins" — the first occurrence in the list keeps its slot, subsequent duplicates are errored. To recover:

1. Remove the offending spec: `files[i] = null`
2. Re-add with corrected values

The error is visible in logs as a `WARN`-level message identifying the conflicting spec.

## Error Handling

Per-file errors are reported via `load_error` state rather than crashing the component. A file is marked `load_error` when:

- The data file doesn't exist or cannot be opened
- The `.sigmf-meta` file is malformed JSON
- The data format is invalid (zero bytes per sample, malformed datatype)
- The path conflicts with another spec in the list
- The `stream_id` conflicts with another spec in the list
- The resolved `(destination_ip, destination_port)` conflicts with another spec in the list
- An endianness swap fails on remap (rare)

Errored files do not emit data and do not affect siblings — other files in `files` continue streaming normally.

To recover from an error: remove the spec (`files[i] = null`) and re-add with corrected configuration.

## Streaming Gate

The `streaming` property controls emission across all configured files:

- `streaming = false` (default): files can be configured, validated, and held in `m_runtime`, but no data is emitted. Memory-mapped regions are released to free file handles.
- `streaming = true`: per-file playback begins. Memory-mapped regions are re-acquired. The metrics thread starts logging per-stream health.

Files can be added, edited, or removed regardless of the streaming state. Configuration changes take effect immediately; emission status is gated separately.

When streaming transitions false → true, each non-errored file restarts playback from offset 0.

## File Path Handling

The `path` field in `file_spec` accepts several formats:

### SigMF and Raw Files

| Input | Meta File Searched | Data File Used |
|-------|-------------------|----------------|
| `/path/to/file` | `/path/to/file.sigmf-meta` | `/path/to/file.sigmf-data` |
| `/path/to/file.sigmf-meta` | `/path/to/file.sigmf-meta` | `/path/to/file.sigmf-data` |
| `/path/to/file.sigmf-data` | `/path/to/file.sigmf-meta` | `/path/to/file.sigmf-data` |
| `/path/to/recording.bin` (exists) | `/path/to/recording.bin.sigmf-meta` | `/path/to/recording.bin` |

When the file exists as-is (without `.sigmf-` extension), it's used directly as the data file. This allows streaming arbitrary binary files when combined with the `overrides` properties.

### MIDAS Blue Files

MIDAS Blue files (`.blue`) are automatically detected by file extension or magic bytes ("BLUE" header).

| Input | Data File Used |
|-------|----------------|
| `/path/to/file.blue` | `/path/to/file.blue` |
| `/path/to/file` (with "BLUE" magic) | `/path/to/file` |

The component reads the 512-byte Header Control Block to extract:
- **Format code** (e.g., "CF" = complex float, "CI" = complex int16)
- **Data offset** and size
- **Sample rate** from xDelta (if available)
- **Endianness** information

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

## MIDAS Blue Format Codes

MIDAS Blue files use a 2-character format code in the header: `[Rank][Type]`

| Rank Code | Description |
|-----------|-------------|
| `C` | Complex (I/Q pairs) |
| `R` | Real |
| `S` | Scalar |

| Type Code | Description | Size |
|-----------|-------------|------|
| `B` | Signed byte | 1 byte |
| `O` | Unsigned byte | 1 byte |
| `I` | Signed int16 | 2 bytes |
| `U` | Unsigned int16 | 2 bytes |
| `L` | Signed int32 | 4 bytes |
| `V` | Unsigned int32 | 4 bytes |
| `F` | Float32 | 4 bytes |
| `D` | Float64 (double) | 8 bytes |
| `X` | Signed int64 | 8 bytes |

**Examples:**
- `CF` - Complex float32 (most common)
- `CI` - Complex signed int16
- `SF` - Scalar (real) float32

## Usage Examples

### Basic SigMF File Playback

```json
{
    "id": "source",
    "library": "libsigmf_source.so",
    "properties": {
        "files": [
          {
            "path": "/data/recording.sigmf-data",
            "stream_id": 0
          }
        ],
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
        "files": [
          {
            "path": "/data/recording.sigmf-data",
            "stream_id": 0,
            "loop": true,
            "rate_control": true
          }
        ],
        "chunk_samples": 4096,
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
        "files": [
          {
            "path": "/data/recording.sigmf-data",
            "stream_id": 0,
            "rate_control": true,
            "max_sample_rate": 1000000.0
          }
        ],
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
        "files": [
          {
            "path": "/data/capture.bin",
            "stream_id": 0,
            "overrides": {
              "sample_rate": 10000000.0,
              "center_frequency": 915000000.0,
              "datatype": "ci16_le"
            }
          }
        ],
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
        "files": [
          {
            "path": "/data/recording.sigmf-data",
            "stream_id": 0,
            "overrides": {
              "center_frequency": 100000000.0
            }
          }
        ],
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
        "files": [
          {
            "path": "/data/recording",
            "stream_id": 0,
            "rate_control": false
          }
        ],
        "streaming": true
    }
}
```

### MIDAS Blue File (Auto-Detected)

MIDAS Blue files are automatically detected by extension or magic bytes:

```json
{
    "id": "source",
    "library": "libsigmf_source.so",
    "properties": {
        "files": [
          {
            "path": "/data/capture.blue",
            "stream_id": 0
          }
        ],
        "streaming": true
    }
}
```

### MIDAS Blue File with Overrides

Override sample rate or other metadata from the Blue header:

```json
{
    "id": "source",
    "library": "libsigmf_source.so",
    "properties": {
        "files": [
          {
            "path": "/data/capture.blue",
            "stream_id": 0,
            "overrides": {
              "sample_rate": 20000000.0,
              "center_frequency": 2400000000.0
            }
          }
        ],
        "streaming": true
    }
}
```

### Force File Type

Use `overrides.filetype` to explicitly specify the file type on load:

```json
{
    "id": "source",
    "library": "libsigmf_source.so",
    "properties": {
        "files": [
          {
            "path": "/data/recording.dat",
            "stream_id": 0,
            "overrides": {
              "filetype": "bluefile-1000"
            }
          }
        ],
        "streaming": true
    }
}
```

### Multi-Stream Playback

Stream three files concurrently, each with independent configuration. Each gets
its own group address on the shared port, so a receiver joining one group is not
forwarded the other two:

```json
{
    "id": "source",
    "library": "libsigmf_source.so",
    "properties": {
        "files": [
            {
                "path": "/data/recording_a.sigmf-data",
                "stream_id": 0,
                "destination_ip": "239.0.10.3",
                "loop": true
            },
            {
                "path": "/data/recording_b.sigmf-data",
                "stream_id": 1,
                "destination_ip": "239.0.10.4",
                "loop": true,
                "max_sample_rate": 5000000.0
            },
            {
                "path": "/data/recording_c.sigmf-data",
                "stream_id": 2,
                "destination_ip": "239.0.10.5",
                "loop": false
            }
        ],
        "destination_port": 5000,
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

**Add File**
```bash
curl -X PUT http://localhost:8080/components/source/properties/files
  -d '{"files": [{"path": "/data/rec1.sigmf-data", "stream_id": 0}]'
```

**Remove File**
```bash
curl -X PUT http://localhost:8080/components/source/properties/files
  -d '{"files[0]: null}]'
```

**Edit File Properties**
```bash
# Toggle loop to false
curl -X PATCH http://localhost:8080/components/source/properties/files
  -d '{"files[0].loop: false}]'

# Toggle rate control to false
curl -X PATCH http://localhost:8080/components/source/properties/files
  -d '{"files[0].rate_control: false}]'
```

## Implementation Notes

- **Memory Mapping**: Files are memory-mapped with `MAP_PRIVATE` when endianness conversion is needed, allowing in-place byte swapping without modifying the original file
- **Rate Control Accuracy**: Uses wall-clock tracking with catch-up logic to maintain long-term sample rate accuracy despite sleep jitter
- **Chunk Bundling**: At very high sample rates, multiple chunks are bundled per wakeup to avoid excessive thread scheduling overhead (minimum 500μs wakeup interval)
- **Metadata Events**: Per-stream metadata is sent on the output port with every emitted chunk, ensuring downstream consumers always have current routing/format information even when files are added or modified mid-stream.
