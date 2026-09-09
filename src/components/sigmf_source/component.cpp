/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * SigMF file source component with rate-controlled playback.
 * Uses mmap for zero-copy file reading at high sample rates.
 */

#include "component.hpp"

#include <composite/core/register.hpp>
#include "blue/BlueFile.hpp"

#include <composite/properties/property_set.hpp>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <pthread.h>
#include <stdexcept>
#include <thread>

sigmf_source::sigmf_source(std::string_view id)
    // The base creates and registers the output port; passing the name keeps the
    // wire contract ("data_out") that existing graphs connect to.
    : base_t(id, "data_out") {
    using enum composite::properties::config_type;

    add_property("destination_port", m_destination_port, RUNTIME);
    
    // chunk_samples feeds calculate_timing (chunk_interval, chunks_per_wakeup) and
    // the per-chunk read size, so an edit has to re-derive timing. Without a
    // listener the new value applied to reads while timing stayed on the old one.
    add_property("chunk_samples", m_chunk_samples, RUNTIME)
        .on_change([this](const composite::properties::json&) { on_files_changed(); });

    // File spec - starts empty, append files on every allocation added
    // One listener now: 0.5.2 delivers a JSON diff rather than an element index, and
    // the handler re-derives its own path-keyed diff from m_files regardless.
    add_property("files", m_files, RUNTIME)
        .on_change([this](const composite::properties::json&) { on_files_changed(); });

    // Streaming control - starts disabled, set streaming=true when ready to stream
    add_property("streaming", m_streaming, RUNTIME)
        .on_change([this](const composite::properties::json&) { on_streaming_changed(); });
}

auto sigmf_source::initialize() -> void {
    // Component starts with streaming disabled - configure properties then set streaming=true
    logger()->info("sigmf_source initialized (streaming disabled, set streaming=true after configuring file_path)");
}

auto sigmf_source::on_worker_start() -> void {
    // Required: the base drops any packet held from a prior run, so a restart does
    // not replay a stale buffer ahead of fresh data.
    base_t::on_worker_start();

    logger()->info("sigmf_source starting");

    // Always track start time (needed for wallclock_timestamps and rate_control)
    auto now = std::chrono::steady_clock::now();

    // Capture UTC epoch for absolute timestamps
    auto utc_now = std::chrono::system_clock::now();
    uint32_t utc_epoch = static_cast<uint32_t>(
        std::chrono::duration_cast<std::chrono::seconds>(utc_now.time_since_epoch()).count());

    // reconfigure runtime vector
    {
        std::unique_lock<std::shared_mutex> lock(m_runtime_mutex);
        for (auto& rt : m_runtime) {
            if (rt.load_error) continue;
            reset_runtime_for_playback(rt, now, utc_epoch);
            if (m_streaming) remap_runtime(rt);
        }
    }
    if (m_streaming) {
        spawn_metrics_thread();
    }
    m_next_file = 0;
}

auto sigmf_source::on_worker_stop() -> void {
    stop_metrics_thread();

    std::unique_lock<std::shared_mutex> lock(m_runtime_mutex);
    for (auto& rt : m_runtime) {
        rt.mmap.reset();
    }
}

auto sigmf_source::produce() -> produce_result {
    std::unique_lock<std::shared_mutex> lock(m_runtime_mutex);

    if (!m_streaming || m_runtime.empty()) {
        return produce_result::idle();
    }

    const auto now = std::chrono::steady_clock::now();
    const std::size_t n = m_runtime.size();

    // A file that could still yield data (now or later).
    bool any_live = false;
    // A file that loaded successfully and has run to its end. Distinguishing this
    // from "never worked" is what keeps a spec of only-bad files recoverable: it
    // idles for a corrected spec instead of finishing the component.
    bool any_completed = false;

    // Round-robin from where we stopped: one packet per call means a fixed scan
    // order would let stream 0 monopolise a saturated output.
    for (std::size_t k = 0; k < n; ++k) {
        const std::size_t i = (m_next_file + k) % n;
        auto& rt = m_runtime[i];

        if (rt.load_error) {
            continue;
        }
        if (rt.eof && !rt.loop) {
            any_completed = true;
            continue;
        }

        any_live = true;

        // Rate control is now purely "am I behind schedule?". Pacing needs no sleep
        // here: returning idle() parks the worker on the doorbell, and while we are
        // behind, the base calls produce() back-to-back so catch-up is immediate.
        if (rt.rate_control) {
            const auto elapsed = std::chrono::duration<double>(now - rt.start_time).count();
            const auto expected = static_cast<uint64_t>(elapsed * rt.effective_sample_rate);
            if (rt.samples_sent >= expected) {
                continue;
            }
        }

        auto r = next_chunk(rt);
        if (r.status == produce_status::data) {
            m_next_file = (i + 1) % n;
            return r;
        }
        // Hit its end on this call; let the loop consider the remaining files.
        if (rt.eof && !rt.loop) {
            any_completed = true;
        }
    }

    // Every loaded file has run out and none can restart: the stream is genuinely
    // over. done() makes the base send EOS (out-of-band, never gated by
    // backpressure) and FINISH, so the downstream graph completes instead of
    // idling forever -- which a plain component could not express.
    if (!any_live && any_completed) {
        return produce_result::done();
    }

    // Backpressure is deliberately absent from all of the above: the base holds an
    // unaccepted packet and re-sends it, so this never has to poll can_send().
    return produce_result::idle();
}

bool sigmf_source::on_files_changed() {
    logger()->debug("sigmf_source called on_files_changed with file_size={}", m_files.size());

    std::unique_lock<std::shared_mutex> lock(m_runtime_mutex);

    // Handle duplicate file paths
    std::unordered_set<std::string> seen_paths;
    std::unordered_set<std::string> path_conflicts;
    for (const auto& s : m_files) {
        if (!seen_paths.insert(s.path).second) {
            path_conflicts.insert(s.path);
            logger()->warn("sigmf_source: duplicate file_path '{}' in files list, "
                           "marking as load_error", s.path);
        }
    }

    // Handle duplicate stream ids. stream_id is the VITA-49 stream identity,
    // so it must be unique regardless of where the stream is addressed.
    std::unordered_set<int32_t> seen_stream_ids;
    std::unordered_set<std::string> stream_id_conflicts;
    for (const auto& s : m_files) {
        if (!seen_stream_ids.insert(s.stream_id).second) {
            stream_id_conflicts.insert(s.path);
            logger()->warn("sigmf_source: duplicate stream_id {} on path '{}', "
                           "marking this spec as load_error. To recover, send "
                           "files[i]=null and re-add with a unique stream_id.",
                           s.stream_id, s.path);
        }
    }

    // Handle duplicate egress destinations. Two specs resolving to the same
    // group and port interleave on the wire, which no receiver can separate.
    // Specs with no destination_ip share udp_sink's single default group, so
    // only one such spec can be addressed unambiguously.
    std::unordered_set<std::string> seen_destinations;
    std::unordered_set<std::string> destination_conflicts;
    for (const auto& s : m_files) {
        const auto port = s.destination_port != 0 ? s.destination_port : m_destination_port;
        const auto ip = s.destination_ip.empty() ? "<udp_sink default>" : s.destination_ip;
        if (!seen_destinations.insert(std::format("{}:{}", ip, port)).second) {
            destination_conflicts.insert(s.path);
            logger()->warn("sigmf_source: duplicate destination {}:{} on path '{}', "
                           "marking this spec as load_error. Streams are separated by "
                           "group address; assign each spec a unique destination_ip.",
                           ip, port, s.path);
        }
    }

    std::unordered_map<std::string, const struct_props::file_spec*> desired;
    for (const auto& s : m_files) desired.emplace(s.path, &s);

    std::size_t removed = 0;
    std::erase_if(m_runtime, [&](const file_runtime& rt) {
        bool gone = !desired.contains(rt.file_path);
        if (gone) ++removed;
        return gone;
    });

    std::size_t added_ok = 0;
    std::size_t added_err = 0;
    for (const auto& s : m_files) {
        // Handle new requests
        bool exists = std::ranges::any_of(m_runtime,
            [&](const file_runtime& rt){ return rt.file_path == s.path; });
        if (exists) continue;

        // Check for duplicate file names, stream ids, and destinations
        file_runtime rt;
        if (stream_id_conflicts.contains(s.path) || path_conflicts.contains(s.path)
            || destination_conflicts.contains(s.path)) {
            rt.file_path = s.path;
            rt.load_error = true;
        } else if (!load_file(s, rt)) {
            rt.file_path = s.path;
            rt.load_error = true;
        }
        if (rt.load_error) ++added_err;
        else ++added_ok;
        m_runtime.push_back(std::move(rt));
    }

    // Handle spec changes for existing entries
    for (auto& rt : m_runtime) {
        if (rt.load_error) continue;
        if (stream_id_conflicts.contains(rt.file_path) || path_conflicts.contains(rt.file_path)
            || destination_conflicts.contains(rt.file_path)) {
            rt.load_error = true;
            continue;
        }
        const auto* spec = desired.at(rt.file_path);
        refresh_runtime_from_spec(*spec, rt);
    }

    if (added_ok || added_err || removed) {
        logger()->info("sigmf_source: files reconfigured: +{} (ok={} err={}) -{} (total {})",
                       added_ok + added_err, added_ok, added_err, removed, m_runtime.size());
    }

    return true;
}

bool sigmf_source::on_streaming_changed() {
    logger()->info("sigmf_source: streaming changed with streaming={}", m_streaming);
    bool streaming_now = m_streaming;

    if (streaming_now) {
        spawn_metrics_thread();
    } else {
        stop_metrics_thread();
    }

    std::unique_lock lock(m_runtime_mutex);

    if (streaming_now) {
        // Always track start time (needed for wallclock_timestamps and rate_control)
        auto now = std::chrono::steady_clock::now();

        // Capture UTC epoch for absolute timestamps
        auto utc_now = std::chrono::system_clock::now();
        uint32_t utc_epoch = static_cast<uint32_t>(
            std::chrono::duration_cast<std::chrono::seconds>(utc_now.time_since_epoch()).count());

        for (auto& rt : m_runtime) {
            if (rt.load_error) continue;
            reset_runtime_for_playback(rt, now, utc_epoch);
            remap_runtime(rt);
        }
    } else {
        for (auto& rt : m_runtime) {
            if (rt.load_error) continue;
            rt.mmap.reset();
        }
    }
    return true;
}

bool sigmf_source::load_file(const struct_props::file_spec& spec, file_runtime& rt) {
    logger()->info("sigmf_source: load_file() called with file_path='{}'", spec.path);

    // Can't load runtime if spec doesn't contain a file path
    if (spec.path.empty()) {
        logger()->warn("sigmf_source: cannot load - file_path is empty");
        return false;
    }

    // Call parse metadata
    logger()->info("sigmf_source: calling parse_metadata()");
    parsed_metadata meta;
    if (!parse_metadata(spec, meta)) return false;

    if (meta.filetype == "bluefile-1000") {
        // Configure as MIDAS Blue file using spec and runtime
        return load_blue_file(spec, meta, rt);
    }

    // Derive the actual data file path. Three accepted spellings, matching what
    // parse_metadata already accepts for the sidecar: the .sigmf-data file itself,
    // the .sigmf-meta sidecar, or the bare SigMF basename. The bare stem is the
    // form graphs and the controller actually use, and it used to fall through to
    // the not-found path below because only the .sigmf-meta rewrite was handled.
    std::string data_path = spec.path;
    if (data_path.ends_with(".sigmf-meta")) {
        data_path = data_path.substr(0, data_path.length() - 11) + ".sigmf-data";
    } else if (!std::filesystem::exists(data_path) &&
               std::filesystem::exists(data_path + ".sigmf-data")) {
        data_path += ".sigmf-data";
    }

    if (!std::filesystem::exists(data_path)) {
        logger()->warn("sigmf_source: no data file for spec path '{}' (looked for '{}'); "
                       "give the .sigmf-data file, the .sigmf-meta sidecar, or their shared basename",
                       spec.path, data_path);
        return false;
    }

    rt.data_path = data_path;

    // Determine if we need to byte-swap for endianness
    constexpr bool host_is_little_endian = (std::endian::native == std::endian::little);
    bool needs_swap = (meta.format.is_big_endian == host_is_little_endian) && (meta.format.bitwidth >= 16);
    rt.needs_swap = needs_swap;

    // Calculate bytes per sample from format
    rt.bytes_per_sample = meta.format.bytes_per_sample();
    if (rt.bytes_per_sample == 0) {
        logger()->warn("sigmf_source: datatype '{}' yields a zero-byte sample for '{}'",
                       meta.format.datatype_str, spec.path);
        return false;
    }

    // Memory-map the file as raw bytes, read-only.
    logger()->info("sigmf_source: memory-mapping file {} (byteswap={}, bytes_per_sample={})",
                  data_path, needs_swap, rt.bytes_per_sample);

    // Catch runtime error from MmapRegion on failure
    try {
        rt.mmap = std::make_shared<MmapRegion<std::byte>>(data_path);
    } catch (const std::exception& e) {
        logger()->warn("sigmf_source: mmap failed for '{}': {}", data_path, e.what());
        return false;
    }

    rt.total_samples = rt.mmap->file_size() / rt.bytes_per_sample;

    build_metadata(spec, rt, meta);

    // The swap itself happens per chunk in next_chunk(), on the copy. The metadata
    // still advertises native order because that is what a consumer receives; only
    // the bytes on disk remain foreign.
    if (needs_swap) {
        logger()->info("sigmf_source: {} data will be byte-swapped per chunk on emission",
                       data_path);
        meta.format.is_big_endian = (std::endian::native == std::endian::big);
        rt.cached_meta.format.endianness = std::endian::native;
        rt.md_dirty = true;
    }

    rt.file_path = spec.path;
    rt.start_time = std::chrono::steady_clock::now();
    auto utc_now = std::chrono::system_clock::now();
    rt.utc_epoch_seconds = static_cast<uint32_t>(
                    std::chrono::duration_cast<std::chrono::seconds>(utc_now.time_since_epoch()).count());
    
    // Pre-seed spec mirrors so refresh's loop-transition snapshot
    // (was_looping = rt.loop) sees the spec's value and the transition
    // branch is a no-op on this initial load.
    rt.loop = spec.loop;
    rt.wallclock_timestamps = spec.wallclock_timestamps;
    rt.rate_control = spec.rate_control;

    // Single source of truth for spec mirrors + timing.
    refresh_runtime_from_spec(spec, rt);

    return true;
}

bool sigmf_source::load_blue_file(const struct_props::file_spec& spec, parsed_metadata& meta, file_runtime& rt) {
    logger()->info("sigmf_source: loading MIDAS Blue file");

    // Derive the .blue file path from spec
    std::string blue_path = spec.path;
    if (blue_path.ends_with(".sigmf-meta")) {
        blue_path = blue_path.substr(0, blue_path.length() - 11) + ".blue";
    } else if (blue_path.ends_with(".sigmf-data")) {
        blue_path = blue_path.substr(0, blue_path.length() - 11) + ".blue";
    } else if (!blue_path.ends_with(".blue")) {
        if (!std::filesystem::exists(blue_path) || !blue::isBlueFile(blue_path)) {
            blue_path = blue_path + ".blue";
        }
    }

    if (!std::filesystem::exists(blue_path)) {
        logger()->warn("sigmf_source: Blue file not found: {}", blue_path);
        return false;
    }

    rt.data_path = blue_path;

    // Parse the Blue file header
    auto blueInfo = blue::parseBlueFile(blue_path);
    if (!blueInfo || !blueInfo->valid) {
        logger()->warn("sigmf_source: failed to parse Blue file: {}", blue_path);
        return false;
    }

    // Convert Blue format into meta.format (writes into the locally-passed meta)
    meta.format.is_complex = blueInfo->format.isComplex;
    meta.format.bitwidth = blueInfo->format.scalarSize * 8;
    meta.format.is_big_endian = blueInfo->needsDataSwap;
    rt.blue_type_code = blueInfo->format.typeCode;

    switch (blueInfo->format.typeCode) {
        case 'F': meta.format.datatype = SigmfFormat::DataType::FLOAT; break;
        case 'D': meta.format.datatype = SigmfFormat::DataType::FLOAT; break;
        case 'I': case 'L': case 'X':
            meta.format.datatype = SigmfFormat::DataType::SIGNED_INT; break;
        case 'B': meta.format.datatype = SigmfFormat::DataType::SIGNED_INT; break;
        case 'O': case 'U': case 'V':
            meta.format.datatype = SigmfFormat::DataType::UNSIGNED_INT; break;
        default:
            meta.format.datatype = SigmfFormat::DataType::FLOAT;
    }

    // parse_metadata already applied spec.overrides.datatype, and the block above
    // has just overwritten it with the Blue header's own view. The override
    // exists precisely for metadata that lies, so it has to win here too --
    // silently discarding it was the bug.
    //
    // With one exception: the Blue swap counts elements by the effective
    // bit_width but strides by the header's type code (see byteswap_chunk), so an
    // override that changes the width on a byte-swapped file makes those two
    // disagree and would mis-swap. That combination is refused rather than
    // half-honoured.
    if (spec.overrides.datatype.has_value()) {
        auto overridden = parse_datatype(*spec.overrides.datatype);
        if (!overridden) {
            logger()->warn("sigmf_source: datatype override '{}' is not a datatype this "
                           "component can read; refusing '{}'",
                           *spec.overrides.datatype, blue_path);
            return false;
        }
        const bool width_differs = overridden->bitwidth != meta.format.bitwidth;
        if (blueInfo->needsDataSwap && width_differs) {
            logger()->warn("sigmf_source: refusing '{}': datatype override '{}' is {} bits but the "
                           "Blue header describes {} bits and its data needs byte swapping; the swap "
                           "stride and the sample width would disagree. Convert the file, or drop the "
                           "override.",
                           blue_path, *spec.overrides.datatype, overridden->bitwidth,
                           meta.format.bitwidth);
            return false;
        }
        logger()->info("sigmf_source: datatype override '{}' takes precedence over the Blue "
                       "header's {}-bit {} sample for '{}'",
                       *spec.overrides.datatype, meta.format.bitwidth,
                       meta.format.is_complex ? "complex" : "real", blue_path);
        meta.format = *overridden;
    }

    // Build datatype string for logging
    char rc = meta.format.is_complex ? 'c' : 'r';
    char tc = (meta.format.datatype == SigmfFormat::DataType::FLOAT) ? 'f' :
              (meta.format.datatype == SigmfFormat::DataType::SIGNED_INT) ? 'i' : 'u';
    meta.format.datatype_str = std::format("{}{}{}", rc, tc, meta.format.bitwidth);

    // Use sample rate from Blue header if not already set (and not overridden)
    if (meta.sample_rate <= 0 && blueInfo->sampleRate > 0) {
        meta.sample_rate = blueInfo->sampleRate;
        logger()->info("sigmf_source: using sample rate {} Hz from Blue header", meta.sample_rate);
    }

    // Identity field
    rt.file_path = spec.path;

    // Format-derived fields
    rt.bytes_per_sample = blueInfo->format.sampleSize;
    if (rt.bytes_per_sample == 0) {
        logger()->warn("sigmf_source: invalid Blue format - bytes_per_sample is 0");
        return false;
    }

    // Memory-map the file
    bool needs_swap = blueInfo->needsDataSwap && (meta.format.bitwidth >= 16);
    rt.needs_swap = needs_swap;

    logger()->info("sigmf_source: memory-mapping Blue file {} (byteswap={}, data_offset={}, data_size={})",
                  blue_path, needs_swap, blueInfo->dataOffset, blueInfo->dataSize);

    // Catch runtime error from MmapRegion on failure
    try {
        rt.mmap = std::make_shared<MmapRegion<std::byte>>(blue_path);
    } catch (const std::exception& e) {
        logger()->warn("sigmf_source: mmap failed for '{}': {}", blue_path, e.what());
        return false;
    }

    rt.data_start_offset = blueInfo->dataOffset;
    rt.current_byte_offset = blueInfo->dataOffset;
    rt.total_samples = blueInfo->sampleCount;

    // Build cached metadata for downstream
    build_metadata(spec, rt, meta);

    // Swapped per chunk on emission (see next_chunk). Chunks are read from
    // data_start_offset onward, so the header block is never in a swapped span --
    // the previous whole-region swap had to skip it explicitly.
    if (needs_swap) {
        logger()->info("sigmf_source: Blue data for {} will be byte-swapped per chunk on emission",
                       blue_path);
        rt.cached_meta.format.endianness = std::endian::native;
        rt.md_dirty = true;
    }

    rt.file_path = spec.path;
    rt.start_time = std::chrono::steady_clock::now();
    auto utc_now = std::chrono::system_clock::now();
    rt.utc_epoch_seconds = static_cast<uint32_t>(
                    std::chrono::duration_cast<std::chrono::seconds>(utc_now.time_since_epoch()).count());

    // Pre-seed spec mirrors so refresh's loop-transition snapshot
    // (was_looping = rt.loop) sees the spec's value and the transition
    // branch is a no-op on this initial load.
    rt.loop = spec.loop;
    rt.wallclock_timestamps = spec.wallclock_timestamps;
    rt.rate_control = spec.rate_control;

    // Single source of truth for spec mirrors + timing.
    refresh_runtime_from_spec(spec, rt);

    return true;
}

bool sigmf_source::parse_metadata(const struct_props::file_spec& spec, parsed_metadata& meta) {
    // extract path from spec
    std::string meta_path = spec.path;

    // Extract sigmf-meta file from specs file path
    if (meta_path.ends_with(".sigmf-data")) {
        meta_path = meta_path.substr(0, meta_path.length() - 11) + ".sigmf-meta";
    } else if (meta_path.ends_with(".blue")) {
        // Blue file - derive meta path by replacing .blue with .sigmf-meta
        meta_path = meta_path.substr(0, meta_path.length() - 5) + ".sigmf-meta";
    } else if (!meta_path.ends_with(".sigmf-meta")) {
        // Check if the file exists as-is (raw binary file without .sigmf- extension)
        if (!std::filesystem::exists(spec.path) || std::filesystem::is_directory(spec.path)) {
            meta_path = meta_path + ".sigmf-meta";
        } else {
            // File exists as-is, derive meta path by appending .sigmf-meta
            meta_path = spec.path + ".sigmf-meta";
        }
    }

    std::ifstream meta_file(meta_path);
    if (!meta_file) {
        logger()->warn("sigmf_source: no metadata file found at {}, will use overrides/defaults", meta_path);
        // Don't return - fall through to apply overrides
    } else {
        try {
            nlohmann::json doc;
            meta_file >> doc;

            if (!doc.contains("global")) {
                logger()->warn("sigmf_source: no 'global' section in metadata");
            } else {
                auto& global = doc["global"];

                // Sample rate from file (may be overridden below)
                if (global.contains("core:sample_rate")) {
                    meta.sample_rate = global["core:sample_rate"].get<double>();
                }

                // Optional fields with defaults
                meta.center_frequency = global.value("core:frequency", 0.0);
                meta.description = global.value("core:description", std::string{});

                // Check for filetype extension (for MIDAS Blue support)
                // Supports both namespaced and non-namespaced field names
                for (const auto& key : {"filetype", "core:filetype"}) {
                    if (global.contains(key)) {
                        meta.filetype = global[key].get<std::string>();
                        logger()->debug("sigmf_source: found filetype '{}' in metadata", meta.filetype);
                        break;
                    }
                }

                // Parse datatype for format/endianness info
                if (global.contains("core:datatype")) {
                    std::string datatype = global["core:datatype"].get<std::string>();
                    auto parsed = parse_datatype(datatype);
                    if (parsed) {
                        meta.format = *parsed;
                        logger()->debug("sigmf_source: parsed datatype '{}' - complex={}, big_endian={}, bits={}",
                                       datatype, meta.format.is_complex, meta.format.is_big_endian, meta.format.bitwidth);
                    } else {
                        logger()->warn("sigmf_source: failed to parse datatype '{}', using defaults", datatype);
                    }
                }

                // Check captures array for frequency override
                if (doc.contains("captures") && doc["captures"].is_array() && !doc["captures"].empty()) {
                    auto& cap0 = doc["captures"][0];
                    if (cap0.contains("core:frequency")) {
                        meta.center_frequency = cap0["core:frequency"].get<double>();
                    }
                }
            }

        } catch (const nlohmann::json::exception& e) {
            logger()->error("sigmf_source: JSON parse error in '{}': {}", meta_path, e.what());
            // Return false which sets load error to true
            return false;
        }
    }

    // Apply property overrides (these take precedence over file metadata)
    // These are always applied, even if no metadata file was found
    if (spec.overrides.sample_rate.has_value()) {
        logger()->info("sigmf_source: overriding sample_rate {} -> {}", meta.sample_rate, *spec.overrides.sample_rate);
        meta.sample_rate = *spec.overrides.sample_rate;
    }
    if (spec.overrides.center_frequency.has_value()) {
        logger()->info("sigmf_source: overriding center_frequency {} -> {}", meta.center_frequency, *spec.overrides.center_frequency);
        meta.center_frequency = *spec.overrides.center_frequency;
    }
    if (spec.overrides.datatype.has_value()) {
        auto parsed = parse_datatype(*spec.overrides.datatype);
        if (parsed) {
            logger()->info("sigmf_source: overriding datatype -> '{}'", *spec.overrides.datatype);
            meta.format = *parsed;
        } else {
            logger()->warn("sigmf_source: failed to parse override datatype '{}', keeping previous", *spec.overrides.datatype);
        }
    }
    if (spec.overrides.filetype.has_value()) {
        logger()->info("sigmf_source: overriding filetype -> '{}'", *spec.overrides.filetype);
        meta.filetype = *spec.overrides.filetype;
    }
    return true;
}

void sigmf_source::stamp_destination(const struct_props::file_spec& spec, file_runtime& rt) {
    rt.cached_meta.annotations["stream_id"] = std::to_string(spec.stream_id);

    // Group address is the per-stream discriminator. When the spec carries no
    // destination_ip, emit nothing and let udp_sink fall back to its
    // default_dest_ip -- erase rather than leave a stale value behind, since
    // this also runs on live spec edits.
    if (!spec.destination_ip.empty()) {
        rt.cached_meta.annotations["destination_ip"] = spec.destination_ip;
    } else {
        rt.cached_meta.annotations.erase("destination_ip");
    }

    // Flat port: every stream shares it. Separation is by group, so offsetting
    // the port by stream_id would only defeat receivers' IGMP filtering.
    const auto port = spec.destination_port != 0 ? spec.destination_port : m_destination_port;
    rt.cached_meta.annotations["destination_port"] = std::to_string(port);

    rt.md_dirty = true;
}

void sigmf_source::build_metadata(const struct_props::file_spec& spec, file_runtime& rt, const parsed_metadata& meta) {
    stamp_destination(spec, rt);

    rt.cached_meta.sample_rate = meta.sample_rate;
    rt.cached_meta.center_frequency = meta.center_frequency;
    rt.cached_meta.bandwidth = spec.overrides.bandwidth.value_or(meta.sample_rate);

    if (!meta.description.empty()) {
        rt.cached_meta.annotations["description"] = meta.description;
    }

    rt.cached_meta.format = to_composite_format(meta.format);

    rt.md_dirty = true;
}

void sigmf_source::reset_runtime_for_playback(file_runtime& rt, std::chrono::steady_clock::time_point now, uint32_t utc_epoch_seconds) {
    rt.current_byte_offset = rt.data_start_offset;
    rt.samples_sent = 0;
    rt.eof = false;
    rt.start_time = now;
    rt.utc_epoch_seconds = utc_epoch_seconds;

    // Reset metrics state
    rt.prev_samples_sent = 0;
    rt.prev_tick_time = {};
    rt.recent_rates.fill(0.0);
    rt.recent_idx = 0;
    rt.recent_count = 0;
    rt.rolling_rate_sps = 0.0;
    rt.prev_health = file_runtime::health::ok;
    rt.prev_eof = false;
}

bool sigmf_source::remap_runtime(file_runtime& rt) {
    if (rt.mmap && rt.mmap->valid()) return true;  // already mapped

    try {
        // No re-swap here any more. The mapping is never mutated, so a fresh one
        // needs no fixing up -- where this previously re-swapped the entire file
        // on every remap (each start, and each loop re-anchor).
        rt.mmap = std::make_shared<MmapRegion<std::byte>>(rt.data_path);
        return true;
    } catch (const std::exception& e) {
        logger()->error("sigmf_source: failed to remap '{}': {}", rt.file_path, e.what());
        rt.load_error = true;
        return false;
    }
}

void sigmf_source::refresh_runtime_from_spec(const struct_props::file_spec& spec, file_runtime& rt) {
    bool prev_loop = rt.loop;

    // Update live RUNTIME fields
    rt.loop = spec.loop;
    rt.wallclock_timestamps = spec.wallclock_timestamps;
    rt.rate_control = spec.rate_control;
    stamp_destination(spec, rt);

    // Dynamic overrides: sample_rate, center_frequency, bandwidth
    // datatype and filetype are load-time-only
    if (spec.overrides.center_frequency) {
        rt.cached_meta.center_frequency = *spec.overrides.center_frequency;
    }
    const double new_rate = spec.overrides.sample_rate.value_or(rt.cached_meta.sample_rate);
    rt.cached_meta.sample_rate = new_rate;
    rt.cached_meta.bandwidth = spec.overrides.bandwidth.value_or(new_rate);
    rt.md_dirty = true;

    // Recompute timing - handles rate_control on/off and max_sample_rate changes
    parsed_metadata meta;
    meta.sample_rate = rt.cached_meta.sample_rate;
    calculate_timing(spec, rt, meta);

    // Prevents stale start time and memory spike
    // due to process thinking it owes thousands of chunks of data
    if (rt.loop && !prev_loop) {
        rt.start_time = std::chrono::steady_clock::now();
        rt.samples_sent = 0;
        auto utc_now = std::chrono::system_clock::now();
        rt.utc_epoch_seconds = static_cast<uint32_t>(
            std::chrono::duration_cast<std::chrono::seconds>(utc_now.time_since_epoch()).count());
        logger()->info("sigmf_source: re-anchoring '{}' on loop=true transition", rt.file_path);
    }
}

void sigmf_source::calculate_timing(const struct_props::file_spec& spec, file_runtime& rt, parsed_metadata& meta) {
    // Rate control off
    if (!spec.rate_control) {
        rt.effective_sample_rate = 0.0;
        rt.chunk_interval = std::chrono::microseconds(0);
        rt.chunks_per_wakeup = 1;
        rt.rate_control = false;
        return;
    }


    // Determine effective sample rate
    rt.effective_sample_rate = meta.sample_rate;
    if (spec.max_sample_rate > 0 && spec.max_sample_rate < meta.sample_rate) {
        rt.effective_sample_rate = spec.max_sample_rate;
        logger()->info("sigmf_source: limiting playback rate from {} to {} Hz",
                      meta.sample_rate, rt.effective_sample_rate);
    }

    if (rt.effective_sample_rate <= 0 || m_chunk_samples == 0) {
        rt.rate_control = false;
        rt.effective_sample_rate = 0.0;
        rt.chunk_interval = std::chrono::microseconds(0);
        rt.chunks_per_wakeup = 1;
        logger()->warn("sigmf_source: invalid sample rate or chunk size, disabling rate control");
        return;
    }

    // Calculate chunks per second and ideal interval
    double chunks_per_sec = rt.effective_sample_rate / static_cast<double>(m_chunk_samples);
    double interval_us = 1e6 / chunks_per_sec;

    // Minimum wakeup interval (500 microseconds) to avoid excessive thread scheduling
    constexpr double MIN_WAKE_US = 500.0;

    if (interval_us >= MIN_WAKE_US) {
        // Slow enough to send one chunk per wakeup
        rt.chunks_per_wakeup = 1;
        rt.chunk_interval = std::chrono::microseconds(static_cast<uint64_t>(interval_us + 0.5));
    } else {
        // Bundle multiple chunks to maintain minimum wakeup interval
        rt.chunks_per_wakeup = static_cast<uint32_t>(std::ceil(MIN_WAKE_US / interval_us));
        rt.chunk_interval = std::chrono::microseconds(
            static_cast<uint64_t>(interval_us * rt.chunks_per_wakeup + 0.5));
    }

    rt.rate_control = true;
    logger()->info("sigmf_source: rate control enabled - {} Hz, {} samples/chunk, {} chunks/wakeup, {} us interval",
                  rt.effective_sample_rate, m_chunk_samples, rt.chunks_per_wakeup, rt.chunk_interval.count());
}

auto sigmf_source::next_chunk(file_runtime& rt) -> produce_result {
    if (rt.eof && !rt.loop) {
        return produce_result::idle();
    }

    if (rt.eof && rt.loop) {
        // Reset to beginning for looping (accounting for data offset in Blue files)
        logger()->trace("sigmf_source: looping back to start");
        rt.current_byte_offset = rt.data_start_offset;
        rt.eof = false;
        rt.loop_count++;
    }

    // Calculate samples to read (working in sample units, converting to bytes for mmap)
    // Account for data_start_offset (non-zero for Blue files where header precedes data)
    std::size_t current_sample = (rt.current_byte_offset - rt.data_start_offset) / rt.bytes_per_sample;
    std::size_t samples_remaining = rt.total_samples - current_sample;
    std::size_t samples_to_read = std::min(m_chunk_samples, samples_remaining);

    if (samples_to_read == 0) {
        rt.eof = true;
        if (!rt.loop) {
            logger()->info("sigmf_source: reached end of file after {} samples", rt.samples_sent);
        }
        return produce_result::idle();
    }

    // Convert sample count to byte count for buffer copy
    std::size_t bytes_to_read = samples_to_read * rt.bytes_per_sample;

    // Copy chunk bytes from the mmap region into a shared buffer.
    auto* src = rt.mmap->data() + rt.current_byte_offset;
    auto chunk = std::make_shared<std::vector<std::byte>>(src, src + bytes_to_read);

    // Convert to host order on the copy. Doing it here rather than once over the
    // mapping is what lets the mapping stay read-only: the previous approach faulted
    // a private copy of the whole recording into memory before emitting a byte.
    if (rt.needs_swap) {
        byteswap_chunk(rt, std::span<std::byte>{*chunk});
    }

    // Captured BEFORE the counters advance: the packet timestamps the first sample
    // it carries. Reading samples_sent after the increment dated every packet one
    // full chunk late.
    const uint64_t chunk_start_sample = rt.samples_sent;

    rt.current_byte_offset += bytes_to_read;
    rt.samples_sent += samples_to_read;

    // Create timestamp (absolute UTC)
    // Offset from start is computed as either elapsed wall-clock or sample-based time,
    // then anchored to the UTC epoch captured at start.
    composite::timestamp ts;
    double offset_seconds;

    if (rt.wallclock_timestamps) {
        // Wall-clock mode: offset based on real elapsed time since start
        auto now = std::chrono::steady_clock::now();
        offset_seconds = std::chrono::duration<double>(now - rt.start_time).count();
    } else {
        // File-time mode (default): position on the RECORDING's timeline, so this
        // uses the file's true rate. effective_sample_rate is the throttled playback
        // rate -- deriving file time from it let max_sample_rate silently rewrite the
        // recording's timestamps.
        const double file_rate = rt.cached_meta.sample_rate > 0.0
            ? rt.cached_meta.sample_rate
            : rt.effective_sample_rate;
        offset_seconds = file_rate > 0.0
            ? static_cast<double>(chunk_start_sample) / file_rate
            : 0.0;
    }

    // Split offset into integer seconds and sub-second fractional part
    auto offset_int = static_cast<uint32_t>(offset_seconds);
    double offset_frac = offset_seconds - offset_int;

    ts.seconds = rt.utc_epoch_seconds + offset_int;
    ts.picoseconds = static_cast<uint64_t>(offset_frac * 1e12);

    // send_metadata() is gone in 0.5.2: metadata travels with the packet. Re-latch
    // only when it actually changed, so the hot path reuses one shared instance.
    if (rt.md_dirty || !rt.latched_md) {
        rt.latched_md = composite::make_metadata(rt.cached_meta);
        rt.md_dirty = false;
    }

    // chunk is a copy of the mmap region (intentional, not a view), which keeps
    // mmap.reset() a true release independent of the downstream pipeline.
    composite::immutable_buffer<std::byte> buf(chunk);
    return produce_result::emit(std::move(buf), ts, rt.latched_md);
}

// Deliberate trade: swapping per chunk costs steady-state throughput to save
// memory. Measured on this fleet with ci16 looping, unthrottled: ~4.4 GB/s
// unswapped vs ~1.5 GB/s swapped, i.e. roughly 3x. Swapping the whole mapping
// once at load (the previous design) paid nothing per chunk but faulted a private
// copy of the entire recording into RSS, and re-ran in full on every loop
// re-anchor. Single-pass playback does the same total work either way; looping
// playback of a big-endian recording is where this costs.
//
// The gap is an artefact of the loops below, not of swapping per chunk: they go
// byte-at-a-time through a uint8_t*, which does not vectorise. Rewriting them
// over sized loads with __builtin_bswap* should recover most of it, and the
// swap-correctness tests in tests/sigmf_source_bluefile_tests.cpp pin the
// observable behaviour for exactly that change.
void sigmf_source::byteswap_chunk(const file_runtime& rt, std::span<std::byte> chunk) {
    if (chunk.empty()) {
        return;
    }

    auto* bytes = reinterpret_cast<uint8_t*>(chunk.data());
    const std::size_t num_bytes = chunk.size();
    const uint32_t bit_width = rt.cached_meta.format.bit_width;
    const std::size_t elem = bit_width / 8U;

    // 8-bit samples have no byte order. A width this code cannot address should
    // never reach here -- parse_datatype restricts widths to 8/16/32/64 -- but
    // bail rather than divide by zero if it ever did.
    if (elem <= 1) {
        return;
    }

    // A chunk is always a whole number of samples, and a sample is a whole number
    // of scalars, so a chunk never splits an element across a boundary. That is
    // what makes swapping per chunk equivalent to swapping the whole region.
    if (rt.blue_type_code != 0) {
        // Blue keeps striding by the header's type code, which is the authority on
        // the on-disk scalar for these files.
        blue::byteswapData(bytes, num_bytes / elem, rt.blue_type_code);
        return;
    }

    switch (bit_width) {
    case 16:
        for (std::size_t i = 0; i + 1 < num_bytes; i += 2) {
            std::swap(bytes[i], bytes[i + 1]);
        }
        break;
    case 32:
        for (std::size_t i = 0; i + 3 < num_bytes; i += 4) {
            std::swap(bytes[i], bytes[i + 3]);
            std::swap(bytes[i + 1], bytes[i + 2]);
        }
        break;
    case 64:
        for (std::size_t i = 0; i + 7 < num_bytes; i += 8) {
            std::swap(bytes[i], bytes[i + 7]);
            std::swap(bytes[i + 1], bytes[i + 6]);
            std::swap(bytes[i + 2], bytes[i + 5]);
            std::swap(bytes[i + 3], bytes[i + 4]);
        }
        break;
    default:
        break;
    }
}

void sigmf_source::spawn_metrics_thread() {
    if (m_metrics_thread.joinable()) return;

    m_metrics_thread = std::jthread([this](std::stop_token st) {
        pthread_setname_np(pthread_self(), "sigmf_metrics");
        while (!st.stop_requested()) {
            std::this_thread::sleep_for(kMetricsTickInterval);
            log_metrics();
        }
    });
}

void sigmf_source::stop_metrics_thread() {
    m_metrics_thread.request_stop();
    if (m_metrics_thread.joinable()) {
        m_metrics_thread.join();
    }
}

void sigmf_source::log_metrics() {
    std::shared_lock lock(m_runtime_mutex);

    if (m_runtime.empty()) {
        return;
    }

    auto now = std::chrono::steady_clock::now();

    // First pass: update rate state and classify health, fire edge-triggered alarms
    std::size_t count_ok = 0;
    std::size_t count_warn = 0;
    std::size_t count_err = 0;

    for (auto& rt : m_runtime) {
        using health = file_runtime::health;

        // Error state takes precedence and short-circuits rate tracking
        if (rt.load_error) {
            if (rt.prev_health != health::err) {
                logger()->warn("sigmf_source: stream={} path='{}' entered ERROR state",
                               stream_id_of(rt), rt.file_path);
                rt.prev_health = health::err;
            }
            count_err++;
            continue;
        }

        // Compute instantaneous rate (samples/sec since last tick)
        double instant_rate = 0.0;
        if (rt.prev_tick_time.time_since_epoch().count() > 0) {
            double dt = std::chrono::duration<double>(now - rt.prev_tick_time).count();
            if (dt > 0) {
                instant_rate = static_cast<double>(rt.samples_sent - rt.prev_samples_sent) / dt;
            }
        }
        rt.prev_samples_sent = rt.samples_sent;
        rt.prev_tick_time = now;

        // Update rolling-window buffer
        rt.recent_rates[rt.recent_idx] = instant_rate;
        rt.recent_idx = (rt.recent_idx + 1) % file_runtime::kRollingWindow;
        rt.recent_count = std::min(rt.recent_count + 1, file_runtime::kRollingWindow);
        rt.rolling_rate_sps = std::accumulate(
            rt.recent_rates.begin(),
            rt.recent_rates.begin() + rt.recent_count,
            0.0) / static_cast<double>(rt.recent_count);

        // EOF edge for non-looping
        if (rt.eof && !rt.loop && !rt.prev_eof) {
            logger()->info("sigmf_source: stream={} path='{}' reached EOF",
                           stream_id_of(rt), rt.file_path);
        }
        rt.prev_eof = rt.eof;

        // Classify health — only after warm-up so partial window doesn't trip the alarm
        health current = health::ok;
        if (rt.rate_control
            && rt.effective_sample_rate > 0
            && rt.recent_count == file_runtime::kRollingWindow) {
            double deviation = std::abs(rt.rolling_rate_sps - rt.effective_sample_rate)
                             / rt.effective_sample_rate;
            if (deviation > kRateDeviationThreshold) {
                current = health::warn;
            }
        }

        // Edge-triggered: fire WARN on entry to warn, INFO on recovery
        if (current != rt.prev_health) {
            if (current == health::warn) {
                double deviation_pct = std::abs(rt.rolling_rate_sps - rt.effective_sample_rate)
                                     / rt.effective_sample_rate * 100.0;
                logger()->warn("sigmf_source: stream={} path='{}' rate deviating: "
                               "rolling={} expected={} ({:.1f}%)",
                               stream_id_of(rt), rt.file_path,
                               format_rate(rt.rolling_rate_sps),
                               format_rate(rt.effective_sample_rate),
                               deviation_pct);
            } else if (current == health::ok && rt.prev_health == health::warn) {
                logger()->info("sigmf_source: stream={} path='{}' rate recovered",
                               stream_id_of(rt), rt.file_path);
            }
            rt.prev_health = current;
        }

        if (current == health::warn) count_warn++;
        else count_ok++;
    }

    // Summary INFO line — one per tick at steady state
    logger()->info("sigmf_source: streams={} ok={} warn={} err={}",
                   m_runtime.size(), count_ok, count_warn, count_err);

    // Per-stream DEBUG detail
    for (const auto& rt : m_runtime) {
        if (rt.load_error) {
            logger()->debug("  stream={} path='{}' ERROR",
                            stream_id_of(rt), rt.file_path);
            continue;
        }

        // Progress: handle looping vs non-looping correctly
        double progress = 0.0;
        if (rt.total_samples > 0) {
            uint64_t effective_samples = rt.loop
                ? (rt.samples_sent % rt.total_samples)
                : std::min(rt.samples_sent, rt.total_samples);
            progress = static_cast<double>(effective_samples) / rt.total_samples * 100.0;
        }

        // Pull the most recent instantaneous rate from the ring buffer
        double instant = 0.0;
        if (rt.recent_count > 0) {
            std::size_t last_idx = (rt.recent_idx + file_runtime::kRollingWindow - 1)
                                 % file_runtime::kRollingWindow;
            instant = rt.recent_rates[last_idx];
        }

        logger()->debug("  stream={} path='{}' "
                        "sr={}(inst)/{}(avg)/{}(exp) "
                        "pos={:.1f}% loops={}",
                        stream_id_of(rt), rt.file_path,
                        format_rate(instant),
                        format_rate(rt.rolling_rate_sps),
                        format_rate(rt.effective_sample_rate),
                        progress,
                        rt.loop_count);
    }
}

auto sigmf_source::parse_datatype(std::string_view sv) -> std::optional<SigmfFormat> {
    // Parse SigMF datatype format: [r|c][f|i|u]{bitwidth}[_le|_be]
    // Examples: cf32_le, ri16_be, cu8
    SigmfFormat fmt;
    fmt.datatype_str = std::string(sv);
    fmt.is_big_endian = false;  // Default to little-endian
    fmt.is_complex = true;      // Default to complex

    // Check for endianness suffix
    if (sv.ends_with("_le")) {
        sv.remove_suffix(3);
        fmt.is_big_endian = false;
    } else if (sv.ends_with("_be")) {
        sv.remove_suffix(3);
        fmt.is_big_endian = true;
    }

    if (sv.empty()) return std::nullopt;

    // First char: real/complex specifier
    switch (sv.front()) {
        case 'c': case 'C': fmt.is_complex = true; break;
        case 'r': case 'R': fmt.is_complex = false; break;
        default: return std::nullopt;
    }
    sv.remove_prefix(1);

    if (sv.empty()) return std::nullopt;

    // Second char: data type specifier
    switch (sv.front()) {
        case 'f': case 'F': fmt.datatype = SigmfFormat::DataType::FLOAT; break;
        case 'i': case 'I': fmt.datatype = SigmfFormat::DataType::SIGNED_INT; break;
        case 'u': case 'U': fmt.datatype = SigmfFormat::DataType::UNSIGNED_INT; break;
        default: return std::nullopt;
    }
    sv.remove_prefix(1);

    if (sv.empty()) return std::nullopt;

    // Remaining: bitwidth
    uint32_t bw{};
    auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), bw);
    if (ec != std::errc{} || ptr != sv.data() + sv.size()) {
        return std::nullopt;
    }
    // Only whole-byte, power-of-two widths are addressable here. The byte swap
    // strides by bitwidth/8 and bytes_per_sample() divides by 8, so a packed
    // width such as ci12 truncates to one byte per scalar and would walk the
    // buffer at the wrong stride while metadata advertised 12 bits.
    if (bw != 8 && bw != 16 && bw != 32 && bw != 64) {
        return std::nullopt;
    }
    fmt.bitwidth = bw;

    return fmt;
}

auto sigmf_source::to_composite_format(const SigmfFormat& fmt) -> composite::data_format {
    composite::data_format df;
    df.is_complex = fmt.is_complex;
    df.bit_width = fmt.bitwidth;
    df.endianness = fmt.is_big_endian ? std::endian::big : std::endian::little;

    switch (fmt.datatype) {
        case SigmfFormat::DataType::FLOAT:
            df.type = composite::data_type::floating_point;
            break;
        case SigmfFormat::DataType::SIGNED_INT:
            df.type = composite::data_type::signed_integer;
            break;
        case SigmfFormat::DataType::UNSIGNED_INT:
            df.type = composite::data_type::unsigned_integer;
            break;
    }

    return df;
}

std::string sigmf_source::format_rate(double sps) {
    if (sps >= 1e9) return std::format("{:.1f}G", sps / 1e9);
    if (sps >= 1e6) return std::format("{:.1f}M", sps / 1e6);
    if (sps >= 1e3) return std::format("{:.1f}k", sps / 1e3);
    return std::format("{:.0f}", sps);
}

std::string sigmf_source::stream_id_of(const file_runtime& rt) {
    auto it = rt.cached_meta.annotations.find("stream_id");
    return it != rt.cached_meta.annotations.end() ? it->second.to_string() : "?";
}

#ifndef UNIT_TESTS
// Emits create() plus composite_abi_version. Uses make_component (not make_shared):
// its deleter stops the worker while the leaf vtable is still intact, which a
// source_component requires because produce() is pure in the base.
COMPOSITE_REGISTER_SIMPLE(sigmf_source)
#endif
