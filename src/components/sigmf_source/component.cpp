/*
 * Copyright (C) 2025 Geon Technologies, LLC
 *
 * SigMF file source component with rate-controlled playback.
 * Uses mmap for zero-copy file reading at high sample rates.
 */

#include "component.hpp"
#include "blue/BlueFile.hpp"

#include <composite/properties/property_set.hpp>
#include <composite/properties/property.hpp>

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
    : composite::component(id) {
    using enum composite::properties::config_type;
    add_port(&m_out_port);

    add_property("destination_port", m_destination_port, RUNTIME);
    
    add_property("chunk_samples", m_chunk_samples, RUNTIME);

    // File spec - starts empty, append files on every allocation added
    add_property("files", m_files, RUNTIME)
        .change_listener([this](std::size_t idx) { return on_files_changed(); })
        // Two change listeners needed:
        // - Indexed: framework fires this for per-element changes (add/edit/single erase)
        // - No-arg: framework fires this for whole-list reset, e.g., setting "files": null
        // Both forward to the same handler since our diff is path-keyed
        .change_listener([this]() { return on_files_changed(); });

    // Streaming control - starts disabled, set streaming=true when ready to stream
    add_property("streaming", m_streaming, RUNTIME)
        .change_listener([this]() { return on_streaming_changed(); });
}

auto sigmf_source::initialize() -> void {
    // Component starts with streaming disabled - configure properties then set streaming=true
    logger()->info("sigmf_source initialized (streaming disabled, set streaming=true after configuring file_path)");
}

auto sigmf_source::start() -> void {
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
    composite::component::start();
}

auto sigmf_source::stop() -> void {
    stop_metrics_thread();

    std::unique_lock<std::shared_mutex> lock(m_runtime_mutex);
    for (auto& rt : m_runtime) {
        rt.mmap.reset();
    }
    composite::component::stop();
}

auto sigmf_source::process() -> composite::retval {
    using enum composite::retval;

    std::unique_lock<std::shared_mutex> lock(m_runtime_mutex);

    if (!m_streaming || m_runtime.empty()) {
        return NOOP;
    }

    auto now = std::chrono::steady_clock::now();
    auto earliest_due = std::chrono::steady_clock::time_point::max();

    // --- Pass 1: classify each runtime ---
    // Collect indices of runtimes that have work ready RIGHT NOW.
    // For ones not ready, accumulate the earliest "due" time.
    std::vector<std::size_t> ready_now;
    ready_now.reserve(m_runtime.size());

    for (std::size_t i = 0; i < m_runtime.size(); ++i) {
        const auto& rt = m_runtime[i];

        if (rt.load_error) continue;
        if (rt.eof && !rt.loop) continue;  // no more work

        constexpr auto kBackoffWhenDownstreamFull = std::chrono::microseconds(500);

        if (!rt.rate_control) {
            if (m_out_port.can_send()) {
                // As-fast-as-possible mode: always ready
                ready_now.push_back(i);
            } else {
                earliest_due = std::min(earliest_due, now + kBackoffWhenDownstreamFull);
            }
            continue;
        }

        // Rate-controlled: ready only if behind schedule
        auto elapsed = std::chrono::duration<double>(now - rt.start_time).count();
        auto expected_samples = static_cast<uint64_t>(elapsed * rt.effective_sample_rate);

        if (rt.samples_sent < expected_samples) {
            ready_now.push_back(i);
        } else {
            // Ahead of schedule — compute when this file's next chunk is due
            double next_due_sec = static_cast<double>(rt.samples_sent) / rt.effective_sample_rate;
            auto next_due = rt.start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(next_due_sec));
            earliest_due = std::min(earliest_due, next_due);
        }
    }

    // --- Pass 2: execute the ready set ---
    bool any_work_done = false;

    if (!ready_now.empty()) {
        // Per-iteration budget: cap how many chunks each ready runtime sends per process() call
        // to keep latency fair across streams. For rate-controlled, use chunks_per_wakeup * 2
        // (matches old catch-up cap). For non-rate-controlled, use a fixed small budget.
        for (auto idx : ready_now) {
            auto& rt = m_runtime[idx];

            uint32_t budget = rt.rate_control
                ? rt.chunks_per_wakeup * 2
                : 1;  // non-rate-controlled: one chunk per process(), no hogging

            uint32_t sent = 0;
            while (sent < budget) {
                if (rt.rate_control) {
                    // Re-check we're still behind (catch-up loop terminates when current)
                    auto elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - rt.start_time).count();
                    auto expected = static_cast<uint64_t>(elapsed * rt.effective_sample_rate);
                    if (rt.samples_sent >= expected && sent >= rt.chunks_per_wakeup) {
                        break;
                    }
                }

                if (process_chunk(rt) != NORMAL) break;
                sent++;
            }

            if (sent > 0) any_work_done = true;
        }
    }

    // --- Sleep if no one was ready and we have a future due time ---
    if (!any_work_done && earliest_due != std::chrono::steady_clock::time_point::max()) {
        lock.unlock();
        auto now2 = std::chrono::steady_clock::now();
        if (earliest_due > now2) {
            auto sleep_duration = std::clamp(
                std::chrono::duration_cast<std::chrono::microseconds>(earliest_due - now2),
                std::chrono::microseconds(250),
                std::chrono::microseconds(10000));
            std::this_thread::sleep_for(sleep_duration);
        }
    }

    return any_work_done ? NORMAL : NOOP;
}

bool sigmf_source::on_files_changed() {
    logger()->debug("sigmf_source called on_files_changed with file_size={}", m_files.size());

    std::unique_lock<std::shared_mutex> lock(m_runtime_mutex);

    // Handle duplicate file paths
    std::unordered_set<std::string> seen_paths;
    std::unordered_set<std::string> path_conflicts;
    for (const auto& s : m_files) {
        if (!seen_paths.insert(s.file_path).second) {
            path_conflicts.insert(s.file_path);
            logger()->warn("sigmf_source: duplicate file_path '{}' in files list, "
                           "marking as load_error", s.file_path);
        }
    }

    // Handle duplicate stream ids. stream_id is the VITA-49 stream identity,
    // so it must be unique regardless of where the stream is addressed.
    std::unordered_set<int32_t> seen_stream_ids;
    std::unordered_set<std::string> stream_id_conflicts;
    for (const auto& s : m_files) {
        if (!seen_stream_ids.insert(s.stream_id).second) {
            stream_id_conflicts.insert(s.file_path);
            logger()->warn("sigmf_source: duplicate stream_id {} on path '{}', "
                           "marking this spec as load_error. To recover, send "
                           "files[i]=null and re-add with a unique stream_id.",
                           s.stream_id, s.file_path);
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
            destination_conflicts.insert(s.file_path);
            logger()->warn("sigmf_source: duplicate destination {}:{} on path '{}', "
                           "marking this spec as load_error. Streams are separated by "
                           "group address; assign each spec a unique destination_ip.",
                           ip, port, s.file_path);
        }
    }

    std::unordered_map<std::string, const struct_props::file_spec*> desired;
    for (const auto& s : m_files) desired.emplace(s.file_path, &s);

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
            [&](const file_runtime& rt){ return rt.file_path == s.file_path; });
        if (exists) continue;

        // Check for duplicate file names, stream ids, and destinations
        file_runtime rt;
        if (stream_id_conflicts.contains(s.file_path) || path_conflicts.contains(s.file_path)
            || destination_conflicts.contains(s.file_path)) {
            rt.file_path = s.file_path;
            rt.load_error = true;
        } else if (!load_file(s, rt)) {
            rt.file_path = s.file_path;
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
    logger()->info("sigmf_source: load_file() called with file_path='{}'", spec.file_path);

    // Can't load runtime if spec doesn't contain a file path
    if (spec.file_path.empty()) {
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

    // Derive the actual data file path
    std::string data_path = spec.file_path;
    if (data_path.ends_with(".sigmf-meta")) {
        data_path = data_path.substr(0, data_path.length() - 11) + ".sigmf-data";
    }

    // Verify file exists
    if (!std::filesystem::exists(data_path)) return false;

    rt.data_path = data_path;

    // Determine if we need to byte-swap for endianness
    constexpr bool host_is_little_endian = (std::endian::native == std::endian::little);
    bool needs_swap = (meta.format.is_big_endian == host_is_little_endian) && (meta.format.bitwidth >= 16);
    rt.needs_swap = needs_swap;

    // Calculate bytes per sample from format
    rt.bytes_per_sample = meta.format.bytes_per_sample();
    if (rt.bytes_per_sample == 0) return false;

    // Memory-map the file as raw bytes
    // Use writable=true if we need to swap bytes (MAP_PRIVATE allows in-place modification)
    logger()->info("sigmf_source: memory-mapping file {} (writable={}, bytes_per_sample={})",
                  data_path, needs_swap, rt.bytes_per_sample);


    // Catch runtime error from MmapRegion on failure
    try {
        rt.mmap = std::make_shared<MmapRegion<std::byte>>(data_path, needs_swap);
    } catch (const std::exception& e) {
        logger()->warn("sigmf_source: mmap failed for '{}': {}", data_path, e.what());
        return false;
    }

    rt.total_samples = rt.mmap->file_size() / rt.bytes_per_sample;

    build_metadata(spec, rt, meta);

    // Apply endianness conversion in-place if needed
    if (needs_swap) {
        logger()->info("sigmf_source: applying endianness swap on {} bytes", rt.mmap->file_size());
        if (!apply_endianness_swap_raw(rt)) return false;
        meta.format.is_big_endian = (std::endian::native == std::endian::big);
        rt.cached_meta.format.endianness = std::endian::native;
    }

    rt.file_path = spec.file_path;
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
    std::string blue_path = spec.file_path;
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
    rt.file_path = spec.file_path;

    // Format-derived fields
    rt.bytes_per_sample = blueInfo->format.sampleSize;
    if (rt.bytes_per_sample == 0) {
        logger()->warn("sigmf_source: invalid Blue format - bytes_per_sample is 0");
        return false;
    }

    // Memory-map the file
    bool needs_swap = blueInfo->needsDataSwap && (meta.format.bitwidth >= 16);
    rt.needs_swap = needs_swap;

    logger()->info("sigmf_source: memory-mapping Blue file {} (writable={}, data_offset={}, data_size={})",
                  blue_path, needs_swap, blueInfo->dataOffset, blueInfo->dataSize);


    // Catch runtime error from MmapRegion on failure
    try {
        rt.mmap = std::make_shared<MmapRegion<std::byte>>(blue_path, needs_swap);
    } catch (const std::exception& e) {
        logger()->warn("sigmf_source: mmap failed for '{}': {}", blue_path, e.what());
        return false;
    }

    rt.data_start_offset = blueInfo->dataOffset;
    rt.current_byte_offset = blueInfo->dataOffset;
    rt.total_samples = blueInfo->sampleCount;

    // Build cached metadata for downstream
    build_metadata(spec, rt, meta);

    // Apply endianness conversion on the data portion only
    if (needs_swap) {
        logger()->info("sigmf_source: applying endianness swap on Blue data ({} bytes)", blueInfo->dataSize);
        if (!apply_endianness_swap_blue(rt)) { return false; }
        rt.cached_meta.format.endianness = std::endian::native;
    }

    rt.file_path = spec.file_path;
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
    std::string meta_path = spec.file_path;

    // Extract sigmf-meta file from specs file path
    if (meta_path.ends_with(".sigmf-data")) {
        meta_path = meta_path.substr(0, meta_path.length() - 11) + ".sigmf-meta";
    } else if (meta_path.ends_with(".blue")) {
        // Blue file - derive meta path by replacing .blue with .sigmf-meta
        meta_path = meta_path.substr(0, meta_path.length() - 5) + ".sigmf-meta";
    } else if (!meta_path.ends_with(".sigmf-meta")) {
        // Check if the file exists as-is (raw binary file without .sigmf- extension)
        if (!std::filesystem::exists(spec.file_path) || std::filesystem::is_directory(spec.file_path)) {
            meta_path = meta_path + ".sigmf-meta";
        } else {
            // File exists as-is, derive meta path by appending .sigmf-meta
            meta_path = spec.file_path + ".sigmf-meta";
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
        rt.mmap = std::make_shared<MmapRegion<std::byte>>(rt.data_path, rt.needs_swap);
        if (rt.needs_swap) {
            bool is_blue = (rt.blue_type_code != 0);
            bool ok = is_blue ? apply_endianness_swap_blue(rt) : apply_endianness_swap_raw(rt);
            if (!ok) {
                logger()->error("sigmf_source: failed endianness swap on remap for '{}'", rt.file_path);
                rt.load_error = true;
                return false;
            }
        }
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

auto sigmf_source::process_chunk(file_runtime& rt) -> composite::retval {
    using enum composite::retval;

    if (rt.eof && !rt.loop) {
        return NOOP;
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
        return NOOP;
    }

    // Convert sample count to byte count for buffer copy
    std::size_t bytes_to_read = samples_to_read * rt.bytes_per_sample;

    // Copy chunk bytes from the mmap region into a shared buffer.
    auto* src = rt.mmap->data() + rt.current_byte_offset;
    auto chunk = std::make_shared<std::vector<std::byte>>(src, src + bytes_to_read);
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
        // File-time mode (default): offset based on sample position
        if (rt.effective_sample_rate > 0) {
            offset_seconds = static_cast<double>(rt.samples_sent) / rt.effective_sample_rate;
        } else {
            offset_seconds = 0.0;
        }
    }

    // Split offset into integer seconds and sub-second fractional part
    auto offset_int = static_cast<uint32_t>(offset_seconds);
    double offset_frac = offset_seconds - offset_int;

    ts.seconds = rt.utc_epoch_seconds + offset_int;
    ts.picoseconds = static_cast<uint64_t>(offset_frac * 1e12);

    // Send metadata for specific file
    m_out_port.send_metadata(rt.cached_meta);

    // Send data - chunk is a copy of the mmap region (intentional, not a view).
    // Keeps mmap.reset() as a true release independent of downstream pipeline
    composite::immutable_buffer<std::byte> buf(chunk);
    m_out_port.send_data(std::move(buf), ts);

    return NORMAL;
}

bool sigmf_source::apply_endianness_swap_raw(file_runtime& rt) {
    if (!rt.mmap || !rt.mmap->valid()) {
        return false;
    }

    if (rt.cached_meta.format.endianness == std::endian::native) {
        // Already native postcondition is satisfied without swapping
        return true;
    }

    auto* bytes = reinterpret_cast<uint8_t*>(rt.mmap->data());
    const std::size_t num_bytes = rt.mmap->file_size();

    if (rt.cached_meta.format.bit_width == 16) {
        for (std::size_t i = 0; i + 1 < num_bytes; i += 2) {
            std::swap(bytes[i], bytes[i + 1]);
        }
        logger()->debug("sigmf_source: applied 16-bit byte swap for endianness on mmap region");
    } else if (rt.cached_meta.format.bit_width == 32) {
        for (std::size_t i = 0; i + 3 < num_bytes; i += 4) {
            std::swap(bytes[i], bytes[i + 3]);
            std::swap(bytes[i + 1], bytes[i + 2]);
        }
        logger()->debug("sigmf_source: applied 32-bit byte swap for endianness on mmap region");
    } else if (rt.cached_meta.format.bit_width == 64) {
        for (std::size_t i = 0; i + 7 < num_bytes; i += 8) {
            std::swap(bytes[i], bytes[i + 7]);
            std::swap(bytes[i + 1], bytes[i + 6]);
            std::swap(bytes[i + 2], bytes[i + 5]);
            std::swap(bytes[i + 3], bytes[i + 4]);
        }
        logger()->debug("sigmf_source: applied 64-bit byte swap for endianness on mmap region");
    }
    return true;
}

// blue endianness swap
bool sigmf_source::apply_endianness_swap_blue(file_runtime& rt) {
    if (!rt.mmap || !rt.mmap->valid()) return false;
    if (rt.blue_type_code == 0) return false;

    auto* data_start = reinterpret_cast<uint8_t*>(rt.mmap->data()) + rt.data_start_offset;
    std::size_t data_bytes = rt.total_samples * rt.bytes_per_sample;
    uint32_t bitwidth = rt.cached_meta.format.bit_width;

    blue::byteswapData(data_start, data_bytes / (bitwidth / 8), rt.blue_type_code);

    logger()->debug("sigmf_source: applied Blue byte swap on data region ({} bytes)", data_bytes);

    return true;
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
    return it != rt.cached_meta.annotations.end() ? it->second : "?";
}

#ifndef UNIT_TESTS
extern "C" {
auto create(std::string_view id) -> std::shared_ptr<composite::component> {
    return std::make_shared<sigmf_source>(id);
}
}
#endif
