/*
 * MIDAS Blue File Format Support
 *
 * Adapted from REDHAWK blueFileLib (GPL v3)
 * Simplified header-only implementation for sigmf_source component.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace blue {

// Endianness magic values
constexpr uint32_t IEEE = 0x49454545;  // "IEEE" - native endianness
constexpr uint32_t EEEI = 0x45454549;  // "EEEI" - reversed endianness

// Type 1000 adjunct (signal files)
struct Adjunct1000 {
    double xStart;      // Abscissa value for first sample
    double xDelta;      // Abscissa interval between samples (1/sample_rate for time-domain)
    uint32_t xUnits;    // Units code for abscissa values
};

// Adjunct for byte swapping (covers all adjunct types)
struct AdjunctSwap {
    int64_t x1;   // 256
    int64_t x2;   // 264
    int32_t x3;   // 272
    int32_t x4;   // 276
    int64_t x5;   // 280
    int64_t x6;   // 288
    int32_t x7;   // 296
    int32_t x8;   // 300
};

// 512-byte Header Control Block structure
struct alignas(1) HeaderControlBlock {
    char version[4];              // 0: "BLUE"

    union {
        char headerRep[4];
        uint32_t headerEndianness;
    };                            // 4: Header endianness (IEEE/EEEI)

    union {
        char dataRep[4];
        uint32_t dataEndianness;
    };                            // 8: Data endianness

    uint32_t detached;            // 12: Detached header flag
    uint32_t isProtected;         // 16: Protected from overwrite
    uint32_t pipe;                // 20: Pipe mode (N/A)
    uint32_t extStart;            // 24: Extended header start (512-byte blocks)
    uint32_t extSize;             // 28: Extended header size in bytes

    union {
        double dataStart;
        int64_t dataStart_i;
    };                            // 32: Data start offset in bytes

    union {
        double dataSize;
        int64_t dataSize_i;
    };                            // 40: Data size in bytes

    uint32_t typeCode;            // 48: File type code (1000, 2000, etc.)

    union {
        char formatCode[2];
        uint16_t format_u16;
    };                            // 52: Data format code (e.g., "CF")

    int16_t flagMask;             // 54: 16-bit flag mask

    union {
        double timeCode;
        int64_t timeCode_i;
    };                            // 56: Time code field

    uint16_t inLets;              // 64: Inlet owner
    uint16_t outLets;             // 66: Number of outlets
    uint32_t outMask;             // 68: Outlet async mask
    uint32_t pipeLoc;             // 72: Pipe location
    uint32_t pipeSize;            // 76: Pipe size in bytes

    union {
        double inBytes;
        int64_t inBytes_i;
    };                            // 80: Next input byte

    union {
        double outByte;
        int64_t outByte_i;
    };                            // 88: Next out byte (cumulative)

    union {
        double outBytes[8];
        int64_t outBytes_i[8];
    };                            // 96: Next out byte (each outlet)

    int32_t keywordLength;        // 160: Length of keyword string
    char keywords[92];            // 164: User-defined keyword string

    union {
        char adjunct[256];
        Adjunct1000 adjunct1000;
        AdjunctSwap adjunctSwap;
    };                            // 256: Type-specific adjunct
};

static_assert(sizeof(HeaderControlBlock) == 512, "HCB must be 512 bytes");

// Byte swap utilities
namespace detail {

inline void swap16(void* ptr) {
    auto* p = static_cast<uint8_t*>(ptr);
    std::swap(p[0], p[1]);
}

inline void swap32(void* ptr) {
    auto* p = static_cast<uint8_t*>(ptr);
    std::swap(p[0], p[3]);
    std::swap(p[1], p[2]);
}

inline void swap64(void* ptr) {
    auto* p = static_cast<uint8_t*>(ptr);
    std::swap(p[0], p[7]);
    std::swap(p[1], p[6]);
    std::swap(p[2], p[5]);
    std::swap(p[3], p[4]);
}

template<typename T>
inline void swapN(T* ptr, int count) {
    for (int i = 0; i < count; ++i) {
        if constexpr (sizeof(T) == 2) swap16(ptr + i);
        else if constexpr (sizeof(T) == 4) swap32(ptr + i);
        else if constexpr (sizeof(T) == 8) swap64(ptr + i);
    }
}

} // namespace detail

// Byte swap the header in-place
inline void byteswapHeader(HeaderControlBlock* hcb) {
    detail::swapN(&hcb->detached, 5);           // detached through extSize
    detail::swap64(&hcb->dataStart_i);
    detail::swap64(&hcb->dataSize_i);
    detail::swap32(&hcb->typeCode);
    detail::swap16(&hcb->flagMask);
    detail::swap64(&hcb->timeCode_i);
    detail::swap16(&hcb->inLets);
    detail::swap16(&hcb->outLets);
    detail::swapN(&hcb->outMask, 3);            // outMask, pipeLoc, pipeSize
    detail::swap64(&hcb->inBytes_i);
    detail::swap64(&hcb->outByte_i);
    detail::swapN(hcb->outBytes_i, 8);
    detail::swap32(&hcb->keywordLength);

    // Adjunct swap
    detail::swap64(&hcb->adjunctSwap.x1);
    detail::swap64(&hcb->adjunctSwap.x2);
    detail::swap32(&hcb->adjunctSwap.x3);
    detail::swap32(&hcb->adjunctSwap.x4);
    detail::swap64(&hcb->adjunctSwap.x5);
    detail::swap64(&hcb->adjunctSwap.x6);
    detail::swap32(&hcb->adjunctSwap.x7);
    detail::swap32(&hcb->adjunctSwap.x8);
}

// Byte swap data buffer based on format code
inline void byteswapData(void* buffer, std::size_t count, char formatCode) {
    switch (formatCode) {
        case 'I':  // int16
        case 'U':  // uint16
            detail::swapN(static_cast<int16_t*>(buffer), static_cast<int>(count));
            break;
        case 'L':  // int32
        case 'V':  // uint32
        case 'F':  // float32
            detail::swapN(static_cast<int32_t*>(buffer), static_cast<int>(count));
            break;
        case 'X':  // int64
        case 'D':  // float64
            detail::swapN(static_cast<int64_t*>(buffer), static_cast<int>(count));
            break;
        // 8-bit types (B, O) don't need swapping
    }
}

// Format code information
struct FormatInfo {
    char rankCode;      // 'C' = complex, 'R' = real, 'S' = scalar
    char typeCode;      // 'B', 'I', 'L', 'F', 'D', etc.
    bool isComplex;
    uint32_t scalarSize;  // Bytes per scalar element
    uint32_t sampleSize;  // Bytes per sample (scalar * 2 if complex)

    // Get scalar size from type code
    static uint32_t getScalarSize(char tc) {
        switch (tc) {
            case 'B': case 'O': return 1;   // byte, unsigned byte
            case 'I': case 'U': return 2;   // int16, uint16
            case 'L': case 'V': case 'F': return 4;  // int32, uint32, float
            case 'X': case 'D': return 8;   // int64, double
            default: return 0;
        }
    }

    // Parse format code string (e.g., "CF", "CI", "SF")
    static std::optional<FormatInfo> parse(const char* fc) {
        if (!fc || fc[0] == '\0' || fc[1] == '\0') {
            return std::nullopt;
        }

        FormatInfo info;
        info.rankCode = fc[0];
        info.typeCode = fc[1];

        // Determine if complex
        switch (info.rankCode) {
            case 'C': info.isComplex = true; break;
            case 'R': case 'S': info.isComplex = false; break;
            default:
                // Numeric rank codes (0-9) or letter codes (A-Z for predefined counts)
                if (info.rankCode >= '0' && info.rankCode <= '9') {
                    info.isComplex = false;  // Treat as vector
                } else {
                    info.isComplex = false;
                }
        }

        info.scalarSize = getScalarSize(info.typeCode);
        if (info.scalarSize == 0) {
            return std::nullopt;
        }

        info.sampleSize = info.isComplex ? info.scalarSize * 2 : info.scalarSize;
        return info;
    }
};

// High-level Blue file reader result
struct BlueFileInfo {
    bool valid{false};
    std::size_t dataOffset{0};      // Byte offset where data starts
    std::size_t dataSize{0};        // Data size in bytes
    std::size_t sampleCount{0};     // Number of samples
    FormatInfo format;
    double sampleRate{0.0};         // Derived from xDelta if available
    bool needsDataSwap{false};      // True if data endianness differs from host
    uint32_t typeCode{0};
};

// Check if a file is a MIDAS Blue file by reading magic
inline bool isBlueFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    char magic[4];
    in.read(magic, 4);
    return in.gcount() == 4 && std::strncmp(magic, "BLUE", 4) == 0;
}

// Parse Blue file header and return info
inline std::optional<BlueFileInfo> parseBlueFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }

    HeaderControlBlock hcb{};
    in.read(reinterpret_cast<char*>(&hcb), sizeof(hcb));
    if (in.gcount() != sizeof(hcb)) {
        return std::nullopt;
    }

    // Verify magic
    if (std::strncmp(hcb.version, "BLUE", 4) != 0) {
        return std::nullopt;
    }

    // Check and fix header endianness
    bool headerSwapped = (hcb.headerEndianness == EEEI);
    if (headerSwapped) {
        byteswapHeader(&hcb);
    }

    // Parse format code
    auto formatOpt = FormatInfo::parse(hcb.formatCode);
    if (!formatOpt) {
        return std::nullopt;
    }

    BlueFileInfo info;
    info.valid = true;
    info.format = *formatOpt;
    info.dataOffset = static_cast<std::size_t>(hcb.dataStart);
    info.dataSize = static_cast<std::size_t>(hcb.dataSize);
    info.sampleCount = info.dataSize / info.format.sampleSize;
    info.typeCode = hcb.typeCode;

    // Check if data needs byte swapping (data endianness differs from host)
    // After header swap, headerEndianness should be IEEE (native)
    // dataEndianness tells us about the data
    constexpr bool hostLittleEndian = (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
    bool dataIsBigEndian = (hcb.dataEndianness == EEEI) != headerSwapped;
    // If header was swapped, the original data rep was opposite of what we read
    // Actually simpler: after byteswap, if dataEndianness == EEEI, data is non-native
    info.needsDataSwap = (hcb.dataEndianness == EEEI);

    // Extract sample rate from xDelta if this is a Type 1000 file
    if ((hcb.typeCode / 1000) == 1 && hcb.adjunct1000.xDelta > 0) {
        info.sampleRate = 1.0 / hcb.adjunct1000.xDelta;
    }

    return info;
}

} // namespace blue