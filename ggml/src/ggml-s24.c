// Private GGUF type 63. The compact file layout is shared by CPU and Vulkan.
#include "ggml-s24.h"
#include "s24.h"

#include <stdint.h>
#include <string.h>

#define S24_BASE6_LIMIT UINT64_C(2821109907456) // 6^16

static uint64_t compact_word(const uint8_t * block, unsigned slot) {
    const unsigned bit = slot * 42;
    uint64_t word = 0;
    for (unsigned byte = 0; byte < 6; ++byte) {
        word |= (uint64_t) block[2 + bit / 8 + byte] << (8 * byte);
    }
    return (word >> (bit % 8)) & ((UINT64_C(1) << 42) - 1);
}

static unsigned direct_index(const uint8_t * block, unsigned group) {
    const unsigned bit = 3 * group;
    unsigned word = block[2 + bit / 8];
    if (bit % 8 > 5) {
        word |= (unsigned) block[3 + bit / 8] << 8;
    }
    return (word >> (bit % 8)) & 7;
}

static bool valid_scale(const uint8_t * block) {
    const unsigned half = block[0] | ((unsigned) block[1] << 8);
    // Independent of initialization of GGML conversion tables. Signed zero is
    // numerically nonnegative and is preserved to keep the wire format exact.
    return (half & 0x7c00) != 0x7c00 && (!(half & 0x8000) || !(half & 0x7fff));
}

bool ggml_s24_validate_compact(const void * src, size_t nbytes) {
    if (!src || nbytes == 0 || nbytes % 39) {
        return false;
    }
    const uint8_t * bytes = src;
    for (size_t offset = 0; offset < nbytes; offset += 39) {
        const uint8_t * block = bytes + offset;
        if (!valid_scale(block)) {
            return false;
        }
        for (unsigned slot = 0; slot < 4; ++slot) {
            if (compact_word(block, slot) >= S24_BASE6_LIMIT) {
                return false;
            }
        }
    }
    return true;
}

bool ggml_s24_validate_direct42(const void * src, size_t nbytes) {
    if (!src || nbytes == 0 || nbytes % 42) {
        return false;
    }
    const uint8_t * bytes = src;
    for (size_t offset = 0; offset < nbytes; offset += 42) {
        const uint8_t * block = bytes + offset;
        if (!valid_scale(block)) {
            return false;
        }
        for (unsigned group = 0; group < 64; ++group) {
            if (direct_index(block, group) >= 6) {
                return false;
            }
        }
    }
    return true;
}

static bool conversion_buffers(const void * src, size_t src_bytes, size_t src_block,
                                void * dst, size_t dst_bytes, size_t dst_block) {
    if (!src || !dst || src_bytes == 0 || src_bytes % src_block ||
        src_bytes / src_block > SIZE_MAX / dst_block ||
        dst_bytes != src_bytes / src_block * dst_block) {
        return false;
    }
    // Difference-based overlap checks avoid wrapping pointer-end arithmetic.
    const uintptr_t source = (uintptr_t) src;
    const uintptr_t target = (uintptr_t) dst;
    return source < target ? target - source >= src_bytes : source - target >= dst_bytes;
}

bool ggml_s24_compact_to_direct42(const void * src, size_t src_bytes,
                                 void * dst, size_t dst_bytes) {
    if (!conversion_buffers(src, src_bytes, 39, dst, dst_bytes, 42) ||
        !ggml_s24_validate_compact(src, src_bytes)) {
        return false;
    }
    const uint8_t * input = src;
    uint8_t * output = dst;
    for (size_t block = 0; block < src_bytes / 39; ++block) {
        const uint8_t * in = input + block * 39;
        uint8_t * out = output + block * 42;
        memset(out, 0, 42);
        memcpy(out, in, 2);
        memcpy(out + 26, in + 23, 16);
        for (unsigned slot = 0; slot < 4; ++slot) {
            uint64_t word = compact_word(in, slot);
            for (unsigned digit = 0; digit < 16; ++digit) {
                const unsigned index = (unsigned) (word % 6);
                word /= 6;
                const unsigned bit = 3 * (slot * 16 + digit);
                out[2 + bit / 8] |= (uint8_t) (index << (bit % 8));
                if (bit % 8 > 5) {
                    out[3 + bit / 8] |= (uint8_t) (index >> (8 - bit % 8));
                }
            }
        }
    }
    return true;
}

bool ggml_s24_direct42_to_compact(const void * src, size_t src_bytes,
                                 void * dst, size_t dst_bytes) {
    if (!conversion_buffers(src, src_bytes, 42, dst, dst_bytes, 39) ||
        !ggml_s24_validate_direct42(src, src_bytes)) {
        return false;
    }
    const uint8_t * input = src;
    uint8_t * output = dst;
    for (size_t block = 0; block < src_bytes / 42; ++block) {
        const uint8_t * in = input + block * 42;
        uint8_t * out = output + block * 39;
        memset(out, 0, 39);
        memcpy(out, in, 2);
        memcpy(out + 23, in + 26, 16);
        for (unsigned slot = 0; slot < 4; ++slot) {
            uint64_t word = 0;
            for (int digit = 15; digit >= 0; --digit) {
                word = word * 6 + direct_index(in, slot * 16 + (unsigned) digit);
            }
            const unsigned bit = slot * 42;
            word <<= bit % 8;
            for (unsigned byte = 0; byte < 6; ++byte) {
                out[2 + bit / 8 + byte] |= (uint8_t) (word >> (8 * byte));
            }
        }
    }
    return true;
}

void quantize_row_s24_ref(const float * src, void * dst, int64_t n) {
    GGML_ASSERT(n > 0 && n % 256 == 0 && (uint64_t) n / 256 <= SIZE_MAX / 39);
    GGML_ASSERT(s24_pack(src, NULL, dst, 1, n) == (size_t) (n / 256) * 39);
}

size_t quantize_s24(const float * src, void * dst, int64_t rows, int64_t columns, const float * importance) {
    GGML_ASSERT(rows > 0 && columns > 0 && columns % 256 == 0);
    GGML_ASSERT((uint64_t) rows <= INT64_MAX / (uint64_t) columns);
    const uint64_t blocks = (uint64_t) rows * (uint64_t) columns / 256;
    GGML_ASSERT(blocks <= SIZE_MAX / 39);
    const size_t count = s24_pack(src, importance, dst, rows, columns);
    GGML_ASSERT(count == (size_t) blocks * 39);
    return count;
}

void dequantize_row_s24(const void * src, float * dst, int64_t n) {
    GGML_ASSERT(s24_unpack(src, dst, n));
}

bool validate_s24_data(const void * src, size_t nbytes) {
    return ggml_s24_validate_compact(src, nbytes);
}
