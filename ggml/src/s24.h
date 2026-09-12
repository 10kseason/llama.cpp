#ifndef GGML_S24_CODEC_H
#define GGML_S24_CODEC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define S24_BLOCK_WEIGHTS 256
#define S24_BLOCK_BYTES 39

/* Research codec, not an upstream GGUF tensor type. Each block is exactly:
 * 2 little-endian FP16 scale bytes, 21 support bytes, 16 sign bytes.
 * importance is NULL or ncols nonnegative finite floats, repeated per row.
 * nrows must be positive and ncols must be a positive multiple of 256.
 * Caller supplies (nrows*ncols/256)*39 writable bytes. Returns that size on
 * success, or 0 for invalid arguments/weights/importance/FP16 scale overflow.
 * Source and destination must not overlap. Discard destination on failure.
 */
size_t s24_pack(const float *values, const float *importance, uint8_t *packed,
                int64_t nrows, int64_t ncols);

/* Caller supplies n_elements/256*39 readable bytes and n_elements writable
 * floats; n_elements must be a positive multiple of 256. Rejects nonfinite or
 * negative scales and unused base-six codes. Returns 1 on success, 0 on error.
 * Source and destination must not overlap. Invalid payloads leave output alone.
 * The caller owns buffer lengths; this interface cannot detect short buffers.
 */
int s24_unpack(const uint8_t *packed, float *values, int64_t n_elements);

#ifdef __cplusplus
}
#endif

#endif
