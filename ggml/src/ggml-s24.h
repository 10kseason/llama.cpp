#pragma once
#include "ggml.h"
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif
GGML_API void quantize_row_s24_ref(const float * src, void * dst, int64_t n);
GGML_API void dequantize_row_s24(const void * src, float * dst, int64_t n);
GGML_API size_t quantize_s24(const float * src, void * dst, int64_t rows, int64_t columns, const float * importance);
GGML_API bool validate_s24_data(const void * src, size_t nbytes);

// S24 stores 256 weights in 39 bytes. Direct42 is an upload-only representation,
// not a different GGUF tensor type. Validators reject null/empty/truncated data,
// nonfinite or negative scales, unused base-six codes and direct indices 6/7.
GGML_API bool ggml_s24_validate_compact(const void * src, size_t nbytes);
GGML_API bool ggml_s24_validate_direct42(const void * src, size_t nbytes);

// Exact lengths and disjoint buffers are required. Invalid input leaves all of
// dst unchanged. Padding belongs to the caller; exclude it from dst_bytes.
// Conversion preserves scale/sign bits and round-trips valid data byte-for-byte.
GGML_API bool ggml_s24_compact_to_direct42(const void * src, size_t src_bytes,
                                         void * dst, size_t dst_bytes);
GGML_API bool ggml_s24_direct42_to_compact(const void * src, size_t src_bytes,
                                         void * dst, size_t dst_bytes);
#ifdef __cplusplus
}
#endif
