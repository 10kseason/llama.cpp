#pragma once

#include "ggml.h"

// GGML CPU internal header. The encoder/decoder remain in ggml-base.
#ifdef __cplusplus
extern "C" {
#endif

void ggml_vec_dot_s24_q8_K(int n, float * result, size_t bs, const void * vx, size_t bx,
                          const void * vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
