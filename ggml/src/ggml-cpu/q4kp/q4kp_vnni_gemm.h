#ifndef Q125_Q4KP_VNNI_GEMM_H
#define Q125_Q4KP_VNNI_GEMM_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Same P6 block_q4_Kx8 / block_q8_Kx4 GEMM contract as q4kp_gemm.
 * The public entry checks contracts and ISA before entering its targeted body.
 * No recoding, allocations, or floating-point reassociation occur.
 */
void q4kp_vnni_gemm(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);
/* Original ggml 8x8 packed layout; must NOT receive P6 metadata. */
void q4kp_vnni_original_gemm(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);
#ifdef __cplusplus
}
#endif
#endif
