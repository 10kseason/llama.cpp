#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Fixed effective mode: 0 off, 1 P6, 2 VNNI/P6, 3 wide/P6, 4 VNNI/original.
int q4kp_runtime_mode(void);
int q4kp_runtime_enabled(void);
// 0 recoded tensors; 1 metadata bytes recoded in place (not extra memory);
// 2 matrix operations using new kernels; 3 extra allocation bytes (zero);
// 4 selected tensor uploads, including original-layout VNNI (no recode).
uint64_t q4kp_runtime_stat(int index);
#ifdef __cplusplus
}
#endif
