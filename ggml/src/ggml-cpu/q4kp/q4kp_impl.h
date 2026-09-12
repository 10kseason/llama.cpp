#pragma once
#include <stddef.h>

// Internal ISA bodies. Public callers must use the checked C entries.
#define Q4KP_DECLARE_IMPL(name) void name(int, float *, size_t, const void *, const void *, int, int)
Q4KP_DECLARE_IMPL(q4kp_gemv_impl);
Q4KP_DECLARE_IMPL(q4kp_gemm_impl);
Q4KP_DECLARE_IMPL(q4kp_vnni_gemv_impl);
Q4KP_DECLARE_IMPL(q4kp_vnni_gemm_impl);
Q4KP_DECLARE_IMPL(q4kp_vnni_original_gemv_impl);
Q4KP_DECLARE_IMPL(q4kp_vnni_original_gemm_impl);
Q4KP_DECLARE_IMPL(q4kp_wide_gemv_impl);
#undef Q4KP_DECLARE_IMPL
