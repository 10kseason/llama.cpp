// Compile this translation unit without optional ISA flags. Contract failures
// must reach GGML_ABORT before entering any ISA-targeted implementation.
#include "q4kp_impl.h"
#include "ggml.h"
#include "q4kp_kernel.h"
#include "q4kp_vnni_kernel.h"
#include "q4kp_vnni_gemm.h"
#include "q4kp_wide_kernel.h"
#include <cstdint>

using kernel = void (*)(int, float *, size_t, const void *, const void *, int, int);

extern "C" int q4kp_cpu_supported(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi2") &&
        __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
}

extern "C" int q4kp_vnni_supported(void) {
    return q4kp_cpu_supported() && __builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni");
}

extern "C" int q4kp_wide_supported(void) {
    return q4kp_vnni_supported() && __builtin_cpu_supports("avx512bw");
}

static size_t checked_mul(size_t a, size_t b) {
    GGML_ASSERT(b == 0 || a <= SIZE_MAX / b);
    return a * b;
}

static uintptr_t checked_end(const void * p, size_t bytes) {
    const auto start = reinterpret_cast<uintptr_t>(p);
    GGML_ASSERT(bytes <= UINTPTR_MAX - start);
    return start + bytes;
}

static void invoke(kernel fn, int isa, bool gemm, int n, float * s, size_t bs,
                   const void * x, const void * y, int nr, int nc) {
    GGML_ASSERT(n > 0 && n % 256 == 0);
    GGML_ASSERT(nc > 0 && nc % 8 == 0);
    GGML_ASSERT(gemm ? (nr > 0 && nr % 4 == 0) : nr == 1);
    GGML_ASSERT(!gemm || bs >= size_t(nc));
    GGML_ASSERT(s != nullptr && x != nullptr && y != nullptr);
    GGML_ASSERT(reinterpret_cast<uintptr_t>(s) % alignof(float) == 0);
    GGML_ASSERT(reinterpret_cast<uintptr_t>(x) % alignof(uint16_t) == 0);
    // GEMM uses aligned 16-byte loads for block_q8_Kx4.d.
    GGML_ASSERT(reinterpret_cast<uintptr_t>(y) % (gemm ? 16 : alignof(float)) == 0);
    const size_t prior_rows = gemm ? checked_mul(size_t(nr - 1), bs) : 0;
    GGML_ASSERT(size_t(nc) <= SIZE_MAX - prior_rows);
    const size_t output_bytes = checked_mul(prior_rows + size_t(nc), sizeof(float));
    const size_t weight_bytes = checked_mul(checked_mul(size_t(n / 256), size_t(nc / 8)), 1152);
    const size_t activation_bytes = checked_mul(checked_mul(size_t(n / 256), size_t(gemm ? nr / 4 : 1)), gemm ? 1168 : 292);
    const auto se = checked_end(s, output_bytes);
    const auto xe = checked_end(x, weight_bytes);
    const auto ye = checked_end(y, activation_bytes);
    GGML_ASSERT(se <= reinterpret_cast<uintptr_t>(x) || xe <= reinterpret_cast<uintptr_t>(s));
    GGML_ASSERT(se <= reinterpret_cast<uintptr_t>(y) || ye <= reinterpret_cast<uintptr_t>(s));
    GGML_ASSERT(isa == 2 ? q4kp_wide_supported() : isa == 1 ? q4kp_vnni_supported() : q4kp_cpu_supported());
    fn(n, s, bs, x, y, nr, nc);
}

#define ENTRY(name, isa, matrix) \
    extern "C" void name(int n, float * s, size_t bs, const void * x, const void * y, int nr, int nc) { \
        invoke(name##_impl, isa, matrix, n, s, bs, x, y, nr, nc); \
    }
ENTRY(q4kp_gemv, 0, false)
ENTRY(q4kp_gemm, 0, true)
ENTRY(q4kp_vnni_gemv, 1, false)
ENTRY(q4kp_vnni_gemm, 1, true)
ENTRY(q4kp_vnni_original_gemv, 1, false)
ENTRY(q4kp_vnni_original_gemm, 1, true)
ENTRY(q4kp_wide_gemv, 2, false)
#undef ENTRY
