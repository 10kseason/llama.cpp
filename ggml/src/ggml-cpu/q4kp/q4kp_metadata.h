#pragma once
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#define Q4KP_VNNI_TARGET __attribute__((target("avx2,bmi2,fma,f16c,avx512f,avx512vl,avx512vnni")))

// Both specializations return identical scale/min lanes. Only metadata decode
// differs; the VNNI dot products and FP32 accumulation share the same body.
template<bool P6>
static inline Q4KP_VNNI_TARGET __m128i q4kp_load_group(const uint8_t * code) {
    if constexpr (P6) {
        uint64_t first, last;
        std::memcpy(&first, code, 8);
        std::memcpy(&last, code + 4, 8);
        const uint64_t mask = 0x3f3f3f3f3f3f3f3full;
        return _mm_set_epi64x(static_cast<int64_t>(_pdep_u64(last >> 16, mask)),
                             static_cast<int64_t>(_pdep_u64(first, mask)));
    } else {
        // Original ggml 8x8 packed metadata, without recoding or PDEP.
        uint32_t u[4];
        std::memcpy(u, code, 12);
        u[3] = ((u[2] >> 4) & 0x0f0f0f0fu) | (((u[1] >> 6) & 0x03030303u) << 4);
        const uint32_t mins = u[1] & 0x3f3f3f3fu;
        u[1] = (u[2] & 0x0f0f0f0fu) | (((u[0] >> 6) & 0x03030303u) << 4);
        u[2] = mins;
        u[0] &= 0x3f3f3f3fu;
        return _mm_set_epi32(u[3], u[2], u[1], u[0]);
    }
}
