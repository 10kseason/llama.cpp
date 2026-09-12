// S24 stays packed throughout the dot product; the fixed lookup table is 4.5 KiB.
#include "s24-dot.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#if defined(__AVX2__)
#include <immintrin.h>

// Two four-weight groups: each has six supports and two signs. Encode the
// ternary values as unsigned bytes {0,1,2} = {-1,0,1}+1, allowing maddubs to
// multiply directly by signed Q8 bytes, including -128. The selected positions
// start at zero; each positive sign changes its byte to two.
#define S24_GROUP(P, Q, S) \
    ((UINT32_C(0x01010101) ^ ((UINT32_C(1) << (8*(P))) | (UINT32_C(1) << (8*(Q))))) | \
     (((uint32_t)(S) & 1u) << (8*(P)+1)) | ((((uint32_t)(S) >> 1) & 1u) << (8*(Q)+1)))
#define S24_ENTRY(P0, Q0, P1, Q1, S) \
    ((uint64_t)S24_GROUP(P0, Q0, (S)&3) | ((uint64_t)S24_GROUP(P1, Q1, (S)>>2) << 32))
#define S24_LUT16(P0, Q0, P1, Q1) { \
    S24_ENTRY(P0,Q0,P1,Q1, 0), S24_ENTRY(P0,Q0,P1,Q1, 1), \
    S24_ENTRY(P0,Q0,P1,Q1, 2), S24_ENTRY(P0,Q0,P1,Q1, 3), \
    S24_ENTRY(P0,Q0,P1,Q1, 4), S24_ENTRY(P0,Q0,P1,Q1, 5), \
    S24_ENTRY(P0,Q0,P1,Q1, 6), S24_ENTRY(P0,Q0,P1,Q1, 7), \
    S24_ENTRY(P0,Q0,P1,Q1, 8), S24_ENTRY(P0,Q0,P1,Q1, 9), \
    S24_ENTRY(P0,Q0,P1,Q1,10), S24_ENTRY(P0,Q0,P1,Q1,11), \
    S24_ENTRY(P0,Q0,P1,Q1,12), S24_ENTRY(P0,Q0,P1,Q1,13), \
    S24_ENTRY(P0,Q0,P1,Q1,14), S24_ENTRY(P0,Q0,P1,Q1,15) }
#define S24_FIRST_GROUPS(P1, Q1) \
    S24_LUT16(0,1,P1,Q1), S24_LUT16(0,2,P1,Q1), S24_LUT16(0,3,P1,Q1), \
    S24_LUT16(1,2,P1,Q1), S24_LUT16(1,3,P1,Q1), S24_LUT16(2,3,P1,Q1)

// support index = first mask + 6*second mask; sign bits retain file order.
static const uint64_t s24_trits[36][16] = {
    S24_FIRST_GROUPS(0,1), S24_FIRST_GROUPS(0,2), S24_FIRST_GROUPS(0,3),
    S24_FIRST_GROUPS(1,2), S24_FIRST_GROUPS(1,3), S24_FIRST_GROUPS(2,3)
};

#undef S24_FIRST_GROUPS
#undef S24_LUT16
#undef S24_ENTRY
#undef S24_GROUP
#endif

static inline uint64_t s24_support_word(const uint8_t * x, unsigned slot) {
    const unsigned bit = slot * 42;
    uint64_t word = 0;
    // Read six bytes only: the final 42-bit field ends at byte 22. No block
    // padding or readable bytes beyond the 39-byte payload are assumed.
    for (unsigned byte = 0; byte < 6; ++byte) {
        word |= (uint64_t)x[2 + bit / 8 + byte] << (8 * byte);
    }
    word = (word >> (bit % 8)) & ((UINT64_C(1) << 42) - 1);
    GGML_ASSERT(word < UINT64_C(2821109907456));
    return word;
}

#if defined(__AVX2__)
static inline int s24_dot_block(const uint8_t * x, const block_q8_K * y) {
    __m256i accumulator = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    for (unsigned slot = 0; slot < 4; ++slot) {
        uint64_t word = s24_support_word(x, slot);
        uint32_t signs = (uint32_t)x[23 + 4*slot] | ((uint32_t)x[24 + 4*slot] << 8) |
                         ((uint32_t)x[25 + 4*slot] << 16) | ((uint32_t)x[26 + 4*slot] << 24);
        for (unsigned part = 0; part < 2; ++part) {
            // Decode eight support masks via four base-36 digits, halving the
            // serial quotient chain without expanding/caching model weights.
            const uint64_t a = s24_trits[word % 36][signs & 15]; word /= 36; signs >>= 4;
            const uint64_t b = s24_trits[word % 36][signs & 15]; word /= 36; signs >>= 4;
            const uint64_t c = s24_trits[word % 36][signs & 15]; word /= 36; signs >>= 4;
            const uint64_t d = s24_trits[word % 36][signs & 15]; word /= 36; signs >>= 4;
            const __m256i trits = _mm256_set_epi64x((int64_t)d, (int64_t)c, (int64_t)b, (int64_t)a);
            const __m256i q8 = _mm256_loadu_si256((const __m256i *)(y->qs + 64*slot + 32*part));
            // A pair lies in [-512,508], so maddubs cannot saturate int16.
            const __m256i pairs = _mm256_maddubs_epi16(trits, q8);
            accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(pairs, ones));
        }
    }
    // Remove the +1 offset once using Q8_K's exact sums of each 16 activations.
    const __m256i sums = _mm256_loadu_si256((const __m256i *)y->bsums);
    accumulator = _mm256_sub_epi32(accumulator, _mm256_madd_epi16(sums, ones));
    __m128i low = _mm_add_epi32(_mm256_castsi256_si128(accumulator),
                               _mm256_extracti128_si256(accumulator, 1));
    low = _mm_add_epi32(low, _mm_shuffle_epi32(low, _MM_SHUFFLE(1,0,3,2)));
    low = _mm_add_epi32(low, _mm_shuffle_epi32(low, _MM_SHUFFLE(2,3,0,1)));
    return _mm_cvtsi128_si32(low);
}
#else
static inline int s24_dot_block(const uint8_t * x, const block_q8_K * y) {
    static const uint8_t pairs[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
    int dot = 0;
    for (unsigned slot = 0; slot < 4; ++slot) {
        uint64_t word = s24_support_word(x, slot);
        for (unsigned digit = 0; digit < 16; ++digit) {
            const unsigned mask = (unsigned)(word % 6);
            word /= 6;
            const unsigned group = slot * 16 + digit;
            const unsigned sign_bit = group * 2;
            const unsigned signs = (x[23 + sign_bit / 8] >> (sign_bit % 8)) & 3;
            const int a = y->qs[4 * group + pairs[mask][0]];
            const int b = y->qs[4 * group + pairs[mask][1]];
            dot += (signs & 1) ? a : -a;
            dot += (signs & 2) ? b : -b;
        }
    }
    return dot;
}
#endif

void ggml_vec_dot_s24_q8_K(int n, float * result, size_t bs, const void * vx, size_t bx,
                          const void * vy, size_t by, int nrc) {
    GGML_ASSERT(nrc == 1 && n % 256 == 0);
    (void)bs; (void)bx; (void)by;
    const uint8_t * x = vx;
    const block_q8_K * y = vy;
    float sum = 0;
    for (int block = 0; block < n / 256; ++block, x += 39) {
        const ggml_half half = (ggml_half)((uint16_t)x[0] | ((uint16_t)x[1] << 8));
        const float scale = GGML_FP16_TO_FP32(half);
        const int dot = s24_dot_block(x, y + block);
        // CPU variants may enable FMA. Keep the original kernel's two rounded
        // multiplies and ordered float additions, even in those variants.
        const volatile float product = scale * y[block].d;
        const volatile float term = product * (float)dot;
        sum += term;
    }
    *result = sum;
}
