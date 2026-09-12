#include "s24.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>

static const uint8_t s24_pairs[6][2] = {
    {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
};

#define S24_BASE6_LIMIT UINT64_C(2821109907456) /* 6**16 */
#define S24_CODE_MASK ((UINT64_C(1) << 42) - 1)

static void s24_select(const double score[4], uint8_t pair[2]) {
    unsigned first = 0, second = 1;
    if (score[second] > score[first]) {
        first = 1;
        second = 0;
    }
    for (unsigned i = 2; i < 4; ++i) {
        if (score[i] > score[first]) {
            second = first;
            first = i;
        } else if (score[i] > score[second]) {
            second = i;
        }
    }
    /* Strict comparisons retain the lowest indices when scores tie. */
    pair[0] = (uint8_t) (first < second ? first : second);
    pair[1] = (uint8_t) (first < second ? second : first);
}

static int s24_half_scale(double value, ggml_fp16_t *result) {
    /* NumPy fits in double before storing half. A float intermediate can round
     * onto a half midpoint, so compare adjacent half values after the GGML
     * conversion to retain direct round-to-nearest, ties-to-even semantics. */
    if (!isfinite(value) || value < 0 || value >= 65520.0) {
        return 0;
    }
    ggml_fp16_t candidate = GGML_FP32_TO_FP16((float) value);
    if (candidate > UINT16_C(0x7bff)) {
        candidate = UINT16_C(0x7bff);
    }
    ggml_fp16_t best = candidate;
    double error = fabs(value - (double) GGML_FP16_TO_FP32(candidate));
    const unsigned low = candidate > 0 ? candidate - 1 : 0;
    const unsigned high = candidate < 0x7bff ? candidate + 1 : 0x7bff;
    for (unsigned bits = low; bits <= high; ++bits) {
        const double distance = fabs(value - (double) GGML_FP16_TO_FP32((ggml_fp16_t) bits));
        if (distance < error || (distance == error && !(bits & 1u))) {
            best = (ggml_fp16_t) bits;
            error = distance;
        }
    }
    *result = best;
    return 1;
}

static int s24_fit(const float *x, const float *importance,
                   uint8_t best_pairs[64][2], ggml_fp16_t *best_scale) {
    double magnitude[256], h[256];
    uint8_t pairs[64][2];
    int observed = 0;
    for (unsigned i = 0; i < 256; ++i) {
        magnitude[i] = fabs((double) x[i]);
        h[i] = importance ? (double) importance[i] : 1.0;
        observed |= h[i] > 0;
    }
    if (!observed) {
        for (unsigned i = 0; i < 256; ++i) {
            h[i] = 1.0;
        }
    }
    for (unsigned group = 0; group < 64; ++group) {
        s24_select(magnitude + group * 4, pairs[group]);
    }

    double best_loss = INFINITY;
    for (unsigned iteration = 0; iteration < 4; ++iteration) {
        double numerator = 0.0, denominator = 0.0;
        for (unsigned group = 0; group < 64; ++group) {
            for (unsigned selected = 0; selected < 2; ++selected) {
                const unsigned i = group * 4 + pairs[group][selected];
                numerator += h[i] * magnitude[i];
                denominator += h[i];
            }
        }
        ggml_fp16_t stored_scale;
        if (!s24_half_scale(denominator > 0 ? numerator / denominator : 0, &stored_scale)) {
            return 0;
        }
        const double scale = GGML_FP16_TO_FP32(stored_scale);
        double loss = 0;
        for (unsigned group = 0; group < 64; ++group) {
            for (unsigned lane = 0; lane < 4; ++lane) {
                const unsigned i = group * 4 + lane;
                const double represented = lane == pairs[group][0] || lane == pairs[group][1] ? scale : 0;
                const double difference = magnitude[i] - represented;
                loss += h[i] * difference * difference;
            }
        }
        /* Evaluate the stored, decoded half scale, and retain the first best
         * iterate. Refinement therefore cannot worsen this weighted objective. */
        if (loss < best_loss) {
            best_loss = loss;
            *best_scale = stored_scale;
            memcpy(best_pairs, pairs, sizeof(pairs));
        }
        for (unsigned group = 0; group < 64; ++group) {
            double gain[4];
            for (unsigned lane = 0; lane < 4; ++lane) {
                const unsigned i = group * 4 + lane;
                gain[lane] = h[i] * (2 * scale * magnitude[i] - scale * scale);
            }
            s24_select(gain, pairs[group]);
        }
    }
    return 1;
}

static unsigned s24_pair_code(const uint8_t pair[2]) {
    for (unsigned code = 0; code < 6; ++code) {
        if (s24_pairs[code][0] == pair[0] && s24_pairs[code][1] == pair[1]) {
            return code;
        }
    }
    return 0; /* Only called with pairs produced by s24_select. */
}

static uint64_t s24_read_code(const uint8_t *block, unsigned slot) {
    const unsigned bit = slot * 42;
    uint64_t word = 0;
    for (unsigned byte = 0; byte < 6; ++byte) {
        word |= (uint64_t) block[2 + bit / 8 + byte] << (8 * byte);
    }
    return (word >> (bit % 8)) & S24_CODE_MASK;
}

size_t s24_pack(const float *values, const float *importance, uint8_t *packed,
                int64_t nrows, int64_t ncols) {
    if (!values || !packed || nrows <= 0 || ncols <= 0 || ncols % 256 ||
        (uint64_t) nrows > INT64_MAX / (uint64_t) ncols) {
        return 0;
    }
    const uint64_t count = (uint64_t) nrows * (uint64_t) ncols;
    if (count > SIZE_MAX / sizeof(float) || count / 256 > SIZE_MAX / 39) {
        return 0;
    }
    for (uint64_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    if (importance) {
        for (int64_t col = 0; col < ncols; ++col) {
            if (!isfinite(importance[col]) || importance[col] < 0) {
                return 0;
            }
        }
    }
    for (uint64_t offset = 0; offset < count; offset += 256) {
        uint8_t pairs[64][2];
        ggml_fp16_t scale;
        const float *h = importance ? importance + offset % (uint64_t) ncols : NULL;
        if (!s24_fit(values + offset, h, pairs, &scale)) {
            return 0;
        }
        uint8_t *block = packed + offset / 256 * 39;
        memset(block, 0, 39);
        block[0] = (uint8_t) scale;
        block[1] = (uint8_t) (scale >> 8);
        for (unsigned slot = 0; slot < 4; ++slot) {
            uint64_t word = 0;
            for (int digit = 15; digit >= 0; --digit) {
                word = word * 6 + s24_pair_code(pairs[slot * 16 + (unsigned) digit]);
            }
            const unsigned bit = slot * 42;
            word <<= bit % 8;
            for (unsigned byte = 0; byte < 6; ++byte) {
                block[2 + bit / 8 + byte] |= (uint8_t) (word >> (8 * byte));
            }
        }
        for (unsigned group = 0; group < 64; ++group) {
            for (unsigned selected = 0; selected < 2; ++selected) {
                const unsigned bit = group * 2 + selected;
                if (values[offset + group * 4 + pairs[group][selected]] >= 0) {
                    block[23 + bit / 8] |= (uint8_t) (1u << (bit % 8));
                }
            }
        }
    }
    return (size_t) (count / 256 * 39);
}

int s24_unpack(const uint8_t *packed, float *values, int64_t n_elements) {
    if (!packed || !values || n_elements <= 0 || n_elements % 256 ||
        (uint64_t) n_elements > SIZE_MAX / sizeof(float) ||
        (uint64_t) n_elements / 256 > SIZE_MAX / 39) {
        return 0;
    }
    const uint64_t blocks = (uint64_t) n_elements / 256;
    /* Validate every block before touching the destination, including support
     * codes that occupy valid 42-bit fields but do not represent 16 base-six digits. */
    for (uint64_t b = 0; b < blocks; ++b) {
        const uint8_t *block = packed + b * 39;
        const ggml_fp16_t half = (ggml_fp16_t) (block[0] | ((uint16_t) block[1] << 8));
        const float scale = GGML_FP16_TO_FP32(half);
        if (!isfinite(scale) || scale < 0) {
            return 0;
        }
        for (unsigned slot = 0; slot < 4; ++slot) {
            if (s24_read_code(block, slot) >= S24_BASE6_LIMIT) {
                return 0;
            }
        }
    }
    memset(values, 0, (size_t) n_elements * sizeof(float));
    for (uint64_t b = 0; b < blocks; ++b) {
        const uint8_t *block = packed + b * 39;
        const ggml_fp16_t half = (ggml_fp16_t) (block[0] | ((uint16_t) block[1] << 8));
        const float scale = GGML_FP16_TO_FP32(half);
        for (unsigned slot = 0; slot < 4; ++slot) {
            uint64_t word = s24_read_code(block, slot);
            for (unsigned digit = 0; digit < 16; ++digit) {
                const unsigned code = (unsigned) (word % 6);
                word /= 6;
                const unsigned group = slot * 16 + digit;
                for (unsigned selected = 0; selected < 2; ++selected) {
                    const unsigned bit = group * 2 + selected;
                    const float signed_scale = block[23 + bit / 8] & (1u << (bit % 8)) ? scale : -scale;
                    values[b * 256 + group * 4 + s24_pairs[code][selected]] = signed_scale;
                }
            }
        }
    }
    return 1;
}
