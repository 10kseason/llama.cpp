#include "ggml.h"
#include "ggml-s24.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { throw std::runtime_error(#expr); } ++checks; } while (0)
static size_t checks;

static void set_direct_index(uint8_t * block, unsigned group, unsigned value) {
    // Deliberately bit-by-bit, independent of the codec's byte-shift packing.
    for (unsigned digit = 0; digit < 3; ++digit) {
        const unsigned bit = 16 + group * 3 + digit;
        block[bit / 8] = (uint8_t) ((block[bit / 8] & ~(1u << (bit % 8))) |
                                  (((value >> digit) & 1u) << (bit % 8)));
    }
}

static void set_compact_word(uint8_t * block, unsigned slot, uint64_t word) {
    for (unsigned digit = 0; digit < 42; ++digit) {
        const unsigned bit = 16 + slot * 42 + digit;
        block[bit / 8] = (uint8_t) ((block[bit / 8] & ~(1u << (bit % 8))) |
                                  (((word >> digit) & 1u) << (bit % 8)));
    }
}

static void test_roundtrip() {
    static const unsigned pairs[6][2] = {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    static const unsigned scales[] = {0, 0x8000, 1, 0x3c00, 0x7bff};
    for (unsigned count : {1u, 2u, 7u, 33u}) {
        std::vector<uint8_t> direct(count * 42), compact(count * 39 + 2, 0xa5);
        std::vector<uint8_t> restored(count * 42 + 2, 0x5a);
        std::vector<float> decoded(count * 256 + 2, 123.0f);
        for (unsigned block = 0; block < count; ++block) {
            auto * bytes = direct.data() + block * 42;
            const unsigned scale = scales[block % 5];
            bytes[0] = (uint8_t) scale;
            bytes[1] = (uint8_t) (scale >> 8);
            for (unsigned group = 0; group < 64; ++group) {
                set_direct_index(bytes, group, (group + block) % 6);
            }
            for (unsigned i = 0; i < 16; ++i) { bytes[26 + i] = (uint8_t) (i * 17 + block * 31); }
        }
        CHECK(ggml_s24_validate_direct42(direct.data(), direct.size()));
        CHECK(ggml_s24_direct42_to_compact(direct.data(), direct.size(), compact.data() + 1, count * 39));
        CHECK(compact.front() == 0xa5 && compact.back() == 0xa5);
        CHECK(ggml_s24_validate_compact(compact.data() + 1, count * 39));
        CHECK(ggml_validate_row_data(GGML_TYPE_S24, compact.data() + 1, count * 39));
        CHECK(ggml_s24_compact_to_direct42(compact.data() + 1, count * 39, restored.data() + 1, count * 42));
        CHECK(restored.front() == 0x5a && restored.back() == 0x5a);
        CHECK(std::equal(direct.begin(), direct.end(), restored.begin() + 1));
        dequantize_row_s24(compact.data() + 1, decoded.data() + 1, count * 256);
        CHECK(decoded.front() == 123.0f && decoded.back() == 123.0f);
        for (unsigned block = 0; block < count; ++block) {
            const float scale = ggml_fp16_to_fp32((ggml_fp16_t) scales[block % 5]);
            for (unsigned group = 0; group < 64; ++group) {
                const unsigned code = (group + block) % 6;
                const unsigned signs = (direct[block * 42 + 26 + group / 4] >> (2 * (group % 4))) & 3;
                for (unsigned lane = 0; lane < 4; ++lane) {
                    float expected = 0;
                    if (lane == pairs[code][0]) { expected = (signs & 1) ? scale : -scale; }
                    if (lane == pairs[code][1]) { expected = (signs & 2) ? scale : -scale; }
                    const float actual = decoded[1 + block * 256 + group * 4 + lane];
                    CHECK(std::memcmp(&actual, &expected, sizeof(float)) == 0);
                }
            }
        }
    }
}

static void test_invalid() {
    std::array<uint8_t, 84> direct{};
    std::array<uint8_t, 78> compact{};
    std::array<uint8_t, 84> output;
    output.fill(0xa5);
    const auto guard = output;
    CHECK(!ggml_s24_validate_compact(nullptr, 39));
    CHECK(!ggml_s24_validate_direct42(nullptr, 42));
    CHECK(!ggml_s24_validate_compact(compact.data(), 0));
    CHECK(!ggml_s24_validate_direct42(direct.data(), 0));
    CHECK(!ggml_s24_validate_compact(compact.data(), 38));
    CHECK(!ggml_s24_validate_direct42(direct.data(), 41));
    CHECK(!ggml_s24_compact_to_direct42(compact.data(), 39, output.data(), 43));
    CHECK(!ggml_s24_direct42_to_compact(direct.data(), 42, output.data(), 38));
    CHECK(!ggml_s24_compact_to_direct42(compact.data(), 78, compact.data(), 84));
    CHECK(!ggml_s24_compact_to_direct42(compact.data(), 39, compact.data() + 1, 42));
    CHECK(!ggml_s24_direct42_to_compact(direct.data() + 1, 42, direct.data(), 39));
    CHECK(output == guard);
    for (unsigned half = 0; half <= 0xffff; ++half) {
        compact[0] = direct[0] = (uint8_t) half;
        compact[1] = direct[1] = (uint8_t) (half >> 8);
        const bool expected = std::isfinite(ggml_fp16_to_fp32((ggml_fp16_t) half)) &&
                              ggml_fp16_to_fp32((ggml_fp16_t) half) >= 0;
        CHECK(ggml_s24_validate_compact(compact.data(), 39) == expected);
        CHECK(ggml_s24_validate_direct42(direct.data(), 42) == expected);
    }
    compact.fill(0);
    direct.fill(0);
    for (unsigned group = 0; group < 64; ++group) {
        for (unsigned reserved : {6u, 7u}) {
            set_direct_index(direct.data() + 42, group, reserved);
            CHECK(!ggml_s24_validate_direct42(direct.data(), direct.size()));
            CHECK(!ggml_s24_direct42_to_compact(direct.data(), direct.size(), output.data(), 78));
            CHECK(output == guard); // even a valid earlier block was not written
            set_direct_index(direct.data() + 42, group, 0);
        }
    }
    for (unsigned slot = 0; slot < 4; ++slot) {
        set_compact_word(compact.data() + 39, slot, UINT64_C(2821109907455));
        CHECK(ggml_s24_validate_compact(compact.data(), compact.size()));
        for (uint64_t invalid : {UINT64_C(2821109907456), (UINT64_C(1) << 42) - 1}) {
            set_compact_word(compact.data() + 39, slot, invalid);
            CHECK(!ggml_s24_validate_compact(compact.data(), compact.size()));
            CHECK(!ggml_s24_compact_to_direct42(compact.data(), compact.size(), output.data(), 84));
            CHECK(output == guard);
        }
        set_compact_word(compact.data() + 39, slot, 0);
    }
    // Newly reserved holes below the private type must reject without dividing by zero.
    CHECK(!ggml_validate_row_data((ggml_type) 62, compact.data(), 39));
}

static void test_quantizer() {
    std::array<float, 512> input{};
    std::array<uint8_t, 78> compact{};
    std::array<float, 512> decoded{};
    for (unsigned i = 0; i < input.size(); ++i) {
        if (i % 4 < 2) { input[i] = (i % 8 < 4 ? 0.5f : -0.5f); }
    }
    CHECK(quantize_s24(input.data(), compact.data(), 2, 256, nullptr) == compact.size());
    CHECK(ggml_s24_validate_compact(compact.data(), compact.size()));
    dequantize_row_s24(compact.data(), decoded.data(), decoded.size());
    CHECK(input == decoded);
    CHECK(ggml_type_size(GGML_TYPE_S24) == 39);
    CHECK(ggml_blck_size(GGML_TYPE_S24) == 256);
    CHECK(std::strcmp(ggml_type_name(GGML_TYPE_S24), "s24") == 0);
}

int main() {
    try {
        test_roundtrip();
        test_invalid();
        test_quantizer();
        std::printf("S24 codec PASS: %zu checks\n", checks);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "S24 codec FAIL after %zu checks: %s\n", checks, error.what());
        return 1;
    }
}
