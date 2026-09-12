// Standalone untimed CUDA equivalent of s24_direct42_gemv.comp.
// The runtime-only 42-byte support expansion does not alter the 39-byte GGUF.
// No CUDA headers, dense weight expansion, tensor cores, or fused FP32 FMA.
__device__ __forceinline__ float decode_half(unsigned short bits) {
    float value;
    asm("cvt.f32.f16 %0, %1;" : "=f"(value) : "h"(bits));
    return value;
}

extern "C" __global__ void s24_direct42_gemv(
        const unsigned char *weights, const float *inputs, float *outputs,
        unsigned rows, unsigned columns, unsigned input_count, unsigned output_stride) {
    const unsigned row = blockIdx.x;
    const unsigned input = blockIdx.y;
    const unsigned lane = threadIdx.x;
    if (row >= rows || input >= input_count) return;
    const unsigned blocks = columns / 256;
    const unsigned first_lane[6] = {0, 0, 0, 1, 1, 2};
    const unsigned second_lane[6] = {1, 2, 3, 2, 3, 3};
    __shared__ float partials[32];
    float total = 0.0f;
    for (unsigned item = lane; item < blocks * 4; item += 32) {
        const unsigned block = item / 4;
        const unsigned slot = item % 4;
        const unsigned base = (row * blocks + block) * 42;
        const unsigned short bits = (unsigned short)(weights[base] | (unsigned(weights[base + 1]) << 8));
        const float scale = decode_half(bits);
        for (unsigned digit = 0; digit < 16; ++digit) {
            const unsigned group = slot * 16 + digit;
            const unsigned bit = group * 3;
            const unsigned support = unsigned(weights[base + 2 + bit / 8]) |
                                     (unsigned(weights[base + 3 + bit / 8]) << 8);
            const unsigned mask = (support >> (bit % 8)) & 7;
            const unsigned signs = unsigned(weights[base + 26 + (group * 2) / 8]) >> ((group * 2) % 8);
            const unsigned column = input * columns + block * 256 + group * 4;
            const float first = (signs & 1) ? scale : -scale;
            const float second = (signs & 2) ? scale : -scale;
            // Explicit RN intrinsics retain the GLSL precise accumulation order.
            total = __fadd_rn(total, __fmul_rn(first, inputs[column + first_lane[mask]]));
            total = __fadd_rn(total, __fmul_rn(second, inputs[column + second_lane[mask]]));
        }
    }
    partials[lane] = total;
    __syncthreads();
    for (unsigned width = 16; width > 0; width /= 2) {
        if (lane < width) partials[lane] = __fadd_rn(partials[lane], partials[lane + width]);
        __syncthreads();
    }
    if (lane == 0) outputs[input * output_stride + 8 + row] = partials[0];
}
