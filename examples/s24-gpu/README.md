# S24 GPU kernel source archive

This directory preserves three standalone experimental GEMV kernels under the repository MIT license. It is source-only: there is no llama.cpp backend dispatch, GGUF type registration, model loader, upload converter or executable target here. The Q4KP CPU option does not enable these kernels. Development and verification were assisted by Codex.

## Files and storage

- `s24_gemv.comp`: Vulkan compact39 reference, requiring shaderInt64.
- `s24_direct42_gemv.comp`: Vulkan runtime-expanded support indices, without shaderInt64.
- `s24_direct42.cu`: CUDA equivalent of the direct42 reference.

Both layouts represent 256 weights with 2 nonzeros per group of 4. Each block has one finite, nonnegative FP16 scale; two independent sign bits select the signs of that group's two nonzeros. Support indices 0..5 select `(0,1)`, `(0,2)`, `(0,3)`, `(1,2)`, `(1,3)`, `(2,3)` respectively.

Compact39 stores 2 scale bytes, 21 support bytes (four 42-bit base-six words, 16 digits each) and 16 sign bytes. Direct42 stores 2 scale bytes, 24 support bytes (64 little-endian 3-bit indices) and 16 sign bytes. Each base-six word must be below `6^16`; direct indices 6 and 7 are invalid. Scales and sign ordering are unchanged. Direct42 is an upload-time layout, not a replacement on-disk GGUF type. No upstream quantization type is assigned by this archive.

## Launch contract

The caller must validate inputs before dispatch. The kernels are not validators for untrusted data.

- Use exactly 32 local invocations or CUDA threads per workgroup/block, and a grid of `(rows, input_count, 1)`. Do not over-dispatch Vulkan groups.
- Columns must be positive and divisible by 256. Store row-major weight blocks and row-major F32 input vectors. Keep all products and offsets within the shaders' unsigned 32-bit indexing range.
- Pad the Vulkan weight SSBO to a multiple of 4 bytes for its `uint` reads. Supply finite input values in the tested normal F32 domain; tiny products and F32 denormal preservation are not certified.
- Vulkan bindings 0/1/2 are weights/input/output. Push constants are four uints: rows, columns, input_count, output_stride.
- Allocate `input_count * output_stride` F32 output elements with `output_stride >= rows + 16`. Both kernels write at `input * output_stride + 8 + row`; eight-element guards surround each result vector.
- Preserve precise FP32 addition/multiplication and the shared-memory reduction order. CUDA was compiled with `--fmad=false --ftz=false`; the code uses explicit round-to-nearest arithmetic intrinsics. Vulkan was compiled for Vulkan 1.2 with GLSL `precise`.

Example shader compilation with an installed SDK:

```sh
glslc --target-env=vulkan1.2 -O s24_gemv.comp -o s24_gemv.spv
glslc --target-env=vulkan1.2 -O s24_direct42_gemv.comp -o s24_direct42_gemv.spv
```

## Measured scope

On one RTX 4060 Ti, the actual 10752x2048 S24 matrix had 3,354,624 compact39 bytes versus 3,612,672 direct42 bytes: an extra 258,048 GPU-resident bytes. Vulkan median device intervals were 140.368 to 84.608 microseconds when resident, and 143.312 to 85.920 microseconds after a separate 128 MiB read/write cache-pressure dispatch. Five interleaved rounds and 20 samples per layout/condition yielded 400 samples, with 1,000 warmups per execution. Pressure, upload, compilation, CPU waits and readback were outside the timestamp interval. Pressure does not prove a cold cache.

Correctness covered eight actual-matrix inputs and synthetic K=256/512/2304 cases. The Vulkan layouts produced bit-identical tested outputs (172,158 combined outputs), and the CUDA direct42 reference matched the Vulkan direct42 outputs (86,079 outputs). Independent FP64 error bounds and output guards passed. The largest actual-matrix absolute error was about 6.90e-7. Timing runs checked outputs before and after measurement, not after every timed dispatch.

The CUDA reference measured 93.184/96.256 microseconds in resident/pressure conditions using graph events. Empty event-boundary time was 5.12 microseconds and was not subtracted. Different API measurement overhead, sequential execution, changing GPU clocks and ordinary desktop activity prevent a general CUDA/Vulkan ranking. This is not a comparison against the optimized llama.cpp CUDA backend.

These are F32-activation single-matrix experiments. The separate CPU S24 implementation uses Q8_K activations, so tested GPU equivalence does not establish full-model CPU/GPU equivalence, task quality or token-generation speed. AMD/iGPU and HX370 execution are untested. In the accompanying LFM pilot, S24-containing allocations lost tool/agent checks; speed alone was not a reason to select them.

Local launchers, model tensors, raw prompts, outputs and device logs are deliberately not included in this public source archive.
