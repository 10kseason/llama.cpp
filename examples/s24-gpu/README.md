# S24 Vulkan backend tool and GPU kernel archive

`llama-s24-bench` loads a selected S24 tensor from a GGUF and executes `ggml_mul_mat` through this fork's Vulkan backend with deterministic F32 inputs. The backend expands compact39 support indices to direct42 during validated upload, and reverses that expansion during tensor readback. This uses the real ggml allocator, upload callbacks, graph dispatch and Vulkan shader; there is no CPU fallback inside the tool. The Q4KP CPU option is independent. Development and verification were assisted by Codex.

S24 is this fork's experimental GGUF tensor type 63, not an upstream format assignment. Loading and computing a tensor does not establish full-model compatibility or a model-quality target. The three original standalone kernels remain unchanged as historical references under the repository MIT license.

## Build and run

Use an installed C++17 toolchain, CMake, and Vulkan SDK with `glslc`. This example requires `GGML_VULKAN=ON` and `GGML_BACKEND_DL=OFF` because it links the backend directly. From the repository root:

```sh
cmake -S . -B build-s24 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_VULKAN=ON -DGGML_BACKEND_DL=OFF -DGGML_CPU_Q4KP=OFF \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
cmake --build build-s24 --config Release --target llama-s24-bench llama-quantize llama-completion -j 4
./build-s24/bin/llama-s24-bench --self-test
./build-s24/bin/llama-s24-bench --model model-mixed-s24.gguf \
  --tensor blk.2.ffn_gate.weight --inputs 1 --warmup 5 --repetitions 20 > result.json
```

For a multi-configuration generator, the executable may be in `bin/Release`. On Windows, append `.exe` and enter shell commands on one line or use PowerShell continuation syntax. `--device N` selects a Vulkan device by backend index, starting at zero. The tool does not download models, select a private model path, or modify its input GGUF. Normal backend diagnostics go to stderr; the result JSON goes to stdout.

The tested Windows build used GCC 15.1, Ninja, Vulkan SDK 1.4.309.0, `GGML_NATIVE=ON`, and `GGML_OPENMP=OFF`. For MinGW, select `gcc`/`g++` explicitly and add `-pipe -D_WIN32_WINNT=0x0A00` to both C and C++ flags. If CMake cannot find SPIRV-Headers, set `SPIRV-Headers_DIR` to the installed SDK's `Lib/cmake` directory. The shader-generator sub-build inherits the selected native compiler; generated-file arguments are relative to support Unicode checkout paths. Put temporary files in a writable directory. Linux, Clang, MSVC and AMD execution have not been tested for this integration.

The model mode reads GGUF metadata and only the named tensor's payload. It rejects other tensor types, non-2D shapes, invalid compact bytes and unsupported backend shapes. The tool bounds rows to 2..65535, columns to positive multiples of 256 up to 262144, input count to 1..1024, compact weights to 256 MiB, and each F32 input/output buffer to 32 MiB. The FP64 check is also capped at 536,870,912 terms. The backend applies the actual device's dispatch and storage-buffer limits. Warmups are 0..1000 and measured repetitions are 1..1000.

## Create a candidate with the fork's quantizer

Starting with a BF16/F16/F32 GGUF, an example with one selected S24 FFN tensor and higher precision elsewhere is:

```sh
./build-s24/bin/llama-quantize \
  --token-embedding-type q8_0 --output-tensor-type q8_0 \
  --tensor-type '^blk[.]2[.]ffn_gate[.]weight$=s24' \
  model-bf16.gguf model-mixed-s24.gguf Q6_K
```

Use a tensor name and architecture actually present in the source GGUF. The explicit tensor selector is a regular expression. `S24` is also recognized as an experimental overall quantization choice, but neither that choice nor the example allocation carries a quality guarantee. The quantizer is a separate operation; `llama-s24-bench` never requantizes the selected payload. Other backends and applications need this type's support to load it.

A mixed model can also execute through the ordinary completion target:

```sh
GGML_VK_PERF_LOGGER=1 ./build-s24/bin/llama-completion -m model-mixed-s24.gguf \
  -ngl 99 -c 512 -b 64 -ub 64 -t 4 -tb 4 -n 32 --temp 0 --seed 42 \
  --single-turn -cnv -p "Give one sentence explaining why tensor shape validation matters." \
  --no-warmup --no-display-prompt
```

The perf logger is optional and adds diagnostic overhead. Look for `MUL_MAT s24` and `MUL_MAT_VEC s24` to verify S24 prefill/decode dispatch; `-ngl 99` alone does not prove dispatch. On PowerShell, set `$env:GGML_VK_PERF_LOGGER = '1'` before the executable. `GGML_CPU_Q4KP` does not enable this GPU path: `GGML_VULKAN` and eligible S24 weights do.

The loader validates S24 even when general tensor checking is disabled. It stages a complete tensor instead of sending generic asynchronous byte chunks. The Vulkan backend accepts only owned contiguous 2D S24 weights with columns divisible by 256, F32 input/output and supported device limits/alignment. Full uploads validate compact bytes, expand them, validate direct indices 0..5, and mark the allocation ready only after the upload completes. Views, partial writes and use before a validated upload fail explicitly; unsupported operations are refused by the backend support query. Readback converts to canonical compact bytes. The integrated shader also guards indices 6/7 before accessing its position table. This is not arbitrary S24 operation or architecture coverage.

SYCL and Metal explicitly refuse operations with S24 input/output before their broader operation checks. They have no S24 device decoder in this fork; the scheduler must choose a supported backend. RPC and VirtGPU also refuse S24 because their broad admission paths cannot confirm support for this private type remotely. These admission guards were reviewed in source, but those backend builds and hardware execution are untested. The integrated S24 GPU runtime verified here is Vulkan.

## Validation and timing definitions

`--self-test` covers synthetic K=256/512/2304, different output-row and input-vector counts, every support pair, both signs, zero scales and finite FP16 edge scales. For every output, it compares the actual GPU result with CPU compact dequantization followed by FP64 products/summation and a per-output forward-error bound. It verifies exact compact upload/readback, partial compact readback, host readback guards, and identical output bits before and after repeated graph execution. It also requires support-query rejection of S24 CONCAT, VIEW and CPY operations and an unaligned F32 input view. The guard check is a host readback check, not a GPU out-of-bounds detector.

The model mode performs the same full output and roundtrip checks. It reports compact bytes, direct42 payload bytes, actual backend allocation sizes, every timing sample, mean, median and sample standard deviation. A validation dispatch precedes the requested warmups. Timed intervals surround synchronized `ggml_backend_graph_compute` calls, so they include graph submission and host synchronization overhead. Upload, initial execution, readback and the FP64 reference are excluded. These are graph wall times, not the device timestamp intervals in the historical results below, and not model tokens per second. CPU S24 uses Q8_K activations; this F32-input comparison is not full-graph CPU/GPU equivalence.

With tests enabled, run:

```sh
ctest --test-dir build-s24 -C Release -R '^s24-vulkan-' --output-on-failure
```

When Python 3 is available at configure time, the suite includes four subprocess tests. Invalid scales, invalid base-six support codes, partial uploads and tensor views must trigger the backend's S24 assertion. Each child first proves that a valid tensor uploads and reads back; a missing GPU or initialization failure cannot count as a successful rejection. Expected assertions are confined to children. These are GPU tests and need an available Vulkan device.

For API validation, configure `GGML_VULKAN_VALIDATE=ON` with the Khronos validation layer installed. Validation warnings remain on stderr; API errors abort instead of leaving a successful result. With an older SDK lacking the corresponding extension declarations, this validation build explicitly disables the optional decode-vector and internally synchronized queue features that its layer cannot validate. Normal builds retain their original feature selection. This is a verification configuration, not a fastest-backend baseline.

## Integrated execution record

The Windows configuration above was verified on an RTX 4060 Ti 16GB, driver 616.92. Core codec/loader and Vulkan self-test/rejection CTest targets passed 4/4. The codec checked 142,547 conditions, including all FP16 bit patterns and invalid direct indices. Four backend child processes rejected an invalid scale, an invalid compact support code, a partial upload and a tensor view after first proving valid upload/readback.

The synthetic GPU self-test checked 76 outputs across three shapes. A real LFM2.5-2.6B mixed GGUF's `blk.2.ffn_gate.weight` (10752 x 2048) checked 86,016 outputs for eight F32 input vectors. Compact roundtrips, partial reads and repeated output bits passed; maximum absolute error against the FP64 reference was 1.4901161193847656e-7, within its forward-error bound.

For that eight-input graph, five synchronized wall-time samples were 919.0, 865.1, 867.0, 870.4 and 947.5 microseconds after two warmups and one validation dispatch. Median was 870.4 microseconds, mean 893.8 and sample SD 37.44. The run used API validation and ordinary desktop clocks. These values are not comparable to the historical single-input device timestamps below. Machine-readable output and reproduction settings are in [MEASUREMENTS.json](MEASUREMENTS.json).

The same mixed model also loaded through `llama-completion` and generated 32 tokens. Backend logs confirmed one S24 prefill operation with 19 input tokens and 31 S24 single-token operations. API validation reported zero errors. Best-practice warnings remained (30 in the self-test and 41 in the full-model smoke); these were retained, not suppressed. This demonstrates functional model execution on that allocation, not completed-answer quality, a 94% score, universal model support or a model-speed gain.

## Historical kernels and storage

- `s24_gemv.comp`: Vulkan compact39 reference, requiring shaderInt64.
- `s24_direct42_gemv.comp`: Vulkan runtime-expanded support indices, without shaderInt64.
- `s24_direct42.cu`: CUDA equivalent of the direct42 reference.

Both layouts represent 256 weights with 2 nonzeros per group of 4. Each block has one finite, nonnegative FP16 scale; two independent sign bits select the signs of that group's two nonzeros. Support indices 0..5 select `(0,1)`, `(0,2)`, `(0,3)`, `(1,2)`, `(1,3)`, `(2,3)` respectively.

Compact39 stores 2 scale bytes, 21 support bytes (four 42-bit base-six words, 16 digits each) and 16 sign bytes. Direct42 stores 2 scale bytes, 24 support bytes (64 little-endian 3-bit indices) and 16 sign bytes. Each base-six word must be below `6^16`; direct indices 6 and 7 are invalid. Scales and sign ordering are unchanged. Direct42 is an upload-time layout, not a replacement on-disk GGUF type.

## Historical standalone launch contract

This section describes the three archived files, not the integrated backend shader. Their caller must validate inputs before dispatch; the archived kernels are not validators for untrusted data. The integrated backend instead validates full owned contiguous 2D S24 tensor uploads, rejects views and partial writes, and uses the standard contiguous ggml output layout without the archived eight-element device guards.

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

## Historical measured scope

On one RTX 4060 Ti, the actual 10752x2048 S24 matrix had 3,354,624 compact39 bytes versus 3,612,672 direct42 bytes: an extra 258,048 GPU-resident bytes. Vulkan median device intervals were 140.368 to 84.608 microseconds when resident, and 143.312 to 85.920 microseconds after a separate 128 MiB read/write cache-pressure dispatch. Five interleaved rounds and 20 samples per layout/condition yielded 400 samples, with 1,000 warmups per execution. Pressure, upload, compilation, CPU waits and readback were outside the timestamp interval. Pressure does not prove a cold cache.

Correctness covered eight actual-matrix inputs and synthetic K=256/512/2304 cases. The Vulkan layouts produced bit-identical tested outputs (172,158 combined outputs), and the CUDA direct42 reference matched the Vulkan direct42 outputs (86,079 outputs). Independent FP64 error bounds and output guards passed. The largest actual-matrix absolute error was about 6.90e-7. Timing runs checked outputs before and after measurement, not after every timed dispatch.

The CUDA reference measured 93.184/96.256 microseconds in resident/pressure conditions using graph events. Empty event-boundary time was 5.12 microseconds and was not subtracted. Different API measurement overhead, sequential execution, changing GPU clocks and ordinary desktop activity prevent a general CUDA/Vulkan ranking. This is not a comparison against the optimized llama.cpp CUDA backend.

These are F32-activation single-matrix experiments. The separate CPU S24 implementation uses Q8_K activations, so tested GPU equivalence does not establish full-model CPU/GPU equivalence, task quality or token-generation speed. AMD/iGPU and HX370 execution are untested. In the accompanying LFM pilot, S24-containing allocations lost tool/agent checks; speed alone was not a reason to select them.

The original historical launchers and their device logs are not included. The integrated launcher, backend, converter, loader, synthetic tests and individual numeric measurement samples are included. Model tensors, private prompts, outputs and workstation paths are excluded. The portable tool accepts user-supplied GGUFs and prints new results without packaging model data.
