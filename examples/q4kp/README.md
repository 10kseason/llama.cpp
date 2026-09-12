# Experimental Q4_K P6 / VNNI kernels

This fork keeps an opt-in CPU experiment for existing Q4_K models. P6 rearranges the six-bit scale/minimum metadata in the CPU's packed weights. The GGUF, quantized values, FP16 scales and allocation size stay unchanged. It is not a new quantization type or a GPU kernel.

The default build and runtime leave this experiment disabled. The kernels derive from `ggml/src/ggml-cpu/arch/x86/repack.cpp` under the repository's MIT license. Development and tests were assisted by Codex. This is a maintained experiment in this fork, not an upstream submission.

## Build

Current scope: x86-64, GCC-compatible GCC/Clang, static libraries, a single AVX2 CPU backend. The machine must support AVX2, BMI2, FMA and F16C. MSVC, ARM, dynamic backend loading and `GGML_NATIVE=ON` are deliberately rejected when this option is enabled. The fixed AVX2 configuration keeps the original reference and P6 GEMM accumulation order comparable.

From the repository root, with CMake, Ninja and a C/C++ compiler installed:

```sh
cmake -S . -B build-q4kp -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_CPU_Q4KP=ON -DGGML_NATIVE=OFF \
  -DGGML_AVX2=ON -DGGML_BMI2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_AVX512=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-q4kp --target llama-completion llama-bench -j 4
```

On PowerShell, enter the configure command on one line, or use PowerShell's continuation syntax. For MinGW, add `-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ "-DCMAKE_C_FLAGS=-pipe -D_WIN32_WINNT=0x0A00" "-DCMAKE_CXX_FLAGS=-pipe -D_WIN32_WINNT=0x0A00"`. The Windows target version makes `CreateFile2` available to the current upstream HTTP library. Put build and temporary directories on a writable drive. Within this experimental build, do not enable AVX-512 globally: only the VNNI/wide functions use those instructions after the runtime checks. A separate performance baseline should use the best supported original backend configuration.

## Run

Set `GGML_Q4KP` before starting a new process:

| Value | Behavior |
| --- | --- |
| unset / `off` | Original CPU repacking and kernels |
| `p6` | P6 layout with AVX2 GEMV/GEMM |
| `vnni` | P6 with 256-bit AVX-512 VNNI GEMV and GEMM |
| `wide` | Experimental 512-bit GEMV; the same 256-bit VNNI GEMM as `vnni` |

Unsupported VNNI falls back to P6. Unsupported wide falls back to VNNI when available, then P6. An unknown value prints a warning and disables the experiment. The mode is fixed on first use; restart the process to change it.

VNNI GEMM uses `dpbusd` and int32 scale multiplication in both the 16-row body and 4-row tail. It preserves the AVX2 path's FP32 FMA order and minimum correction. There is no 512-bit GEMM variant.

```sh
GGML_Q4KP=vnni ./build-q4kp/bin/llama-completion -m /path/to/model.gguf -ngl 0 -p "Hello" -n 64
GGML_Q4KP=off ./build-q4kp/bin/llama-bench -m /path/to/model.gguf -ngl 0
GGML_Q4KP=vnni ./build-q4kp/bin/llama-bench -m /path/to/model.gguf -ngl 0
```

PowerShell equivalent:

```powershell
$env:GGML_Q4KP = 'vnni'
./build-q4kp/bin/llama-completion.exe -m ./model.gguf -ngl 0 -p 'Hello' -n 64
```

Only owned, contiguous, two-dimensional Q4_K tensors with columns divisible by 256 and rows divisible by 8 are recoded. Other types, noneligible shapes and MoE weights keep their normal path. Views inherit the owner's actual layout. Full-column views aligned to eight-row groups are allowed; incompatible views of recoded weights are refused before calculation. P6 data must never reach a standard packed-weight decoder.

## Check correctness

The tests use synthetic data only. Install NumPy in your chosen Python environment; configure `-DLLAMA_BUILD_TESTS=ON` and optionally `-DPython3_EXECUTABLE=/path/to/python` using the same build flags above. No model or dataset download is required.

```sh
cmake --build build-q4kp --target test-q4kp-kernels test-q4kp-runtime -j 4
ctest --test-dir build-q4kp -R '^q4kp-' --output-on-failure
```

The kernel suite compares the freshly built original AVX2 implementation, P6, supported VNNI/wide kernels and an independent scalar oracle bit for bit. It exercises six-bit metadata round trips, Q8 and nibble extrema, finite FP16 edge values, both AVX2 and VNNI GEMM branches, row tails, a 512-row prompt batch, output guards and invalid arguments. Tests for unavailable VNNI/wide instructions report a skip. Runtime checks cover allocation size, tensor/view ownership, repeated loading and canonical Q4_K graph calculations.

These checks establish implementation equivalence within the tested AVX2 configuration. They do not certify every model, backend, compiler or task accuracy. GPU arithmetic and globally enabled AVX-512 kernels have separate numerical behavior.

## Measure performance and confirm dispatch

With the test configuration above and `LLAMA_BUILD_TOOLS=ON`, build `q4kp-bench`. It accepts the same arguments as `llama-bench`, keeps the requested benchmark format on stdout, and appends a `Q4KP_STATS` line to stderr after the benchmark returns:

```sh
cmake -S . -B build-q4kp -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build-q4kp --target q4kp-bench -j 4
GGML_Q4KP=vnni ./build-q4kp/bin/q4kp-bench -m /path/to/model.gguf -ngl 0 -t 4 -p 512 -n 128 -r 1 -o json > vnni.json 2> vnni.log
```

On PowerShell, set `$env:GGML_Q4KP = 'vnni'` first and run `./build-q4kp/bin/q4kp-bench.exe` with the same arguments. The stats line has fields `enabled`, `tensors`, `metadata_bytes` and `matrix_ops`. Counters are cumulative for the process, including warmups and repeated runs; `metadata_bytes` counts bytes recoded in place, not extra memory or bytes read during inference. Check `tensors` and `matrix_ops` are nonzero before attributing a result to these kernels. `enabled=1` alone does not show that the model had eligible tensors. Stderr also contains normal backend logs and any ISA fallback warning.

Use fresh processes for interleaved `off` / `p6` / `vnni` runs at 2, 4 and 8 threads, with at least five repetitions and recorded dispersion. Compare prompt processing and token generation separately. Also compare with a separate `GGML_CPU_Q4KP=OFF` build using the original backend's best supported ISA configuration; the fixed AVX2 correctness build is not an optimized upstream performance baseline.

P6 and VNNI still read the same 1152-byte weight blocks. They do not reduce model size or weight traffic, so bandwidth-bound token generation may see little benefit. The [LFM2.5-2.6B benchmark](BENCHMARKS.md) records five interleaved repetitions per mode and thread count against a separate native/AVX512/VNNI original build. At four threads on the tested 7800X3D, VNNI improved pp512 by 20.3%; tg128 differed by only +1.0%. This does not establish the same result on other hardware.

## Layout and maintenance

- One block contains eight rows of 256 weights in 1152 bytes.
- Bytes 0-31 hold the original FP16 scales/minima; bytes 128-1151 retain the original quantized values.
- The 96 metadata bytes contain eight groups of 12 bytes. Each group is six little-endian bytes of eight packed six-bit scales, followed by six bytes of minima.
- BMI2 PDEP reconstructs the original byte lanes. VNNI changes the integer dot-product stage, preserving FP32 multiplication and FMA order.
- Recode runs once after the original Q4_K repacker. It is not idempotent. Repeated `tensor_set` must start with canonical Q4_K bytes and perform original repacking again.
- `q4kp_runtime_stat(1)` counts metadata bytes rewritten in place, not additional memory. No sidecar weight allocation is added.

When updating from upstream, review packed struct sizes and offsets, original GEMV/GEMM arithmetic, tensor trait dispatch, view initialization and repacking. Rebuild the original reference from that checkout and rerun the synthetic tests. Do not reuse an older static library as proof for a new revision. Keep the CPU and runtime switches off by default until correctness and performance have been measured on the intended device.

Wide is a research option: it can be slower on larger matrices. Lower kernel time does not imply the same percentage gain in end-to-end token generation. Compare identical models, thread counts, contexts and offload settings with repeated interleaved runs. Report aggregate tokens divided by aggregate time. There is no universal 20-45% speedup claim.

The publication contains source, synthetic tests and documentation only. Model weights, prompts, datasets, logits, personal system reports, absolute workstation paths and build artifacts are excluded.
