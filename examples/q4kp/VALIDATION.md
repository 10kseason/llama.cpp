# Fork integration and VNNI GEMM validation

Base revision: `d230ddd763ffe27781c7ffd237ea78b639b36b6d`.

The documented static AVX2 configuration was built from this checkout with GCC 15.1.0 on Windows x86-64. No prebuilt experimental library was used as a reference.

| Check | Result |
| --- | --- |
| `test-q4kp-kernels`, `test-q4kp-runtime` | Built |
| `llama-completion`, `llama-bench` | Built; version/help smoke passed |
| CTest `q4kp-*` | 6/6 passed |
| Python synthetic kernel suite | 21/21 passed, no ISA skips on the test machine |
| Runtime modes off/p6/vnni/wide/invalid | 231 checks per mode passed |
| Separate `GGML_CPU_Q4KP=OFF` CPU backend | Built |

The Python suite compared the original AVX2 kernels with P6 and independent scalar outputs (45 GEMV cases, 116 GEMM cases), VNNI (104 GEMV cases), and wide (122 GEMV cases), including output guards and input immutability. The VNNI GEMM extension added 77 cases and 55,680 bit-identical FP32 outputs against original AVX2, P6 AVX2 and the scalar oracle. These cases cover the 16-row body, 4-row tail, a 512-row prompt batch, integer/FP16 extrema and output guards. Runtime checks used canonical Q4_K data, repeated tensor loading, aligned and rejected views, and two-thread graphs with 1, 4, 5 and 20 activation rows. Both `vnni` and `wide` now dispatch GEMM through the 256-bit VNNI implementation after ISA checks; `p6` retains the AVX2 GEMM.

These results establish correctness for the tested configuration. The weight block remains 1152 bytes, so the new integer arithmetic does not reduce weight traffic. Separate [LFM2.5-2.6B measurements](BENCHMARKS.md) compare 60 interleaved runs against an optimized original native/AVX512/VNNI build. Four-thread pp512 improved 20.3% on the tested 7800X3D; tg128 differed by +1.0%, within substantial run-to-run dispersion. See the `q4kp-bench` procedure in [README.md](README.md#measure-performance-and-confirm-dispatch) for benchmark output and stderr dispatch counters. This is not HX370, GPU or universal model evidence.

Unrelated upstream MinGW warnings remain in the subprocess helper and benchmark date formatting. The current server UI embedding tool failed with a non-ASCII build path; the documented build uses completion and benchmark tools with server/app disabled. Neither warning-free upstream builds nor server/UI support is claimed here.

Linux PIC requirements were checked in source and addressed for the shared synthetic test library; Linux and Clang execution were not performed. Unsupported ISA fallback was reviewed in source, but no separate machine lacking VNNI was used. This record is not a model accuracy certification or an end-to-end speedup claim. Re-run the tests after source, compiler or CPU changes.

## Checked entries and original-layout VNNI follow-up

The new Release build replaces silent invalid-input returns with checked scalar entries for P6 GEMV/GEMM, VNNI GEMV/GEMM, wide GEMV and both original-layout VNNI entries. The child-process suite exercises 132 invalid calls (dimensions, null/aligned pointers, stride overflow, address wrap and output/input overlap). Every child must exit through the ggml abort callback with an assertion diagnostic; a silent return or access violation fails. This tests `NDEBUG` behavior, not only debug assertions. Actual execution on an unsupported-ISA CPU is not tested on this host.

The numerical fixtures now include `vnni-original` alongside P6 VNNI, original AVX2 and the scalar oracle. The same VNNI arithmetic and accumulation code is instantiated with the two metadata decoders. Runtime graph checks cover both layouts, views, repeated loads and counters. Original-layout uploads must match the original packed bytes exactly and keep the recode counters at zero.

Verification on GCC 15.1, Release `-O3 -DNDEBUG`, fixed AVX2 backend: 7/7 CTest targets passed (18 Python tests plus off/P6/VNNI/original-VNNI/wide/invalid-mode graph runs). Object inspection found no VEX/EVEX vector instructions or PDEP in the scalar entry translation unit. Original-layout GEMV and GEMM contain `vpdpbusd` and no PDEP, while P6 contains both. These results establish a usable ablation path and fail-fast behavior; they do not assign a speed gain to the new mode.
