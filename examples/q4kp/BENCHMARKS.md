# LFM2.5-2.6B CPU benchmark

Measured on 2026-09-12 with an AMD Ryzen 7 7800X3D, GCC 15.1.0, Release static builds and OpenMP disabled. These are CPU results on one machine; HX370, its integrated GPU and other architectures have not been measured.

The model was the same BF16-derived LFM2.5-2.6B Q4_K_M GGUF in every run: 1,674,455,104 bytes, 2,697,198,592 parameters. Its SHA-256 is `814544faffbb4a767dda73e7ea83c11fbad816787dd63a5e8ae45955e96380a6`.

## Method

- Four modes: original fixed-AVX2 (`off`), P6 AVX2, P6 with 256-bit VNNI GEMV/GEMM (`vnni`), and pristine original base `d230ddd763ffe27781c7ffd237ea78b639b36b6d` built with `GGML_NATIVE=ON` and AVX512/VNNI enabled.
- Five fresh-process repetitions per mode at 2, 4 and 8 threads; thread and mode order rotated between repetitions. Each invocation measured pp512 and tg128 with the built-in warmup, batch/ubatch 512, `-ngl 0`, `-fa 0`, and `-r 1`.
- No concurrent model evaluation, quantization or compilation during the measured runs. Ordinary OS activity and scheduling remain uncontrolled; no CPU affinity or frequency lock was applied.
- Every active P6/VNNI process reported 148 recoded tensors and nonzero matrix operations. Every off process reported zero recoded tensors and matrix operations. These counts establish dispatch coverage, not speed.
- Throughput is total tokens divided by total measured time. The spread shown is the sample standard deviation of the five individual token/s values, not a confidence interval.

## Results

| Threads | Phase | Original AVX2 | P6 | VNNI GEMV/GEMM | Original native | VNNI vs native |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | pp512 | 40.87 ± 4.83 | 36.97 ± 6.74 | 60.28 ± 3.42 | 50.67 ± 7.89 | +19.0% |
| 2 | tg128 | 19.94 ± 2.81 | 23.03 ± 1.17 | 22.79 ± 1.09 | 22.34 ± 0.72 | +2.0% |
| 4 | pp512 | 64.78 ± 7.22 | 74.85 ± 6.13 | 104.20 ± 3.22 | 86.59 ± 4.36 | +20.3% |
| 4 | tg128 | 25.45 ± 0.64 | 25.43 ± 2.05 | 25.34 ± 1.69 | 25.10 ± 1.98 | +1.0% |
| 8 | pp512 | 103.99 ± 9.74 | 104.95 ± 8.19 | 158.19 ± 10.01 | 140.25 ± 4.29 | +12.8% |
| 8 | tg128 | 22.39 ± 2.32 | 22.99 ± 1.98 | 23.86 ± 1.10 | 24.36 ± 1.17 | -2.1% |

Values are tokens/s. `±` is run-to-run standard deviation.

## Interpretation

- 2 threads: VNNI prompt processing +63.0% versus P6 and +19.0% versus original native; token generation +2.0% versus original native.
- 4 threads: VNNI prompt processing +39.2% versus P6 and +20.3% versus original native; token generation +1.0% versus original native.
- 8 threads: VNNI prompt processing +50.7% versus P6 and +12.8% versus original native; token generation -2.1% versus original native.

P6 and VNNI still read the same 1152-byte packed blocks. These results do not demonstrate reduced weight traffic or a universal 20–45% token-generation gain. Prompt-processing improvements must not be relabeled decode improvements. `vnni` changes both GEMV and GEMM; this matrix does not isolate VNNI on the original non-P6 layout.

The synthetic suite establishes bitwise equivalence to the fixed AVX2 kernel for tested inputs. This throughput run is separate from BF16/Q6_K/mixed model-quality evaluation. In particular, the private S24 mixed models are not the Q4_K_M model benchmarked here.

## Reproduce

Use the experimental build from [README.md](README.md), with tests/tools enabled for `q4kp-bench`. For the native reference, use a separate checkout of the base above, set `GGML_CPU_Q4KP=OFF`, `GGML_NATIVE=ON`, `GGML_AVX512=ON`, `GGML_AVX512_VNNI=ON`, and match the other build settings.

```sh
GGML_Q4KP=vnni ./build-q4kp/bin/q4kp-bench -m model.gguf -ngl 0 -t 4 -p 512 -n 128 -b 512 -ub 512 -fa 0 -r 1 -o json > result.json 2> dispatch.log
```

Repeat with off/p6 and the separate native executable, interleaving all modes five times per thread count. Keep each JSON and stderr file, record model/binary hashes and aggregate the `samples_ns` values separately for pp and tg. Windows uses the same arguments after setting the environment variable in PowerShell. Raw local logs, model files and personal paths are excluded from this document.
