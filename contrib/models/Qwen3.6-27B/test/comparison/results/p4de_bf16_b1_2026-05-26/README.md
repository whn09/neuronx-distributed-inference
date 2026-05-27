# P4DE (A100 80GB SXM4) BF16 batch=1 sweep — 2026-05-26

Companion to `../p4de_fp8_b1_2026-05-26/`, same host + launcher, `Qwen/Qwen3.6-27B`
(BF16 HF checkpoint), TP=1, max-model-len 9216.

## Environment

- Instance: P4DE (A100 80GB SXM4 ×8), `/opt/pytorch` bundled vLLM
- CUDA: `/opt/pytorch/cuda` (cu13, libs in `lib/`)
- Launcher: `start_vllm_p4de.sh`
- Single GPU (TP=1, `--gpu-memory-utilization 0.92`).

## Sweep

`vllm bench serve`, `PREFILL_ISLS="1023 2047 4095 8191"`, `DECODE_CONC="1"`
(decode used ISL=1024 OSL=1024).

| Test                              | TTFT (ms) | TPOT (ms) | OTPS (tok/s) |
|-----------------------------------|----------:|----------:|-------------:|
| Prefill ISL=1023                  |     500.5 |        — |           — |
| Prefill ISL=2047                  |     483.3 |        — |           — |
| Prefill ISL=4095                  |    1533.7 |        — |           — |
| Prefill ISL=8191                  |    1973.8 |        — |           — |
| Decode ISL=1024 OSL=1024 conc=1   |     258.0 |     35.14 |        28.28 |

## P4DE FP8 (Marlin) vs BF16, same host

| Test                                 |  BF16 |   FP8 | FP8 / BF16 |
|--------------------------------------|------:|------:|-----------:|
| Prefill ISL=1023 TTFT (ms)           |   500 |   596 |   1.19x    |
| Prefill ISL=2047 TTFT (ms)           |   483 |   694 |   1.44x    |
| Prefill ISL=4095 TTFT (ms)           |  1534 |  1928 |   1.26x    |
| Prefill ISL=8191 TTFT (ms)           |  1974 |  2817 |   1.43x    |
| Decode TPOT (ms)                     |  35.1 |  20.7 |   0.59x    |
| Decode OTPS (tok/s)                  |  28.3 |  47.6 |   1.68x    |

A100 has no native FP8 tensor cores. The Marlin weight-only kernel cuts the
weight bandwidth (memory-bound decode wins ~40%), but adds dequantize overhead
to the compute-bound prefill path (slower TTFT). The comparison is the GPU-side
mirror of the Trn2 MLP-weight-only result.
