# P5 (H100 80GB SXM5) BF16 batch=1 sweep — 2026-06-02

`Qwen/Qwen3.6-27B` (BF16 HF checkpoint) on a P5 host (8× H100 80GB HBM3),
TP=1, single GPU, max-model-len 9216. Same `vllm bench serve` client and
flags as the A100 (P4DE) and Trn2 sweeps, so prompt construction, sampling,
and metric computation are identical across platforms.

## Environment

- Instance: P5 (H100 80GB SXM5 ×8), `/opt/pytorch` bundled vLLM
- vLLM 0.22.0, torch 2.11.0+cu130, flashinfer 0.6.11.post2
- CUDA: `/opt/pytorch/cuda` (cu13, libs in `lib/`, symlinked `lib64` for
  flashinfer JIT — see the FP8 README for the link fix)
- Driver 595.71.05
- Launcher: P5 copy of `gpu/start_vllm.sh` (CUDA_HOME=`/opt/pytorch/cuda`,
  `--max-num-seqs 256` to satisfy the hybrid DeltaNet/Mamba cache-block cap)
- Single GPU (TP=1, `--gpu-memory-utilization 0.92`), GPU 1.

## Sweep

`vllm bench serve`, `PREFILL_ISLS="1023 2047 4095 8191"`, `DECODE_CONC="1"`
(decode used ISL=1024 OSL=1024), `--random-range-ratio 0`.

| Test                              | TTFT (ms) | TPOT (ms) | OTPS (tok/s) |
|-----------------------------------|----------:|----------:|-------------:|
| Prefill ISL=1023                  |      96.3 |        — |           — |
| Prefill ISL=2047                  |     186.9 |        — |           — |
| Prefill ISL=4095                  |     352.7 |        — |           — |
| Prefill ISL=8191                  |     687.8 |        — |           — |
| Decode ISL=1024 OSL=1024 conc=1   |    1071.2 |     20.15 |        47.22 |

> **Note on ISL=1023:** the first prefill call of the sweep pays a one-time
> flashinfer-autotune + CUDA-graph warmup cost (~1.6 s cold). The value above
> is the warm re-measurement, taken after the server was hot, so it is
> apples-to-apples with the other (already-warm) ISL points and with the A100
> baseline. The raw cold value is preserved in `bf16_prefill.log`.

## vs A100 (P4DE, same harness, BF16)

| Test                       |  A100 |  H100 | H100 / A100 |
|----------------------------|------:|------:|------------:|
| Prefill ISL=1023 TTFT (ms) |   500 |    96 |   **5.2x faster** |
| Prefill ISL=8191 TTFT (ms) |  1974 |   688 |   **2.9x faster** |
| Decode TPOT (ms)           |  35.1 |  20.2 |   **1.74x faster** |
| Decode OTPS (tok/s)        |  28.3 |  47.2 |   **1.67x faster** |

H100 BF16 is uniformly faster than A100 BF16 on this model — higher HBM3
bandwidth (3.35 TB/s vs 2.0 TB/s) drives the decode win, and the larger
SM count + higher clocks drive the prefill win. See
`../p5_fp8_b1_2026-06-02/` for the native-FP8 comparison on the same host.
