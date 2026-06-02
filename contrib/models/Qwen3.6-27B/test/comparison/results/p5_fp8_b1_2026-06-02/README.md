# P5 (H100 80GB SXM5) FP8 batch=1 sweep — 2026-06-02

vLLM `Qwen/Qwen3.6-27B-FP8` (HF native FP8 checkpoint) on a P5 host (8× H100
80GB HBM3), TP=1, single GPU, max-model-len 9216. Same `vllm bench serve`
client and flags as the A100 / Trn2 sweeps.

## Native FP8 — this is a *real* FP8 compute test

Unlike the A100 (P4DE) FP8 run, which fell back to the Marlin **weight-only**
kernel (FP8 weights dequantized to BF16 at compute time — no FP8 tensor-core
math), the H100 has **native FP8 tensor cores**. vLLM JIT-compiles flashinfer's
`fp8_blockscale_gemm_90` (SM90) kernel and runs true FP8×FP8 GEMMs. So the
H100 FP8-vs-BF16 deltas below reflect actual FP8 compute speedup, and the sign
is the opposite of the A100 result (where FP8 made prefill *slower*).

## Environment

- Instance: P5 (H100 80GB SXM5 ×8), `/opt/pytorch` bundled vLLM
- vLLM 0.22.0, torch 2.11.0+cu130, flashinfer 0.6.11.post2
- CUDA: `/opt/pytorch/cuda` → `nvidia/cu13` (libs in `lib/`, not `lib64/`)
- Driver 595.71.05
- Single GPU (TP=1, `--gpu-memory-utilization 0.92`), GPU 0.

### flashinfer FP8-GEMM JIT link fix (H100-only)

The native FP8 path JIT-builds `fp8_blockscale_gemm_90`, whose ninja link step
hardcodes `-L${CUDA_HOME}/lib64` and `-lnvrtc`. This box keeps CUDA libs in
`lib/` with only versioned `libnvrtc.so.13`, so the link failed with
`cannot find -lnvrtc`. Fixed with two symlinks under the user-owned
`/opt/pytorch/cuda`:

```
ln -sfn lib  /opt/pytorch/cuda/lib64          # lib64 -> lib
ln -sf  libnvrtc.so.13  /opt/pytorch/cuda/lib/libnvrtc.so
```

A100 never hit this because Marlin weight-only needs no FP8-GEMM JIT.

## Sweep

`vllm bench serve`, `PREFILL_ISLS="1023 2047 4095 8191"`, `DECODE_CONC="1"`
(decode used ISL=1024 OSL=1024), `--random-range-ratio 0`.

| Test                              | TTFT (ms) | TPOT (ms) | OTPS (tok/s) |
|-----------------------------------|----------:|----------:|-------------:|
| Prefill ISL=1023                  |      92.8 |        — |           — |
| Prefill ISL=2047                  |     145.0 |        — |           — |
| Prefill ISL=4095                  |     259.3 |        — |           — |
| Prefill ISL=8191                  |     499.5 |        — |           — |
| Decode ISL=1024 OSL=1024 conc=1   |    1100.5 |     12.87 |        71.75 |

> **Note on ISL=1023:** warm re-measurement (the first prefill call of the
> sweep paid a one-time ~1.6 s flashinfer-autotune + CUDA-graph warmup). Raw
> cold value preserved in `fp8_prefill.log`.

## H100 FP8 vs H100 BF16, same host

| Test                       |  BF16 |   FP8 | FP8 / BF16 |
|----------------------------|------:|------:|-----------:|
| Prefill ISL=1023 TTFT (ms) |  96.3 |  92.8 |   0.96x    |
| Prefill ISL=2047 TTFT (ms) | 186.9 | 145.0 |   0.78x    |
| Prefill ISL=4095 TTFT (ms) | 352.7 | 259.3 |   0.74x    |
| Prefill ISL=8191 TTFT (ms) | 687.8 | 499.5 |   0.73x    |
| Decode TPOT (ms)           | 20.15 | 12.87 |   0.64x    |
| Decode OTPS (tok/s)        | 47.22 | 71.75 |   1.52x    |

Native FP8 wins on **both** prefill (compute-bound: ~27% faster at ISL≥4K) and
decode (bandwidth-bound: TPOT −36%, OTPS +52%). Contrast with A100, where FP8
helped decode (~40% via weight-bandwidth) but *hurt* prefill (dequant overhead
on the compute path). This is the expected consequence of H100 having real
FP8 tensor cores and A100 not.
