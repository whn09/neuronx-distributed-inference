# Trn2 BF16 batch=1 Probe H prefill sweep — 2026-05-27

Re-runs the same batch=1 multi-bucket BF16 sweep as
`trn2_bf16_b1_decayrefactor_2026-05-27/`, on top of a kernel-level
optimization that eliminates the per-round transposes inside the fused
DeltaNet kernel's Neumann power-doubling loop.

## What changed

Each round of the Neumann series previously did two `nc_transpose` calls:
one to compute `A_pow^2 = transpose(A_pow) @ A_pow`, and one to compute
`(I+A_pow) @ P_acc = transpose(I+A_pow) @ P_acc`. Across 6 rounds that is
12 transposes per chunk, executed on the Vector Engine.

The refactor maintains `A_pow_T` alongside `A_pow` across rounds:

    A_pow_new   = A_pow_T^T @ A_pow             (one TE matmul)
    A_pow_T_new = A_pow^T   @ A_pow_T           (one TE matmul)
    IpA_T       = I + A_pow_T                   (no transpose: eye is symmetric)

Trade per round: 1 extra TE matmul, but 2 fewer VE transposes. Per chunk:
+6 matmuls and +1 initial `transpose(A_mat)` to seed `A_pow_T`, but −12
transposes across the loop. The kernel is VE-bound at b=1, so trading VE
work for TE work is a net win.

`nc_transpose` count per chunk drops from 15 (post-decay-refactor) to 9
(−40%); cumulative drop vs the original fused kernel is 25 → 9 (−64%).

Numerical equivalence verified offline at fp64 across 5 random
lower-triangular `A` matrices (max diff = 0 vs the original loop, since
both `A_pow` paths use the same identity `(A·A)^T = A^T·A^T`).

## Environment

- Instance: trn2.3xlarge, SDK 2.29 (`neuronx-cc 2.25.3371.0+f524f7f8`)
- Compiler venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- Server venv: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16`
- `NEURON_RT_VISIBLE_CORES=0-3`, TP=4, LNC=2
- DeltaNet prefill flags (defaults): same as 2026-05-27 decay-refactor.
  Only the kernel internals changed.

## Compile

```
qwen36_27b_compile_bf16.py
  --seq-len 9216 --batch-size 1
  --context-encoding-buckets 1024,2048,4096,8192
  --tp-degree 4 --logical-nc-config 2
```

Wall time: ~8 min on this host. Artifact: 52 GB.

## Sweep

`vllm bench serve` at `--random-input-len {1023,2047,4095,8191}
--random-output-len 1 --num-prompts 1 --max-concurrency 1
--random-range-ratio 0`, single shared server.

| ISL  | Decay-refactor (ms) | Probe H (ms) | Δ vs decay-refactor | Δ vs fused-hybrid (3dff2a7) |
|-----:|--------------------:|-------------:|--------------------:|----------------------------:|
| 1023 |              465.64 |       445.63 |               −4.3% |                       −7.0% |
| 2047 |              740.62 |       695.73 |               −6.1% |                       −9.0% |
| 4095 |             1382.23 |      1289.86 |               −6.7% |                       −9.9% |
| 8191 |             2938.34 |      2762.92 |               −6.0% |                       −9.3% |

Cross-platform (vs A100 BF16 P4DE TP=1):

- ISL=1K: Trn2 446 vs A100 500 → **Trn2 1.12x faster**
- ISL=8K: Trn2 2763 vs A100 1974 → Trn2 1.40x slower (was 1.49x with
  decay-refactor, 1.54x with fused-hybrid)

## Files

- `probeH_prefill_isl{1023,2047,4095,8191}.json` — raw vllm-bench-serve
  output

## Why this is faster

The fused DeltaNet kernel runs ~12 nc_matmul (Tensor Engine) and ~25
nc_transpose (Vector Engine) per (chunk, head) pre-refactor. After the
decay-refactor the count was ~14 matmuls / ~15 transposes; Probe H takes
that to ~20 matmuls / ~9 transposes. At b=1 the Tensor Engine is far
from peak (single-head matmuls don't fill it) so spending more matmuls
to free VE time is profitable.

The win scales sub-linearly with ISL (more chunks ⇒ more Neumann rounds,
but launch overhead amortizes). At ISL=8K we save ~175 ms over the
previous default kernel (decay-refactor); the absolute gap to A100 BF16
shrinks by another ~10%.
