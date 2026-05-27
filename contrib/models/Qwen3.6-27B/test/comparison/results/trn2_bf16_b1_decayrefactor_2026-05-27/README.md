# Trn2 BF16 batch=1 decay-refactor prefill sweep — 2026-05-27

Re-runs the same batch=1 multi-bucket BF16 sweep as
`trn2_bf16_b1_fusedhybrid_2026-05-27/`, but on top of a kernel-level
optimization that removes redundant transposes from the fused DeltaNet
prefill kernel.

## What changed

`nki_deltanet_fused.py` previously did the per-chunk decay scaling twice:
once for `QK_decay` (used to build the Neumann correction matrix `A`) and
once for `qk_decay` (used to build `attn_intra`). Each path did the same
sequence of three ops: row-scale by `exp(gc)` → transpose → col-scale by
`exp(-gc)` → transpose back. That is two `nc_transpose` calls per path,
four per chunk total.

The refactor builds a single shared decay matrix once per chunk:

    decay_lo[i, j] = Lmask[i, j] * exp(gc[i]) * exp(-gc[j])

which costs the same two `nc_transpose` calls as before, but is then
reused by both downstream paths via element-wise multiply. The
lower-with-diag variant `decay_lo_d = decay_lo + I` is cheap (one
tensor_tensor add) because the diagonal of the decay factor is
`exp(gc[i]) * exp(-gc[i]) = 1`.

Net: `nc_transpose` count per chunk drops from 25 to 15 (-40%).

Numerical equivalence verified offline at fp32 (max diff ~1e-7 for both
the `A` matrix and `attn_intra`) before recompiling.

## Environment

- Instance: trn2.3xlarge, SDK 2.29 (`neuronx-cc 2.25.3371.0+f524f7f8`)
- Compiler venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- Server venv: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16`
- `NEURON_RT_VISIBLE_CORES=0-3`, TP=4, LNC=2
- DeltaNet prefill flags (defaults): same as 2026-05-27 fused-hybrid
  baseline. Only the kernel internals changed.

## Compile

```
qwen36_27b_compile_bf16.py
  --seq-len 9216 --batch-size 1
  --context-encoding-buckets 1024,2048,4096,8192
  --tp-degree 4 --logical-nc-config 2
```

Wall time: ~13 min on this host (NEFF cache cold for the new kernel
signature). Artifact: 52 GB.

## Sweep

`vllm bench serve` at `--random-input-len {1023,2047,4095,8191}
--random-output-len 1 --num-prompts 1 --max-concurrency 1
--random-range-ratio 0`, single shared server.

| ISL  | Fused-hybrid (ms) | Decay-refactor (ms) | Δ vs fused-hybrid |
|-----:|------------------:|--------------------:|------------------:|
| 1023 |               479 |              465.64 | -2.8%             |
| 2047 |               764 |              740.62 | -3.1%             |
| 4095 |              1431 |             1382.23 | -3.4%             |
| 8191 |              3046 |             2938.34 | -3.5%             |

The win scales with ISL (more chunks = more transposes saved). At
ISL=8191 we save ~108 ms over the previous default kernel; the absolute
gap to A100 BF16 (1974 ms) shrinks from 1.54x to 1.49x.

## Files

- `decayrefactor_multi_prefill_isl{1023,2047,4095,8191}.json` — raw
  vllm-bench-serve output

## Why this is faster

The fused DeltaNet kernel runs ~12 nc_matmul (Tensor Engine) and ~25
nc_transpose (Vector Engine) per (chunk, head). At b=1 the kernel is far
from TE peak — VE work and the SBUF bookkeeping behind it dominates.
Cutting 10 transposes per chunk frees 40% of the VE work the compiler
otherwise serialized with TE matmuls, and the improvement compounds
linearly with chunk count (i.e. with ISL).

## Why this is *only* 3% (and not the 40% transpose reduction would
suggest)

The compiler was already overlapping a chunk of the eliminated
transposes with the main TE matmul wave. After the refactor, the
remaining transposes (Neumann power-doubling, q/k_cumdecay/attn_intra
input conditioning) still leave ~5 transposes on the critical path, so
the visible TTFT drop is ~3% rather than ~10%. Further wins would
require either (a) restructuring the Neumann round so each iteration
uses one fewer transpose, or (b) growing chunk_size to amortize per-chunk
fixed costs.
