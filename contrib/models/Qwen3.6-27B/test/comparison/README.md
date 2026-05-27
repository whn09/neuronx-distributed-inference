# Qwen3.6-27B GPU vs Trn2 comparison harness

Apples-to-apples benchmark scripts for comparing the contrib Qwen3.6-27B model on
A100 (and other GPUs) against the same model on Trn2. Both sides use vLLM's
standard `vllm bench serve` client so the request methodology, sampling, and
metrics are identical.

```
test/comparison/
├── gpu/    # vLLM CUDA backend (FP8 + BF16) + bench harness
└── trn2/   # vLLM Neuron backend (FP8 + BF16) + bench harness
```

The two sides share the same prompt distribution (`--dataset-name random
--random-range-ratio`) and the same sweep grids:

* **Prefill sweep**: ISL ∈ {8K, 16K, 32K, 64K, 128K, 256K}, OSL = 1, concurrency = 1.
* **Decode sweep**: ISL = 1K, OSL = 1K, concurrency ∈ {1, 2, 4, 8, 16, 32, 64}.

Stop conditions per side: TTFT > 120 s for prefill, TTFT > 10 s for decode.

See `gpu/README.md` and `trn2/README.md` for one-host run instructions.

## Results summary (b=1, single host)

All numbers are median across the same 4 prefill ISLs (1023/2047/4095/8191)
and a single-concurrency 1K-in/1K-out decode. Each row links to its own
result directory under `results/`, where the raw JSONs and per-run README
live.

### BF16 head-to-head: Trn2 vs A100 (b=1)

A100 has no native FP8 tensor cores, so its FP8 path is Marlin weight-only —
not a fair comparison against Trn2's native FP8. The honest cross-platform
comparison is BF16 vs BF16:

| Platform                                   | Prefill ISL=1023 (ms) | Prefill ISL=8191 (ms) | Decode TPOT (ms) | Decode OTPS (tok/s) |
|--------------------------------------------|----------------------:|----------------------:|-----------------:|--------------------:|
| A100 (P4DE, TP=1, BF16)                    |                   500 |                  1974 |            35.14 |                28.3 |
| Trn2 (trn2.3xlarge, TP=4, BF16, fused)     |                   479 |                  3046 |            33.01 |                27.4 |
| **Trn2 / A100 ratio**                      |                **0.96x** |                **1.54x** |        **0.94x** |             **0.97x** |

Trn2 numbers are the fused-hybrid prefill path (commit `3dff2a7`,
`results/trn2_bf16_b1_fusedhybrid_2026-05-27/`). The earlier shipped
chunked-NKI prefill path (`results/trn2_bf16_b1_2026-05-26/`) clocked
1900 ms / 14473 ms at the same two ISLs — see the next section for
the full sweep and why this changed.

**Reading the comparison:**

- **Decode is at parity** (Trn2 33.0 ms vs A100 35.1 ms TPOT, ~3% gap
  either way). Both are weight-bandwidth bound at b=1.
- **Prefill at 1K is at parity** (Trn2 479 vs A100 500 ms). At 8K
  Trn2 is ~1.5x slower, and the gap is from Trn2's near-linear
  per-chunk scan vs A100's sublinear vLLM prefill — not from launch
  overhead anymore.

#### Why the BF16 numbers changed vs the earlier sweep

The chunked-prefill path on Qwen3.6 has 47 DeltaNet (`linear_attention`)
layers out of 64. The shipped DeltaNet prefill kernel
(`_nki_chunked_forward`) launches one NKI kernel per `(batch, head,
chunk)` — at ISL=8K with `chunk_size=128` that is 64 launches × 47
layers = 3008 launches, and that launch overhead dominated TTFT.

The new path (`_fused_chunked_forward` from `nki_deltanet_fused.py`)
launches **one kernel per (batch, head)** for the whole sequence and
keeps the recurrent state in SBUF across all chunks. To make this
usable under the hybrid cache manager — which seeds DeltaNet prefill
with prior recurrent state — `3dff2a7` plumbs an `initial_state`
argument through the kernel and routes the hybrid-cache prefill code
path through the fused kernel by default. This drops launch count from
~3000 to ~47 per layer-stack and gives the 4-5x speedup tabulated
below.

This was also the test case where we tried `jim`'s PR141 v17 fused
kernel (8-block forward-substitution Neumann, tuned for Qwen3.5-2B).
On Qwen3.6 dimensions v17 came in at 639 ms vs 443 ms for the original
6-round Neumann fused kernel, so v17 was reverted in `3dff2a7`.

#### Trn2 BF16 prefill: shipped vs fused-hybrid

| ISL  | Shipped chunked-NKI (ms) | Fused-hybrid (ms) | Speedup |
|-----:|-------------------------:|------------------:|--------:|
| 1023 |                     1900 |               479 |  3.97x  |
| 2047 |                     3616 |               764 |  4.74x  |
| 4095 |                     7132 |              1431 |  4.98x  |
| 8191 |                    14473 |              3046 |  4.75x  |

- `results/trn2_bf16_b1_fusedhybrid_2026-05-27/` (default code path)
- `results/trn2_bf16_b1_2026-05-26/` (earlier shipped path, kept for
  reference)
- `results/p4de_bf16_b1_2026-05-26/`

### Trn2 FP8 sweep (Trn2-internal, b=1)

This sweep is Trn2-only — it characterizes the *scope* of FP8 quantization
on Qwen3.6, not a cross-platform comparison.

> **Note:** the FP8 numbers below were collected on the older
> chunked-NKI prefill path (pre-`3dff2a7`). The BF16 row is reproduced
> here as the matching baseline so the deltas are apples-to-apples. The
> *current* Trn2 BF16 prefill (fused-hybrid) is in the head-to-head
> table above. FP8 has not been re-swept on the new path yet — it would
> show the same ~5x prefill speedup since the FP8 paths don't touch the
> DeltaNet kernel.

| Tier                                   | Prefill ISL=1023 (ms) | Prefill ISL=8191 (ms) | Decode TPOT (ms) | Decode OTPS (tok/s) | HBM scope                                     |
|----------------------------------------|----------------------:|----------------------:|-----------------:|--------------------:|-----------------------------------------------|
| BF16 (chunked-NKI baseline)            |                  1900 |                 14473 |            33.01 |                27.4 | all BF16                                      |
| FP8 weight-only MLP                    |                  1835 |                 14389 |            30.55 |                29.4 | MLP weights FP8, BF16 compute                 |
| FP8 dynamic MLP                        |                  1849 |                 14287 |            27.47 |                32.4 | + MLP activations FP8 (real FP8×FP8)          |
| FP8 dynamic MLP+attn                   |                  1805 |                 14299 |            27.33 |                32.5 | + std-attn QKV/O FP8 on full_attention layers |

- `results/trn2_fp8_b1_2026-05-26/` (weight-only MLP)
- `results/trn2_fp8_dynamic_mlp_b1_2026-05-27/` (T1)
- `results/trn2_fp8_dynamic_mlp_attn_b1_2026-05-27/` (T2)

**How to read the FP8 sweep:**

- **Weight-only → dynamic MLP** captures the actual FP8 compute win at b=1
  (decode TPOT −10%, OTPS +10%); prefill barely moves because the
  DeltaNet kernel itself is unchanged.
- **Dynamic MLP → dynamic MLP+attn** is a wash on throughput (±0.5%) but
  buys ~1.4 GB of HBM headroom on the 17 full_attention layers; the value
  shows up at higher batch / longer context, not at b=1.
- FP8 and the fused-hybrid prefill path are independent levers — FP8
  helps decode bandwidth, the fused kernel helps prefill launch
  overhead. Stacking the two would compose.
