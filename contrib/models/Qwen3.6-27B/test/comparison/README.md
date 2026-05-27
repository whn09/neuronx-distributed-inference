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

### Trn2 (trn2.3xlarge, TP=4, LNC=2)

| Tier                            | Prefill ISL=1023 (ms) | Prefill ISL=8191 (ms) | Decode TPOT (ms) | Decode OTPS (tok/s) | HBM scope                                     |
|---------------------------------|----------------------:|----------------------:|-----------------:|--------------------:|-----------------------------------------------|
| BF16                            |                  1900 |                 14473 |            33.01 |                27.4 | all BF16                                      |
| FP8 weight-only MLP             |                  1835 |                 14389 |            30.55 |                29.4 | MLP weights FP8, BF16 compute                 |
| FP8 dynamic MLP                 |                  1849 |                 14287 |            27.47 |                32.4 | + MLP activations FP8 (real FP8×FP8)          |
| FP8 dynamic MLP+attn            |                  1805 |                 14299 |            27.33 |                32.5 | + std-attn QKV/O FP8 on full_attention layers |

- `results/trn2_bf16_b1_2026-05-26/`
- `results/trn2_fp8_b1_2026-05-26/` (weight-only MLP)
- `results/trn2_fp8_dynamic_mlp_b1_2026-05-27/` (T1)
- `results/trn2_fp8_dynamic_mlp_attn_b1_2026-05-27/` (T2)

### A100 (P4DE, TP=1, max-model-len 9216)

| Tier         | Prefill ISL=1023 (ms) | Prefill ISL=8191 (ms) | Decode TPOT (ms) | Decode OTPS (tok/s) |
|--------------|----------------------:|----------------------:|-----------------:|--------------------:|
| BF16         |                   500 |                  1974 |            35.14 |                28.3 |
| FP8 (Marlin) |                   596 |                  2817 |            20.68 |                47.6 |

- `results/p4de_bf16_b1_2026-05-26/`
- `results/p4de_fp8_b1_2026-05-26/` (Marlin weight-only — A100 has no native FP8 cores)

### How to read the FP8 sweep

- **Weight-only → dynamic MLP** captures the actual FP8 compute win at b=1
  (decode TPOT −10%, OTPS +10%); prefill is unchanged because the chunked
  DeltaNet prefill stays BF16.
- **Dynamic MLP → dynamic MLP+attn** is a wash on throughput (±0.5%) but
  buys ~1.4 GB of HBM headroom on the 17 full_attention layers; the value
  shows up at higher batch / longer context, not at b=1.
- See each per-run README for honest context: A100 FP8 is Marlin
  weight-only (no native FP8 tensor cores), so the GPU-side decode win is
  bandwidth, not FP8 compute — the same paradigm as the Trn2 weight-only
  baseline.
