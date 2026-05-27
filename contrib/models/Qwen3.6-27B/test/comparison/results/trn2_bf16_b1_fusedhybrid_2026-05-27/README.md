# Trn2 BF16 batch=1 fused-hybrid prefill sweep — 2026-05-27

Re-runs the same batch=1 multi-bucket BF16 sweep as
`trn2_bf16_b1_2026-05-26/`, but on the new fused-hybrid prefill path
(commit `3dff2a7`): `_fused_chunked_forward` is now the default
DeltaNet prefill kernel under the hybrid cache manager (with
`initial_state` plumbed through), replacing the per-chunk
`_nki_chunked_forward` launches.

## Environment

- Instance: trn2.3xlarge, SDK 2.29 (`neuronx-cc 2.25.3371.0+f524f7f8`)
- Compiler venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- Server venv: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16`
- `NEURON_RT_VISIBLE_CORES=0-3`, TP=4, LNC=2
- DeltaNet prefill flags (defaults): `use_hybrid_cache_manager=True`,
  `use_qwen_hybrid_chunked_prefill=True`,
  `use_qwen_hybrid_chunked_prefill_nki=True` — same flags as the
  2026-05-26 baseline; only the kernel routing inside
  `modeling_qwen35.py` changed.

## Compile

```
qwen36_27b_compile_bf16.py
  --seq-len 9216 --batch-size 1
  --context-encoding-buckets 1024,2048,4096,8192
  --tp-degree 4 --logical-nc-config 2
```

Wall time: ~13 min on this host (NEFF cache mostly cold for the new
kernel signature, but the compiled graph is structurally similar to
the 2026-05-26 artifact). Artifact: 52 GB.

## Sweep

`vllm bench serve` at `--random-input-len {1023,2047,4095,8191}
--random-output-len 1 --num-prompts 1 --max-concurrency 1
--random-range-ratio 0`, single shared server.

| ISL  | Shipped chunked-NKI (ms) | Fused-hybrid (ms) | Speedup |
|-----:|-------------------------:|------------------:|--------:|
| 1023 |                     1900 |               479 |  3.97x  |
| 2047 |                     3616 |               764 |  4.74x  |
| 4095 |                     7132 |              1431 |  4.98x  |
| 8191 |                    14473 |              3046 |  4.75x  |

(`Shipped chunked-NKI` column is verbatim from
`trn2_bf16_b1_2026-05-26/README.md`.)

The fused-hybrid path is ~4-5x faster across all ISLs, and TTFT
scales near-linearly (~370 µs/token) versus the shipped path's
~1.78 s/1K tokens — same linear regime, ~5x lower constant.

## Why this is faster

The shipped path launches a NKI kernel **per (batch, head, chunk)** for
the 47 DeltaNet layers; the fused kernel launches **once per (batch,
head)** and keeps the recurrent state in SBUF across all chunks. The
chunk count grows with ISL (8 chunks at 1K, 64 at 8K), so launch
overhead dominated at every ISL. The fused kernel removes that
overhead.

The kernel itself is the original `nki_deltanet_fused.py` (6-round
Neumann power-doubling), not jim's PR141 v17 variant. v17 was
benched separately at ISL=1023 and came in at 639 ms (vs 443 ms for
this kernel) — its 8-block forward-substitution was tuned for
Qwen3.5-2B's shapes, not Qwen3.6-27B's 47-layer DeltaNet stack. v17
was reverted in `3dff2a7`.

## Files

- `fusedhybrid_multi_prefill_isl{1023,2047,4095,8191}.json` — raw
  vllm-bench-serve output
- `fusedhybrid_multi_bench.log` — combined stdout from the four
  bench invocations
