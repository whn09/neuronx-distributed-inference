# Trn2 FP8 dynamic-MLP+attn batch=1 sweep — 2026-05-27

Same compile recipe and host as `../trn2_fp8_dynamic_mlp_b1_2026-05-27/`,
extending FP8 scope to also cover **standard self-attention QKV/O** on the
`full_attention` layers. DeltaNet (`linear_attn`), `output_gate_proj`, norms,
embeddings, lm_head, and KV cache stay BF16.

Driven by `test/comparison/trn2/drive_b1.sh` with `PRECISION=fp8` and
`FP8_TIER=dynamic_mlp_attn`.

## What changed vs. T1 (`dynamic_mlp`)

T1 only converts the MLP linear layers (`gate_proj` / `up_proj` / `down_proj`)
to FP8 and runs them as real FP8 × FP8 with dynamic activation scales.

T2 additionally converts `self_attn.q_proj` / `k_proj` / `v_proj` / `o_proj`
on every `full_attention` layer to FP8. To keep this scope-only and avoid
re-shaping any modeling code paths:

1. **Pre-split doubled q_proj before quantize.** Qwen3.6's HF
   `self_attn.q_proj.weight` is `(num_heads · head_dim · 2, hidden)` with the
   interleaved layout `[head0_query | head0_gate | head1_query | …]`. NxDI's
   per-channel FP8 scale (`channel_axis=0`) is `(num_heads · head_dim · 2, 1)`
   — but at load time `convert_qwen35_hf_to_neuron_state_dict` splits the
   tensor into `q_proj.weight` (query, `(num_heads · head_dim, hidden)`) and
   `output_gate_proj.weight` (gate, same shape). Splitting *after* quantize
   would orphan half of the scale rows. So `qwen36_27b_compile_fp8.py` now
   pre-splits the doubled q_proj **before** calling `quantize_fp8_per_channel`,
   and emits the gate half as a plain BF16 `output_gate_proj.weight` in the
   FP8 checkpoint shard.
2. **Make the runtime split idempotent.** `convert_qwen35_hf_to_neuron_state_dict`
   now skips the split when `output_gate_proj.weight` is already present in
   the state dict (the FP8 pre-split case). Both behaviors are covered by
   `test/unit/test_weight_conversion.py::TestQProjSplitIdempotent`.
3. **Keep `output_gate_proj` BF16.** It's added to `modules_to_not_convert`
   for the `dynamic_mlp_attn` tier so NxDI's runtime quant pass leaves it
   alone (no weight scale exists for it; replacing the Linear would trip the
   load-time scale lookup).

## Correctness gate

`drive_b1.sh` ran `correctness_gate.py` between compile and sweep. Both
prompts produced ≥12 distinct new tokens with no degenerate repetition:

    GATE PASS prompt='The capital of France is' new_tokens=20 distinct=12 repetitive=False
    GATE PASS prompt='Hello, I am a language model'  new_tokens=20 distinct=16 repetitive=False

Compile-time we also see `MANUAL_FP8_QUANT_COUNT tier=dynamic_mlp_attn
count=263 q_proj_split=17`, where 17 matches the number of `full_attention`
layers in this 64-layer config.

## Results

| Test                            | dynamic-MLP FP8 | dynamic-MLP+attn FP8 |   delta |
|---------------------------------|----------------:|---------------------:|--------:|
| Prefill ISL=1023 TTFT (ms)      |          1849.1 |               1804.6 |   −2.4% |
| Prefill ISL=2047 TTFT (ms)      |          3550.3 |               3555.1 |   +0.1% |
| Prefill ISL=4095 TTFT (ms)      |          7079.3 |               7091.7 |   +0.2% |
| Prefill ISL=8191 TTFT (ms)      |         14287.1 |              14299.4 |   +0.1% |
| Decode TPOT (ms)                |           27.47 |                27.33 |   −0.5% |
| Decode OTPS (tok/s)             |           32.37 |                32.51 |   +0.4% |

Compared to the BF16 baseline (`../trn2_bf16_b1_2026-05-26/`) and the
weight-only FP8 baseline (`../trn2_fp8_b1_2026-05-26/`):

| Test                            |  BF16 | weight-only FP8 | dynamic-MLP FP8 | dynamic-MLP+attn FP8 |
|---------------------------------|------:|----------------:|----------------:|---------------------:|
| Prefill ISL=1023 TTFT (ms)      |  1900 |          1835.3 |          1849.1 |               1804.6 |
| Prefill ISL=8191 TTFT (ms)      | 14473 |         14388.9 |         14287.1 |              14299.4 |
| Decode TPOT (ms)                | 33.01 |           30.55 |           27.47 |                27.33 |
| Decode OTPS (tok/s)             | 27.39 |           29.39 |           32.37 |                32.51 |

Reading honestly:

- **vs T1 (dynamic-MLP):** wash at b=1 (decode TPOT −0.5%, prefill within
  ±0.5%). At this batch size decode per-token is dominated by *DeltaNet*
  state updates and KV-cache reads on the `linear_attention` layers —
  still BF16 — so converting QKV/O on the 17 `full_attention` layers does
  not move the wall clock.
- **vs BF16:** decode keeps the ~18% throughput win that T1 already
  unlocked (TPOT 33.0 → 27.3 ms, OTPS 27.4 → 32.5 tok/s). Prefill stays
  noise-level vs BF16 because the chunked DeltaNet prefill + RoPE +
  softmax dominate and remain BF16 in all FP8 tiers here.

**Where T2 still earns its keep: HBM.** Quantizing the 17 full_attention
QKV/O projections to FP8 saves ~1.4 GB of weight memory on top of T1
(17 layers × ~84 MB/layer at hidden=5120, q_proj 6144 + k_proj/v_proj/o_proj
shapes, BF16→FP8 halves the per-tensor footprint). At b=1 this does not
translate to throughput because we're not KV-cache-pressured, but it does
free that headroom for larger batch / longer context configurations where
the bottleneck shifts onto the standard-attention path.

The outcome confirms the expected ordering for *throughput* at b=1 — the
highest-leverage scope for FP8 on Qwen3.6 is the MLP path. T2 is a clean
architectural extension (correctness gate passes, no extra runtime
overhead) and a memory-footprint win; it just does not unlock further
throughput in this single-batch configuration.

## Trade-off vs T1

Compile time roughly doubles for T2 vs T1 because every `full_attention`
QKV/O matmul now expands into FP8-quant + matmul + FP8-dequant in HLO, which
the neuronx-cc Tensorizer/SimplifierBackendPasses re-process per-bucket. The
8K CTE bucket alone took ~50 minutes for T2 vs ~25 minutes for T1.
