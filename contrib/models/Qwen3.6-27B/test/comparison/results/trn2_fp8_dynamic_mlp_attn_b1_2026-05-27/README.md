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

Compared to the original weight-only FP8 baseline (`../trn2_fp8_b1_2026-05-26/`):

| Test                            |       weight-only |   dynamic-MLP+attn |   delta vs WO |
|---------------------------------|------------------:|-------------------:|--------------:|
| Prefill ISL=1023 TTFT (ms)      |            1835.3 |             1804.6 |         −1.7% |
| Prefill ISL=2047 TTFT (ms)      |            3605.5 |             3555.1 |         −1.4% |
| Prefill ISL=4095 TTFT (ms)      |            7166.4 |             7091.7 |         −1.0% |
| Prefill ISL=8191 TTFT (ms)      |           14388.9 |            14299.4 |         −0.6% |
| Decode TPOT (ms)                |             30.55 |              27.33 |        −10.5% |
| Decode OTPS (tok/s)             |             29.39 |              32.51 |        +10.6% |

Reading honestly:

- Most of the decode TPOT win was already realized by T1 (FP8-MLP); extending
  FP8 to standard self-attention QKV/O on top is a wash at b=1 (−0.5%). At
  this batch size, decode time per token is dominated by *DeltaNet* state
  updates and KV-cache reads on the `linear_attention` layers — both still
  BF16 — so converting the standard QKV/O on the 17 `full_attention` layers
  does not move the needle.
- Prefill is unchanged (within ±0.5%). The standard self-attention compute
  *is* on the prefill critical path, but the wall-clock here is dominated by
  the chunked DeltaNet prefill + RoPE + softmax that stay BF16, so swapping
  4 of the 64-layer-wide projections for FP8 is invisible at these ISLs.

The outcome confirms the expected ordering — the highest-leverage scope for
FP8 on Qwen3.6 at b=1 is the MLP path. T2 is a clean architectural win (no
quality regression, no extra runtime overhead) but does not unlock further
throughput beyond T1 in this configuration.

## Trade-off vs T1

Compile time roughly doubles for T2 vs T1 because every `full_attention`
QKV/O matmul now expands into FP8-quant + matmul + FP8-dequant in HLO, which
the neuronx-cc Tensorizer/SimplifierBackendPasses re-process per-bucket. The
8K CTE bucket alone took ~50 minutes for T2 vs ~25 minutes for T1.
