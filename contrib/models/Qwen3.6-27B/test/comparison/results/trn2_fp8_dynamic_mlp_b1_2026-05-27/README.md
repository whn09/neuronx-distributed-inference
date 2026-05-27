# Trn2 FP8 dynamic-MLP batch=1 sweep — 2026-05-27

Same compile recipe and host as the FP8 weight-only baseline
(`../trn2_fp8_b1_2026-05-26/`), changing only the FP8 *scope*: this run also
quantizes MLP **activations** to FP8 (`activation_quantization_type="dynamic"`)
on top of the FP8 MLP weights, so MLP matmuls are real FP8 × FP8 instead of
BF16 × dequantized-FP8. Attention QKV/O, DeltaNet, norms, embeddings, lm_head,
and KV cache stay BF16.

Driven by `test/comparison/trn2/drive_b1.sh` with `PRECISION=fp8` and
`FP8_TIER=dynamic_mlp`.

## NxDI scale_dequantize fix (compile-time monkey-patch)

NxDI's `scale_dequantize(tensor, scale, ...)` always does
`scale.unsqueeze(len(scale.shape) - 1)` before broadcasting. That is correct
for the 2-D weight scale `[1, out]` against a 3-D matmul output `(B, S, out)`.
But for the *input* side of the dynamic FP8 path,
`quantize_fp8_per_channel(x, channel_axis=1)` over a 3-D activation
`(B, S, H)` produces a 3-D scale `[1, S, 1]`; the unsqueeze rewrites it to
`[1, S, 1, 1]` and the in-place broadcast onto a 3-D matmul output trips
neuronx-cc's `input_sizes <= output_sizes` rank check at HLO time:

    RuntimeError: Check failed: input_sizes.size() <= output_sizes.size()

`qwen36_27b_compile_fp8.py::_patch_scale_dequantize_for_3d_activations()`
swaps in a shim that only `unsqueeze`s when `scale.ndim < tensor.ndim`,
leaving the 2-D weight-scale path untouched. The patch is applied unconditionally
when `--quantization-tier` is `dynamic_mlp` or `dynamic_mlp_attn`.

## Results

| Test                            | weight-only FP8 | dynamic-MLP FP8 |   delta |
|---------------------------------|----------------:|----------------:|--------:|
| Prefill ISL=1023 TTFT (ms)      |          1835.3 |          1849.1 |   +0.8% |
| Prefill ISL=2047 TTFT (ms)      |          3605.5 |          3550.3 |   −1.5% |
| Prefill ISL=4095 TTFT (ms)      |          7166.4 |          7079.3 |   −1.2% |
| Prefill ISL=8191 TTFT (ms)      |         14388.9 |         14287.1 |   −0.7% |
| Decode TPOT (ms)                |           30.55 |           27.47 |  −10.1% |
| Decode OTPS (tok/s)             |           29.39 |           32.37 |  +10.1% |

Reading honestly:

- Decode TPOT improves ~10%. At b=1 decode is memory-bandwidth-bound on the
  MLP weight loads, but it also performs an MLP matmul per token; replacing
  the BF16 × dequantized-FP8 matmul with FP8 × FP8 + a small per-row scale
  apply trims that cost.
- Prefill is essentially unchanged (within ~1.5%), which is what we'd expect
  for an MLP-only FP8 path — prefill is dominated by the *attention* compute
  on the full_attention layers, which is still BF16 in this tier.

## Correctness gate

`drive_b1.sh` ran `correctness_gate.py` between compile and sweep; both
prompts produced ≥12 distinct words with no degenerate repetition (see
`trn2_fp8_dynamic_mlp_b1_gate.log`).

## Next step (T2 — `dynamic_mlp_attn`)

Extending FP8 to standard self-attention QKV/O (DeltaNet `linear_attn` stays
BF16) needs a modeling-side change first: Qwen3.6's `self_attn.q_proj` is
*doubled* (query + gate); the HF→Neuron converter splits it at load time, but
that split runs *after* quantization in the current FP8 compile path, so the
weight scale would not match. Tracked as task #32 / #34.
