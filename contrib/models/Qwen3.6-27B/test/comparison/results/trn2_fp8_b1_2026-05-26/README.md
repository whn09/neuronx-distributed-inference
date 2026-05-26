# Trn2 FP8 batch=1 sweep — 2026-05-26

Same recipe as the BF16 batch=1 sweep on the same host (see
`../trn2_bf16_b1_2026-05-26/`), but with the contrib FP8 compile path. Driven
end-to-end by `test/comparison/trn2/drive_b1.sh` with `PRECISION=fp8`.

## What "FP8" means here — IMPORTANT CAVEAT

`qwen36_27b_compile_fp8.py` only quantizes the MLP `gate_proj/up_proj/down_proj`
weights to FP8 (e4m3, per-channel). Everything else stays BF16:

- Attention QKV/O, DeltaNet, embeddings, lm_head, norms — BF16
- KV cache — BF16 (`kv_cache_quant=False`)
- Activations — BF16 (`activation_quantization_type=None`)
- `quantized_mlp_kernel_enabled=False` → FP8 MLP weights are dequantized to
  BF16 at compute time

So this is a **memory-footprint experiment, not an FP8 compute throughput
test**. Treat the speed numbers as "BF16 with smaller MLP weight loads," not
as a true FP8 vs BF16 comparison. Real FP8 compute would require enabling the
quantized MLP kernel and probably also quantizing attention + KV cache.

## Environment

- Instance: trn2.3xlarge, SDK 2.29 (`neuronx-cc 2.25.3371.0+f524f7f8`)
- Compiler venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- Server venv: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16`
  (NxDI MODEL_TYPES patched via `vllm/install_qwen36_vllm.sh`)
- `NEURON_RT_VISIBLE_CORES=0-3`, TP=4, LNC=2

## Compile

```
qwen36_27b_compile_fp8.py
  --seq-len 9216 --batch-size 1
  --context-encoding-buckets 1024,2048,4096,8192
  --tp-degree 4 --logical-nc-config 2
  --quantized-checkpoints-path .../quantized/Qwen3.6-27B-fp8-mlp
```

Wall time: ~62 min (HLO 6 min, NEFF 56 min — FP8 e4m3 path is slightly slower
to compile than BF16). Artifact: 36 GB (vs 52 GB BF16 — the saving comes from
the MLP weights stored as fp8e4m3fn instead of bfloat16).

## Server

```
start_vllm_server.sh
  --max-model-len 9216 --seq-len 9216
  --context-encoding-buckets 1024,2048,4096,8192
  --max-num-seqs 1 --tensor-parallel-size 4 --logical-nc-config 2
  --port 8101
```

## Sweep

`vllm bench serve` driven by `test/comparison/trn2/run_sweeps.sh` with
`PREFILL_ISLS="1023 2047 4095 8191"` and `DECODE_CONC="1"`. Decode used
ISL=1024 OSL=1024, the harness default.

| Test                                 | TTFT (ms) | TPOT (ms) | OTPS (tok/s) |
|--------------------------------------|----------:|----------:|-------------:|
| Prefill ISL=1023                     |     1835 |        — |           — |
| Prefill ISL=2047                     |     3605 |        — |           — |
| Prefill ISL=4095                     |     7166 |        — |           — |
| Prefill ISL=8191                     |    14389 |        — |           — |
| Decode ISL=1024 OSL=1024 conc=1      |     3591 |     30.5 |        29.4 |

## FP8 (MLP-weight-only) vs BF16, same host, batch=1

| Test                                 |  BF16 |   FP8 | FP8 / BF16 |
|--------------------------------------|------:|------:|-----------:|
| Prefill ISL=1023 TTFT (ms)           |  1900 |  1835 |  0.97x     |
| Prefill ISL=2047 TTFT (ms)           |  3616 |  3605 |  1.00x     |
| Prefill ISL=4095 TTFT (ms)           |  7132 |  7166 |  1.00x     |
| Prefill ISL=8191 TTFT (ms)           | 14473 | 14389 |  0.99x     |
| Decode TPOT (ms)                     |  33.0 |  30.5 |  0.92x     |
| Decode OTPS (tok/s)                  |  27.4 |  29.4 |  1.07x     |

Reading these numbers honestly:

- Prefill is essentially unchanged. Prefill is matmul-heavy (compute-bound)
  and the FP8 weights are dequantized to BF16 before the matmul, so we
  shouldn't expect a speedup — and we don't see one.
- Decode is ~7% faster. Decode at b=1 is heavily memory-bandwidth-bound on
  the MLP weight loads, so smaller weights → fewer bytes per token → modest
  TPOT improvement. This is the *only* signal that smells like real FP8
  benefit, and it's exactly what MLP-weight-only quantization should give.
- Artifact size dropped 31% (52 GB → 36 GB), as expected from converting only
  the MLP weights from BF16 to FP8.

If anyone reads these as "Trn2 FP8 ~= Trn2 BF16, why bother?" — the answer is
"this isn't real FP8 compute yet." Enabling the FP8 MLP kernel and quantizing
attention/KV would be the next step before drawing throughput conclusions.
