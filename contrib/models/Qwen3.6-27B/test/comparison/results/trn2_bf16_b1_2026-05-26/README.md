# Trn2 BF16 batch=1 sweep — 2026-05-26

Re-run on a fresh trn2.3xlarge after the previous host was rotated. Reproduces
the proven batch=1 multi-bucket BF16 path; numbers match the earlier host's
batch=1 sweep, confirming the reproduction recipe is stable.

## Environment

- Instance: trn2.3xlarge, SDK 2.29 (`neuronx-cc 2.25.3371.0+f524f7f8`)
- Compiler venv: `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference`
- Server venv: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16`
  (NxDI MODEL_TYPES patched via `vllm/install_qwen36_vllm.sh`)
- `NEURON_RT_VISIBLE_CORES=0-3`, TP=4, LNC=2

## Compile

```
qwen36_27b_compile_bf16.py
  --seq-len 9216 --batch-size 1
  --context-encoding-buckets 1024,2048,4096,8192
  --tp-degree 4 --logical-nc-config 2
```

Wall time: ~58 min (HLO trace 6 min, NEFF compile 52 min — 8192 bucket is the
long pole). Artifact: 52 GB.

## Server

```
start_vllm_server.sh
  --max-model-len 9216 --seq-len 9216
  --context-encoding-buckets 1024,2048,4096,8192
  --max-num-seqs 1 --tensor-parallel-size 4 --logical-nc-config 2
```

## Sweep

`vllm bench serve` driven by `test/comparison/trn2/run_sweeps.sh` with
`PREFILL_ISLS="1023 2047 4095 8191"` and `DECODE_CONC="1"`. Decode used
ISL=1024 OSL=1024, the harness default.

| Test                                 | TTFT (ms) | TPOT (ms) | OTPS (tok/s) |
|--------------------------------------|----------:|----------:|-------------:|
| Prefill ISL=1023                     |     1900 |        — |           — |
| Prefill ISL=2047                     |     3616 |        — |           — |
| Prefill ISL=4095                     |     7132 |        — |           — |
| Prefill ISL=8191                     |    14473 |        — |           — |
| Decode ISL=1024 OSL=1024 conc=1      |     3614 |     33.0 |        27.4 |

Prefill scales linearly at ~1.78s per 1K tokens. Decode TPOT 33 ms and OTPS 27
match the previous host's batch=1 numbers within noise.

## Notes

A batch=4 single-bucket compile was attempted first on this host (seq=4096,
buckets=1024/2048/4096) and produced unreliable bench numbers: prefill was
~4× slower than batch=1, and decode at concurrency≥2 hit
`sample_tokens() called without prior execute_model()` (a known vllm-neuron v1
race). The batch=1 path here is the trustworthy reference.
