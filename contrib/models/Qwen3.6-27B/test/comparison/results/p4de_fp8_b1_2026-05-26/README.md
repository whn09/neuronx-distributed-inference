# P4DE (A100 80GB SXM4) FP8 batch=1 sweep — 2026-05-26

vLLM `Qwen/Qwen3.6-27B-FP8` (HF native FP8 checkpoint) on a P4DE host with
TP=1, BF16 KV cache, max-model-len 9216. A100 has no native FP8 tensor cores,
so vLLM falls back to the Marlin weight-only kernel — FP8 weights are
dequantized to BF16 at compute time. This is the same paradigm as the Trn2
weight-only FP8 baseline, not a real FP8 compute test.

## Environment

- Instance: P4DE (A100 80GB SXM4 ×8), `/opt/pytorch` bundled vLLM
- CUDA: `/opt/pytorch/cuda` (cu13, libs in `lib/`, not `lib64/`)
- Launcher: `start_vllm_p4de.sh` with `LD_LIBRARY_PATH` + `LIBRARY_PATH` both
  pointing at `${CUDA_HOME}/lib` so flashinfer's JIT ninja builds can find
  `-lcudart` at link time.
- Single GPU (TP=1, `--gpu-memory-utilization 0.92`).

## Sweep

`vllm bench serve` with `PREFILL_ISLS="1023 2047 4095 8191"` and
`DECODE_CONC="1"` (decode used ISL=1024 OSL=1024).

| Test                              | TTFT (ms) | TPOT (ms) | OTPS (tok/s) |
|-----------------------------------|----------:|----------:|-------------:|
| Prefill ISL=1023                  |     596.4 |        — |           — |
| Prefill ISL=2047                  |     693.5 |        — |           — |
| Prefill ISL=4095                  |    1928.3 |        — |           — |
| Prefill ISL=8191                  |    2816.7 |        — |           — |
| Decode ISL=1024 OSL=1024 conc=1   |     349.0 |     20.68 |        47.62 |
