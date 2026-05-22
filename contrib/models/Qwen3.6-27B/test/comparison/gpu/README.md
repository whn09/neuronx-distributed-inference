# GPU side: Qwen3.6-27B vLLM benchmark harness

Standard `vllm bench serve` driver for an A100 / Hopper / Blackwell host running
the contrib Qwen3.6-27B model (FP8 or BF16) under vLLM's CUDA backend.

## Files

* `start_vllm.sh` — launch one vLLM server on a single GPU (parameterised
  `MODEL_PATH`, `SERVED_NAME`, `PORT`, `GPUS`, `TP`, `MAX_MODEL_LEN`).
* `launch_both.sh` — convenience wrapper to start FP8 on GPU 0 and BF16 on
  GPU 1 simultaneously.
* `run_sweeps.sh` — drive prefill + decode sweeps with `vllm bench serve`.

## Prerequisites

* CUDA toolkit on PATH so flashinfer JIT can compile (`/usr/local/cuda-13.2`
  via apt is what we tested on AWS DLAMI).
* `ninja` available on PATH (`pip install ninja`, then ensure `<venv>/bin` is on
  PATH).
* vLLM ≥ 0.21 with the `vllm bench` subcommand.
* `Qwen3.6-27B-FP8` and `Qwen3.6-27B-BF16` weights downloaded locally.

## One-host run

```bash
# 1. start servers
FP8_MODEL_PATH=/path/to/Qwen3.6-27B-FP8 \
BF16_MODEL_PATH=/path/to/Qwen3.6-27B-BF16 \
LOG_DIR=/tmp/qwen36_vllm \
./launch_both.sh
# wait for "Application startup complete" in both vllm_*.log files

# 2. run sweeps (FP8 first, then BF16)
TAG=fp8  PORT=8000 SERVED_NAME=Qwen3.6-27B-FP8  OUTDIR=/tmp/qwen36_sweeps/fp8  ./run_sweeps.sh
TAG=bf16 PORT=8001 SERVED_NAME=Qwen3.6-27B-BF16 OUTDIR=/tmp/qwen36_sweeps/bf16 ./run_sweeps.sh
```

Each sweep emits one JSON result file per (ISL or concurrency) point in
`OUTDIR`, plus a single `${TAG}_prefill.log` / `${TAG}_decode.log` with the
parsed `SUMMARY ...` lines. Auto-stop:
* prefill stops when median TTFT > `PREFILL_TTFT_LIMIT_S` (default 120 s);
* decode stops when max TTFT > `DECODE_TTFT_LIMIT_S` (default 10 s).

## Methodology

The sweep uses vLLM's standard random dataset:

```
vllm bench serve \
  --dataset-name random \
  --random-input-len ${ISL} \
  --random-output-len ${OSL} \
  --random-range-ratio ${RANGE_RATIO}   # default 0, i.e. exact ISL
  --num-prompts ${P} \
  --max-concurrency ${C} \
  --ignore-eos                          # force exact OSL
```

Both sides of the comparison (GPU and Trn2) call this same client with the same
flags so prompt construction, sampling, and metric computation are identical.
