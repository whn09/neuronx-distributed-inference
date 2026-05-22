#!/usr/bin/env bash
# Launch a Qwen3.6-27B vLLM server on a single CUDA GPU.
#
# Required env vars:
#   MODEL_PATH      Path to HF weights (FP8 or BF16).
#   SERVED_NAME     Model id to expose (e.g. Qwen3.6-27B-FP8).
#
# Optional env vars:
#   PORT            Default 8000.
#   GPUS            CUDA_VISIBLE_DEVICES, default 0.
#   TP              Tensor-parallel size, default 1.
#   MAX_MODEL_LEN   Default 262144.
#   GPU_UTIL        gpu-memory-utilization, default 0.92.
#   CUDA_HOME       Default /usr/local/cuda-13.2 (apt cuda-toolkit-13-2).
#   FLASHINFER_DIR  Workspace dir per server, default /tmp/flashinfer_$PORT.
#
# CUDA_HOME and PATH are wired so flashinfer's JIT can find nvcc + ninja.
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH required}"
: "${SERVED_NAME:?SERVED_NAME required}"

export PORT="${PORT:-8000}"
export GPUS="${GPUS:-0}"
export TP="${TP:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
export GPU_UTIL="${GPU_UTIL:-0.92}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
export FLASHINFER_DIR="${FLASHINFER_DIR:-/tmp/flashinfer_${PORT}}"

mkdir -p "${FLASHINFER_DIR}"

export PATH="/opt/pytorch/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_DIR}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

exec vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --mm-encoder-tp-mode data \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --port "${PORT}"
