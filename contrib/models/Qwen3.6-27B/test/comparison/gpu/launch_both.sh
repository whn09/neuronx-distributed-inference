#!/usr/bin/env bash
# Launch FP8 + BF16 vLLM servers on two GPUs (one per server) so the prefill
# and decode sweeps can be driven concurrently.
#
# Required env vars:
#   FP8_MODEL_PATH    HF FP8 weights
#   BF16_MODEL_PATH   HF BF16 weights
#
# Optional:
#   FP8_GPU=0 BF16_GPU=1 FP8_PORT=8000 BF16_PORT=8001 LOG_DIR=/tmp/qwen36_vllm
set -uo pipefail

: "${FP8_MODEL_PATH:?FP8_MODEL_PATH required}"
: "${BF16_MODEL_PATH:?BF16_MODEL_PATH required}"

FP8_GPU="${FP8_GPU:-0}"
BF16_GPU="${BF16_GPU:-1}"
FP8_PORT="${FP8_PORT:-8000}"
BF16_PORT="${BF16_PORT:-8001}"
LOG_DIR="${LOG_DIR:-/tmp/qwen36_vllm}"
mkdir -p "${LOG_DIR}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pkill -f 'vllm serve' 2>/dev/null || true
sleep 5

PORT="${FP8_PORT}" GPUS="${FP8_GPU}" TP=1 \
SERVED_NAME=Qwen3.6-27B-FP8 MODEL_PATH="${FP8_MODEL_PATH}" \
nohup "${HERE}/start_vllm.sh" > "${LOG_DIR}/vllm_fp8.log" 2>&1 &
echo "FP8_PID=$!"
disown

PORT="${BF16_PORT}" GPUS="${BF16_GPU}" TP=1 \
SERVED_NAME=Qwen3.6-27B-BF16 MODEL_PATH="${BF16_MODEL_PATH}" \
nohup "${HERE}/start_vllm.sh" > "${LOG_DIR}/vllm_bf16.log" 2>&1 &
echo "BF16_PID=$!"
disown

echo "Logs: ${LOG_DIR}/vllm_fp8.log  ${LOG_DIR}/vllm_bf16.log"
echo "Wait for 'Application startup complete' on both before running run_sweeps.sh."
