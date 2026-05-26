#!/usr/bin/env bash
# Start a Qwen3.6-27B Trn2 vLLM/Neuron server pinned to a single 4-core
# logical device. Reuses contrib/.../vllm/start_vllm_server.sh.
#
# Required env vars:
#   REPO              Path to the neuronx-distributed-inference checkout
#   SERVER_VENV       Path to neuronx_venv_pytorch_inference_vllm_* venv
#   MODEL_PATH        HF safetensors directory for Qwen3.6-27B
#   COMPILED_PATH     Compiled artifact directory (matches the compile step)
#
# Optional env vars (defaults shown):
#   PORT=8100
#   MAX_MODEL_LEN=9216
#   SEQ_LEN=9216
#   CTE_BUCKETS_CSV=1024,2048,4096,8192
#   TP_DEGREE=4
#   LOGICAL_NC_CONFIG=2
#   MAX_NUM_SEQS=1
#   NEURON_RT_VISIBLE_CORES=0-3
set -euo pipefail

: "${REPO:?REPO required}"
: "${SERVER_VENV:?SERVER_VENV required}"
: "${MODEL_PATH:?MODEL_PATH required}"
: "${COMPILED_PATH:?COMPILED_PATH required}"

PORT="${PORT:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
SEQ_LEN="${SEQ_LEN:-9216}"
CTE_BUCKETS_CSV="${CTE_BUCKETS_CSV:-1024,2048,4096,8192}"
TP_DEGREE="${TP_DEGREE:-4}"
LOGICAL_NC_CONFIG="${LOGICAL_NC_CONFIG:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export NEURON_RT_VISIBLE_CORES="${NEURON_RT_VISIBLE_CORES:-0-3}"

# shellcheck disable=SC1091
source "${SERVER_VENV}/bin/activate"
cd "${REPO}"

exec contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path "${MODEL_PATH}" \
  --compiled-artifacts "${COMPILED_PATH}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --seq-len "${SEQ_LEN}" \
  --context-encoding-buckets "${CTE_BUCKETS_CSV}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --tensor-parallel-size "${TP_DEGREE}" \
  --logical-nc-config "${LOGICAL_NC_CONFIG}" \
  --port "${PORT}"
