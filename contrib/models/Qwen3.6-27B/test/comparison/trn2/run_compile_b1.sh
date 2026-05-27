#!/usr/bin/env bash
# Compile a Qwen3.6-27B Trn2 artifact (BF16 or FP8-MLP-only) at batch_size=1
# with multi-bucket context-encoding. Mirrors the proven b=1 path.
#
# Required env vars:
#   PRECISION         "bf16" or "fp8"  (fp8 = MLP weight-only e4m3, BF16 elsewhere)
#   REPO              Path to the neuronx-distributed-inference checkout
#   COMPILER_VENV     Path to neuronx_venv_pytorch_*_nxd_inference venv
#   MODEL_PATH        HF safetensors directory for Qwen3.6-27B
#   COMPILED_PATH     Output artifact directory
#
# Optional env vars (defaults shown):
#   SEQ_LEN=9216
#   CTE_BUCKETS_CSV=1024,2048,4096,8192
#   TP_DEGREE=4
#   LOGICAL_NC_CONFIG=2
#   WORK_ROOT=/tmp/qwen36_compile_${PRECISION}_b1   (BASE_COMPILE_WORK_DIR)
#   CACHE_DIR=${WORK_ROOT}_cache                    (NEURON_COMPILE_CACHE_URL)
#   TMPDIR_OVERRIDE=${WORK_ROOT}_tmp                (TMPDIR)
#   QUANTIZED_CKPT_PATH                             (FP8 only; required for FP8)
#   FP8_TIER=weight_only_mlp                        (FP8 only; weight_only_mlp |
#                                                    dynamic_mlp | dynamic_mlp_attn)
set -euo pipefail

: "${PRECISION:?PRECISION must be bf16 or fp8}"
: "${REPO:?REPO required}"
: "${COMPILER_VENV:?COMPILER_VENV required}"
: "${MODEL_PATH:?MODEL_PATH required}"
: "${COMPILED_PATH:?COMPILED_PATH required}"

SEQ_LEN="${SEQ_LEN:-9216}"
CTE_BUCKETS_CSV="${CTE_BUCKETS_CSV:-1024,2048,4096,8192}"
TP_DEGREE="${TP_DEGREE:-4}"
LOGICAL_NC_CONFIG="${LOGICAL_NC_CONFIG:-2}"
WORK_ROOT="${WORK_ROOT:-/tmp/qwen36_compile_${PRECISION}_b1}"
CACHE_DIR="${CACHE_DIR:-${WORK_ROOT}_cache}"
TMPDIR_OVERRIDE="${TMPDIR_OVERRIDE:-${WORK_ROOT}_tmp}"

mkdir -p "${WORK_ROOT}" "${CACHE_DIR}" "${TMPDIR_OVERRIDE}"

# shellcheck disable=SC1091
source "${COMPILER_VENV}/bin/activate"
cd "${REPO}"

export BASE_COMPILE_WORK_DIR="${WORK_ROOT}"
export NEURON_COMPILE_CACHE_URL="${CACHE_DIR}"
export TMPDIR="${TMPDIR_OVERRIDE}"

case "${PRECISION}" in
  bf16)
    exec python3 contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_bf16.py \
      --model-path "${MODEL_PATH}" \
      --compiled-path "${COMPILED_PATH}" \
      --seq-len "${SEQ_LEN}" --batch-size 1 \
      --context-encoding-buckets "${CTE_BUCKETS_CSV}" \
      --tp-degree "${TP_DEGREE}" --logical-nc-config "${LOGICAL_NC_CONFIG}"
    ;;
  fp8)
    : "${QUANTIZED_CKPT_PATH:?QUANTIZED_CKPT_PATH required for PRECISION=fp8}"
    FP8_TIER="${FP8_TIER:-weight_only_mlp}"
    exec python3 contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py \
      --model-path "${MODEL_PATH}" \
      --compiled-path "${COMPILED_PATH}" \
      --quantized-checkpoints-path "${QUANTIZED_CKPT_PATH}" \
      --quantization-tier "${FP8_TIER}" \
      --seq-len "${SEQ_LEN}" --batch-size 1 \
      --context-encoding-buckets "${CTE_BUCKETS_CSV}" \
      --tp-degree "${TP_DEGREE}" --logical-nc-config "${LOGICAL_NC_CONFIG}"
    ;;
  *)
    echo "ERROR: PRECISION must be bf16 or fp8 (got: ${PRECISION})" >&2
    exit 2
    ;;
esac
