#!/usr/bin/env bash
# End-to-end driver: compile -> launch server -> wait ready -> sweep -> stop.
# Suitable for unattended overnight runs on a Trn2 host.
#
# Required env vars (forwarded to run_compile_b1.sh + start_vllm_b1.sh +
# run_sweeps.sh):
#   PRECISION         "bf16" or "fp8"
#   REPO              Path to the neuronx-distributed-inference checkout
#   COMPILER_VENV     nxd_inference compiler venv
#   SERVER_VENV       vllm_0_16 server venv
#   MODEL_PATH        HF model directory
#   COMPILED_PATH     Output artifact directory
#   QUANTIZED_CKPT_PATH  (FP8 only)
#
# Optional env vars (defaults match the proven b=1 path):
#   PORT=8100  TAG=trn2_${PRECISION}_b1
#   OUTDIR=${REPO_PARENT}/sweeps/${TAG}
#   LOG_DIR=${OUTDIR}            # also receives compile/server logs
#   PREFILL_ISLS="1023 2047 4095 8191"
#   DECODE_CONC="1"
#   PREFILL_TTFT_LIMIT_S=180  DECODE_TTFT_LIMIT_S=30
#   READY_TIMEOUT_S=600
set -euo pipefail

: "${PRECISION:?PRECISION required (bf16|fp8)}"
: "${REPO:?REPO required}"
: "${COMPILER_VENV:?COMPILER_VENV required}"
: "${SERVER_VENV:?SERVER_VENV required}"
: "${MODEL_PATH:?MODEL_PATH required}"
: "${COMPILED_PATH:?COMPILED_PATH required}"

TAG="${TAG:-trn2_${PRECISION}_b1}"
PORT="${PORT:-8100}"
OUTDIR="${OUTDIR:-/tmp/${TAG}_sweeps}"
LOG_DIR="${LOG_DIR:-${OUTDIR}}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-600}"
PREFILL_ISLS="${PREFILL_ISLS:-1023 2047 4095 8191}"
DECODE_CONC="${DECODE_CONC:-1}"
PREFILL_TTFT_LIMIT_S="${PREFILL_TTFT_LIMIT_S:-180}"
DECODE_TTFT_LIMIT_S="${DECODE_TTFT_LIMIT_S:-30}"

mkdir -p "${OUTDIR}" "${LOG_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMPILE_LOG="${LOG_DIR}/${TAG}_compile.log"
SERVER_LOG="${LOG_DIR}/${TAG}_server.log"
SWEEP_LOG="${LOG_DIR}/${TAG}_sweep.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    sleep 5
  fi
  pkill -TERM -f start_vllm_server.sh 2>/dev/null || true
  pkill -TERM -f serve_qwen36 2>/dev/null || true
}
trap cleanup EXIT

echo "=== [${TAG}] STEP 1: COMPILE ===" | tee -a "${COMPILE_LOG}"
PRECISION="${PRECISION}" REPO="${REPO}" COMPILER_VENV="${COMPILER_VENV}" \
  MODEL_PATH="${MODEL_PATH}" COMPILED_PATH="${COMPILED_PATH}" \
  QUANTIZED_CKPT_PATH="${QUANTIZED_CKPT_PATH:-}" \
  bash "${SCRIPT_DIR}/run_compile_b1.sh" >> "${COMPILE_LOG}" 2>&1
echo "=== [${TAG}] COMPILE_DONE ==="

echo "=== [${TAG}] STEP 2: START SERVER ==="
PRECISION="${PRECISION}" REPO="${REPO}" SERVER_VENV="${SERVER_VENV}" \
  MODEL_PATH="${MODEL_PATH}" COMPILED_PATH="${COMPILED_PATH}" \
  PORT="${PORT}" \
  nohup bash "${SCRIPT_DIR}/start_vllm_b1.sh" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "server pid=${SERVER_PID}"

echo "=== [${TAG}] STEP 3: WAIT READY (timeout=${READY_TIMEOUT_S}s) ==="
deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
ready=0
while (( $(date +%s) < deadline )); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1; break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: server process exited before becoming ready"
    tail -80 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 5
done
if [[ "${ready}" -ne 1 ]]; then
  echo "ERROR: server never became ready within ${READY_TIMEOUT_S}s"
  tail -80 "${SERVER_LOG}" || true
  exit 1
fi
echo "server ready"

echo "=== [${TAG}] STEP 4: SWEEP ==="
# `vllm bench serve` lives in SERVER_VENV; activate it before running the
# sweep so the client binary is on PATH. Use a subshell so the activation
# does not leak into later cleanup steps.
(
  # shellcheck disable=SC1091
  source "${SERVER_VENV}/bin/activate"
  TAG="${TAG}" HOST=127.0.0.1 PORT="${PORT}" \
    SERVED_NAME="${MODEL_PATH}" \
    OUTDIR="${OUTDIR}" \
    PREFILL_ISLS="${PREFILL_ISLS}" \
    DECODE_CONC="${DECODE_CONC}" \
    PREFILL_TTFT_LIMIT_S="${PREFILL_TTFT_LIMIT_S}" \
    DECODE_TTFT_LIMIT_S="${DECODE_TTFT_LIMIT_S}" \
    bash "${SCRIPT_DIR}/run_sweeps.sh"
) >> "${SWEEP_LOG}" 2>&1
echo "=== [${TAG}] SWEEP_DONE ==="

echo "=== [${TAG}] ALL_DONE ==="
date
ls -la "${OUTDIR}"
