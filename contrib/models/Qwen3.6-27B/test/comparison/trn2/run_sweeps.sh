#!/usr/bin/env bash
# Drive prefill + decode sweeps against a running Trn2 vLLM/Neuron server using
# the standard `vllm bench serve` client. Wire-compatible with the GPU side.
#
# Required env vars:
#   SERVED_NAME       Model id served by the Neuron vLLM server (HF model dir
#                     name, since `vllm serve <model_path>` uses the path as
#                     the served-model-name unless overridden).
#   TAG               Sweep tag, e.g. "trn2_fp8" / "trn2_bf16".
#   OUTDIR            Directory for raw JSON + log output.
#
# Optional env vars:
#   HOST=127.0.0.1 PORT=8000 RANGE_RATIO=0
#   PREFILL_ISLS="8191 16383 32767 65535 131071 262143"
#   DECODE_CONC="1 2 4 8 16 32 64"
#   PREFILL_TTFT_LIMIT_S=120  DECODE_TTFT_LIMIT_S=10
#
# RANGE_RATIO defaults to 0 (exact ISL); set to >0 only if your largest
# compiled CTE bucket comfortably covers ISL*(1+RANGE_RATIO) — otherwise
# prompts will overshoot the largest bucket and prefill will fail.
#
# PREFILL_ISLS notes: NxDI's first_fit bucket selection uses strict less-than
# (`required_len < bucket`) — see neuronx_distributed_inference/models/
# model_wrapper.py:_get_seq_bucket. So ISL exactly equal to a compiled bucket
# boundary falls through to the next-larger bucket. We default to `bucket - 1`
# so each ISL actually exercises the bucket of the same name. The largest
# value (262143) follows the same pattern, matching the upstream default.
#
# Auto-stop conditions match the GPU harness:
#   * prefill stops when median TTFT > PREFILL_TTFT_LIMIT_S;
#   * decode stops when max TTFT > DECODE_TTFT_LIMIT_S.
set -uo pipefail

: "${SERVED_NAME:?SERVED_NAME required}"
: "${TAG:?TAG required}"
: "${OUTDIR:?OUTDIR required}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RANGE_RATIO="${RANGE_RATIO:-0}"
PREFILL_ISLS="${PREFILL_ISLS:-8191 16383 32767 65535 131071 262143}"
DECODE_CONC="${DECODE_CONC:-1 2 4 8 16 32 64}"
PREFILL_TTFT_LIMIT_S="${PREFILL_TTFT_LIMIT_S:-120}"
DECODE_TTFT_LIMIT_S="${DECODE_TTFT_LIMIT_S:-10}"

mkdir -p "${OUTDIR}"
PREFILL_LOG="${OUTDIR}/${TAG}_prefill.log"
DECODE_LOG="${OUTDIR}/${TAG}_decode.log"
: > "${PREFILL_LOG}"
: > "${DECODE_LOG}"

json_get() {
    python3 - "$1" "$2" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        data = json.load(f)
    val = data.get(key)
    if val is None:
        sys.exit(0)
    print(val)
except Exception:
    sys.exit(0)
PY
}

cmp_gt() {
    awk -v a="$1" -v b="$2" 'BEGIN{ if (a > b) print "yes"; else print "no" }'
}

run_bench() {
    local result_file="$1" isl="$2" osl="$3" n="$4" conc="$5"
    vllm bench serve \
        --host "${HOST}" \
        --port "${PORT}" \
        --model "${SERVED_NAME}" \
        --served-model-name "${SERVED_NAME}" \
        --dataset-name random \
        --random-input-len "${isl}" \
        --random-output-len "${osl}" \
        --random-range-ratio "${RANGE_RATIO}" \
        --num-prompts "${n}" \
        --max-concurrency "${conc}" \
        --ignore-eos \
        --save-result \
        --result-dir "${OUTDIR}" \
        --result-filename "${result_file}"
}

echo "=== ${TAG}: PREFILL SWEEP (osl=1, conc=1) ===" | tee -a "${PREFILL_LOG}"
for isl in ${PREFILL_ISLS}; do
    rf="${TAG}_prefill_isl${isl}.json"
    echo "[prefill] ISL=${isl}" | tee -a "${PREFILL_LOG}"
    run_bench "${rf}" "${isl}" 1 1 1 2>&1 | tee -a "${PREFILL_LOG}"
    rp="${OUTDIR}/${rf}"
    ttft_ms="$(json_get "${rp}" median_ttft_ms)"
    [ -z "${ttft_ms}" ] && ttft_ms="$(json_get "${rp}" mean_ttft_ms)"
    if [ -n "${ttft_ms}" ]; then
        ttft_s="$(awk -v m="${ttft_ms}" 'BEGIN{ printf "%.4f", m/1000 }')"
        echo "SUMMARY ${TAG} prefill isl=${isl} ttft_s=${ttft_s}" | tee -a "${PREFILL_LOG}"
        if [ "$(cmp_gt "${ttft_s}" "${PREFILL_TTFT_LIMIT_S}")" = "yes" ]; then
            echo "PREFILL_OVER_${PREFILL_TTFT_LIMIT_S}S_AT_ISL=${isl} (ttft=${ttft_s}) — stopping" \
                | tee -a "${PREFILL_LOG}"
            break
        fi
    else
        echo "WARN: no ttft in ${rp}" | tee -a "${PREFILL_LOG}"
    fi
done
echo "=== ${TAG} :: PREFILL_DONE ===" | tee -a "${PREFILL_LOG}"

echo "=== ${TAG}: DECODE SWEEP (isl=1024, osl=1024) ===" | tee -a "${DECODE_LOG}"
for c in ${DECODE_CONC}; do
    rf="${TAG}_decode_c${c}.json"
    echo "[decode] conc=${c}" | tee -a "${DECODE_LOG}"
    run_bench "${rf}" 1024 1024 "${c}" "${c}" 2>&1 | tee -a "${DECODE_LOG}"
    rp="${OUTDIR}/${rf}"
    ttft_ms="$(json_get "${rp}" max_ttft_ms)"
    [ -z "${ttft_ms}" ] && ttft_ms="$(json_get "${rp}" p99_ttft_ms)"
    if [ -n "${ttft_ms}" ]; then
        ttft_s="$(awk -v m="${ttft_ms}" 'BEGIN{ printf "%.4f", m/1000 }')"
        echo "SUMMARY ${TAG} decode conc=${c} ttft_max_s=${ttft_s}" | tee -a "${DECODE_LOG}"
        if [ "$(cmp_gt "${ttft_s}" "${DECODE_TTFT_LIMIT_S}")" = "yes" ]; then
            echo "TTFT_OVER_${DECODE_TTFT_LIMIT_S}S_AT_CONC=${c} (ttft_max=${ttft_s}) — stopping" \
                | tee -a "${DECODE_LOG}"
            break
        fi
    else
        echo "WARN: no ttft in ${rp}" | tee -a "${DECODE_LOG}"
    fi
done
echo "=== ${TAG} :: DECODE_DONE ===" | tee -a "${DECODE_LOG}"
