#!/usr/bin/env bash
# Drive prefill + decode sweeps against a running vLLM server using the
# standard `vllm bench serve` client.
#
# Required env vars:
#   HOST              Server host, default 127.0.0.1.
#   PORT              Server port, default 8000.
#   SERVED_NAME       Model id served by the server (must match start_vllm.sh).
#   TAG               Sweep tag, used in result filenames (e.g. "fp8" / "bf16").
#   OUTDIR            Directory for raw JSON + log output.
#
# Optional env vars:
#   RANGE_RATIO       --random-range-ratio, default 0 (exact ISL). Raise only
#                     if your serving config can absorb ISL*(1+RANGE_RATIO)
#                     without overshooting compiled buckets.
#   PREFILL_ISLS      Space-separated, default "8192 16384 32768 65536 131072 262143".
#   DECODE_CONC       Space-separated, default "1 2 4 8 16 32 64".
#   PREFILL_TTFT_LIMIT_S   Stop prefill sweep when median TTFT exceeds this (default 120).
#   DECODE_TTFT_LIMIT_S    Stop decode sweep when max TTFT exceeds this (default 10).
set -uo pipefail

: "${SERVED_NAME:?SERVED_NAME required}"
: "${TAG:?TAG required}"
: "${OUTDIR:?OUTDIR required}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RANGE_RATIO="${RANGE_RATIO:-0}"
PREFILL_ISLS="${PREFILL_ISLS:-8192 16384 32768 65536 131072 262143}"
DECODE_CONC="${DECODE_CONC:-1 2 4 8 16 32 64}"
PREFILL_TTFT_LIMIT_S="${PREFILL_TTFT_LIMIT_S:-120}"
DECODE_TTFT_LIMIT_S="${DECODE_TTFT_LIMIT_S:-10}"

mkdir -p "${OUTDIR}"
PREFILL_LOG="${OUTDIR}/${TAG}_prefill.log"
DECODE_LOG="${OUTDIR}/${TAG}_decode.log"
: > "${PREFILL_LOG}"
: > "${DECODE_LOG}"

# Read a numeric field out of a vllm-bench-serve JSON result.
# Tolerant of schema drift: returns "" when the key is missing.
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

run_bench() {
    # $1 result_filename, $2 isl, $3 osl, $4 num_prompts, $5 max_concurrency
    local result_file="$1"
    local isl="$2"
    local osl="$3"
    local n="$4"
    local conc="$5"

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

cmp_gt() {  # $1 > $2 ?  echo "yes" / "no" without bc
    awk -v a="$1" -v b="$2" 'BEGIN{ if (a > b) print "yes"; else print "no" }'
}

echo "=== ${TAG}: PREFILL SWEEP (osl=1, conc=1) ===" | tee -a "${PREFILL_LOG}"
for isl in ${PREFILL_ISLS}; do
    rf="${TAG}_prefill_isl${isl}.json"
    echo "[prefill] ISL=${isl}" | tee -a "${PREFILL_LOG}"
    run_bench "${rf}" "${isl}" 1 1 1 2>&1 | tee -a "${PREFILL_LOG}"
    rp="${OUTDIR}/${rf}"
    ttft_ms="$(json_get "${rp}" median_ttft_ms)"
    if [ -z "${ttft_ms}" ]; then
        ttft_ms="$(json_get "${rp}" mean_ttft_ms)"
    fi
    if [ -n "${ttft_ms}" ]; then
        ttft_s="$(awk -v m="${ttft_ms}" 'BEGIN{ printf "%.4f", m/1000 }')"
        echo "SUMMARY ${TAG} prefill isl=${isl} ttft_s=${ttft_s}" | tee -a "${PREFILL_LOG}"
        if [ "$(cmp_gt "${ttft_s}" "${PREFILL_TTFT_LIMIT_S}")" = "yes" ]; then
            echo "PREFILL_OVER_${PREFILL_TTFT_LIMIT_S}S_AT_ISL=${isl} (ttft=${ttft_s}) — stopping prefill sweep" \
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
    if [ -z "${ttft_ms}" ]; then
        ttft_ms="$(json_get "${rp}" p99_ttft_ms)"
    fi
    if [ -n "${ttft_ms}" ]; then
        ttft_s="$(awk -v m="${ttft_ms}" 'BEGIN{ printf "%.4f", m/1000 }')"
        echo "SUMMARY ${TAG} decode conc=${c} ttft_max_s=${ttft_s}" | tee -a "${DECODE_LOG}"
        if [ "$(cmp_gt "${ttft_s}" "${DECODE_TTFT_LIMIT_S}")" = "yes" ]; then
            echo "TTFT_OVER_${DECODE_TTFT_LIMIT_S}S_AT_CONC=${c} (ttft_max=${ttft_s}) — stopping decode sweep" \
                | tee -a "${DECODE_LOG}"
            break
        fi
    else
        echo "WARN: no ttft in ${rp}" | tee -a "${DECODE_LOG}"
    fi
done
echo "=== ${TAG} :: DECODE_DONE ===" | tee -a "${DECODE_LOG}"
