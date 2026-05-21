#!/bin/bash
# LTX-2.3 Unified Compilation Script
#
# Compiles all components needed for LTX-2.3 inference on Trainium2:
#   - Gemma3 12B text encoder (TP=4, parallel_model_trace)
#   - DiT backbone — full-resolution (default) and optionally half-resolution
#     for two-stage mode (HALFRES=1 also compiles the upsample stage)
#   - VAE decoder (TP=4, tiled)
#
# Usage:
#   bash compile.sh                                  # all components, defaults
#   bash compile.sh /path/to/output /path/to/workdir # custom output dirs
#   COMPONENT=encoder bash compile.sh                # only the text encoder
#   COMPONENT=transformer bash compile.sh            # only the DiT
#   COMPONENT=vae bash compile.sh                    # only the VAE decoder
#   HEIGHT=384 WIDTH=512 NUM_FRAMES=25 bash compile.sh
#   HALFRES=1 bash compile.sh                        # also compile half-res DiT
#
# Required env: MODEL_PATH must point to ltx-2.3-22b-distilled.safetensors,
# GEMMA_PATH must point to the Gemma 3 12B HF directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# --- Configuration -----------------------------------------------------------
COMPILED_MODELS_DIR="${1:-/opt/dlami/nvme/compiled_models_ltx23}"
COMPILER_WORKDIR="${2:-/opt/dlami/nvme/compiler_workdir_ltx23}"

# Video settings (must match inference)
HEIGHT="${HEIGHT:-384}"
WIDTH="${WIDTH:-512}"
NUM_FRAMES="${NUM_FRAMES:-25}"
TEXT_SEQ_LEN="${TEXT_SEQ_LEN:-1024}"
TP_DEGREE="${TP_DEGREE:-4}"

# Half-resolution DiT for the two-stage pipeline (192x256, video_seq=192)
HALFRES="${HALFRES:-0}"

# Which components to build (encoder | transformer | vae | all)
COMPONENT="${COMPONENT:-all}"

# Required model paths
MODEL_PATH="${MODEL_PATH:-/opt/dlami/nvme/work/ltx23/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors}"
GEMMA_PATH="${GEMMA_PATH:-/opt/dlami/nvme/work/ltx23/models/gemma-3-12b-it}"

# Latent dims derived from pixel dims (VAE downsamples 32x spatially)
LATENT_H=$((HEIGHT / 32))
LATENT_W=$((WIDTH / 32))

echo "=============================================="
echo "LTX-2.3 Compilation"
echo "=============================================="
echo "Output:         ${COMPILED_MODELS_DIR}"
echo "Workdir:        ${COMPILER_WORKDIR}"
echo "Resolution:     ${HEIGHT}x${WIDTH}  (latent ${LATENT_H}x${LATENT_W})"
echo "Frames:         ${NUM_FRAMES}"
echo "Text seq len:   ${TEXT_SEQ_LEN}"
echo "TP degree:      ${TP_DEGREE}"
echo "Component(s):   ${COMPONENT}"
echo "Half-res DiT:   ${HALFRES}"
echo "=============================================="

mkdir -p "${COMPILED_MODELS_DIR}" "${COMPILER_WORKDIR}"

ENC_DIR="${COMPILED_MODELS_DIR}/gemma3_encoder"
DIT_DIR="${COMPILED_MODELS_DIR}/backbone"
DIT_HR_DIR="${COMPILED_MODELS_DIR}/backbone_halfres"
VAE_DIR="${COMPILED_MODELS_DIR}/vae"

want() {
    [ "${COMPONENT}" = "all" ] || [ "${COMPONENT}" = "$1" ]
}

# --- 1. Gemma3 text encoder --------------------------------------------------
if want encoder; then
    echo
    echo "[encoder] Gemma3 12B (TP=${TP_DEGREE}, seq=${TEXT_SEQ_LEN})"
    if [ -f "${ENC_DIR}/tp_0.pt" ]; then
        echo "  Skipping: ${ENC_DIR}/tp_0.pt already exists"
    else
        NEURON_FUSE_SOFTMAX=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
            python3 "${SCRIPT_DIR}/compile.py" encoder \
                --tp-degree "${TP_DEGREE}" \
                --seq-len "${TEXT_SEQ_LEN}" \
                --compile-dir "${ENC_DIR}"
    fi
fi

# --- 2. DiT backbone (full-res) ----------------------------------------------
if want transformer; then
    echo
    echo "[transformer] DiT backbone full-res (latent ${LATENT_H}x${LATENT_W}, TP=${TP_DEGREE})"
    if [ -f "${DIT_DIR}/tp_0.pt" ] || [ -d "${DIT_DIR}/compiler_workdir" ]; then
        echo "  Skipping: ${DIT_DIR} already populated"
    else
        NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
            torchrun --nproc_per_node="${TP_DEGREE}" "${SCRIPT_DIR}/compile.py" transformer \
                --latent-h "${LATENT_H}" \
                --latent-w "${LATENT_W}" \
                --tp-degree "${TP_DEGREE}" \
                --model-path "${MODEL_PATH}" \
                --compile-dir "${DIT_DIR}"
    fi
fi

# --- 2b. DiT backbone (half-res) for two-stage pipeline ----------------------
if want transformer && [ "${HALFRES}" = "1" ]; then
    echo
    echo "[transformer] DiT backbone half-res (192x256, TP=${TP_DEGREE})"
    if [ -f "${DIT_HR_DIR}/tp_0.pt" ] || [ -d "${DIT_HR_DIR}/compiler_workdir" ]; then
        echo "  Skipping: ${DIT_HR_DIR} already populated"
    else
        NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
            torchrun --nproc_per_node="${TP_DEGREE}" "${SCRIPT_DIR}/compile.py" transformer \
                --halfres \
                --tp-degree "${TP_DEGREE}" \
                --model-path "${MODEL_PATH}" \
                --compile-dir "${DIT_HR_DIR}"
    fi
fi

# --- 3. VAE decoder (tiled) --------------------------------------------------
if want vae; then
    echo
    echo "[vae] TP-sharded video decoder (tile ${HEIGHT}x${WIDTH}, frames=${NUM_FRAMES})"
    if [ -f "${VAE_DIR}/tp_0.pt" ] || [ -d "${VAE_DIR}/compiler_workdir" ]; then
        echo "  Skipping: ${VAE_DIR} already populated"
    else
        NEURON_RT_VISIBLE_CORES=0-$((TP_DEGREE - 1)) \
            python3 "${SCRIPT_DIR}/compile.py" vae \
                --height "${HEIGHT}" \
                --width "${WIDTH}" \
                --num-frames "${NUM_FRAMES}" \
                --tp-degree "${TP_DEGREE}" \
                --model-path "${MODEL_PATH}" \
                --compile-dir "${VAE_DIR}" \
                --compiler-workdir "${COMPILER_WORKDIR}/vae"
    fi
fi

echo
echo "=============================================="
echo "Compilation complete: ${COMPILED_MODELS_DIR}"
echo "=============================================="
