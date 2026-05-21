# Contrib Model: LTX-2.3

NeuronX adaptation of [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) for AWS Trainium 2 inference. LTX-2.3 is a 22B-parameter DiT (diffusion transformer) that generates synchronized video + audio from a text prompt, with optional image-to-video conditioning. Runs on top of the native [`ltx-core`](https://github.com/Lightricks/LTX-2) framework — not Diffusers.

## Model Information

- **HuggingFace ID:** `Lightricks/LTX-2.3` (model file: `ltx-2.3-22b-distilled.safetensors`, 8-step distilled, 43 GB)
- **Model Type:** DiT for joint audio-video generation (text-to-video and image-to-video)
- **Architecture:** 48 transformer blocks, 32 heads, 4096 video dim, 2048 audio dim, bidirectional A/V cross-attention, gated attention, QK-RMSNorm, split RoPE, flow matching
- **License:** See HuggingFace model card
- **Framework:** Native `ltx-core` (not Diffusers)

## Architecture Details

| Component | Model | Parameters | Neuron Parallelism |
|-----------|-------|-----------:|-------------------|
| Text Encoder | Gemma 3 12B | ~12 B | TP=4, parallel_model_trace |
| Transformer | DiT (audio+video, 48 blocks) | ~22 B | TP=4 |
| Video VAE Decoder | LTX 3D Conv decoder | ~330 M | TP=4 tiled (1×16 latent, H≤64 SRAM limit) |
| Audio VAE + Vocoder | Conv1d / HiFi-GAN | ~50 M | CPU |
| Spatial / Temporal upscalers (optional) | Conv3D | 498 M / 131 M | CPU |

Key parameters:

- **Denoising steps**: 8 (distilled checkpoint), flow-matching Euler integrator, distilled sigma schedule.
- **Default resolution**: 384×512 / 25 frames (single-stage). Two-stage mode generates at 192×256 / 8 steps then refines at 384×512 / 3 steps after a CPU 2× spatial upsample.
- **Encoder–DiT scheduling**: Gemma 3 (TP=4) and DiT (TP=4) share the same 4 NeuronCores and run sequentially — Gemma 3 loads, encodes, unloads, then DiT loads and runs the denoise loop. Loading both at once thrashes (144s+ for the first denoise step instead of 0.3s).

## Performance

Numbers from a single trn2.3xlarge run, SDK 2.28, 384×512 / 25 frames / 8 steps, single-stage T2V with Neuron-compiled Gemma 3:

| Phase | Time | Notes |
|---|---:|---|
| `text_encode` (Neuron Gemma 3, warm) | ~1.3 s | Tokenize + forward + post-process; cold first call adds ~16 s NEFF warmup |
| `transformer_warmup` (DiT, 1 call) | 138.5 s | First NEFF load onto cores |
| `transformer_forward` step 1 (cold) | 174.6 s | Neuron device initialization |
| `transformer_forward` steps 2–8 (warm) | **0.3 s / step** | Steady-state |
| Total denoising (8 steps) | 176.9 s | Dominated by step-1 cold start |
| `video_decode` (CPU VAE) | 7.2 s | 25 frames @ 384×512 |
| `video_decode` (Neuron tiled VAE, TP=4) | 23.5 s @ 1024×1536 / 121 f | 33 tiles × 610 ms; 3.3× faster than CPU VAE at that resolution |
| `audio_decode` (CPU) | 2.5 s | Stereo 48 kHz WAV |

Per-step warm DiT detail (384×512, with the AdaLN dedup + step-invariant caching CPU optimizations enabled):

| Component | Time | % of step |
|---|---:|---:|
| CPU preprocess | 33.1 ms | 11.9 % |
| Neuron backbone forward | 244.1 ms | 87.4 % |
| Euler step | 2.1 ms | 0.7 % |
| **Total per step** | **279.3 ms** | 100 % |

Two CPU preprocessing optimizations close the gap to the Neuron forward:

1. **AdaLN deduplication** — when all tokens share the same sigma (T2V), the AdaLN MLP is computed once instead of per-token (768 tokens). Saves ~47 ms / step.
2. **Step-invariant caching** — RoPE embeddings, context projection, and attention masks are constant across denoising steps; computed once on step 1 and reused. Saves ~57 ms / step (context projection alone is ~55 ms).

Overall per-step improvement vs unoptimized baseline: 330 ms → 279 ms (15.4 % reduction).

### Phase-timer output

Every `generate_ltx23.py` run emits a `=== Phase breakdown ===` summary at exit, populated by `PhaseTimer` (mirrors the helper used in `contrib/models/Wan2.2-TI2V-5B/src/run_wan2.2_ti2v.py`). Sample:

```
=== Phase breakdown (E2E) ===
  text_encode           1.30s  (  0.7%)  calls=  1  avg= 1304.2ms
  transformer_warmup  138.50s  ( 70.3%)  calls=  1  avg=138502.6ms
  transformer_forward  37.50s  ( 19.0%)  calls=  8  avg= 4687.8ms
  video_decode          7.20s  (  3.7%)  calls=  1  avg= 7195.4ms
  audio_decode          2.50s  (  1.3%)  calls=  1  avg= 2501.3ms
  (unaccounted)         9.97s  (  5.0%)
  total               197.00s
```

Phases are recorded for both single-stage and two-stage paths (and on the Phase-1-only `--save-s1-latent` early-return path).

### Resolution sweep

A multi-resolution wall-clock sweep (512×384, 720×480, 1280×704 at 25 / 121 frames) is **TODO**. Each row needs a fresh DiT + VAE compile (~30 min) plus an 8-step inference (~3 min cold), so the table will be filled in by a follow-up PR.

## Prerequisites

- **Instance:** trn2.3xlarge (4 NeuronCores) for default 384×512 / 25 f. trn2.48xlarge (32 NeuronCores) is required for the trn2.48xlarge two-phase pipeline (TP=4 → TP=16).
- **Virtual env:** `/opt/aws_neuronx_venv_pytorch_2_9_nxd_inference` (PyTorch 2.9, neuronx-cc ≥ 2.22, neuronx-distributed ≥ 0.16).
- **NVMe:** Mount RAID at `/opt/dlami/nvme/` (the master compile.sh assumes outputs land under that path).
- **System packages:** `sudo apt install -y ffmpeg` (the DLAMI does not bundle it).

## Usage

### 1. Setup

```bash
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core

# torchaudio: install the CPU variant explicitly (DLAMI's torchaudio depends
# on libcudart.so.13 and won't load on Neuron). Required on SDK ≥ 2.27.
pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torchaudio==2.9.1+cpu

sudo apt install -y ffmpeg
```

### 2. Download Model

```bash
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-22b-distilled.safetensors \
  --local-dir /opt/dlami/nvme/models/LTX-2.3/

huggingface-cli download google/gemma-3-12b-it \
  --local-dir /opt/dlami/nvme/models/gemma-3-12b-it

# Optional: upscalers for --upscale (spatial x2 + temporal x2)
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --local-dir /opt/dlami/nvme/models/LTX-2.3/upscalers/
huggingface-cli download Lightricks/LTX-2.3 ltx-2.3-temporal-upscaler-x2-1.0.safetensors \
  --local-dir /opt/dlami/nvme/models/LTX-2.3/upscalers/
```

### 3. Compile All Components

```bash
# Defaults: 384x512, 25 frames, TP=4, all components (encoder + DiT + VAE)
MODEL_PATH=/opt/dlami/nvme/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
GEMMA_PATH=/opt/dlami/nvme/models/gemma-3-12b-it \
  bash src/compile.sh

# Custom resolution
HEIGHT=480 WIDTH=720 NUM_FRAMES=121 \
MODEL_PATH=... GEMMA_PATH=... \
  bash src/compile.sh

# Build only one component (re-runs are idempotent — already-populated dirs are skipped)
COMPONENT=encoder bash src/compile.sh

# Also compile the half-res DiT for two-stage mode
HALFRES=1 bash src/compile.sh

# Custom output directory
bash src/compile.sh /path/to/output /path/to/compiler_workdir
```

`compile.sh` wraps the existing `compile.py {encoder, transformer, vae}` subcommands so every artifact needed by `generate_ltx23.py` is produced from one entry point. Total compile time: ~5 min for the encoder, ~1 min for the DiT, ~10 min for the VAE.

### 4. Pre-shard backbone weights (one-time)

Pre-sharding avoids re-loading the 41 GB safetensors at every generation:

```bash
python3 src/shard_weights.py backbone \
  --model-path /opt/dlami/nvme/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
  --output-dir /opt/dlami/nvme/models/LTX-2.3/backbone_sharded

python3 src/shard_weights.py encoder \
  --gemma-path /opt/dlami/nvme/models/gemma-3-12b-it \
  --output-dir /opt/dlami/nvme/models/gemma-3-12b-it_sharded
```

### 5. Run Inference

```bash
# Text-to-Video (T2V) with Neuron-compiled Gemma 3 (recommended path)
python3 src/generate_ltx23.py \
  --neuron-gemma \
  --model-path /opt/dlami/nvme/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
  --gemma-path /opt/dlami/nvme/models/gemma-3-12b-it \
  --gemma-compiled-dir /opt/dlami/nvme/compiled_models_ltx23/gemma3_encoder \
  --gemma-sharded-dir /opt/dlami/nvme/models/gemma-3-12b-it_sharded \
  --backbone-sharded-dir /opt/dlami/nvme/models/LTX-2.3/backbone_sharded \
  --compile-dir /opt/dlami/nvme/compiled_models_ltx23/backbone \
  --prompt "A golden retriever puppy runs across a sunny green meadow" \
  --output-dir ./out

# Image-to-Video (I2V) — same compiled backbone, just add --image
python3 src/generate_ltx23.py \
  --neuron-gemma \
  --model-path /opt/dlami/nvme/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
  --gemma-path /opt/dlami/nvme/models/gemma-3-12b-it \
  --gemma-compiled-dir /opt/dlami/nvme/compiled_models_ltx23/gemma3_encoder \
  --gemma-sharded-dir /opt/dlami/nvme/models/gemma-3-12b-it_sharded \
  --backbone-sharded-dir /opt/dlami/nvme/models/LTX-2.3/backbone_sharded \
  --compile-dir /opt/dlami/nvme/compiled_models_ltx23/backbone \
  --prompt "The woman turns and smiles warmly at the camera" \
  --image /path/to/photo.png \
  --output-dir ./out

# Two-stage generation (requires HALFRES=1 compile)
python3 src/generate_ltx23.py --two-stage \
  --halfres-compiled-dir /opt/dlami/nvme/compiled_models_ltx23/backbone_halfres \
  --spatial-upscaler-path /opt/dlami/nvme/models/LTX-2.3/upscalers/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  ... (same args as above)

# Quick smoke test with random embeddings (skips Gemma 3 entirely)
python3 src/generate_ltx23.py --no-text-encoder
```

Output: PNG frames + `output.mp4` (via ffmpeg) + `output.wav` + `latents.pt` in `--output-dir`.

## Compatibility Matrix

| Instance / SDK | 2.24 | 2.27 | 2.28 | 2.29 |
|---|:-:|:-:|:-:|:-:|
| trn2.3xlarge (TP=4) | E2E tested | — | Validated | Validated |
| trn2.48xlarge (TP=4 / TP=16) | — | Validated | — | — |

Notes:
- **SDK 2.24 (E2E test, this branch)**: pipeline runs end-to-end at 384×512 / 25 frames; warm step latency ~0.3 s matches earlier SDK validations.
- **SDK 2.29**: torchaudio must be reinstalled as the CPU variant (see step 1). `decode_video` moved to `VideoDecoder.decode_video()` in ltx-core 1.1; the runner handles both the old free function and the new method.

## Testing

```bash
cd contrib/models/LTX-2.3

MODEL_PATH=/opt/dlami/nvme/models/LTX-2.3/ltx-2.3-22b-distilled.safetensors \
COMPILED_MODEL_PATH=/opt/dlami/nvme/compiled_models_ltx23/backbone \
  pytest test/integration/test_model.py -v -s
```

Tests check that the model loads, a forward pass produces non-NaN output, cosine similarity ≥ 0.999 vs CPU reference, and per-step warm latency holds.

Accuracy validation (single forward pass at sigma = 1.0, noise input):

| Component | Cosine similarity | vs CPU reference |
|---|---:|---|
| Video forward pass | 0.999947 | unsharded BF16, native ltx-core |
| Audio forward pass | 0.999867 | same |
| 8-step denoised latent (real text) | 0.972 | normal BF16 TP accumulation drift |

## Key Implementation Notes

1. **Sequential Gemma 3 / DiT execution**: both compile for TP=4 onto the same 4 NeuronCores. The runner explicitly unloads Gemma 3 (NRT resource cleanup) before loading the DiT — co-residency thrashes (144 s+ for the first denoise step instead of 0.3 s).
2. **AdaLN deduplication (T2V)**: when every token shares the same sigma, the AdaLN MLP is computed once and broadcast, not per-token. Saves ~47 ms / step.
3. **Step-invariant caching**: RoPE, context projection, and additive attention masks are constant across the 8 denoising steps; computed once on step 1, reused for steps 2–8 (~57 ms / step).
4. **Tiled VAE decode (TP=4)**: the LTX-2.3 video decoder hits the Neuron SBUF limit at H>64 latent. The TP-sharded decoder is compiled at 1×16 latent (128×512 pixels post-VAE) and tiled with overlap blending (overlap_h=1 latent) for arbitrary output resolutions. 3.3× faster than CPU VAE at 1024×1536.
5. **DistributedRMSNorm bypass**: QK-norm uses the local-only RMSNorm path (compiler bug with the global all-reduce flavor) — same workaround as Wan2.2-TI2V.
6. **Distilled sigma schedule**: the 8-step distilled checkpoint requires the exact sigma list from `ltx_pipelines/utils/constants.py` (`DISTILLED_SIGMA_VALUES`). The runner asserts `len(sigmas) == num_steps + 1` so a wrong schedule fails fast.
7. **PhaseTimer output**: see the “Phase-timer output” section above. The single sample per phase makes regressions in any boundary (text encode, DiT, VAE decode) jump out without hand-grepping the log.

## Known Issues

- **Cold start latency**: DiT warmup ~139 s + step-1 cold ~175 s on a fresh instance (Neuron device init). Steps 2-N are ~0.3 s. Gemma 3 NEFF rehydration adds ~362 s on first load (one-time per instance).
- **CPU video decode at small resolutions**: at 384×512 / 25 frames the CPU video decoder takes ~7 s — over half of warm-state E2E. The Neuron tiled VAE decoder (`--vae-compiled-dir`) collapses this at higher resolutions but isn't a win at 384×512 (overhead beats the savings).
- **Two-stage cold start**: each stage loads its own NEFF; total cold-start overhead is ~2× single-stage.
- **No EFA**: trn2.3xlarge does not use EFA; NCCL/OFI EFA warnings are benign.
- **Frame conversion**: `VideoDecoder.decode_video()` returns float in [0, 1] in ltx-core ≥ 1.1 (not uint8 as the older free function did). The runner handles the conversion explicitly; without the cast `.numpy()` raises on bf16 and a naive `.to(torch.uint8)` truncates every pixel to 0.

## File Structure

```
LTX-2.3/
  README.md
  src/
    compile.sh                            # Master compile driver (this PR)
    compile.py                            # encoder | transformer | vae subcommands
    compile_benchmark.py                  # trn2.48xlarge two-phase compile harness
    shard_weights.py                      # backbone | encoder weight pre-sharding
    generate_ltx23.py                     # E2E runner (text encode → denoise → VAE → MP4 + WAV)
    run_phase2.py                         # Phase 2 standalone (S2 denoise + Neuron VAE)
    pipeline.py                           # NeuronTransformerWrapper + AdaLN dedup + step-cache
    modeling_ltx23.py                     # Backbone TP sharding + DistributedRMSNorm
    modeling_gemma3_encoder.py            # Encoder-only Gemma 3 model (returns 49 hidden states)
    modeling_vae_23.py                    # TP-sharded VAE decoder
    tiled_vae_decode_23.py                # Tile + overlap blending wrapper
    application.py                        # NxDI Application compositor (run-mode helper)
  test/
    integration/
      test_model.py                       # Forward-pass + cosine-similarity test
```

## Example Checkpoints

* [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3) — `ltx-2.3-22b-distilled.safetensors`
* [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it) — text encoder

## Maintainer

Henan Wan (whn09), forked from jimburtoft/contrib/ltx-2.3.

**Last Updated:** 2026-05-21 (PhaseTimer + unified compile.sh)
