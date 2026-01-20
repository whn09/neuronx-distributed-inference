# MiMo-V2-Flash Implementation for neuronx-distributed-inference

## Overview

MiMo-V2-Flash is a large MoE model from Xiaomi with unique architecture features:
- 48 layers
- 256 routed experts, top-8 routing
- Hybrid attention (full + sliding window)
- Different Q/K dim (192) and V dim (128)
- Partial RoPE (34% of dimensions)

Reference: https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash

## Files

### Model Implementation
- `src/neuronx_distributed_inference/models/mimo_v2/__init__.py` - Module exports
- `src/neuronx_distributed_inference/models/mimo_v2/modeling_mimo_v2.py` - Main model implementation

### Demo Script
- `examples/generation_mimo_v2_demo.py` - Compilation and inference demo

## Model Components

### MiMoV2InferenceConfig
Config class with hybrid attention pattern parsing for per-layer attention type.

### MiMoV2RotaryEmbedding
Partial RoPE implementation where only 34% of dimensions use rotary position encoding.

### NeuronMiMoV2Attention
Custom attention handling:
- Different K/V head dimensions (K: 192, V: 128)
- Hybrid full/sliding window attention per layer
- Token generation with prior/active KV decomposition
- KV cache padding workarounds for unified cache format

### NeuronMiMoV2DecoderLayer
Decoder layer supporting:
- Hybrid attention (full or sliding window per layer)
- MoE or dense FFN per layer based on moe_layer_freq

### NeuronMiMoV2Model / NeuronMiMoV2ForCausalLM
Full model implementation with weight conversion support.

## Constraints

### TP Degree

**Standard GQA Mode (TP <= 4)**:
- Must divide the minimum num_kv_heads (4)
- Valid values: **1, 2, 4**

**CONVERT_TO_MHA Mode (TP > 4)**:
- K/V heads are replicated to match num_attention_heads (64)
- Must divide num_attention_heads (64)
- Valid values: **8, 16, 32, 64**
- **Recommended: TP=32** for this large model (fits on trn2.48xlarge with 32 logical cores)

### Memory Requirements

| TP Degree | Memory per Chip | Status |
|-----------|-----------------|--------|
| 4 | ~143GB | Exceeds 24GB HBM |
| 8 | ~18GB | Fits |
| 16 | ~9GB | Fits comfortably |
| 32 | ~4.5GB | Fits easily (recommended) |

### Hybrid Attention
| Attention Type | Original num_kv_heads | With CONVERT_TO_MHA | head_dim | v_head_dim |
|---------------|----------------------|---------------------|----------|------------|
| Full (pattern=0) | 4 | 64 | 192 | 128 |
| Sliding Window (pattern=1) | 8 | 64 | 192 | 128 |

- With CONVERT_TO_MHA (TP>4), all layers have 64 KV heads (same as Q heads)
- KV cache stores 64 heads per layer (split across TP ranks)

## Usage

```bash
# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Set PYTHONPATH for development
export PYTHONPATH=/home/ubuntu/neuronx-distributed-inference/src:$PYTHONPATH

# Compile model with TP=32 (recommended)
python examples/generation_mimo_v2_demo.py \
    --tp-degree 32 \
    --batch-size 1 \
    --max-context-length 128 \
    --seq-len 512 \
    --compile-only

# Run inference (after compilation)
python examples/generation_mimo_v2_demo.py \
    --tp-degree 32 \
    --skip-compile \
    --prompt "Hello, how are you?"
```

## Implementation Notes

### KV Cache Workarounds

1. **Different K/V head dimensions**: V is padded from 128 to 192 when storing to cache, and sliced back when reading.

2. **Hybrid KV heads**: Full attention layers (4 KV heads) pad to 8 KV heads for cache compatibility with sliding window layers.

### Token Generation

Uses decomposed attention approach:
- Prior attention: Compute attention on cached KV
- Active attention: Compute attention on current token's KV
- Combined with proper softmax normalization

## Current Status

**COMPILATION: PASSED** - Both context_encoding_model and token_generation_model compiled successfully with TP=32.

**Solution: CONVERT_TO_MHA with TP=32** - Using high tensor parallelism (TP=32) with GQA CONVERT_TO_MHA mode to distribute the ~143GB model across 32 logical cores on trn2.48xlarge.

Compilation results:
- HLO generation: ~11 seconds
- Context encoding compilation: ~355 seconds
- Token generation compilation: Cached (used previous compile)
- Total build time: ~431 seconds (~7 minutes)

## Implementation Details

### CONVERT_TO_MHA Mode
When TP > num_kv_heads (4 for MiMo), the implementation uses GQA CONVERT_TO_MHA strategy:
- K and V projections output num_attention_heads (64) instead of num_kv_heads (4 or 8)
- This allows proper tensor parallel splitting across TP ranks
- KV cache stores 64 heads per layer (2 per TP rank with TP=32)

### Expert Parallelism Note
The neuronx-distributed MoE module currently has a limitation for EP > 1:
```
NotImplementedError: Selective Loading with Expert parallelism is not supported in token generation.
```
With TP=32, EP is not required since the model fits in memory using tensor parallelism alone.

## Next Steps

1. Test compilation and inference with TP=32 on a 32-chip instance
2. Benchmark inference performance
3. Consider adding FP8 quantization for smaller cluster deployments
