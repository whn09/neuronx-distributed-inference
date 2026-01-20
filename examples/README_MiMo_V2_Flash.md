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
Must divide the minimum num_kv_heads (4). Valid values: **1, 2, 4**.

- Full attention uses num_kv_heads=4
- Sliding window attention uses num_kv_heads=8
- Common divisor is 4

### Memory Requirements
With TP=4, the model requires ~143GB per compilation unit, but Trainium2 has 24GB per chip.

Full compilation requires one of:
- Expert parallelism (EP) to distribute 256 experts across chips
- Multi-node distributed inference
- Model quantization (FP8/INT8)

### Hybrid Attention
| Attention Type | num_kv_heads | head_dim | v_head_dim |
|---------------|--------------|----------|------------|
| Full (pattern=0) | 4 | 192 | 128 |
| Sliding Window (pattern=1) | 8 | 192 | 128 |

The KV cache uses the maximum num_kv_heads (8) for compatibility.

## Usage

```bash
# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Set PYTHONPATH for development
export PYTHONPATH=/home/ubuntu/neuronx-distributed-inference/src:$PYTHONPATH

# Compile model (requires multi-node setup for full compilation)
python examples/generation_mimo_v2_demo.py \
    --tp-degree 4 \
    --batch-size 1 \
    --max-context-length 128 \
    --seq-len 512 \
    --compile-only

# Run inference (after compilation)
python examples/generation_mimo_v2_demo.py \
    --tp-degree 4 \
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

## Next Steps for Full Support

1. Add expert parallelism (EP) configuration to distribute 256 experts
2. Test with multi-node setup (e.g., trn2.48xlarge with 16 chips + EP)
3. Add FP8/INT8 quantization support for memory reduction
4. Optimize KV cache for hybrid attention (per-layer cache sizes)
