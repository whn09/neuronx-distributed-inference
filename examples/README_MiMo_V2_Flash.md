# MiMo-V2-Flash Implementation for neuronx-distributed-inference

## Overview

MiMo-V2-Flash is a large MoE model from Xiaomi with unique architecture features:
- **48 layers** with hybrid attention patterns
- **256 routed experts** per layer, top-8 routing with sigmoid activation
- **Hybrid attention**: Full attention + Sliding window attention (alternating pattern)
- **Different Q/K dim (192) and V dim (128)** - asymmetric head dimensions
- **Partial RoPE**: Only 34% of dimensions use rotary position encoding

Reference: https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash

## Model Architecture

```
MiMo-V2-Flash Architecture
├── Embedding: vocab_size=151936, hidden_size=4096
├── 48 Decoder Layers
│   ├── Attention (hybrid pattern)
│   │   ├── Full Attention (pattern=0): num_kv_heads=4, head_dim=192, v_head_dim=128
│   │   └── Sliding Window (pattern=1): num_kv_heads=8, head_dim=192, v_head_dim=128, window=32768
│   ├── RMSNorm (eps=1e-5)
│   └── MoE Layer
│       ├── Router: 4096 → 256 (sigmoid activation)
│       ├── 256 Experts: gate_proj + up_proj + down_proj (GLU)
│       └── Intermediate size: 1536 per expert
├── Final RMSNorm
└── LM Head (tied with embeddings)

Total Parameters: ~143B (FP8) / ~286B (BF16 equivalent)
```

## Files

### Model Implementation
| File | Description |
|------|-------------|
| `src/neuronx_distributed_inference/models/mimo_v2/__init__.py` | Module exports |
| `src/neuronx_distributed_inference/models/mimo_v2/modeling_mimo_v2.py` | Main model implementation |

### Demo Script
| File | Description |
|------|-------------|
| `examples/generation_mimo_v2_demo.py` | Compilation and inference demo |

## Quick Start

### Prerequisites

The original HuggingFace checkpoint uses FP8 quantized weights. You need to convert them to BF16 first on a GPU machine:

```python
# Run this on a GPU machine with sufficient memory (~200GB)
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "XiaomiMiMo/MiMo-V2-Flash",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.save_pretrained("/path/to/MiMo-V2-Flash-BF16")
# Also copy the tokenizer files from the original checkpoint
```

### Environment Setup
```bash
# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Set PYTHONPATH for development
export PYTHONPATH=/home/ubuntu/neuronx-distributed-inference/src:$PYTHONPATH
```

### Compile and Run

```bash
# Compile model with TP=32 (recommended)
python examples/generation_mimo_v2_demo.py \
    --model-path /home/ubuntu/models/MiMo-V2-Flash-BF16 \
    --compiled-model-path /home/ubuntu/traced_model/MiMo-V2-Flash-BF16 \
    --tp-degree 32 \
    --batch-size 1 \
    --max-context-length 128 \
    --seq-len 512 \
    --compile-only

# Run inference (after compilation)
python examples/generation_mimo_v2_demo.py \
    --compiled-model-path /home/ubuntu/traced_model/MiMo-V2-Flash-BF16 \
    --tp-degree 32 \
    --skip-compile \
    --prompt "Hello, how are you?"
```

## Constraints

### TP Degree Requirements

| Mode | TP Range | Valid Values | Requirement |
|------|----------|--------------|-------------|
| Standard GQA | TP ≤ 4 | 1, 2, 4 | Must divide min(num_kv_heads) = 4 |
| CONVERT_TO_MHA | TP > 4 | 8, 16, 32, 64 | Must divide num_attention_heads = 64 |

**Recommended: TP=32** for trn2.48xlarge (32 Neuron cores)

### Memory Requirements

| TP Degree | Memory per Chip | Sharded Model Size | Status |
|-----------|-----------------|-------------------|--------|
| 4 | ~143GB | ~585GB | Exceeds 24GB HBM |
| 32 | ~4.5GB | ~585GB | Fits (recommended) |

### Hybrid Attention Configuration

| Attention Type | Pattern | num_kv_heads | With CONVERT_TO_MHA | head_dim | v_head_dim | window |
|---------------|---------|--------------|---------------------|----------|------------|--------|
| Full | 0 | 4 | 64 | 192 | 128 | ∞ |
| Sliding Window | 1 | 8 | 64 | 192 | 128 | 32768 |

## Implementation Details

### Model Components

| Component | Description |
|-----------|-------------|
| `MiMoV2InferenceConfig` | Config class with hybrid attention pattern parsing |
| `MiMoV2RotaryEmbedding` | Partial RoPE (34% of dimensions) |
| `NeuronMiMoV2Attention` | Hybrid attention with different K/V head dims |
| `NeuronMiMoV2DecoderLayer` | Decoder layer with hybrid attention + MoE |
| `NeuronMiMoV2Model` | Full transformer model |
| `NeuronMiMoV2ForCausalLM` | Causal LM wrapper with weight conversion |

### CONVERT_TO_MHA Mode

When TP > num_kv_heads (4 for MiMo), the implementation uses GQA CONVERT_TO_MHA strategy:

```
Original (GQA):
  Q: [batch, seq, 64, 192]  (64 heads)
  K: [batch, seq, 4, 192]   (4 heads, shared by 16 Q heads each)
  V: [batch, seq, 4, 128]   (4 heads)

CONVERT_TO_MHA (TP=32):
  Q: [batch, seq, 64, 192]  (64 heads, 2 per TP rank)
  K: [batch, seq, 64, 192]  (replicated to 64 heads, 2 per TP rank)
  V: [batch, seq, 64, 128]  (replicated to 64 heads, 2 per TP rank)
```

K/V replication uses `repeat_interleave` for correct GQA pattern:
```python
# Correct: [h0, h0, ..., h1, h1, ...] (each KV head serves consecutive Q heads)
k_proj = k_proj.repeat_interleave(repeat_factor, dim=0)
```

### KV Cache Workarounds

1. **Different K/V head dimensions**: V (128) is padded to K dimension (192) when storing to cache
2. **Hybrid KV heads**: Full attention (4 heads) pads to sliding window (8 heads) for unified cache format
3. **With CONVERT_TO_MHA**: All layers use 64 KV heads (same as Q heads)

### Token Generation

Uses decomposed attention approach for efficiency:
```
1. Prior attention: Compute attention on cached KV (all previous tokens)
2. Active attention: Compute attention on current token's KV
3. Combine: Weighted sum based on softmax normalization factors
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `TP degree must divide num_kv_heads` | TP ≤ 4 but doesn't divide 4 | Use TP = 1, 2, or 4 |
| `TP degree must divide num_attention_heads` | TP > 4 but doesn't divide 64 | Use TP = 8, 16, 32, or 64 |
| Garbage output | K/V replication using `repeat` instead of `repeat_interleave` | Fixed in current version |
| Garbage output | Missing `attention_sink_bias` in token generation | Fixed in current version |
| Slow checkpoint loading | Large model | Use `save_sharded_checkpoint=True` |
| Repetitive output | Missing EOS token handling | Set `eos_token_id` and use `max_new_tokens` |

### Key Bug Fixes (2025-01-28)

#### Issue: Garbage/Random Output During Inference

**Root Cause**: The token generation path was missing the `attention_sink_bias` handling that was present in the context encoding path.

**Technical Details**:
- MiMo-V2-Flash uses "attention sink bias" for Sliding Window Attention (SWA) layers
- This is a learnable parameter (`attention_sink_bias` with shape `[num_attention_heads]`) that adds an extra "sink" column to attention weights
- The HF implementation applies this in `eager_attention_forward`:
  ```python
  # Add sink column
  sinks = module.attention_sink_bias.reshape(1, -1, 1, 1).expand(...)
  attn_weights = torch.cat([attn_weights, sinks], dim=-1)
  # Subtract max for numerical stability
  attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
  # Softmax
  probs = F.softmax(attn_weights, dim=-1)
  # Drop sink column after softmax
  probs = probs[..., :-1]
  ```

**Fix Applied**:
1. Added `attention_sink_bias` handling to the token generation path (was completely missing)
2. Both context encoding and token generation now follow the HF implementation:
   - Add sink bias column before softmax
   - Subtract max for numerical stability
   - Apply softmax
   - Drop sink column after softmax

**Verification**:
- Config: `add_swa_attention_sink_bias=True` (default), `add_full_attention_sink_bias=False`
- Only SWA layers (39 out of 48) have `attention_sink_bias` in the checkpoint
- Full attention layers (layer 0, 5, 11, 17, 23, 29, 35, 41, 47) do NOT have `attention_sink_bias`

## Expert Parallelism Note

The neuronx-distributed MoE module has a limitation for EP > 1:
```
NotImplementedError: Selective Loading with Expert parallelism is not supported in token generation.
```

With TP=32, EP is not required since the model fits in memory using tensor parallelism alone.

## Native FP8 Support (Not Currently Working)

The original HuggingFace checkpoint uses FP8 quantized weights with block-wise 128x128 scales. Native FP8 inference is **not currently supported** due to scale format incompatibilities:

- **HuggingFace format**: Block-wise quantization with per-block scales (e.g., `[256, 16, 2]` for MoE)
- **Neuron framework expectation**: Per-row or per-tensor scales (e.g., `[1, 1, 128]`)

Converting between these formats is not straightforward as it would require complete re-quantization, which loses the original quantization accuracy. For now, use BF16 mode by converting the checkpoint on a GPU machine first.

## References

- [MiMo-V2-Flash on HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)
- [Xiaomi MiMo Technical Report](https://github.com/XiaomiMiMo/MiMo)
- [neuronx-distributed-inference Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)

## Version History

| Date | Changes |
|------|---------|
| 2025-01-28 | Fixed attention_sink_bias missing in token generation path (was causing garbage output) |
| 2025-01-21 | Simplified to BF16-only mode (native FP8 not supported due to scale format incompatibilities) |
| 2025-01-20 | Implemented CONVERT_TO_MHA for TP=32 support |
| 2025-01-19 | Initial implementation with hybrid attention and MoE |
