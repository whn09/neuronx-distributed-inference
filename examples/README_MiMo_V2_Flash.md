# MiMo-V2-Flash Implementation for neuronx-distributed-inference

## Overview

MiMo-V2-Flash is a large MoE model from Xiaomi with unique architecture features:
- **48 layers** with hybrid attention patterns
- **256 routed experts** per layer, top-8 routing with sigmoid activation
- **Hybrid attention**: Full attention + Sliding window attention (alternating pattern)
- **Different Q/K dim (192) and V dim (128)** - asymmetric head dimensions
- **Partial RoPE**: Only 34% of dimensions use rotary position encoding
- **FP8 quantized weights**: float8_e4m3fn with block-wise 128x128 scales

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
| `src/neuronx_distributed_inference/models/mimo_v2/conversion_script/preprocess_mimo_v2_fp8.py` | FP8 preprocessing script |

### Demo Script
| File | Description |
|------|-------------|
| `examples/generation_mimo_v2_demo.py` | Compilation and inference demo |

## Quick Start

### Environment Setup
```bash
# Activate environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Set PYTHONPATH for development
export PYTHONPATH=/home/ubuntu/neuronx-distributed-inference/src:$PYTHONPATH
```

### Option A: BF16 Mode (Simpler, Larger Memory)

FP8 weights are dequantized to BF16 during loading.

```bash
# Compile model with TP=32 (recommended)
python examples/generation_mimo_v2_demo.py \
    --model-path /home/ubuntu/models/MiMo-V2-Flash \
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

### Option B: Native FP8 Mode (Recommended, Smaller Memory)

Weights remain in FP8 format with preserved scales. Requires preprocessing.

**Step 1: Preprocess FP8 checkpoint (one-time, ~5 hours)**

```bash
python -m neuronx_distributed_inference.models.mimo_v2.conversion_script.preprocess_mimo_v2_fp8 \
    --hf_model_path /home/ubuntu/models/MiMo-V2-Flash \
    --save_path /home/ubuntu/models/preprocessed_mimo_v2_fp8 \
    --tp_degree 32 \
    --convert_to_mha
```

**Step 2: Compile with native FP8**
```bash
python examples/generation_mimo_v2_demo.py \
    --model-path /home/ubuntu/models/MiMo-V2-Flash \
    --compiled-model-path /home/ubuntu/traced_model/MiMo-V2-Flash-FP8 \
    --tp-degree 32 \
    --batch-size 1 \
    --max-context-length 128 \
    --seq-len 512 \
    --compile-only \
    --quantized \
    --quantized-checkpoints-path /home/ubuntu/models/preprocessed_mimo_v2_fp8
```

**Step 3: Run inference**
```bash
python examples/generation_mimo_v2_demo.py \
    --compiled-model-path /home/ubuntu/traced_model/MiMo-V2-Flash-FP8 \
    --tp-degree 32 \
    --skip-compile \
    --quantized \
    --quantized-checkpoints-path /home/ubuntu/models/preprocessed_mimo_v2_fp8 \
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

| Mode | TP Degree | Memory per Chip | Sharded Model Size | Status |
|------|-----------|-----------------|-------------------|--------|
| BF16 | 4 | ~143GB | ~585GB | Exceeds 24GB HBM |
| BF16 | 32 | ~4.5GB | ~585GB | Fits (recommended) |
| FP8 | 4 | ~36GB | ~143GB | Exceeds 24GB HBM |
| FP8 | 32 | ~4.5GB | ~143GB | Fits (recommended) |

### Hybrid Attention Configuration

| Attention Type | Pattern | num_kv_heads | With CONVERT_TO_MHA | head_dim | v_head_dim | window |
|---------------|---------|--------------|---------------------|----------|------------|--------|
| Full | 0 | 4 | 64 | 192 | 128 | ∞ |
| Sliding Window | 1 | 8 | 64 | 192 | 128 | 32768 |

## FP8 Quantization Details

### FP8 Format Differences

| Format | Range | Exponent | Mantissa | Used By |
|--------|-------|----------|----------|---------|
| OCP FP8 E4M3 (e4m3fn) | ±448 | 4 bits | 3 bits | HuggingFace, GPUs |
| IEEE-754 FP8 E4M3 | ±240 | 4 bits | 3 bits | Neuron/Trainium |

**Rescaling Factor**: `FP8_SCALING_FACTOR = 448.0 / 240.0 ≈ 1.867`

### Scale Format Conversion

| Source (HuggingFace) | Target (Neuron) | Conversion Formula |
|---------------------|-----------------|-------------------|
| `weight_scale_inv` | `.scale` | `neuron_scale = weight_scale_inv * FP8_SCALING_FACTOR` |

### Block-wise Quantization

MiMo-V2-Flash uses block-wise FP8 quantization:
- **Block size**: 128 × 128
- **Each block has its own scale factor**
- **Scale tensor shape**: `[ceil(weight_h / 128), ceil(weight_w / 128)]`

Example:
```
Weight: layers.0.self_attn.q_proj.weight
  Shape: [12288, 4096]  (num_attention_heads * head_dim, hidden_size)
  Scale shape: [96, 32]  (12288/128 = 96, 4096/128 = 32)
  Total blocks: 3,072 (each with independent scale)
```

### What the Preprocessing Script Does

1. **Load HuggingFace checkpoint** (~143GB, takes ~30 minutes)
2. **Rescale FP8 weights**: Divide by `FP8_SCALING_FACTOR` to fit Neuron's ±240 range
3. **Convert scales**: `weight_scale_inv` → `.scale` with rescaling
4. **Fuse MoE experts**: Concatenate gate_proj + up_proj for efficient inference
5. **Handle CONVERT_TO_MHA**: Replicate K/V weights and scales for TP > num_kv_heads
6. **Save in safetensors format**: Compatible with neuronx-distributed-inference

**Preprocessing Time**: ~5 hours for full model (48 layers × 256 experts)
**Memory Required**: ~400GB RAM during preprocessing

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

## Compilation Results

**Configuration**: TP=32, batch_size=1, max_context_length=128, seq_len=512

| Stage | Time |
|-------|------|
| HLO generation | ~11 seconds |
| Context encoding compilation | ~355 seconds |
| Token generation compilation | ~170 seconds |
| Pre-sharding checkpoints (BF16) | ~96 minutes |
| **Total** | ~105 minutes |

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `TP degree must divide num_kv_heads` | TP ≤ 4 but doesn't divide 4 | Use TP = 1, 2, or 4 |
| `TP degree must divide num_attention_heads` | TP > 4 but doesn't divide 64 | Use TP = 8, 16, 32, or 64 |
| Out of memory during preprocessing | Model is large | Need ~400GB RAM |
| Garbage output | K/V replication using `repeat` instead of `repeat_interleave` | Fixed in current version |
| Slow checkpoint loading | Large model | Use `save_sharded_checkpoint=True` |

### Monitoring Preprocessing

```bash
# Check if preprocessing is running
ps aux | grep preprocess_mimo | grep -v grep

# Check memory usage
ps aux | grep preprocess_mimo | grep -v grep | awk '{print "RSS:", $6/1024/1024 "GB"}'

# Check output directory (created when complete)
ls -la /home/ubuntu/models/preprocessed_mimo_v2_fp8/
```

## Expert Parallelism Note

The neuronx-distributed MoE module has a limitation for EP > 1:
```
NotImplementedError: Selective Loading with Expert parallelism is not supported in token generation.
```

With TP=32, EP is not required since the model fits in memory using tensor parallelism alone.

## References

- [MiMo-V2-Flash on HuggingFace](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)
- [Xiaomi MiMo Technical Report](https://github.com/XiaomiMiMo/MiMo)
- [neuronx-distributed-inference Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)

## Version History

| Date | Changes |
|------|---------|
| 2025-01-20 | Added native FP8 support with preprocessing script |
| 2025-01-20 | Implemented CONVERT_TO_MHA for TP=32 support |
| 2025-01-19 | Initial implementation with hybrid attention and MoE |
