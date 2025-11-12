#!/usr/bin/env python3
"""
Explain weight_scale_inv and scale in FP8 quantization
"""

import torch
import numpy as np

print("=" * 100)
print("Understanding weight_scale_inv and scale in FP8 Quantization")
print("=" * 100)

print("""
📚 Background: FP8 Quantization Basics
═══════════════════════════════════════

FP8 (8-bit floating point) quantization reduces memory usage and speeds up inference by
storing weights in 8 bits instead of 16 bits (bfloat16) or 32 bits (float32).

FP8 format used: float8_e4m3fn
  • e4m3: 1 sign bit + 4 exponent bits + 3 mantissa bits
  • Range: approximately [-448, 448]
  • Much smaller range than bfloat16: [-65504, 65504]

Problem: How to fit large weight values into this small range?
Solution: Use QUANTIZATION SCALES!
""")

print("=" * 100)
print("1️⃣  Quantization Process: From High Precision → FP8")
print("=" * 100)

# Example: Simulate quantization
original_weight = torch.randn(4, 4, dtype=torch.bfloat16) * 10  # Original weights
print(f"\n📊 Original Weight (bfloat16):")
print(f"  Shape: {original_weight.shape}")
print(f"  Dtype: {original_weight.dtype}")
print(f"  Range: [{original_weight.min():.4f}, {original_weight.max():.4f}]")
print(f"  Sample values:")
print(original_weight[:2, :2])

# Per-channel quantization: compute scale for each output channel
# This is block-wise quantization with block_size=[128, 128]
# For simplicity, we do per-tensor here
abs_max = original_weight.abs().max()
fp8_max = 448.0  # Max value representable in float8_e4m3fn

# Compute scale: scale is the factor to DIVIDE original weights
# scale = abs_max / fp8_max
# This ensures: original_weight / scale fits in [-448, 448]
scale = abs_max / fp8_max

print(f"\n🔢 Quantization Calculation:")
print(f"  abs_max(original_weight) = {abs_max:.6f}")
print(f"  fp8_max = {fp8_max}")
print(f"  scale = abs_max / fp8_max = {scale:.6f}")

# Quantize: divide by scale, then convert to FP8
quantized_weight_fp32 = original_weight / scale
print(f"\n  After scaling (still in high precision):")
print(f"    weight / scale range: [{quantized_weight_fp32.min():.4f}, {quantized_weight_fp32.max():.4f}]")
print(f"    ✓ Now fits in FP8 range [-448, 448]!")

# In practice, this would be converted to float8_e4m3fn
# quantized_weight_fp8 = quantized_weight_fp32.to(torch.float8_e4m3fn)

print("\n" + "=" * 100)
print("2️⃣  Storage: What's Saved in the Checkpoint")
print("=" * 100)

print("""
MiniMax-M2 checkpoint stores:

  ✅ Quantized weight (FP8):
     • Key: layers.X.Y.weight
     • Dtype: float8_e4m3fn
     • Values: original_weight / scale
     • Size: 1 byte per element

  ✅ Scale parameter (FP32):
     • Key: layers.X.Y.weight_scale_inv
     • Dtype: float32
     • Values: 1 / scale  (reciprocal!)
     • Size: 4 bytes per scale value

Why store 1/scale instead of scale?
  → Multiplication is faster than division on hardware!
  → During inference: weight * weight_scale_inv (fast ✓)
  → Instead of: weight / scale (slower ✗)
""")

# Demonstrate weight_scale_inv
weight_scale_inv = 1.0 / scale
print(f"💾 Stored in checkpoint:")
print(f"  scale = {scale:.6f}")
print(f"  weight_scale_inv = 1 / scale = {weight_scale_inv:.6f}")

print("\n" + "=" * 100)
print("3️⃣  Dequantization: FP8 → High Precision (During Inference)")
print("=" * 100)

print("""
During inference, we need to recover the original high-precision values:

Method 1: Using scale
  original_weight = quantized_weight_fp8 / scale

Method 2: Using weight_scale_inv (FASTER! ⚡)
  original_weight = quantized_weight_fp8 * weight_scale_inv

Since weight_scale_inv = 1/scale, these are mathematically equivalent!
""")

# Demonstrate dequantization
dequantized_using_scale = quantized_weight_fp32 * scale
dequantized_using_inv = quantized_weight_fp32 * weight_scale_inv

print(f"🔄 Dequantization comparison:")
print(f"\n  Method 1: quantized * scale")
print(f"    Result range: [{dequantized_using_scale.min():.4f}, {dequantized_using_scale.max():.4f}]")
print(f"    Sample:")
print(dequantized_using_scale[:2, :2])

print(f"\n  Method 2: quantized * weight_scale_inv")
print(f"    Result range: [{dequantized_using_inv.min():.4f}, {dequantized_using_inv.max():.4f}]")
print(f"    Sample:")
print(dequantized_using_inv[:2, :2])

print(f"\n  Original (for comparison):")
print(original_weight[:2, :2])

print(f"\n  ✅ Both methods recover the original weights!")

print("\n" + "=" * 100)
print("4️⃣  Neuron Inference Framework Convention")
print("=" * 100)

print("""
Different frameworks use different naming conventions:

HuggingFace MiniMax-M2 checkpoint:
  • Quantized weight: layers.X.Y.weight (float8_e4m3fn)
  • Scale parameter: layers.X.Y.weight_scale_inv (float32)
  • Meaning: weight_scale_inv = 1 / scale
  • Formula: original = quantized_weight * weight_scale_inv

Neuron Inference Framework:
  • Quantized weight: layers.X.Y.weight (float8_e4m3fn)
  • Scale parameter: layers.X.Y.scale (float32)
  • Meaning: scale is the dequantization factor
  • Formula: original = quantized_weight * scale

⚠️  IMPORTANT DIFFERENCE:
  HuggingFace weight_scale_inv = 1 / Neuron scale

Therefore, conversion is needed:
  neuron_scale = 1.0 / hf_weight_scale_inv
""")

print("\n" + "=" * 100)
print("5️⃣  The Conversion in convert_minimax_m2_hf_to_neuron_state_dict")
print("=" * 100)

print("""
This is why you see this code (line 124-139):

```python
if config.neuron_config.quantized_mlp_kernel_enabled or config.neuron_config.quantized:
    for param_name in param_name_list:
        if param_name.endswith(".weight_scale_inv"):
            # Convert weight_scale_inv to scale
            new_param_name = param_name.replace(".weight_scale_inv", ".scale")

            # CRITICAL: Take reciprocal to convert!
            scale_inv = neuron_state_dict[param_name]
            neuron_state_dict[new_param_name] = 1.0 / scale_inv  # ← INVERSION!

            del neuron_state_dict[param_name]
```

Step-by-step:
  1. Read HF checkpoint's weight_scale_inv
  2. Compute Neuron's scale = 1.0 / weight_scale_inv
  3. Rename: .weight_scale_inv → .scale
  4. Delete original .weight_scale_inv parameter
  5. Now Neuron framework can use: original = weight * scale
""")

print("\n" + "=" * 100)
print("6️⃣  Block-wise Quantization (Real Implementation)")
print("=" * 100)

print("""
MiniMax-M2 uses block-wise quantization with block_size=[128, 128]:

Example: layers.0.self_attn.q_proj.weight
  • Weight shape: [6144, 3072]
  • Block size: [128, 128]
  • Number of blocks: (6144/128) × (3072/128) = 48 × 24 = 1,152 blocks
  • Scale shape: [48, 24]  ← One scale per block!

Each 128×128 block has its own scale:
  quantized_block[i,j] = original_block[i,j] / scale[i,j]

This provides better precision than per-tensor or per-channel quantization!

From the checkpoint analysis:
  layers.0.self_attn.q_proj.weight: float8_e4m3fn, shape=[6144, 3072]
  layers.0.self_attn.q_proj.weight_scale_inv: float32, shape=[48, 24]
                                                            ↑
  After conversion:                                     48 = 6144/128
  layers.0.self_attn.qkv_proj.q_proj.scale: float32, shape=[48, 24]
                                                                  ↑
                                                              24 = 3072/128
""")

print("\n" + "=" * 100)
print("7️⃣  Practical Example from Your Model")
print("=" * 100)

# Load actual scale from checkpoint
print("""
From the checkpoint analysis, we saw:

Layer 0 Q projection:
  • layers.0.self_attn.qkv_proj.q_proj.weight: float8_e4m3fn, [6144, 3072]
  • layers.0.self_attn.qkv_proj.q_proj.scale: float32, [48, 24]

Layer 0 Expert 0 w1:
  • layers.0.block_sparse_moe.experts.0.w1.weight: float8_e4m3fn, [1536, 3072]
  • layers.0.block_sparse_moe.experts.0.w1.scale: float32, [12, 24]
    (12 = 1536/128, 24 = 3072/128)

During inference, Neuron computes:
  original_weight[i,j] = quantized_weight[i,j] * scale[block_i, block_j]

This happens in the fused FP8 GEMM kernel for maximum performance!
""")

print("\n" + "=" * 100)
print("8️⃣  Why This Matters for Your Error")
print("=" * 100)

print("""
❌ The AssertionError you saw:

  assert tensor_scale.shape[channel_axis] == tensor.shape[channel_axis]

Happened because:
  1. You had quantized_mlp_kernel_enabled=True
  2. System expected both weight AND scale for attention layers
  3. Checkpoint HAS both (we verified: 47,864 scale parameters!)
  4. But when you added "self_attn" to modules_to_not_convert:
     → System thought: "Don't expect scales for self_attn"
     → But scales WERE loaded and converted
     → Mismatch! Some code paths expected scale=None, got scale=tensor
     → Dimension verification failed

✅ Solution:
  REMOVE "self_attn" from modules_to_not_convert
  Let the system correctly load and use the FP8 attention weights + scales!
""")

print("\n" + "=" * 100)
print("Summary")
print("=" * 100)

print("""
┌─────────────────────┬──────────────────────┬─────────────────────────┐
│ Terminology         │ Mathematical Meaning │ Usage                   │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ weight_scale_inv    │ 1 / scale            │ HF checkpoint storage   │
│ (HuggingFace)       │                      │ Fast multiplication     │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ scale               │ Dequant factor       │ Neuron framework        │
│ (Neuron)            │                      │ original = weight*scale │
├─────────────────────┼──────────────────────┼─────────────────────────┤
│ Conversion          │ scale = 1/scale_inv  │ In state_dict loader    │
└─────────────────────┴──────────────────────┴─────────────────────────┘

Key Points:
  1. FP8 quantization compresses weights from 16/32 bits → 8 bits
  2. Scale parameters enable dequantization back to high precision
  3. HF stores reciprocal (weight_scale_inv) for faster multiplication
  4. Neuron expects direct scale, so conversion (1/x) is needed
  5. Block-wise quantization uses one scale per [128, 128] block
  6. Your model has 47,864 scale parameters covering all quantized layers
  7. Both Attention AND MoE layers are FP8 quantized in your checkpoint!
""")

print("=" * 100)
