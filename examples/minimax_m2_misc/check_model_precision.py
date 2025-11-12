#!/usr/bin/env python3
"""
Check the precision/dtype of each layer in MiniMax-M2 checkpoint
"""

import torch
from safetensors import safe_open
from pathlib import Path
from collections import defaultdict
import json

model_path = "/home/ubuntu/model_hf/MiniMax-M2/"

def get_safetensor_files(model_path):
    """Find all safetensor files in the model directory"""
    model_dir = Path(model_path)
    safetensor_files = list(model_dir.glob("*.safetensors"))

    # Also check model.safetensors.index.json
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, 'r') as f:
            index_data = json.load(f)
            weight_map = index_data.get('weight_map', {})
            # Get unique safetensor files from index
            unique_files = set(weight_map.values())
            safetensor_files = [model_dir / f for f in unique_files]

    return sorted(safetensor_files)

def analyze_checkpoint_dtypes(model_path):
    """Analyze dtypes of all parameters in the checkpoint"""

    print("=" * 100)
    print("MiniMax-M2 Checkpoint Precision Analysis")
    print("=" * 100)

    safetensor_files = get_safetensor_files(model_path)

    if not safetensor_files:
        print("❌ No safetensor files found!")
        print("Checking for pytorch_model files...")
        model_dir = Path(model_path)
        bin_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("pytorch_model*.bin"))
        if bin_files:
            print(f"Found {len(bin_files)} .bin files - analyzing those instead...")
            return analyze_bin_checkpoint(model_path)
        return

    print(f"\n📁 Found {len(safetensor_files)} safetensor file(s):")
    for f in safetensor_files:
        print(f"  • {f.name}")

    # Statistics
    dtype_counts = defaultdict(int)
    layer_dtypes = {}
    fp8_scale_params = []
    weight_scale_inv_params = []

    print(f"\n🔍 Analyzing parameters...")

    all_keys = []
    for safetensor_file in safetensor_files:
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_keys.append(key)
                tensor = f.get_tensor(key)
                dtype = str(tensor.dtype).replace("torch.", "")

                dtype_counts[dtype] += 1
                layer_dtypes[key] = {
                    'dtype': dtype,
                    'shape': list(tensor.shape),
                    'size': tensor.numel()
                }

                # Check for FP8 scale parameters
                if 'scale' in key.lower():
                    fp8_scale_params.append((key, dtype, list(tensor.shape)))

                if 'weight_scale_inv' in key:
                    weight_scale_inv_params.append((key, dtype, list(tensor.shape)))

    print(f"✓ Analyzed {len(all_keys)} parameters\n")

    # Print dtype statistics
    print("=" * 100)
    print("Data Type Distribution")
    print("=" * 100)
    total_params = sum(dtype_counts.values())
    for dtype, count in sorted(dtype_counts.items()):
        percentage = (count / total_params) * 100
        print(f"  {dtype:20s}: {count:6d} parameters ({percentage:5.1f}%)")

    # Check for FP8 quantization
    print("\n" + "=" * 100)
    print("FP8 Quantization Status")
    print("=" * 100)

    if weight_scale_inv_params:
        print(f"✓ Found {len(weight_scale_inv_params)} FP8 scale parameters (weight_scale_inv)")
        print(f"  → Model has FP8 quantized weights")
        print(f"\n  First 10 FP8 scale parameters:")
        for key, dtype, shape in weight_scale_inv_params[:10]:
            print(f"    • {key:80s} | dtype: {dtype:10s} | shape: {shape}")
        if len(weight_scale_inv_params) > 10:
            print(f"    ... and {len(weight_scale_inv_params) - 10} more")
    else:
        print(f"✗ No weight_scale_inv parameters found")
        print(f"  → Model weights are NOT FP8 quantized")

    if fp8_scale_params and not weight_scale_inv_params:
        print(f"\n  Found {len(fp8_scale_params)} other scale parameters:")
        for key, dtype, shape in fp8_scale_params[:5]:
            print(f"    • {key:80s} | dtype: {dtype:10s} | shape: {shape}")

    # Analyze layer patterns
    print("\n" + "=" * 100)
    print("Layer-wise Precision Breakdown")
    print("=" * 100)

    # Group by layer type
    attention_params = {}
    moe_params = {}
    norm_params = {}
    embedding_params = {}
    other_params = {}

    for key, info in layer_dtypes.items():
        if 'self_attn' in key or 'attention' in key.lower():
            attention_params[key] = info
        elif 'block_sparse_moe' in key or 'expert' in key.lower() or 'gate' in key:
            moe_params[key] = info
        elif 'norm' in key.lower() or 'layernorm' in key.lower():
            norm_params[key] = info
        elif 'embed' in key.lower():
            embedding_params[key] = info
        else:
            other_params[key] = info

    # Print embeddings
    if embedding_params:
        print(f"\n📦 Embedding Layers ({len(embedding_params)} parameters):")
        for key, info in list(embedding_params.items())[:5]:
            print(f"  • {key:80s} | {info['dtype']:10s} | shape: {info['shape']}")

    # Print attention precision (first layer as example)
    print(f"\n🔍 Attention Layers (Layer 0 as example):")
    layer0_attn = {k: v for k, v in attention_params.items() if 'layers.0.self_attn' in k}
    if layer0_attn:
        for key, info in sorted(layer0_attn.items()):
            key_short = key.replace('layers.0.self_attn.', '')
            print(f"  • {key_short:40s} | {info['dtype']:10s} | shape: {info['shape']}")
    else:
        print("  (No attention parameters found for layer 0)")

    # Check if attention has FP8 scales
    attn_has_fp8 = any('weight_scale_inv' in k for k in attention_params.keys())
    print(f"  → Attention precision: {list(set(info['dtype'] for info in layer0_attn.values()))}")
    print(f"  → Has FP8 scales: {'Yes' if attn_has_fp8 else 'No'}")

    # Print MoE precision (first layer, first expert as example)
    print(f"\n🔀 MoE Expert Layers (Layer 0, Expert 0 as example):")
    layer0_expert0 = {k: v for k, v in moe_params.items() if 'layers.0.block_sparse_moe.experts.0' in k}
    if layer0_expert0:
        for key, info in sorted(layer0_expert0.items()):
            key_short = key.replace('layers.0.block_sparse_moe.experts.0.', '')
            print(f"  • {key_short:40s} | {info['dtype']:10s} | shape: {info['shape']}")
    else:
        print("  (No MoE parameters found for layer 0, expert 0)")

    # Check if MoE has FP8 scales
    moe_has_fp8 = any('weight_scale_inv' in k for k in moe_params.keys())
    print(f"  → MoE precision: {list(set(info['dtype'] for info in layer0_expert0.values()))}")
    print(f"  → Has FP8 scales: {'Yes' if moe_has_fp8 else 'No'}")

    # Print router (gate)
    print(f"\n🎯 Router/Gate (Layer 0):")
    layer0_gate = {k: v for k, v in layer_dtypes.items() if 'layers.0.block_sparse_moe.gate' in k}
    if layer0_gate:
        for key, info in layer0_gate.items():
            key_short = key.replace('layers.0.block_sparse_moe.', '')
            print(f"  • {key_short:40s} | {info['dtype']:10s} | shape: {info['shape']}")

    # Print norms
    print(f"\n📐 Normalization Layers (Layer 0 as example):")
    layer0_norms = {k: v for k, v in norm_params.items() if 'layers.0' in k}
    if layer0_norms:
        for key, info in sorted(layer0_norms.items()):
            key_short = key.replace('layers.0.', '')
            print(f"  • {key_short:40s} | {info['dtype']:10s} | shape: {info['shape']}")

    # Summary
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    print(f"\n📊 Parameter Type Breakdown:")
    print(f"  • Embedding params: {len(embedding_params)}")
    print(f"  • Attention params: {len(attention_params)}")
    print(f"  • MoE params: {len(moe_params)}")
    print(f"  • Normalization params: {len(norm_params)}")
    print(f"  • Other params: {len(other_params)}")
    print(f"  • Total: {len(layer_dtypes)}")

    print(f"\n🎯 Quantization Status:")
    if weight_scale_inv_params:
        print(f"  ✓ Model IS FP8 quantized")
        print(f"  • Found {len(weight_scale_inv_params)} FP8 scale parameters")

        # Determine which modules are quantized
        quantized_modules = set()
        for key, _, _ in weight_scale_inv_params:
            if 'self_attn' in key:
                quantized_modules.add('self_attn')
            if 'block_sparse_moe.experts' in key:
                quantized_modules.add('moe_experts')
            if 'lm_head' in key:
                quantized_modules.add('lm_head')

        print(f"  • Quantized modules: {', '.join(quantized_modules)}")

        # Determine which modules are NOT quantized
        not_quantized = []
        if not any('self_attn' in k for k, _, _ in weight_scale_inv_params):
            not_quantized.append('self_attn')
        if not any('gate.weight_scale_inv' in k for k, _, _ in weight_scale_inv_params):
            not_quantized.append('gate')
        if not any('lm_head' in k for k, _, _ in weight_scale_inv_params):
            not_quantized.append('lm_head')

        if not_quantized:
            print(f"  • NOT quantized: {', '.join(not_quantized)}")
    else:
        print(f"  ✗ Model is NOT FP8 quantized")
        print(f"  • All weights are in their original precision")
        print(f"  • Predominant dtype: {max(dtype_counts, key=dtype_counts.get)}")

    print("\n" + "=" * 100)

def analyze_bin_checkpoint(model_path):
    """Analyze .bin checkpoint files"""
    model_dir = Path(model_path)
    bin_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("pytorch_model*.bin"))

    print(f"\n📁 Found {len(bin_files)} .bin file(s)")

    dtype_counts = defaultdict(int)

    for bin_file in bin_files:
        print(f"\nLoading {bin_file.name}...")
        state_dict = torch.load(bin_file, map_location='cpu')

        for key, tensor in state_dict.items():
            dtype = str(tensor.dtype).replace("torch.", "")
            dtype_counts[dtype] += 1

    print("\n" + "=" * 100)
    print("Data Type Distribution")
    print("=" * 100)
    total_params = sum(dtype_counts.values())
    for dtype, count in sorted(dtype_counts.items()):
        percentage = (count / total_params) * 100
        print(f"  {dtype:20s}: {count:6d} parameters ({percentage:5.1f}%)")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    analyze_checkpoint_dtypes(model_path)
