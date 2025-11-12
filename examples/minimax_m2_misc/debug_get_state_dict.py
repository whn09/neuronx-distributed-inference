#!/usr/bin/env python3
"""
Debug get_state_dict to understand the loading flow
"""

import torch
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import (
    MiniMaxM2InferenceConfig,
    NeuronMiniMaxM2ForCausalLM
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

model_path = "/home/ubuntu/model_hf/MiniMax-M2/"
traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2/"

print("=" * 100)
print("Debugging get_state_dict loading flow")
print("=" * 100)

# Create config (same as your generation script)
neuron_config = MoENeuronConfig(
    tp_degree=64,
    batch_size=1,
    max_context_length=128,
    seq_len=1024,
    enable_bucketing=False,
    flash_decoding_enabled=False,
    blockwise_matmul_config={'use_torch_block_wise': True},
    quantized_mlp_kernel_enabled=True,
    modules_to_not_convert=["lm_head", "self_attn"],
    fused_qkv=False,
)

config = MiniMaxM2InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

print(f"\n🔍 Config check:")
print(f"  quantized_mlp_kernel_enabled: {config.neuron_config.quantized_mlp_kernel_enabled}")
print(f"  quantized: {config.neuron_config.quantized}")
print(f"  modules_to_not_convert: {config.neuron_config.modules_to_not_convert}")

# Test 1: Load from original HF model path
print("\n" + "=" * 100)
print("Test 1: Loading from ORIGINAL HF model path")
print("=" * 100)
print(f"Path: {model_path}")

try:
    state_dict_hf = NeuronMiniMaxM2ForCausalLM.get_state_dict(model_path, config)

    # Check for weight_scale_inv and scale parameters
    weight_scale_inv_keys = [k for k in state_dict_hf.keys() if 'weight_scale_inv' in k]
    scale_keys = [k for k in state_dict_hf.keys() if k.endswith('.scale') and 'weight_scale_inv' not in k]
    attn_keys = [k for k in state_dict_hf.keys() if 'self_attn' in k and '.0.' in k]

    print(f"\n📊 Loaded state_dict from HF model:")
    print(f"  Total keys: {len(state_dict_hf)}")
    print(f"  Keys with 'weight_scale_inv': {len(weight_scale_inv_keys)}")
    print(f"  Keys ending with '.scale': {len(scale_keys)}")
    print(f"  Layer 0 attention keys: {len(attn_keys)}")

    if weight_scale_inv_keys:
        print(f"\n  ❌ ERROR: Found {len(weight_scale_inv_keys)} weight_scale_inv keys!")
        print(f"  These should have been converted to .scale")
        print(f"  First 3:")
        for k in weight_scale_inv_keys[:3]:
            print(f"    {k}")

    if scale_keys:
        print(f"\n  ✓ Found {len(scale_keys)} .scale keys (converted from weight_scale_inv)")
        print(f"  First 5:")
        for k in scale_keys[:5]:
            print(f"    {k}")
            print(f"      dtype: {state_dict_hf[k].dtype}, shape: {state_dict_hf[k].shape}")

    if attn_keys:
        print(f"\n  Layer 0 attention parameters:")
        for k in sorted(attn_keys)[:10]:
            tensor = state_dict_hf[k]
            print(f"    {k}: dtype={tensor.dtype}, shape={tensor.shape}")

except Exception as e:
    print(f"  ❌ Error loading from HF model: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Check traced model path
print("\n" + "=" * 100)
print("Test 2: Checking TRACED model path")
print("=" * 100)
print(f"Path: {traced_model_path}")

import os
if os.path.exists(traced_model_path):
    # Check if it has safetensors
    has_safetensors = os.path.exists(os.path.join(traced_model_path, "model.safetensors.index.json"))
    has_single_safetensor = os.path.exists(os.path.join(traced_model_path, "model.safetensors"))

    print(f"  Directory exists: True")
    print(f"  Has sharded safetensors: {has_safetensors}")
    print(f"  Has single safetensor: {has_single_safetensor}")

    if has_safetensors or has_single_safetensor:
        try:
            state_dict_traced = NeuronMiniMaxM2ForCausalLM.get_state_dict(traced_model_path, config)

            weight_scale_inv_keys_traced = [k for k in state_dict_traced.keys() if 'weight_scale_inv' in k]
            scale_keys_traced = [k for k in state_dict_traced.keys() if k.endswith('.scale')]

            print(f"\n📊 Loaded state_dict from traced model:")
            print(f"  Total keys: {len(state_dict_traced)}")
            print(f"  Keys with 'weight_scale_inv': {len(weight_scale_inv_keys_traced)}")
            print(f"  Keys ending with '.scale': {len(scale_keys_traced)}")

            if len(scale_keys_traced) == 0:
                print(f"\n  ⚠️  WARNING: No .scale parameters found in traced model!")
                print(f"  This means FP8 quantization was lost during compilation")
        except Exception as e:
            print(f"  ❌ Error loading from traced model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ⚠️  No safetensors found, likely uses different checkpoint format")
else:
    print(f"  Directory does not exist yet (model not compiled)")

# Analysis
print("\n" + "=" * 100)
print("Analysis")
print("=" * 100)

print("""
🔍 Potential Issues in get_state_dict:

1. Line 124 condition check:
   if config.neuron_config.quantized_mlp_kernel_enabled or config.neuron_config.quantized:

   → This controls whether weight_scale_inv is converted to .scale
   → Your config has quantized_mlp_kernel_enabled=True, so this should be True

2. Line 129 check:
   if param_name.endswith(".weight_scale_inv"):

   → This should match all weight_scale_inv parameters
   → If this doesn't match, possible reasons:
     a) Parameters were already filtered out before reaching this point
     b) Parameter names have a different format
     c) Using wrong checkpoint path (e.g., traced model without FP8 scales)

3. Line 605-606 in get_state_dict:
   if param_name.endswith(".weight_scale"):
       updated_param_name = updated_param_name.replace(".weight_scale", ".scale")

   → This handles .weight_scale but NOT .weight_scale_inv
   → .weight_scale_inv should be handled in convert_minimax_m2_hf_to_neuron_state_dict

4. Common mistake:
   → Loading from traced_model_path instead of original model_path
   → Traced model may have lost FP8 scales if compiled with wrong config
   → Always compile from original HF checkpoint with correct config
""")

print("\n✅ Recommendations:")
print("""
1. Check which path is being loaded:
   - Original HF model: /home/ubuntu/model_hf/MiniMax-M2/
   - Traced model: /home/ubuntu/traced_model/MiniMax-M2/

2. When compiling (skip_compile=False):
   - Should load from ORIGINAL HF model path
   - Should see "Total converted: 47864 FP8 scale parameters"

3. When loading compiled model (skip_compile=True):
   - Loads from traced_model_path
   - May have different checkpoint format (not safetensors)
   - FP8 scales should already be baked into the traced model

4. If you see "Total converted: 0":
   - You're loading from traced model path during compilation (WRONG)
   - Or the checkpoint was corrupted/filtered
   - Re-download the original model or remove traced_model directory
""")

print("=" * 100)
