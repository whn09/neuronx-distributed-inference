#!/usr/bin/env python3
"""
Script to show the correct quantization_config based on checkpoint analysis
"""

import json

config_path = "/home/ubuntu/model_hf/MiniMax-M2/config.json"

print("=" * 100)
print("MiniMax-M2 Quantization Configuration Guide")
print("=" * 100)

print("""
Based on the checkpoint analysis, your model HAS FP8 quantization with the following details:

📊 Checkpoint Facts:
  • Attention layers (q/k/v/o_proj): float8_e4m3fn + weight_scale_inv
  • MoE expert layers (w1/w2/w3): float8_e4m3fn + weight_scale_inv
  • Router (gate): float32 (NOT quantized)
  • LM Head: bfloat16 (NOT quantized)
  • Embedding: bfloat16 (NOT quantized)

🔧 Correct quantization_config to add to config.json:
""")

quantization_config = {
    "activation_scheme": "dynamic",
    "fmt": "float8_e4m3fn",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
    "modules_to_not_convert": [
        "gate",
        "e_score_correction_bias",
        "lm_head"
    ]
}

print(json.dumps({"quantization_config": quantization_config}, indent=2))

print("""
⚠️  IMPORTANT NOTES:

1. When quantization_config is MISSING from config.json:
   → HuggingFace's AutoModel.from_pretrained() will FILTER OUT all FP8 scale parameters
   → You'll get float8_e4m3fn weights WITHOUT their corresponding weight_scale_inv
   → This causes the AssertionError in verify_scale_dimension()

2. When quantization_config is PRESENT:
   → HuggingFace will load both weights AND scales
   → Model will work correctly with FP8 quantization

3. modules_to_not_convert:
   - "gate": Router weights are float32 (not FP8)
   - "e_score_correction_bias": Expert bias correction (not quantized)
   - "lm_head": Output head is bfloat16 (not FP8)

   ⚠️  NOTE: "self_attn" should NOT be in this list!
       Attention layers ARE FP8 quantized in your checkpoint!

4. If you want to use the model WITHOUT quantization_config:
   → You need to convert all float8_e4m3fn weights back to bfloat16/float32
   → OR modify the loading code to handle missing scales
   → This is NOT recommended as you lose the FP8 quantization benefits
""")

print("\n" + "=" * 100)
print("Checking current config.json...")
print("=" * 100)

with open(config_path, 'r') as f:
    config = json.load(f)

if 'quantization_config' in config:
    print("✓ quantization_config is present:")
    print(json.dumps(config['quantization_config'], indent=2))
else:
    print("✗ quantization_config is MISSING")
    print("\nTo restore it, add the following to config.json:")
    print("\nOption 1: Manual edit")
    print("  1. Open /home/ubuntu/model_hf/MiniMax-M2/config.json")
    print('  2. Add the "quantization_config" block shown above')
    print("  3. Save the file")

    print("\nOption 2: Automatic restore (run this command):")
    print("  python3 -c \"")
    print("import json")
    print(f"with open('{config_path}', 'r') as f:")
    print("    config = json.load(f)")
    print("config['quantization_config'] = {")
    print("    'activation_scheme': 'dynamic',")
    print("    'fmt': 'float8_e4m3fn',")
    print("    'quant_method': 'fp8',")
    print("    'weight_block_size': [128, 128],")
    print("    'modules_to_not_convert': ['gate', 'e_score_correction_bias', 'lm_head']")
    print("}")
    print(f"with open('{config_path}', 'w') as f:")
    print("    json.dump(config, f, indent=2)")
    print('  "')

print("\n" + "=" * 100)
print("After restoring quantization_config:")
print("=" * 100)
print("""
✅ Update your generation_minimax_m2_demo.py:

REMOVE this line:
    "self_attn",  # ← REMOVE THIS!

Your modules_to_not_convert should be:
    modules_to_not_convert=[
        "lm_head",
        # Note: "gate" and "e_score_correction_bias" are already excluded in the model architecture
    ],

Or simply match what's in quantization_config:
    modules_to_not_convert=[
        "gate",
        "e_score_correction_bias",
        "lm_head"
    ],
""")

print("=" * 100)
