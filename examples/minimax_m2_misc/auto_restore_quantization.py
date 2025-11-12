#!/usr/bin/env python3
"""
Automatically restore the correct quantization_config to config.json
"""

import json
import shutil
from pathlib import Path

config_path = "/home/ubuntu/model_hf/MiniMax-M2/config.json"

print("=" * 100)
print("Auto-Restore quantization_config")
print("=" * 100)

# Backup original config
backup_path = config_path + ".backup"
shutil.copy2(config_path, backup_path)
print(f"✓ Created backup: {backup_path}")

# Load config
with open(config_path, 'r') as f:
    config = json.load(f)

# Add quantization_config
config['quantization_config'] = {
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

# Save updated config
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Added quantization_config to {config_path}")

# Verify
with open(config_path, 'r') as f:
    updated_config = json.load(f)

if 'quantization_config' in updated_config:
    print("\n✅ Verification passed!")
    print("\nAdded quantization_config:")
    print(json.dumps(updated_config['quantization_config'], indent=2))
else:
    print("\n❌ Verification failed!")

print("\n" + "=" * 100)
print("Next Steps:")
print("=" * 100)
print("""
1. Update generation_minimax_m2_demo.py:
   - REMOVE "self_attn" from modules_to_not_convert
   - Change to: modules_to_not_convert=["lm_head"]

2. Recompile the model:
   python generation_minimax_m2_demo.py

3. The model will now correctly load FP8 quantized attention layers!
""")

print(f"\nℹ️  If you want to revert: cp {backup_path} {config_path}")
print("=" * 100)
