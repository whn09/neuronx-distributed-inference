#!/usr/bin/env python3
"""
Convert MiniMax M2 FP8 checkpoint to BF16 checkpoint.
Run this on a GPU machine with enough memory.

Usage:
    python convert_minimax_m2_fp8_to_bf16.py \
        --input_dir /path/to/fp8/checkpoint \
        --output_dir /path/to/bf16/checkpoint
       
hf download MiniMaxAI/MiniMax-M2 --local-dir /home/ubuntu/model_hf/MiniMax-M2/
 
python convert_minimax_m2_fp8_to_bf16.py \
    --input_dir /opt/dlami/nvme/MiniMax-M2/ \
    --output_dir /opt/dlami/nvme/MiniMax-M2-BF16/
    
Manually remove `quantization_config` from config.json

python -m sglang.launch_server \
    --model-path /opt/dlami/nvme/MiniMax-M2-BF16 \
    --tp-size 8 \
    --ep-size 8 \
    --tool-call-parser minimax-m2 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --reasoning-parser minimax-append-think \
    --port 8000 \
    --mem-fraction-static 0.85
    
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "MiniMax-M2",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": "Who won the world series in 2020?"}]}
        ]
    }'

"""

import argparse
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequantize_fp8_blockwise(weight, scale):
    """
    Dequantize FP8 weight using block-wise scale.

    Args:
        weight: FP8 tensor [H, W]
        scale: Scale tensor [scale_h, scale_w]

    Returns:
        Dequantized tensor in float32
    """
    weight_float = weight.float()
    scale_h, scale_w = scale.shape
    weight_h, weight_w = weight.shape
    block_h = weight_h // scale_h
    block_w = weight_w // scale_w

    dequantized = torch.empty_like(weight_float)

    for i in range(scale_h):
        for j in range(scale_w):
            block_scale = scale[i, j]
            h_start, h_end = i * block_h, (i + 1) * block_h
            w_start, w_end = j * block_w, (j + 1) * block_w
            dequantized[h_start:h_end, w_start:w_end] = \
                weight_float[h_start:h_end, w_start:w_end] * block_scale

    return dequantized


def convert_shard(input_shard_path, output_shard_path, device='cuda'):
    """Convert one safetensors shard from FP8 to BF16."""

    print(f"\nProcessing: {input_shard_path.name}")

    # Load the shard
    tensors = {}
    with safe_open(input_shard_path, framework='pt', device=device) as f:
        keys = list(f.keys())

        # Find all weight_scale_inv parameters
        scale_inv_keys = {k for k in keys if k.endswith('.weight_scale_inv')}

        for key in tqdm(keys, desc="Converting tensors"):
            tensor = f.get_tensor(key)

            # Skip weight_scale_inv parameters (we'll use them but not save them)
            if key in scale_inv_keys:
                continue

            # Check if this weight has a corresponding scale
            if key.endswith('.weight'):
                scale_inv_key = key.replace('.weight', '.weight_scale_inv')

                if scale_inv_key in scale_inv_keys and tensor.dtype == torch.float8_e4m3fn:
                    # Dequantize FP8 weight
                    # Note: Despite the name "weight_scale_inv", it's actually the scale itself!
                    scale = f.get_tensor(scale_inv_key)

                    # Dequantize and convert to bfloat16
                    dequantized = dequantize_fp8_blockwise(tensor, scale)
                    bf16_result = dequantized.to(torch.bfloat16)
                    tensors[key] = bf16_result.cpu()

                    # Validate conversion
                    print(f"  ✓ Dequantized: {key}")
                    print(f"      Weight: {tensor.shape} [{tensor.float().min():.2f}, {tensor.float().max():.2f}]")
                    print(f"      Scale:  {scale.shape} [{scale.min():.6f}, {scale.max():.6f}]")
                    print(f"      Result: {bf16_result.shape} [{bf16_result.float().min():.6f}, {bf16_result.float().max():.6f}]")
                else:
                    # Not FP8, just convert dtype if needed
                    if tensor.dtype == torch.float32:
                        tensors[key] = tensor.to(torch.bfloat16).cpu()
                    else:
                        tensors[key] = tensor.cpu()
            else:
                # Non-weight parameter, keep as is
                tensors[key] = tensor.cpu()

    # Save the converted shard
    save_file(tensors, output_shard_path)
    print(f"  Saved to: {output_shard_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert MiniMax M2 FP8 to BF16')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory with FP8 checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for BF16 checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use for conversion (cuda or cpu)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting MiniMax M2 checkpoint:")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {args.device}")

    # Copy config files
    print("\n=== Copying config files ===")
    for config_file in ['config.json', 'tokenizer_config.json', 'tokenizer.json',
                        'special_tokens_map.json', 'generation_config.json', 
                        'chat_template.jinja', 'configuration_minimax_m2.py', 'modeling_minimax_m2.py']:
        src = input_dir / config_file
        if src.exists():
            shutil.copy2(src, output_dir / config_file)
            print(f"  ✓ Copied: {config_file}")

    # Load and update model index
    index_path = input_dir / 'model.safetensors.index.json'
    if not index_path.exists():
        raise ValueError(f"model.safetensors.index.json not found in {input_dir}")

    with open(index_path) as f:
        index = json.load(f)

    # Remove weight_scale_inv from weight_map
    original_weight_map = index['weight_map'].copy()
    new_weight_map = {}
    removed_count = 0

    for key, shard_file in original_weight_map.items():
        if key.endswith('.weight_scale_inv'):
            removed_count += 1
        else:
            new_weight_map[key] = shard_file

    print(f"\n=== Updating index ===")
    print(f"  Removed {removed_count} weight_scale_inv entries from weight_map")

    index['weight_map'] = new_weight_map

    # Save updated index
    with open(output_dir / 'model.safetensors.index.json', 'w') as f:
        json.dump(index, f, indent=2)
    print(f"  ✓ Saved updated index")

    # Convert all shards
    print("\n=== Converting safetensors shards ===")
    shard_files = sorted(set(original_weight_map.values()))

    for shard_file in shard_files:
        input_shard = input_dir / shard_file
        output_shard = output_dir / shard_file
        convert_shard(input_shard, output_shard, device=args.device)

    print(f"\n✅ Conversion complete!")
    print(f"   Converted {len(shard_files)} shards")
    print(f"   Output saved to: {output_dir}")
    print(f"\nNow you can use the BF16 checkpoint on Neuron without quantization config.")


if __name__ == '__main__':
    main()
