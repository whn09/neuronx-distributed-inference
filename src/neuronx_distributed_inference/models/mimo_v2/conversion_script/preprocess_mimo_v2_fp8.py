"""
Preprocess MiMo-V2-Flash FP8 checkpoint for Neuron inference.

The HuggingFace FP8 checkpoint cannot be directly used for inference on Neuron.
This script preprocesses the checkpoint to make it compatible.

Steps:
1. Rescale FP8 weights from OCP format (range ±448) to Neuron format (range ±240)
2. Convert weight_scale_inv to .scale format (reciprocal + rescaling)
3. Fuse gate/up projections for MoE experts
4. Handle K/V weight and scale replication for CONVERT_TO_MHA mode
5. Save to preprocessed checkpoint directory

Usage:
    python preprocess_mimo_v2_fp8.py \
        --hf_model_path /path/to/MiMo-V2-Flash \
        --save_path /path/to/preprocessed_mimo_v2_fp8 \
        --tp_degree 32 \
        --convert_to_mha
"""

import argparse
import gc
import json
import os
from typing import Dict, Any, List, Optional

import torch

from neuronx_distributed_inference.modules.checkpoint import (
    load_state_dict,
    save_state_dict_safetensors,
)


# FP8 range difference between OCP (HuggingFace) and Neuron (IEEE-754)
# OCP FP8 E4M3/e4m3fn: range ±448
# Neuron FP8 E4M3 (IEEE-754): range ±240
FP8_SCALING_FACTOR = 448.0 / 240.0


def rescale_fp8_weight(weight: torch.Tensor, scale: torch.Tensor):
    """
    Rescale FP8 weight from OCP format to Neuron format.

    Args:
        weight: FP8 weight tensor (float8_e4m3fn)
        scale: Scale tensor (float32 or bfloat16), this is weight_scale_inv (1/scale)

    Returns:
        Tuple of (rescaled_weight, neuron_scale)
        - rescaled_weight: FP8 weight compatible with Neuron
        - neuron_scale: Scale in Neuron format (direct scale, not reciprocal)
    """
    # Convert weight to BF16 for rescaling
    weight_bf16 = weight.bfloat16()

    # Divide by scaling factor to fit in Neuron's smaller range
    rescaled_weight_bf16 = weight_bf16 / FP8_SCALING_FACTOR

    # Convert back to FP8
    rescaled_weight = rescaled_weight_bf16.to(torch.float8_e4m3fn)

    # Convert weight_scale_inv to neuron scale format:
    # neuron_scale = (1 / weight_scale_inv) * FP8_SCALING_FACTOR
    # Because: original = quantized * weight_scale_inv
    #          After rescaling: original = rescaled * neuron_scale
    # We need: rescaled * neuron_scale = (quantized / FP8_SCALING_FACTOR) * neuron_scale = original
    # So: neuron_scale = weight_scale_inv * FP8_SCALING_FACTOR
    # But wait, Neuron expects: original = weight * scale (where scale is dequant factor)
    # And HF stores: weight_scale_inv = 1/scale
    # So: neuron_scale = (1 / weight_scale_inv) * FP8_SCALING_FACTOR = scale * FP8_SCALING_FACTOR
    # But after rescaling the weight by /FP8_SCALING_FACTOR, we need to compensate:
    # original = (weight / FP8_SCALING_FACTOR) * (scale * FP8_SCALING_FACTOR) = weight * scale
    # So: neuron_scale = 1 / weight_scale_inv (take reciprocal, no rescaling needed for scale)

    # Actually, let's think more carefully:
    # Original dequant: original = fp8_weight * weight_scale_inv (HF convention)
    # But weight_scale_inv is 1/scale, so: original = fp8_weight / scale
    # Wait no, looking at explain_fp8_scales.py:
    #   HF: original = quantized_weight * weight_scale_inv
    #   Neuron: original = quantized_weight * scale
    #   So weight_scale_inv IS the dequant factor, just different naming
    #
    # After our rescaling:
    #   rescaled_weight = fp8_weight / FP8_SCALING_FACTOR
    #   We need: original = rescaled_weight * new_scale
    #   So: original = (fp8_weight / FP8_SCALING_FACTOR) * new_scale = fp8_weight * weight_scale_inv
    #   Therefore: new_scale = weight_scale_inv * FP8_SCALING_FACTOR

    neuron_scale = scale.float() * FP8_SCALING_FACTOR

    return rescaled_weight, neuron_scale.to(torch.float32)


def replicate_for_convert_to_mha(
    weight: torch.Tensor,
    scale: Optional[torch.Tensor],
    num_kv_heads: int,
    num_attention_heads: int,
    head_dim: int,
    block_size: List[int] = [128, 128],
):
    """
    Replicate K/V weights and scales for CONVERT_TO_MHA mode.

    When TP > num_kv_heads, we need to replicate K/V heads to match Q heads.
    This uses repeat_interleave to create the correct GQA pattern.

    Args:
        weight: FP8 K or V weight [num_kv_heads * head_dim, hidden_size]
        scale: Scale tensor [scale_h, scale_w] for block-wise quantization
        num_kv_heads: Original number of KV heads
        num_attention_heads: Target number of attention heads
        head_dim: Dimension per head
        block_size: Block size for quantization [128, 128]

    Returns:
        Tuple of (replicated_weight, replicated_scale)
    """
    if num_kv_heads >= num_attention_heads:
        return weight, scale

    repeat_factor = num_attention_heads // num_kv_heads

    # Reshape weight to [num_kv_heads, head_dim, hidden_size]
    weight_reshaped = weight.view(num_kv_heads, head_dim, -1)

    # Replicate using repeat_interleave (correct GQA pattern)
    # This creates [h0, h0, ..., h1, h1, ...] pattern
    weight_replicated = weight_reshaped.repeat_interleave(repeat_factor, dim=0)

    # Reshape back to [num_attention_heads * head_dim, hidden_size]
    weight_replicated = weight_replicated.view(-1, weight_replicated.shape[-1])

    if scale is None:
        return weight_replicated, None

    # Replicate scale for block-wise quantization
    # Scale shape: [scale_h, scale_w] where scale_h = ceil(weight_h / block_size[0])
    # After replication, weight_h increases by repeat_factor
    # So scale_h should also increase

    # Number of scale rows per KV head
    scales_per_head = scale.shape[0] // num_kv_heads

    # Reshape scale to [num_kv_heads, scales_per_head, scale_w]
    scale_reshaped = scale.view(num_kv_heads, scales_per_head, -1)

    # Replicate scales
    scale_replicated = scale_reshaped.repeat_interleave(repeat_factor, dim=0)

    # Reshape back
    scale_replicated = scale_replicated.view(-1, scale_replicated.shape[-1])

    return weight_replicated, scale_replicated


def process_mimo_v2_checkpoint(
    hf_model_path: str,
    save_path: str,
    tp_degree: int = 32,
    convert_to_mha: bool = True,
):
    """
    Process MiMo-V2-Flash checkpoint for Neuron FP8 inference.

    Args:
        hf_model_path: Path to HuggingFace MiMo-V2-Flash checkpoint
        save_path: Path to save preprocessed checkpoint
        tp_degree: Tensor parallelism degree
        convert_to_mha: Whether to replicate K/V for CONVERT_TO_MHA mode
    """
    print(f"Loading checkpoint from: {hf_model_path}", flush=True)
    state_dict = load_state_dict(hf_model_path)

    # Load config
    config_path = os.path.join(hf_model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract model dimensions
    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]
    num_attention_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]  # Full attention: 4
    swa_num_attention_heads = config["swa_num_attention_heads"]  # Sliding window: 32
    swa_num_kv_heads = config["swa_num_key_value_heads"]  # Sliding window: 8
    head_dim = config["head_dim"]  # Q/K head dim: 192
    v_head_dim = config["v_head_dim"]  # V head dim: 128
    swa_head_dim = config.get("swa_head_dim", head_dim)
    swa_v_head_dim = config.get("swa_v_head_dim", v_head_dim)

    # Get hybrid layer pattern
    hybrid_layer_pattern = config.get("hybrid_layer_pattern", [0] * num_layers)

    # MoE configuration
    num_experts = config["n_routed_experts"]  # 256
    moe_intermediate_size = config["moe_intermediate_size"]
    moe_layer_freq = config.get("moe_layer_freq", [1] * num_layers)

    # Block size for quantization
    quant_config = config.get("quantization_config", {})
    block_size = quant_config.get("weight_block_size", [128, 128])

    print(f"\nModel configuration:", flush=True)
    print(f"  num_layers: {num_layers}", flush=True)
    print(f"  hidden_size: {hidden_size}", flush=True)
    print(f"  num_attention_heads: {num_attention_heads}", flush=True)
    print(f"  num_kv_heads (full): {num_kv_heads}", flush=True)
    print(f"  swa_num_kv_heads (sliding): {swa_num_kv_heads}", flush=True)
    print(f"  head_dim (Q/K): {head_dim}", flush=True)
    print(f"  v_head_dim: {v_head_dim}", flush=True)
    print(f"  num_experts: {num_experts}", flush=True)
    print(f"  moe_intermediate_size: {moe_intermediate_size}", flush=True)
    print(f"  block_size: {block_size}", flush=True)
    print(f"  tp_degree: {tp_degree}", flush=True)
    print(f"  convert_to_mha: {convert_to_mha}", flush=True)

    state_dict_keys = set(state_dict.keys())
    new_state_dict = {}

    # Process each layer
    for layer_idx in range(num_layers):
        print(f"\nProcessing layer {layer_idx}...", end="", flush=True)

        prefix = f"model.layers.{layer_idx}."
        is_sliding_window = hybrid_layer_pattern[layer_idx] == 1
        is_moe_layer = moe_layer_freq[layer_idx] == 1

        # Get layer-specific parameters
        if is_sliding_window:
            layer_num_heads = swa_num_attention_heads
            layer_num_kv_heads = swa_num_kv_heads
            layer_head_dim = swa_head_dim
            layer_v_head_dim = swa_v_head_dim
        else:
            layer_num_heads = num_attention_heads
            layer_num_kv_heads = num_kv_heads
            layer_head_dim = head_dim
            layer_v_head_dim = v_head_dim

        attn_type = "sliding_window" if is_sliding_window else "full"
        print(f" ({attn_type}, kv_heads={layer_num_kv_heads})", end="", flush=True)

        # Process attention weights
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            weight_key = f"{prefix}self_attn.{proj}.weight"
            scale_key = f"{prefix}self_attn.{proj}.weight_scale_inv"

            if weight_key not in state_dict_keys:
                continue

            weight = state_dict[weight_key]
            scale = state_dict.get(scale_key)

            # Rescale FP8 weight for Neuron
            if weight.dtype == torch.float8_e4m3fn and scale is not None:
                weight, scale = rescale_fp8_weight(weight, scale)

            # Apply CONVERT_TO_MHA for K and V projections
            if convert_to_mha and proj in ["k_proj", "v_proj"]:
                if tp_degree > layer_num_kv_heads:
                    proj_head_dim = layer_head_dim if proj == "k_proj" else layer_v_head_dim
                    weight, scale = replicate_for_convert_to_mha(
                        weight, scale,
                        layer_num_kv_heads, layer_num_heads,
                        proj_head_dim, block_size
                    )

            # Save with Neuron naming convention
            new_weight_key = f"layers.{layer_idx}.self_attn.{proj}.weight"
            new_state_dict[new_weight_key] = weight

            if scale is not None:
                new_scale_key = f"layers.{layer_idx}.self_attn.{proj}.scale"
                new_state_dict[new_scale_key] = scale

        # Process layer norms (no FP8)
        for norm in ["input_layernorm", "post_attention_layernorm"]:
            weight_key = f"{prefix}{norm}.weight"
            if weight_key in state_dict_keys:
                new_key = f"layers.{layer_idx}.{norm}.weight"
                new_state_dict[new_key] = state_dict[weight_key]

        # Process MoE router
        router_key = f"{prefix}mlp.gate.weight"
        if router_key in state_dict_keys:
            new_key = f"layers.{layer_idx}.mlp.router.linear_router.weight"
            new_state_dict[new_key] = state_dict[router_key]

        # Process MoE experts
        if is_moe_layer:
            # Prepare fused gate_up and down projections
            gate_weights = []
            gate_scales = []
            up_weights = []
            up_scales = []
            down_weights = []
            down_scales = []

            for expert_idx in range(num_experts):
                expert_prefix = f"{prefix}mlp.experts.{expert_idx}."

                # Gate projection
                gate_w_key = f"{expert_prefix}gate_proj.weight"
                gate_s_key = f"{expert_prefix}gate_proj.weight_scale_inv"

                if gate_w_key in state_dict_keys:
                    gate_w = state_dict[gate_w_key]
                    gate_s = state_dict.get(gate_s_key)

                    if gate_w.dtype == torch.float8_e4m3fn and gate_s is not None:
                        gate_w, gate_s = rescale_fp8_weight(gate_w, gate_s)

                    gate_weights.append(gate_w.T)  # Transpose for fusion
                    if gate_s is not None:
                        gate_scales.append(gate_s)

                # Up projection
                up_w_key = f"{expert_prefix}up_proj.weight"
                up_s_key = f"{expert_prefix}up_proj.weight_scale_inv"

                if up_w_key in state_dict_keys:
                    up_w = state_dict[up_w_key]
                    up_s = state_dict.get(up_s_key)

                    if up_w.dtype == torch.float8_e4m3fn and up_s is not None:
                        up_w, up_s = rescale_fp8_weight(up_w, up_s)

                    up_weights.append(up_w.T)  # Transpose for fusion
                    if up_s is not None:
                        up_scales.append(up_s)

                # Down projection
                down_w_key = f"{expert_prefix}down_proj.weight"
                down_s_key = f"{expert_prefix}down_proj.weight_scale_inv"

                if down_w_key in state_dict_keys:
                    down_w = state_dict[down_w_key]
                    down_s = state_dict.get(down_s_key)

                    if down_w.dtype == torch.float8_e4m3fn and down_s is not None:
                        down_w, down_s = rescale_fp8_weight(down_w, down_s)

                    down_weights.append(down_w.T)  # Transpose for fusion
                    if down_s is not None:
                        down_scales.append(down_s)

            # Fuse gate and up projections
            if gate_weights and up_weights:
                # Stack experts: [num_experts, hidden_size, intermediate_size]
                gate_stacked = torch.stack(gate_weights, dim=0)
                up_stacked = torch.stack(up_weights, dim=0)

                # Concatenate gate and up: [num_experts, hidden_size, 2 * intermediate_size]
                gate_up_fused = torch.cat([gate_stacked, up_stacked], dim=2)

                new_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"
                new_state_dict[new_key] = gate_up_fused

                # Fuse scales if present
                if gate_scales and up_scales:
                    # Scales shape after transpose: [scale_h, scale_w]
                    # After stacking: [num_experts, scale_h, scale_w]
                    gate_s_stacked = torch.stack(gate_scales, dim=0)
                    up_s_stacked = torch.stack(up_scales, dim=0)

                    # Concatenate scales along last dim
                    gate_up_scale = torch.cat([gate_s_stacked, up_s_stacked], dim=-1)

                    new_scale_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.scale"
                    new_state_dict[new_scale_key] = gate_up_scale

            # Down projection
            if down_weights:
                # Stack: [num_experts, intermediate_size, hidden_size]
                down_stacked = torch.stack(down_weights, dim=0)

                new_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight"
                new_state_dict[new_key] = down_stacked

                if down_scales:
                    down_s_stacked = torch.stack(down_scales, dim=0)
                    new_scale_key = f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.scale"
                    new_state_dict[new_scale_key] = down_s_stacked

        gc.collect()
        print(" done", flush=True)

    # Process embeddings and final layer norm
    print("\nProcessing embeddings and final norm...", flush=True)

    if "model.embed_tokens.weight" in state_dict_keys:
        new_state_dict["embed_tokens.weight"] = state_dict["model.embed_tokens.weight"]

    if "model.norm.weight" in state_dict_keys:
        new_state_dict["norm.weight"] = state_dict["model.norm.weight"]

    if "lm_head.weight" in state_dict_keys:
        new_state_dict["lm_head.weight"] = state_dict["lm_head.weight"]
    elif "model.embed_tokens.weight" in state_dict_keys:
        # Tied embeddings
        new_state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    # Save preprocessed checkpoint
    print(f"\nSaving preprocessed checkpoint to: {save_path}", flush=True)
    os.makedirs(save_path, exist_ok=True)

    save_state_dict_safetensors(new_state_dict, save_path)

    # Copy config.json
    import shutil
    shutil.copy(config_path, os.path.join(save_path, "config.json"))

    # Copy tokenizer files
    for tokenizer_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src_path = os.path.join(hf_model_path, tokenizer_file)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(save_path, tokenizer_file))

    print(f"\nPreprocessing complete!", flush=True)
    print(f"  Total parameters: {len(new_state_dict)}", flush=True)

    # Print FP8 weight count
    fp8_count = sum(1 for v in new_state_dict.values() if v.dtype == torch.float8_e4m3fn)
    scale_count = sum(1 for k in new_state_dict.keys() if k.endswith(".scale"))
    print(f"  FP8 weights: {fp8_count}", flush=True)
    print(f"  Scale parameters: {scale_count}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MiMo-V2-Flash FP8 checkpoint for Neuron inference"
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to HuggingFace MiMo-V2-Flash checkpoint",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save preprocessed checkpoint",
    )
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=32,
        help="Tensor parallelism degree (default: 32)",
    )
    parser.add_argument(
        "--convert_to_mha",
        action="store_true",
        default=True,
        help="Replicate K/V for CONVERT_TO_MHA mode (default: True)",
    )
    parser.add_argument(
        "--no_convert_to_mha",
        action="store_false",
        dest="convert_to_mha",
        help="Disable K/V replication",
    )

    args = parser.parse_args()

    process_mimo_v2_checkpoint(
        hf_model_path=args.hf_model_path,
        save_path=args.save_path,
        tp_degree=args.tp_degree,
        convert_to_mha=args.convert_to_mha,
    )


if __name__ == "__main__":
    main()
