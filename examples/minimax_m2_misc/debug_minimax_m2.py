"""
Debug script to check if MiniMax M2 model weights are loaded correctly.
This script compares the embedding output from the traced model with expected values.
"""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

from transformers import AutoTokenizer
from safetensors import safe_open
import os
import json

model_path = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights/"

def check_original_weights():
    """Load and check original model weights directly from safetensors."""
    print("\n=== Checking Original Model Weights ===")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Test input
    text = "Who are you?"
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids
    print(f"Input text: '{text}'")
    print(f"Input token IDs: {input_ids[0].tolist()}")

    # Load embedding weights from safetensors
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    with open(index_path, 'r') as f:
        index = json.load(f)

    # Find embed_tokens weight
    embed_file = index['weight_map'].get('model.embed_tokens.weight')
    print(f"\nEmbed tokens weight file: {embed_file}")

    if embed_file:
        shard_path = os.path.join(model_path, embed_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            embed_weight = f.get_tensor('model.embed_tokens.weight')

        print(f"Embedding weight shape: {embed_weight.shape}")
        print(f"Embedding weight dtype: {embed_weight.dtype}")

        # Check embedding for input tokens
        print("\n=== Embedding values for input tokens ===")
        for i, token_id in enumerate(input_ids[0].tolist()):
            token_str = tokenizer.decode([token_id])
            embedding = embed_weight[token_id]
            print(f"Token {i}: '{token_str}' (ID={token_id})")
            print(f"  Embedding stats: mean={embedding.float().mean():.6f}, std={embedding.float().std():.6f}")
            print(f"  First 5 values: {embedding[:5].tolist()}")

    # Also check lm_head weight
    lm_head_file = index['weight_map'].get('lm_head.weight')
    print(f"\n=== LM Head Weight ===")
    print(f"LM head weight file: {lm_head_file}")

    if lm_head_file:
        shard_path = os.path.join(model_path, lm_head_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            lm_head_weight = f.get_tensor('lm_head.weight')

        print(f"LM head weight shape: {lm_head_weight.shape}")
        print(f"LM head weight dtype: {lm_head_weight.dtype}")
        print(f"LM head weight stats: mean={lm_head_weight.float().mean():.6f}, std={lm_head_weight.float().std():.6f}")


def check_layer0_attention_weights():
    """Check layer 0 attention weights."""
    print("\n=== Checking Layer 0 Attention Weights ===")

    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    with open(index_path, 'r') as f:
        index = json.load(f)

    # Check Q, K, V projection weights
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        key = f'model.layers.0.self_attn.{proj}.weight'
        shard_file = index['weight_map'].get(key)
        if shard_file:
            shard_path = os.path.join(model_path, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                weight = f.get_tensor(key)
            print(f"{proj}: shape={weight.shape}, dtype={weight.dtype}, mean={weight.float().mean():.6f}, std={weight.float().std():.6f}")

    # Check qk_norm weights
    for norm in ['q_norm', 'k_norm']:
        key = f'model.layers.0.self_attn.{norm}.weight'
        shard_file = index['weight_map'].get(key)
        if shard_file:
            shard_path = os.path.join(model_path, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                weight = f.get_tensor(key)
            print(f"{norm}: shape={weight.shape}, dtype={weight.dtype}, mean={weight.float().mean():.6f}, std={weight.float().std():.6f}")


def check_moe_weights():
    """Check MoE layer weights."""
    print("\n=== Checking Layer 0 MoE Weights ===")

    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    with open(index_path, 'r') as f:
        index = json.load(f)

    # Check gate weight
    gate_key = 'model.layers.0.block_sparse_moe.gate.weight'
    shard_file = index['weight_map'].get(gate_key)
    if shard_file:
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            weight = f.get_tensor(gate_key)
        print(f"Router gate: shape={weight.shape}, dtype={weight.dtype}, mean={weight.float().mean():.6f}")

    # Check e_score_correction_bias
    bias_key = 'model.layers.0.block_sparse_moe.e_score_correction_bias'
    shard_file = index['weight_map'].get(bias_key)
    if shard_file:
        shard_path = os.path.join(model_path, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            weight = f.get_tensor(bias_key)
        print(f"e_score_correction_bias: shape={weight.shape}, dtype={weight.dtype}, mean={weight.float().mean():.6f}")

    # Check expert 0 weights
    for proj in ['w1', 'w2', 'w3']:
        key = f'model.layers.0.block_sparse_moe.experts.0.{proj}.weight'
        shard_file = index['weight_map'].get(key)
        if shard_file:
            shard_path = os.path.join(model_path, shard_file)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                weight = f.get_tensor(key)
            print(f"Expert 0 {proj}: shape={weight.shape}, dtype={weight.dtype}, mean={weight.float().mean():.6f}")


def simple_forward_test():
    """
    Test a simple forward pass through embedding only.
    This doesn't require the full model to be loaded.
    """
    print("\n=== Simple Embedding Forward Test ===")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load embedding weights
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    with open(index_path, 'r') as f:
        index = json.load(f)

    embed_file = index['weight_map'].get('model.embed_tokens.weight')
    shard_path = os.path.join(model_path, embed_file)
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        embed_weight = f.get_tensor('model.embed_tokens.weight')

    # Create embedding layer
    embed = torch.nn.Embedding.from_pretrained(embed_weight, freeze=True)

    # Test input
    text = "Who are you?"
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids

    # Get embeddings
    embeddings = embed(input_ids)
    print(f"Input: '{text}'")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings mean: {embeddings.float().mean():.6f}")
    print(f"Embeddings std: {embeddings.float().std():.6f}")
    print(f"First token embedding (first 10 values): {embeddings[0, 0, :10].tolist()}")


if __name__ == "__main__":
    check_original_weights()
    check_layer0_attention_weights()
    check_moe_weights()
    simple_forward_test()
