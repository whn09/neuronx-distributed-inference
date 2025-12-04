"""
Detailed weight debugging script for MiniMax M2.
Compares original HF weights with loaded Neuron model weights.
"""
import torch
import sys
import os
import json
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

from safetensors import safe_open
from transformers import AutoTokenizer

model_hf_path = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights/"


def load_hf_weight(key):
    """Load a specific weight from HF model."""
    with open(os.path.join(model_hf_path, 'model.safetensors.index.json')) as f:
        index = json.load(f)

    shard_file = index['weight_map'].get(key)
    if not shard_file:
        return None

    with safe_open(os.path.join(model_hf_path, shard_file), framework='pt', device='cpu') as f:
        return f.get_tensor(key)


def compute_expected_output(input_ids):
    """
    Manually compute what the first layer should output.
    This helps verify if the embedding is correct.
    """
    print("\n" + "="*60)
    print("Manual embedding computation")
    print("="*60)

    # Load embedding weights
    embed_weight = load_hf_weight('model.embed_tokens.weight')
    if embed_weight is None:
        print("ERROR: Could not load embedding weight")
        return

    print(f"Embedding weight shape: {embed_weight.shape}")
    print(f"Embedding weight dtype: {embed_weight.dtype}")

    # Get embedding for our input token
    token_id = input_ids[0, 0].item()
    print(f"\nInput token ID: {token_id}")

    embedding = embed_weight[token_id]
    print(f"Embedding for token {token_id}:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Dtype: {embedding.dtype}")
    print(f"  First 10 values: {embedding[:10].float().tolist()}")
    print(f"  Mean: {embedding.float().mean().item():.6f}")
    print(f"  Std: {embedding.float().std().item():.6f}")

    # Also check a few reference tokens
    print("\n--- Reference token embeddings ---")
    ref_tokens = [0, 1, 2, 911, 367]  # Special tokens and the predicted ones
    for tok in ref_tokens:
        emb = embed_weight[tok]
        print(f"Token {tok}: mean={emb.float().mean().item():.6f}, std={emb.float().std().item():.6f}")

    return embedding


def check_lm_head():
    """Check lm_head weights."""
    print("\n" + "="*60)
    print("LM Head weight analysis")
    print("="*60)

    lm_head = load_hf_weight('lm_head.weight')
    if lm_head is None:
        print("ERROR: Could not load lm_head weight")
        return

    print(f"LM Head weight shape: {lm_head.shape}")
    print(f"LM Head weight dtype: {lm_head.dtype}")

    # Check specific token rows
    print("\n--- LM head row analysis ---")
    # These are the tokens that were predicted
    tokens_to_check = [
        (911, '()'),
        (367, '\\n\\n'),
        (10906, '()\\n\\n'),
        (2675, '()\\n'),
        (343, ' I'),  # Expected response to "Hello"
        (8696, "'m"),
    ]

    for tok_id, tok_str in tokens_to_check:
        row = lm_head[tok_id]
        print(f"Token {tok_id} '{tok_str}': mean={row.float().mean().item():.6f}, std={row.float().std().item():.6f}, first5={row[:5].float().tolist()}")


def check_layer0_attention():
    """Check layer 0 attention weights."""
    print("\n" + "="*60)
    print("Layer 0 attention weight analysis")
    print("="*60)

    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        key = f'model.layers.0.self_attn.{proj}.weight'
        weight = load_hf_weight(key)
        if weight is not None:
            print(f"{proj}: shape={weight.shape}, dtype={weight.dtype}")
            print(f"  mean={weight.float().mean().item():.6f}, std={weight.float().std().item():.6f}")
        else:
            print(f"{proj}: NOT FOUND")

    # Check qk_norm
    for norm in ['q_norm', 'k_norm']:
        key = f'model.layers.0.self_attn.{norm}.weight'
        weight = load_hf_weight(key)
        if weight is not None:
            print(f"{norm}: shape={weight.shape}, dtype={weight.dtype}")
            print(f"  mean={weight.float().mean().item():.6f}, std={weight.float().std().item():.6f}")
            print(f"  first 10 values: {weight[:10].float().tolist()}")


def check_layer0_moe():
    """Check layer 0 MoE weights."""
    print("\n" + "="*60)
    print("Layer 0 MoE weight analysis")
    print("="*60)

    # Router gate
    gate = load_hf_weight('model.layers.0.block_sparse_moe.gate.weight')
    if gate is not None:
        print(f"Router gate: shape={gate.shape}, dtype={gate.dtype}")
        print(f"  mean={gate.float().mean().item():.6f}, std={gate.float().std().item():.6f}")

    # Bias
    bias = load_hf_weight('model.layers.0.block_sparse_moe.e_score_correction_bias')
    if bias is not None:
        print(f"e_score_correction_bias: shape={bias.shape}, dtype={bias.dtype}")
        print(f"  mean={bias.float().mean().item():.6f}, std={bias.float().std().item():.6f}")
        print(f"  first 5 values: {bias[:5].tolist()}")

    # Expert 0
    for proj in ['w1', 'w2', 'w3']:
        key = f'model.layers.0.block_sparse_moe.experts.0.{proj}.weight'
        weight = load_hf_weight(key)
        if weight is not None:
            print(f"Expert 0 {proj}: shape={weight.shape}, dtype={weight.dtype}")
            print(f"  mean={weight.float().mean().item():.6f}, std={weight.float().std().item():.6f}")


def test_simple_forward():
    """
    Test a simple manual forward pass through embedding only.
    This doesn't require the full Neuron model.
    """
    print("\n" + "="*60)
    print("Simple manual forward test")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(model_hf_path)

    # Input
    text = "Hello"
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids

    print(f"Input text: '{text}'")
    print(f"Input token IDs: {input_ids[0].tolist()}")

    # Get embedding
    embedding = compute_expected_output(input_ids)

    if embedding is None:
        return

    # Now compute what the lm_head would output for this embedding
    # (this is a VERY simplified test - ignores all transformer layers)
    print("\n--- Direct embedding -> lm_head test ---")
    print("NOTE: This ignores all transformer layers, just checks embedding/lm_head alignment")

    lm_head = load_hf_weight('lm_head.weight')
    if lm_head is None:
        print("ERROR: Could not load lm_head")
        return

    # Simple matmul: embedding @ lm_head.T
    # This would only work if the model was identity (which it's not)
    # But it can help verify the shapes are correct
    logits = torch.matmul(embedding.float().unsqueeze(0), lm_head.float().T)
    print(f"Direct logits shape: {logits.shape}")

    # Top predictions from direct embedding (won't be meaningful but shows shape is correct)
    top_logits, top_indices = torch.topk(logits[0], 10)
    print(f"Top 10 tokens from direct embedding -> lm_head (meaningless but shows alignment):")
    for i, (logit, idx) in enumerate(zip(top_logits.tolist(), top_indices.tolist())):
        token_str = tokenizer.decode([idx])
        print(f"  {i+1}. Token '{token_str}' (ID={idx}, logit={logit:.2f})")


def check_norm_weights():
    """Check normalization layer weights."""
    print("\n" + "="*60)
    print("Normalization weight analysis")
    print("="*60)

    # Input layernorm
    input_ln = load_hf_weight('model.layers.0.input_layernorm.weight')
    if input_ln is not None:
        print(f"input_layernorm: shape={input_ln.shape}, dtype={input_ln.dtype}")
        print(f"  mean={input_ln.float().mean().item():.6f}, std={input_ln.float().std().item():.6f}")
        print(f"  min={input_ln.float().min().item():.6f}, max={input_ln.float().max().item():.6f}")
        print(f"  first 10: {input_ln[:10].float().tolist()}")

    # Post attention layernorm
    post_attn_ln = load_hf_weight('model.layers.0.post_attention_layernorm.weight')
    if post_attn_ln is not None:
        print(f"post_attention_layernorm: shape={post_attn_ln.shape}, dtype={post_attn_ln.dtype}")
        print(f"  mean={post_attn_ln.float().mean().item():.6f}, std={post_attn_ln.float().std().item():.6f}")

    # Final norm
    norm = load_hf_weight('model.norm.weight')
    if norm is not None:
        print(f"model.norm: shape={norm.shape}, dtype={norm.dtype}")
        print(f"  mean={norm.float().mean().item():.6f}, std={norm.float().std().item():.6f}")


if __name__ == "__main__":
    check_norm_weights()
    check_layer0_attention()
    check_layer0_moe()
    check_lm_head()
    test_simple_forward()
