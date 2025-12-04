"""
Compare MiniMax M2 embedding + first layer output between HF (CPU) and Neuron.
This helps identify if the issue is in weight loading or computation.
"""
import torch
import sys
import os

model_path = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
sys.path.insert(0, model_path)
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

from safetensors import safe_open
import json


def load_hf_weight(key):
    """Load a specific weight from HF model."""
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    with open(index_path) as f:
        index = json.load(f)

    shard_file = index['weight_map'].get(key)
    if not shard_file:
        return None

    with safe_open(os.path.join(model_path, shard_file), framework='pt', device='cpu') as f:
        return f.get_tensor(key)


def test_embedding():
    """Test embedding lookup."""
    print("\n" + "="*60)
    print("Testing embedding lookup")
    print("="*60)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Input
    text = "Hello"
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids

    print(f"Input text: '{text}'")
    print(f"Input token IDs: {input_ids[0].tolist()}")

    # Load embedding weight
    embed_weight = load_hf_weight('model.embed_tokens.weight')
    print(f"\nEmbedding weight shape: {embed_weight.shape}")

    # Get embedding for input token
    token_id = input_ids[0, 0].item()
    embedding = embed_weight[token_id]
    print(f"\nExpected embedding for token {token_id}:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Mean: {embedding.float().mean().item():.6f}")
    print(f"  Std: {embedding.float().std().item():.6f}")
    print(f"  First 10: {embedding[:10].float().tolist()}")

    return embedding, token_id


def test_input_layernorm(embedding):
    """Test input layernorm on the embedding."""
    print("\n" + "="*60)
    print("Testing input layernorm (layer 0)")
    print("="*60)

    # Load input_layernorm weight
    ln_weight = load_hf_weight('model.layers.0.input_layernorm.weight')
    print(f"Input layernorm weight shape: {ln_weight.shape}")
    print(f"Input layernorm weight first 10: {ln_weight[:10].float().tolist()}")

    # Apply RMSNorm manually
    # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    hidden_states = embedding.unsqueeze(0).unsqueeze(0).float()  # [1, 1, 3072]
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
    normalized = ln_weight.float() * hidden_states

    print(f"\nAfter input_layernorm:")
    print(f"  Shape: {normalized.shape}")
    print(f"  Mean: {normalized.mean().item():.6f}")
    print(f"  Std: {normalized.std().item():.6f}")
    print(f"  First 10: {normalized[0, 0, :10].tolist()}")

    return normalized.squeeze(0)


def test_qkv_projection(hidden_states):
    """Test Q, K, V projections."""
    print("\n" + "="*60)
    print("Testing Q, K, V projections (layer 0)")
    print("="*60)

    # Load Q, K, V projection weights
    q_weight = load_hf_weight('model.layers.0.self_attn.q_proj.weight')
    k_weight = load_hf_weight('model.layers.0.self_attn.k_proj.weight')
    v_weight = load_hf_weight('model.layers.0.self_attn.v_proj.weight')

    print(f"Q projection weight shape: {q_weight.shape}")  # [6144, 3072]
    print(f"K projection weight shape: {k_weight.shape}")  # [1024, 3072]
    print(f"V projection weight shape: {v_weight.shape}")  # [1024, 3072]

    # Apply projections (no bias)
    hidden_states_float = hidden_states.float()
    Q = torch.matmul(hidden_states_float, q_weight.T.float())
    K = torch.matmul(hidden_states_float, k_weight.T.float())
    V = torch.matmul(hidden_states_float, v_weight.T.float())

    print(f"\nQ output shape: {Q.shape}")  # [1, 6144]
    print(f"Q mean: {Q.mean().item():.6f}, std: {Q.std().item():.6f}")
    print(f"Q first 10: {Q[0, :10].tolist()}")

    print(f"\nK output shape: {K.shape}")  # [1, 1024]
    print(f"K mean: {K.mean().item():.6f}, std: {K.std().item():.6f}")
    print(f"K first 10: {K[0, :10].tolist()}")

    return Q, K, V


def test_qk_norm(Q, K):
    """Test MiniMax M2's qk_norm (RMSNorm on full Q/K before reshape)."""
    print("\n" + "="*60)
    print("Testing qk_norm (layer 0)")
    print("="*60)

    # Load qk_norm weights
    q_norm_weight = load_hf_weight('model.layers.0.self_attn.q_norm.weight')
    k_norm_weight = load_hf_weight('model.layers.0.self_attn.k_norm.weight')

    print(f"q_norm weight shape: {q_norm_weight.shape}")  # [6144]
    print(f"k_norm weight shape: {k_norm_weight.shape}")  # [1024]

    # Apply RMSNorm on full Q output
    Q_float = Q.float()
    Q_variance = Q_float.pow(2).mean(-1, keepdim=True)
    Q_normalized = Q_float * torch.rsqrt(Q_variance + 1e-6)
    Q_normed = q_norm_weight.float() * Q_normalized

    print(f"\nQ after qk_norm:")
    print(f"  Mean: {Q_normed.mean().item():.6f}")
    print(f"  Std: {Q_normed.std().item():.6f}")
    print(f"  First 10: {Q_normed[0, :10].tolist()}")

    # Apply RMSNorm on full K output
    K_float = K.float()
    K_variance = K_float.pow(2).mean(-1, keepdim=True)
    K_normalized = K_float * torch.rsqrt(K_variance + 1e-6)
    K_normed = k_norm_weight.float() * K_normalized

    print(f"\nK after qk_norm:")
    print(f"  Mean: {K_normed.mean().item():.6f}")
    print(f"  Std: {K_normed.std().item():.6f}")
    print(f"  First 10: {K_normed[0, :10].tolist()}")

    return Q_normed, K_normed


def test_full_forward():
    """Test full forward pass using HuggingFace model on CPU (if memory allows)."""
    print("\n" + "="*60)
    print("Testing full forward pass on CPU")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Try to load just embedding and first layer
        print("Loading model (this may OOM for such a large model)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
            # num_hidden_layers=1,  # Only load 1 layer
        )

        text = "Hello"
        inputs = tokenizer([text], return_tensors="pt")

        with torch.no_grad():
            outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask)
            logits = outputs.logits[0, -1, :]

        top_logits, top_indices = torch.topk(logits.float(), 10)
        print(f"\nHF CPU model top 10 predictions for '{text}':")
        for i, (logit, idx) in enumerate(zip(top_logits.tolist(), top_indices.tolist())):
            token_str = tokenizer.decode([idx])
            print(f"  {i+1}. '{token_str}' (ID={idx}, logit={logit:.2f})")

    except Exception as e:
        print(f"Cannot run full forward pass: {e}")
        print("This is expected for such a large model without GPU.")


if __name__ == "__main__":
    embedding, token_id = test_embedding()
    normalized = test_input_layernorm(embedding)
    Q, K, V = test_qkv_projection(normalized)
    Q_normed, K_normed = test_qk_norm(Q, K)

    # Don't run full forward - model is too large for CPU
    # test_full_forward()
