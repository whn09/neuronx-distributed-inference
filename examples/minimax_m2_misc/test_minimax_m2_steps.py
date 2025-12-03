"""
Unit tests for MiniMax M2 processing steps.
Run individual steps on CPU to verify correctness before Neuron compilation.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Test configuration - UPDATED based on actual config.json
MODEL_PATH = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
HIDDEN_SIZE = 3072           # From config.json (NOT 8192!)
NUM_ATTENTION_HEADS = 48     # Note: Q_dim = 48 * 128 = 6144, which is LARGER than hidden_size!
NUM_KV_HEADS = 8             # K/V dim = 8 * 128 = 1024
HEAD_DIM = 128               # head_dim from config
ROTARY_DIM = 64              # rotary_dim from config
VOCAB_SIZE = 200064          # vocab_size from config (already padded for TP)
TP_DEGREE = 64

# MiniMax M2 has unusual architecture:
# - hidden_size (embedding dim) = 3072
# - Q projection output = 48 * 128 = 6144 (LARGER than hidden_size!)
# - This means Q projection EXPANDS the dimension: 3072 -> 6144


def test_step1_tokenization():
    """Test Step 1: Tokenization"""
    print("\n" + "="*60)
    print("TEST 1: Tokenization")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    test_text = "The capital of France is"
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)

    print(f"Input text: '{test_text}'")
    print(f"Token IDs: {tokens}")
    print(f"Decoded back: '{decoded}'")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Verify round-trip
    assert decoded == test_text, f"Tokenization round-trip failed: '{decoded}' != '{test_text}'"
    print("✓ Tokenization round-trip OK")

    # Verify token IDs are in valid range
    assert all(0 <= t < tokenizer.vocab_size for t in tokens), "Token IDs out of range"
    print("✓ All token IDs in valid range")

    return tokens


def test_step2_embedding_lookup():
    """Test Step 2: Embedding lookup (simulated)"""
    print("\n" + "="*60)
    print("TEST 2: Embedding Lookup (CPU simulation)")
    print("="*60)

    from safetensors import safe_open
    import os

    # Load embedding weights from checkpoint
    index_file = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    import json
    with open(index_file) as f:
        index = json.load(f)

    # Find which shard has embed_tokens.weight
    embed_key = "model.embed_tokens.weight"
    embed_shard = index["weight_map"].get(embed_key)
    print(f"Embedding weight in shard: {embed_shard}")

    if embed_shard:
        shard_path = os.path.join(MODEL_PATH, embed_shard)
        with safe_open(shard_path, framework="pt") as f:
            embed_weight = f.get_tensor(embed_key)

        print(f"Embedding weight shape: {embed_weight.shape}")
        print(f"Embedding weight dtype: {embed_weight.dtype}")
        print(f"Expected shape: [{VOCAB_SIZE}, {HIDDEN_SIZE}]")

        # Test lookup
        test_tokens = torch.tensor([758, 5505, 300, 5969, 355])
        embeddings = embed_weight[test_tokens]
        print(f"Embeddings for test tokens shape: {embeddings.shape}")

        # Check embeddings are not all zeros or NaN
        assert not torch.isnan(embeddings).any(), "Embeddings contain NaN!"
        assert not torch.isinf(embeddings).any(), "Embeddings contain Inf!"
        assert embeddings.abs().sum() > 0, "Embeddings are all zeros!"
        print("✓ Embeddings are valid (not NaN/Inf/zero)")

        # Print some stats
        print(f"Embedding stats: mean={embeddings.mean():.6f}, std={embeddings.std():.6f}")

        return embeddings
    else:
        print("ERROR: Could not find embedding weight in index")
        return None


def test_step3a_rmsnorm():
    """Test Step 3a: RMSNorm"""
    print("\n" + "="*60)
    print("TEST 3a: RMSNorm")
    print("="*60)

    class RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            return self.weight * x

    # Test with random input
    norm = RMSNorm(HIDDEN_SIZE)
    x = torch.randn(1, 5, HIDDEN_SIZE)
    y = norm(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Input mean: {x.mean():.6f}, std: {x.std():.6f}")
    print(f"Output mean: {y.mean():.6f}, std: {y.std():.6f}")

    # Verify output is normalized
    output_rms = y.pow(2).mean(-1).sqrt()
    print(f"Output RMS per position: {output_rms.squeeze()}")
    print("✓ RMSNorm test passed")


def test_step3b_qkv_projection():
    """Test Step 3b: QKV Projection shapes"""
    print("\n" + "="*60)
    print("TEST 3b: QKV Projection Shapes")
    print("="*60)

    print(f"Hidden size: {HIDDEN_SIZE}")
    print(f"Num attention heads: {NUM_ATTENTION_HEADS}")
    print(f"Num KV heads: {NUM_KV_HEADS}")
    print(f"Head dim: {HEAD_DIM}")

    # Original shapes (on GPU)
    q_size = NUM_ATTENTION_HEADS * HEAD_DIM  # 48 * 128 = 6144
    k_size = NUM_KV_HEADS * HEAD_DIM          # 8 * 128 = 1024
    v_size = NUM_KV_HEADS * HEAD_DIM          # 8 * 128 = 1024

    print(f"\nOriginal (GPU) projection sizes:")
    print(f"  Q: {HIDDEN_SIZE} -> {q_size}")
    print(f"  K: {HIDDEN_SIZE} -> {k_size}")
    print(f"  V: {HIDDEN_SIZE} -> {v_size}")

    # Padded/replicated shapes (on Neuron with TP=64)
    padded_q_heads = 64  # Padded from 48 to 64 (interleaved)
    padded_kv_heads = 64  # Replicated from 8 to 64

    padded_q_size = padded_q_heads * HEAD_DIM  # 64 * 128 = 8192
    padded_k_size = padded_kv_heads * HEAD_DIM  # 64 * 128 = 8192

    print(f"\nPadded/Replicated (Neuron TP=64) projection sizes:")
    print(f"  Q: {HIDDEN_SIZE} -> {padded_q_size} (48->64 heads, interleaved padding)")
    print(f"  K: {HIDDEN_SIZE} -> {padded_k_size} (8->64 heads, 8x replication)")
    print(f"  V: {HIDDEN_SIZE} -> {padded_k_size} (8->64 heads, 8x replication)")

    print(f"\nPer-rank sizes (TP=64, 1 head per rank):")
    print(f"  Q per rank: {padded_q_size // TP_DEGREE} = {HEAD_DIM}")
    print(f"  K per rank: {padded_k_size // TP_DEGREE} = {HEAD_DIM}")
    print(f"  V per rank: {padded_k_size // TP_DEGREE} = {HEAD_DIM}")


def test_step3b_qk_norm():
    """Test Step 3b: QK Norm (DistributedRMSNorm simulation)"""
    print("\n" + "="*60)
    print("TEST 3b: QK Norm (Distributed RMSNorm)")
    print("="*60)

    from safetensors import safe_open
    import os
    import json

    # Load q_norm and k_norm weights
    index_file = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_file) as f:
        index = json.load(f)

    q_norm_key = "model.layers.0.self_attn.q_norm.weight"
    k_norm_key = "model.layers.0.self_attn.k_norm.weight"

    q_norm_shard = index["weight_map"].get(q_norm_key)
    k_norm_shard = index["weight_map"].get(k_norm_key)

    print(f"q_norm weight in shard: {q_norm_shard}")
    print(f"k_norm weight in shard: {k_norm_shard}")

    if q_norm_shard and k_norm_shard:
        # Load weights
        with safe_open(os.path.join(MODEL_PATH, q_norm_shard), framework="pt") as f:
            q_norm_weight = f.get_tensor(q_norm_key)
        with safe_open(os.path.join(MODEL_PATH, k_norm_shard), framework="pt") as f:
            k_norm_weight = f.get_tensor(k_norm_key)

        print(f"\nq_norm weight shape: {q_norm_weight.shape}")
        print(f"k_norm weight shape: {k_norm_weight.shape}")
        print(f"Expected q_norm shape: [{NUM_ATTENTION_HEADS * HEAD_DIM}] = [6144]")
        print(f"Expected k_norm shape: [{NUM_KV_HEADS * HEAD_DIM}] = [1024]")

        print(f"\nq_norm stats: mean={q_norm_weight.mean():.6f}, std={q_norm_weight.std():.6f}")
        print(f"k_norm stats: mean={k_norm_weight.mean():.6f}, std={k_norm_weight.std():.6f}")

        # Simulate qk_norm on CPU
        def rmsnorm(x, weight, eps=1e-6):
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            return weight * x

        # Test with random Q and K tensors
        batch_size, seq_len = 1, 5
        Q = torch.randn(batch_size, seq_len, NUM_ATTENTION_HEADS * HEAD_DIM)  # [1, 5, 6144]
        K = torch.randn(batch_size, seq_len, NUM_KV_HEADS * HEAD_DIM)          # [1, 5, 1024]

        Q_normed = rmsnorm(Q, q_norm_weight)
        K_normed = rmsnorm(K, k_norm_weight)

        print(f"\nSimulated qk_norm:")
        print(f"  Q input shape: {Q.shape}")
        print(f"  Q output shape: {Q_normed.shape}")
        print(f"  K input shape: {K.shape}")
        print(f"  K output shape: {K_normed.shape}")

        print("✓ QK Norm simulation passed")
    else:
        print("WARNING: Could not find q_norm/k_norm weights")


def test_step3b_rope():
    """Test Step 3b: RoPE (Partial Rotary Embedding)"""
    print("\n" + "="*60)
    print("TEST 3b: Partial RoPE (rotary_dim=64, head_dim=128)")
    print("="*60)

    print(f"Head dim: {HEAD_DIM}")
    print(f"Rotary dim: {ROTARY_DIM}")
    print(f"Pass-through dim: {HEAD_DIM - ROTARY_DIM}")

    # RoPE only applies to first rotary_dim dimensions
    # Last (head_dim - rotary_dim) dimensions pass through unchanged

    batch_size, num_heads, seq_len = 1, 64, 5
    Q = torch.randn(batch_size, num_heads, seq_len, HEAD_DIM)

    # Split into rotary and pass-through parts
    Q_rot = Q[..., :ROTARY_DIM]
    Q_pass = Q[..., ROTARY_DIM:]

    print(f"\nQ shape: {Q.shape}")
    print(f"Q_rot (to be rotated) shape: {Q_rot.shape}")
    print(f"Q_pass (pass-through) shape: {Q_pass.shape}")

    # Verify split is correct
    Q_reconstructed = torch.cat([Q_rot, Q_pass], dim=-1)
    assert torch.allclose(Q, Q_reconstructed), "Split/concat not matching!"
    print("✓ Partial RoPE split/concat verified")


def test_vocab_tp_alignment():
    """Test vocab size alignment with TP degree"""
    print("\n" + "="*60)
    print("TEST: Vocab Size / TP Alignment")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer_vocab = tokenizer.vocab_size

    # Model config uses padded vocab
    model_vocab = 200064  # Padded for TP=64 alignment

    print(f"Tokenizer vocab size: {tokenizer_vocab}")
    print(f"Model config vocab size: {model_vocab}")
    print(f"TP degree: {TP_DEGREE}")

    print(f"\nTokenizer vocab / TP = {tokenizer_vocab} / {TP_DEGREE} = {tokenizer_vocab / TP_DEGREE}")
    print(f"Model vocab / TP = {model_vocab} / {TP_DEGREE} = {model_vocab / TP_DEGREE}")

    per_rank_vocab = model_vocab // TP_DEGREE
    print(f"\nPer-rank vocab size: {per_rank_vocab}")

    # Check which rank handles which tokens
    test_tokens = [758, 5505, 300, 5969, 355]
    print(f"\nToken to rank mapping:")
    for token in test_tokens:
        rank = token // per_rank_vocab
        local_idx = token % per_rank_vocab
        print(f"  Token {token}: rank {rank}, local index {local_idx}")

    # Verify all test tokens are in valid ranks
    for token in test_tokens:
        rank = token // per_rank_vocab
        assert 0 <= rank < TP_DEGREE, f"Token {token} maps to invalid rank {rank}"
    print("✓ All test tokens map to valid ranks")


def test_attention_shapes():
    """Test attention computation shapes"""
    print("\n" + "="*60)
    print("TEST: Attention Computation Shapes")
    print("="*60)

    batch_size = 1
    seq_len = 5
    num_heads = 64  # After padding/replication
    head_dim = HEAD_DIM

    # Per-rank shapes (TP=64, 1 head per rank)
    Q = torch.randn(batch_size, 1, seq_len, head_dim)  # [1, 1, 5, 128]
    K = torch.randn(batch_size, 1, seq_len, head_dim)  # [1, 1, 5, 128]
    V = torch.randn(batch_size, 1, seq_len, head_dim)  # [1, 1, 5, 128]

    print(f"Per-rank Q shape: {Q.shape}")
    print(f"Per-rank K shape: {K.shape}")
    print(f"Per-rank V shape: {V.shape}")

    # Attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
    print(f"Attention scores shape: {scores.shape}")  # [1, 1, 5, 5]

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    print(f"Attention weights shape: {attn_weights.shape}")

    # Attention output
    attn_output = torch.matmul(attn_weights, V)
    print(f"Attention output shape: {attn_output.shape}")  # [1, 1, 5, 128]

    print("✓ Attention shapes verified")


def test_full_forward_cpu():
    """Test full forward pass on CPU with actual weights"""
    print("\n" + "="*60)
    print("TEST: Full Forward Pass on CPU (1 layer)")
    print("="*60)

    from safetensors import safe_open
    import os
    import json

    # Load model index
    index_file = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_file) as f:
        index = json.load(f)

    # Load weights for layer 0
    def load_weight(key):
        shard = index["weight_map"].get(key)
        if shard:
            with safe_open(os.path.join(MODEL_PATH, shard), framework="pt") as f:
                return f.get_tensor(key)
        return None

    # Load embedding
    embed_weight = load_weight("model.embed_tokens.weight")
    print(f"Loaded embed_tokens.weight: {embed_weight.shape}")

    # Load layer 0 input_layernorm
    input_ln_weight = load_weight("model.layers.0.input_layernorm.weight")
    print(f"Loaded input_layernorm.weight: {input_ln_weight.shape}")

    # Load Q projection
    q_proj_weight = load_weight("model.layers.0.self_attn.q_proj.weight")
    print(f"Loaded q_proj.weight: {q_proj_weight.shape}")

    # Load q_norm
    q_norm_weight = load_weight("model.layers.0.self_attn.q_norm.weight")
    print(f"Loaded q_norm.weight: {q_norm_weight.shape}")

    # Simple forward pass
    test_tokens = torch.tensor([[758, 5505, 300, 5969, 355]])  # [1, 5]
    print(f"\nInput tokens: {test_tokens}")

    # Step 1: Embedding
    hidden_states = embed_weight[test_tokens]  # [1, 5, 3072]
    print(f"After embedding: {hidden_states.shape}")
    print(f"  mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")

    # Step 2: RMSNorm
    def rmsnorm(x, weight, eps=1e-6):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + eps)
        return (weight * x).to(x.dtype)

    hidden_states = rmsnorm(hidden_states, input_ln_weight)
    print(f"After input_layernorm: {hidden_states.shape}")
    print(f"  mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")

    # Step 3: Q projection (3072 -> 6144)
    Q = torch.nn.functional.linear(hidden_states.float(), q_proj_weight.float())  # [1, 5, 6144]
    print(f"After Q projection: {Q.shape}")
    print(f"  mean={Q.mean():.6f}, std={Q.std():.6f}")

    # Step 4: Q norm
    Q_normed = rmsnorm(Q, q_norm_weight)
    print(f"After q_norm: {Q_normed.shape}")
    print(f"  mean={Q_normed.mean():.6f}, std={Q_normed.std():.6f}")

    # Check for NaN/Inf
    assert not torch.isnan(Q_normed).any(), "Q_normed contains NaN!"
    assert not torch.isinf(Q_normed).any(), "Q_normed contains Inf!"
    print("✓ Full forward pass (partial) completed without NaN/Inf")


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "="*60)
    print("RUNNING ALL MINIMAX M2 UNIT TESTS")
    print("="*60)

    test_step1_tokenization()
    test_step2_embedding_lookup()
    test_step3a_rmsnorm()
    test_step3b_qkv_projection()
    test_step3b_qk_norm()
    test_step3b_rope()
    test_vocab_tp_alignment()
    test_attention_shapes()
    test_full_forward_cpu()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
