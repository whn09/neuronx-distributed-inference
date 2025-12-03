"""
Compare CPU vs Neuron embedding output.

Since we can't easily hook into compiled Neuron models,
this script takes a different approach:
1. Load original HF model on CPU (small test with 1 layer if needed)
2. Compare token IDs and verify embedding is correct on CPU
3. Then analyze what could go wrong in Neuron's ParallelEmbedding

For actual Neuron debugging, we need to add debug output during compilation.
"""
import torch
from transformers import AutoTokenizer
from safetensors import safe_open
import os
import json

MODEL_PATH = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
TEST_TEXT = "The capital of France is"


def load_embedding_weight():
    """Load embedding weights from checkpoint."""
    index_file = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_file) as f:
        index = json.load(f)

    embed_key = "model.embed_tokens.weight"
    shard = index["weight_map"][embed_key]
    with safe_open(os.path.join(MODEL_PATH, shard), framework="pt") as f:
        return f.get_tensor(embed_key)


def simulate_parallel_embedding(embed_weight, token_ids, tp_degree=64):
    """
    Simulate how ParallelEmbedding with shard_across_embedding=True works.

    With shard_across_embedding=True:
    - Vocab is sharded across TP ranks
    - Each rank handles vocab_size/tp_degree tokens
    - Lookup: only the owning rank returns non-zero embedding
    - All-reduce sum gathers the result
    """
    vocab_size, hidden_size = embed_weight.shape
    per_rank_vocab = vocab_size // tp_degree

    print(f"\nSimulating ParallelEmbedding:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  tp_degree: {tp_degree}")
    print(f"  per_rank_vocab: {per_rank_vocab}")

    # For each token, determine which rank owns it
    print(f"\nToken ownership:")
    for token_id in token_ids:
        owning_rank = token_id // per_rank_vocab
        local_idx = token_id % per_rank_vocab
        print(f"  Token {token_id}: rank {owning_rank}, local_idx {local_idx}")

    # Simulate the lookup on each rank
    print(f"\nSimulating per-rank lookup and all-reduce:")

    # Standard (non-parallel) lookup for reference
    reference_embeddings = embed_weight[token_ids]
    print(f"  Reference embedding shape: {reference_embeddings.shape}")

    # Simulate parallel lookup
    batch_size = len(token_ids)
    all_rank_outputs = []

    for rank in range(tp_degree):
        rank_start = rank * per_rank_vocab
        rank_end = (rank + 1) * per_rank_vocab

        # This rank's portion of embedding
        rank_embed = embed_weight[rank_start:rank_end]  # [per_rank_vocab, hidden_size]

        # For each token, check if this rank owns it
        rank_output = torch.zeros(batch_size, hidden_size, dtype=embed_weight.dtype)

        for i, token_id in enumerate(token_ids):
            if rank_start <= token_id < rank_end:
                # This rank owns this token
                local_idx = token_id - rank_start
                rank_output[i] = rank_embed[local_idx]

        all_rank_outputs.append(rank_output)

    # All-reduce (sum)
    parallel_embeddings = torch.stack(all_rank_outputs).sum(dim=0)

    # Compare
    print(f"  Parallel embedding shape: {parallel_embeddings.shape}")

    diff = (reference_embeddings.float() - parallel_embeddings.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n  Comparison:")
    print(f"    Max difference: {max_diff}")
    print(f"    Mean difference: {mean_diff}")

    if max_diff < 1e-6:
        print("  ✓ Parallel embedding simulation matches reference!")
    else:
        print("  ✗ MISMATCH detected!")

    return reference_embeddings, parallel_embeddings


def check_embedding_for_test_tokens():
    """Check embedding values for our test tokens."""
    print("=" * 60)
    print("Embedding Check for Test Tokens")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    token_tensor = torch.tensor(tokens)

    print(f"Test text: '{TEST_TEXT}'")
    print(f"Token IDs: {tokens}")

    embed_weight = load_embedding_weight()
    print(f"\nEmbedding weight shape: {embed_weight.shape}")

    # Do lookup
    embeddings = embed_weight[token_tensor]

    print(f"\nEmbedding values for each token:")
    for i, token_id in enumerate(tokens):
        token_str = tokenizer.decode([token_id])
        emb = embeddings[i].float()
        print(f"\n  Token {token_id} ('{token_str}'):")
        print(f"    Shape: {emb.shape}")
        print(f"    Mean: {emb.mean():.6f}")
        print(f"    Std: {emb.std():.6f}")
        print(f"    Norm: {emb.norm():.4f}")
        print(f"    Min: {emb.min():.6f}")
        print(f"    Max: {emb.max():.6f}")
        print(f"    First 5 values: {emb[:5].tolist()}")

    # Simulate parallel embedding
    ref_emb, par_emb = simulate_parallel_embedding(embed_weight, token_tensor, tp_degree=64)

    return embeddings


def analyze_potential_issues():
    """Analyze what could go wrong with ParallelEmbedding."""
    print("\n" + "=" * 60)
    print("Potential Issues Analysis")
    print("=" * 60)

    embed_weight = load_embedding_weight()
    vocab_size, hidden_size = embed_weight.shape
    tp_degree = 64
    per_rank_vocab = vocab_size // tp_degree

    print(f"""
    Configuration:
    - vocab_size: {vocab_size}
    - hidden_size: {hidden_size}
    - tp_degree: {tp_degree}
    - per_rank_vocab: {per_rank_vocab}

    Potential Issues:

    1. VOCAB SIZE PADDING
       - Tokenizer vocab: 200000
       - Model vocab: {vocab_size}
       - Padding: {vocab_size - 200000} extra tokens
       - These extra tokens should be at the END (indices 200000-{vocab_size-1})

    2. PER-RANK VOCAB CALCULATION
       - Each rank should handle {per_rank_vocab} tokens
       - Rank 0: tokens 0-{per_rank_vocab-1}
       - Rank 1: tokens {per_rank_vocab}-{2*per_rank_vocab-1}
       - etc.

    3. ALL-REDUCE CORRECTNESS
       - After local lookup, must do all-reduce SUM
       - If all-reduce is wrong, embeddings will be incorrect

    4. EMBEDDING WEIGHT LOADING
       - NXD might shard embedding weights during loading
       - Each rank should have [per_rank_vocab, hidden_size] slice
       - But with shard_across_embedding=True, the original [vocab_size, hidden_size]
         weights need to be properly distributed
    """)


if __name__ == "__main__":
    embeddings = check_embedding_for_test_tokens()
    analyze_potential_issues()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
    The simulation shows that ParallelEmbedding SHOULD work correctly
    if implemented properly. The issue is likely one of:

    1. Embedding weights not loaded correctly during compilation
    2. All-reduce not working correctly on Neuron
    3. Some other layer (not embedding) is causing the issue

    To debug further:
    - Recompile with skip_compile=False to see weight loading debug output
    - Or add debug output in the model's forward pass
    """)
