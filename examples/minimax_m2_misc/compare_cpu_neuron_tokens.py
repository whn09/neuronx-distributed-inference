"""
Compare CPU vs Neuron tokenization and embedding lookup.
This script verifies that input tokens are processed identically.
"""
import torch
from transformers import AutoTokenizer

MODEL_PATH = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
TEST_TEXT = "The capital of France is"


def test_tokenization():
    """Test that tokenization produces identical results."""
    print("=" * 60)
    print("TEST: Tokenization Comparison")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)

    # Also test with padding (as used in generation)
    inputs = tokenizer([TEST_TEXT], padding=True, return_tensors="pt")

    print(f"Input text: '{TEST_TEXT}'")
    print(f"Token IDs (no special tokens): {tokens}")
    print(f"Token IDs (with padding): {inputs.input_ids[0].tolist()}")
    print(f"Attention mask: {inputs.attention_mask[0].tolist()}")

    # Decode back
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: '{decoded}'")

    # Verify round-trip
    assert decoded == TEST_TEXT, f"Round-trip failed!"
    print("✓ Tokenization is correct")

    return tokens, inputs


def test_embedding_cpu():
    """Load embedding weights and do lookup on CPU."""
    print("\n" + "=" * 60)
    print("TEST: CPU Embedding Lookup")
    print("=" * 60)

    from safetensors import safe_open
    import os
    import json

    # Load index
    index_file = os.path.join(MODEL_PATH, "model.safetensors.index.json")
    with open(index_file) as f:
        index = json.load(f)

    # Load embedding
    embed_key = "model.embed_tokens.weight"
    shard = index["weight_map"][embed_key]
    with safe_open(os.path.join(MODEL_PATH, shard), framework="pt") as f:
        embed_weight = f.get_tensor(embed_key)

    print(f"Embedding weight shape: {embed_weight.shape}")
    print(f"Embedding weight dtype: {embed_weight.dtype}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokens = tokenizer.encode(TEST_TEXT, add_special_tokens=False)
    token_tensor = torch.tensor(tokens)

    # Lookup
    embeddings = embed_weight[token_tensor]

    print(f"\nToken IDs: {tokens}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")

    # Print embedding stats for each token
    print("\nPer-token embedding statistics:")
    for i, token_id in enumerate(tokens):
        emb = embeddings[i].float()
        token_str = tokenizer.decode([token_id])
        print(f"  Token {token_id} ('{token_str}'): mean={emb.mean():.6f}, std={emb.std():.6f}, norm={emb.norm():.4f}")

    # Save for comparison
    torch.save({
        'tokens': token_tensor,
        'embeddings': embeddings,
        'text': TEST_TEXT,
    }, '/tmp/cpu_embeddings.pt')
    print("\n✓ Saved CPU embeddings to /tmp/cpu_embeddings.pt")

    return embeddings


if __name__ == "__main__":
    tokens, inputs = test_tokenization()
    embeddings = test_embedding_cpu()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test text: '{TEST_TEXT}'")
    print(f"Token IDs: {tokens}")
    print(f"Embedding shape per token: [{embeddings.shape[1]}]")
    print(f"\nNext: Run compare_neuron_embedding.py to get Neuron embeddings")
