"""
Diagnostic script to identify the root cause of MiniMax M2 garbled output.
This script checks various components of the model to find where the issue occurs.
"""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

from transformers import AutoTokenizer
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights/"


def diagnose_model():
    """Diagnose the model step by step."""
    print("\n" + "="*60)
    print("MiniMax M2 Diagnostic Script")
    print("="*60)

    # Load model and tokenizer
    print("\n[1] Loading model...")
    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Check model config
    print("\n[2] Checking model config...")
    print(f"  vocab_size: {model.config.vocab_size}")
    print(f"  hidden_size: {model.config.hidden_size}")
    print(f"  num_attention_heads: {model.config.num_attention_heads}")
    print(f"  num_key_value_heads: {model.config.num_key_value_heads}")
    print(f"  num_hidden_layers: {model.config.num_hidden_layers}")
    print(f"  num_local_experts: {model.config.num_local_experts}")
    print(f"  num_experts_per_tok: {model.config.num_experts_per_tok}")
    print(f"  intermediate_size: {model.config.intermediate_size}")
    print(f"  head_dim: {model.config.head_dim}")
    print(f"  rotary_dim: {getattr(model.config, 'rotary_dim', 'NOT SET')}")
    print(f"  use_qk_norm: {getattr(model.config, 'use_qk_norm', 'NOT SET')}")
    print(f"  tp_degree: {model.config.neuron_config.tp_degree}")

    # Test tokenization
    print("\n[3] Testing tokenization...")
    test_text = "Hello"
    inputs = tokenizer([test_text], return_tensors="pt")
    print(f"  Input text: '{test_text}'")
    print(f"  Token IDs: {inputs.input_ids[0].tolist()}")
    print(f"  Decoded back: '{tokenizer.decode(inputs.input_ids[0])}'")

    # Test simple generation
    print("\n[4] Testing generation with single token output...")
    generation_model = HuggingFaceGenerationAdapter(model)

    # Generate just 1 token to see what the model predicts
    outputs_1 = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
    )

    # Analyze the first token prediction
    if hasattr(outputs_1, 'logits') and outputs_1.logits is not None:
        first_logits = outputs_1.logits[0]  # First step logits
        print(f"\n[5] First step logits analysis...")
        print(f"  Logits shape: {first_logits.shape}")
        print(f"  Logits dtype: {first_logits.dtype}")

        # Statistics
        logits_float = first_logits.float()
        print(f"  Mean: {logits_float.mean().item():.4f}")
        print(f"  Std: {logits_float.std().item():.4f}")
        print(f"  Min: {logits_float.min().item():.4f}")
        print(f"  Max: {logits_float.max().item():.4f}")

        # Check for NaN or Inf
        has_nan = torch.isnan(logits_float).any().item()
        has_inf = torch.isinf(logits_float).any().item()
        print(f"  Has NaN: {has_nan}")
        print(f"  Has Inf: {has_inf}")

        # Top-10 predictions
        print(f"\n  Top 10 token predictions:")
        top_logits, top_indices = torch.topk(first_logits[0], 10)
        for i, (logit, idx) in enumerate(zip(top_logits.tolist(), top_indices.tolist())):
            token_str = tokenizer.decode([idx])
            print(f"    {i+1}. Token '{token_str}' (ID={idx}, logit={logit:.2f})")

        # Check if all logits are the same (would indicate no learning)
        unique_logits = len(torch.unique(first_logits))
        print(f"\n  Number of unique logit values: {unique_logits}")
        if unique_logits < 100:
            print("  WARNING: Very few unique logit values - model may not be computing properly!")
    else:
        print("  WARNING: No logits returned!")

    # Generate more tokens
    print("\n[6] Testing generation with 20 tokens...")
    outputs_20 = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=20,
        do_sample=False,
    )

    output_text = tokenizer.decode(outputs_20[0], skip_special_tokens=True)
    print(f"  Generated text: '{output_text}'")
    print(f"  Output token IDs: {outputs_20[0].tolist()}")

    # Decode token by token
    print(f"\n  Token-by-token decoding:")
    for i, token_id in enumerate(outputs_20[0].tolist()):
        token_str = tokenizer.decode([token_id])
        print(f"    Position {i}: ID={token_id} -> '{token_str}'")

    # Check for patterns in output token IDs
    print("\n[7] Checking for patterns in output...")
    output_ids = outputs_20[0].tolist()

    # Check if output IDs have a counting pattern
    diffs = [output_ids[i+1] - output_ids[i] for i in range(len(output_ids)-1)]
    if len(set(diffs)) <= 3:
        print(f"  WARNING: Output shows simple pattern! Differences: {set(diffs)}")
    else:
        print(f"  Output differences (first 10): {diffs[:10]}")

    # Check if output contains many repeated tokens
    from collections import Counter
    token_counts = Counter(output_ids)
    most_common = token_counts.most_common(5)
    print(f"  Most common tokens: {most_common}")

    print("\n" + "="*60)
    print("Diagnostic complete")
    print("="*60)


if __name__ == "__main__":
    diagnose_model()
