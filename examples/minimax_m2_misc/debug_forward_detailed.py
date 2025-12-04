"""
Detailed debug script to trace where the forward pass goes wrong.
This adds hooks to print intermediate values.
"""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

from transformers import AutoTokenizer
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights/"


def debug_embedding():
    """Test just the embedding layer output."""
    print("\n" + "="*60)
    print("Debug: Checking embedding output")
    print("="*60)

    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Input
    text = "Hello"
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids

    print(f"Input text: '{text}'")
    print(f"Input IDs: {input_ids[0].tolist()}")

    # Get the neuron model
    if hasattr(model, 'models') and len(model.models) > 0:
        neuron_model = model.models[0]

        # Check if embed_tokens is accessible
        if hasattr(neuron_model, 'embed_tokens'):
            print("\nEmbed tokens layer found")
            embed = neuron_model.embed_tokens
            print(f"  Type: {type(embed)}")
            if hasattr(embed, 'weight'):
                w = embed.weight
                print(f"  Weight shape: {w.shape}")
                print(f"  Weight dtype: {w.dtype}")
        else:
            print("Cannot access embed_tokens directly from traced model")

    # Do a forward pass and check output
    print("\n" + "="*60)
    print("Debug: Checking generation output token-by-token")
    print("="*60)

    generation_model = HuggingFaceGenerationAdapter(model)

    # Generate 10 tokens one at a time
    all_output_ids = input_ids[0].tolist()
    current_input = input_ids.clone()
    current_attention = inputs.attention_mask.clone()

    for step in range(10):
        outputs = generation_model.generate(
            current_input,
            attention_mask=current_attention,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

        # Get the new token
        new_token_id = outputs.sequences[0, -1].item()
        all_output_ids.append(new_token_id)

        # Analyze logits
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            logits = outputs.logits[0][0]  # First step, first batch
            top_logits, top_indices = torch.topk(logits, 5)

            print(f"\nStep {step + 1}:")
            print(f"  Current input length: {current_input.shape[1]}")
            print(f"  Logits shape: {outputs.logits[0].shape}")
            print(f"  Top 5 predictions:")
            for i, (logit, idx) in enumerate(zip(top_logits.tolist(), top_indices.tolist())):
                token_str = tokenizer.decode([idx])
                marker = " <-- SELECTED" if idx == new_token_id else ""
                print(f"    {i+1}. '{token_str}' (ID={idx}, logit={logit:.2f}){marker}")

        # Update for next iteration
        current_input = outputs.sequences
        current_attention = torch.ones_like(current_input)

    # Final output
    print("\n" + "="*60)
    print("Final output")
    print("="*60)
    print(f"Output IDs: {all_output_ids}")
    print(f"Output text: '{tokenizer.decode(all_output_ids)}'")


def check_model_config():
    """Check model configuration."""
    print("\n" + "="*60)
    print("Debug: Model configuration")
    print("="*60)

    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)

    cfg = model.config
    print(f"vocab_size: {cfg.vocab_size}")
    print(f"hidden_size: {cfg.hidden_size}")
    print(f"num_hidden_layers: {cfg.num_hidden_layers}")
    print(f"num_attention_heads: {cfg.num_attention_heads}")
    print(f"num_key_value_heads: {cfg.num_key_value_heads}")
    print(f"head_dim: {cfg.head_dim}")
    print(f"rotary_dim: {getattr(cfg, 'rotary_dim', 'NOT SET')}")
    print(f"use_qk_norm: {getattr(cfg, 'use_qk_norm', 'NOT SET')}")
    print(f"num_local_experts: {cfg.num_local_experts}")
    print(f"num_experts_per_tok: {cfg.num_experts_per_tok}")
    print(f"intermediate_size: {cfg.intermediate_size}")
    print(f"tp_degree: {cfg.neuron_config.tp_degree}")


if __name__ == "__main__":
    check_model_config()
    debug_embedding()
