"""
Debug script to check Neuron model output step by step.
"""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights/"


def debug_forward_pass():
    """Debug the forward pass of the model."""
    print("\n=== Loading model ===")
    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)

    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Test input
    text = "Hello"
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    print(f"\n=== Input ===")
    print(f"Text: '{text}'")
    print(f"Input IDs: {input_ids[0].tolist()}")
    print(f"Attention mask: {attention_mask[0].tolist()}")

    # Create position_ids
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    print(f"Position IDs: {position_ids[0].tolist()}")

    # Check model config
    print(f"\n=== Model Config ===")
    print(f"vocab_size: {model.config.vocab_size}")
    print(f"hidden_size: {model.config.hidden_size}")
    print(f"num_attention_heads: {model.config.num_attention_heads}")
    print(f"num_key_value_heads: {model.config.num_key_value_heads}")
    print(f"tp_degree: {model.config.neuron_config.tp_degree}")
    print(f"use_qk_norm: {getattr(model.config, 'use_qk_norm', 'NOT FOUND')}")
    print(f"rotary_dim: {getattr(model.config, 'rotary_dim', 'NOT FOUND')}")

    # Try a single forward pass
    print(f"\n=== Forward pass ===")
    generation_model = HuggingFaceGenerationAdapter(model)

    # Use generate with max_new_tokens=1 to see first token
    outputs = generation_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
    )

    print(f"\n=== Output ===")
    print(f"Generated tokens: {outputs.sequences[0].tolist()}")
    print(f"Decoded: '{tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)}'")

    # Check logits
    if hasattr(outputs, 'logits') and outputs.logits is not None:
        print(f"\n=== Logits analysis ===")
        for i, logits in enumerate(outputs.logits):
            print(f"Step {i}: logits shape = {logits.shape}")
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(logits[0], 5)
            print(f"  Top 5 token predictions:")
            for j, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
                token = tokenizer.decode([idx])
                print(f"    {j+1}. '{token}' (ID={idx}, logit={prob:.2f})")

            # Check if logits look normal
            logits_mean = logits.mean().item()
            logits_std = logits.std().item()
            logits_min = logits.min().item()
            logits_max = logits.max().item()
            print(f"  Logits stats: mean={logits_mean:.2f}, std={logits_std:.2f}, min={logits_min:.2f}, max={logits_max:.2f}")


if __name__ == "__main__":
    debug_forward_pass()
