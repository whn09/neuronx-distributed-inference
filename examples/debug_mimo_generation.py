"""
Debug script to diagnose MiMo-V2-Flash generation issues.
"""

import argparse
import torch
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled-model-path", type=str,
                        default="/mnt/nvme/traced_model/MiMo-V2-Flash-BF16")
    parser.add_argument("--prompt", type=str, default="你好")
    args = parser.parse_args()

    print("=" * 60)
    print("MiMo-V2-Flash Generation Debug")
    print("=" * 60)

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.compiled_model_path, trust_remote_code=True)
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"   EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    print(f"   BOS token: {tokenizer.bos_token} (id={getattr(tokenizer, 'bos_token_id', None)})")

    # Tokenize input
    print(f"\n2. Tokenizing prompt: '{args.prompt}'")
    inputs = tokenizer([args.prompt], padding=True, return_tensors="pt")
    print(f"   Input IDs: {inputs.input_ids.tolist()}")
    print(f"   Attention mask: {inputs.attention_mask.tolist()}")

    # Decode the repeated token
    print("\n3. Analyzing the repeated token (118076)...")
    repeated_token_id = 118076
    repeated_text = tokenizer.decode([repeated_token_id])
    print(f"   Token {repeated_token_id} decodes to: '{repeated_text}'")

    # Check nearby tokens
    print("\n4. Checking nearby tokens...")
    for tid in [118074, 118075, 118076, 118077, 118078]:
        text = tokenizer.decode([tid])
        print(f"   Token {tid}: '{text}'")

    # Load model and check config
    print("\n5. Loading model...")
    from neuronx_distributed_inference.models.mimo_v2.modeling_mimo_v2 import (
        NeuronMiMoV2ForCausalLM,
    )
    model = NeuronMiMoV2ForCausalLM(args.compiled_model_path)
    model.load(args.compiled_model_path)

    config = model.config
    neuron_config = config.neuron_config

    print(f"\n6. Model configuration:")
    print(f"   TP degree: {neuron_config.tp_degree}")
    print(f"   Batch size: {neuron_config.batch_size}")
    print(f"   Max context length: {neuron_config.max_context_length}")
    print(f"   Sequence length: {neuron_config.seq_len}")
    print(f"   On-device sampling: {neuron_config.on_device_sampling_config is not None}")
    if neuron_config.on_device_sampling_config:
        odc = neuron_config.on_device_sampling_config
        print(f"     do_sample: {odc.do_sample}")
        print(f"     temperature: {getattr(odc, 'temperature', 'N/A')}")
        print(f"     top_k: {getattr(odc, 'top_k', 'N/A')}")
        print(f"     top_p: {getattr(odc, 'top_p', 'N/A')}")

    # Check KV cache configuration
    print(f"\n7. KV cache configuration:")
    # The model wrapper uses 'models' which is a list of compiled models
    inner_model = None
    if hasattr(model, 'models') and model.models:
        # models is a list, get the first available model
        inner_model = model.models[0] if isinstance(model.models, list) else None

    if inner_model:
        print(f"   num_key_value_heads (model): {getattr(inner_model, 'num_key_value_heads', 'N/A')}")
        print(f"   n_positions: {getattr(inner_model, 'n_positions', 'N/A')}")
    else:
        print("   Could not access inner model")

    # Check layer configuration
    print(f"\n8. Layer configuration (first 3 layers):")
    layers = getattr(inner_model, 'layers', []) if inner_model else []
    for i in range(min(3, len(layers))):
        layer = layers[i]
        attn = getattr(layer, 'self_attn', None)
        print(f"   Layer {i}:")
        print(f"     attention_type: {getattr(layer, 'attention_type', 'N/A')}")
        print(f"     uses_moe: {getattr(layer, 'uses_moe', 'N/A')}")
        if attn:
            print(f"     local_num_heads: {getattr(attn, 'local_num_heads', 'N/A')}")
            print(f"     local_num_kv_heads: {getattr(attn, 'local_num_kv_heads', 'N/A')}")
            print(f"     use_gqa_convert_to_mha: {getattr(attn, 'use_gqa_convert_to_mha', 'N/A')}")
            print(f"     attn_head_dim: {getattr(attn, 'attn_head_dim', 'N/A')}")
            print(f"     attn_v_head_dim: {getattr(attn, 'attn_v_head_dim', 'N/A')}")
        else:
            print(f"     (no self_attn found)")

    # Try generation with manual stepping
    print("\n9. Testing generation...")
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    generation_model = HuggingFaceGenerationAdapter(model)

    # Generate just 5 tokens to see the pattern
    print(f"   Generating 5 tokens...")
    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=5,
        do_sample=False,
    )
    print(f"   Output IDs: {outputs[0].tolist()}")
    print(f"   Output text: '{tokenizer.decode(outputs[0], skip_special_tokens=True)}'")

    # Try with sampling enabled
    print("\n10. Testing with do_sample=True, temperature=0.7...")
    outputs_sample = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=5,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    print(f"   Output IDs: {outputs_sample[0].tolist()}")
    print(f"   Output text: '{tokenizer.decode(outputs_sample[0], skip_special_tokens=True)}'")

    print("\n" + "=" * 60)
    print("Debug complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
