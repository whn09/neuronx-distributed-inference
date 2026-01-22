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
    print(f"   Input IDs (raw): {inputs.input_ids.tolist()}")
    print(f"   Attention mask: {inputs.attention_mask.tolist()}")

    # Also try with chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            messages = [{"role": "user", "content": args.prompt}]
            chat_input = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            print(f"   Input IDs (chat template): {chat_input.tolist()}")
            print(f"   Decoded chat input: '{tokenizer.decode(chat_input[0])}'")
        except Exception as e:
            print(f"   Chat template not available: {e}")

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

    # Print head_dim configuration (critical for KV cache sizing)
    print(f"\n6a. Head dimension configuration (critical for KV cache):")
    print(f"   config.head_dim (from HF config): {getattr(config, 'head_dim', 'NOT SET')}")
    print(f"   config.v_head_dim: {getattr(config, 'v_head_dim', 'NOT SET')}")
    print(f"   config.swa_head_dim: {getattr(config, 'swa_head_dim', 'NOT SET')}")
    print(f"   config.swa_v_head_dim: {getattr(config, 'swa_v_head_dim', 'NOT SET')}")
    print(f"   config.hidden_size: {getattr(config, 'hidden_size', 'NOT SET')}")
    print(f"   config.num_attention_heads: {getattr(config, 'num_attention_heads', 'NOT SET')}")
    # Computed head_dim if not set
    if hasattr(config, 'hidden_size') and hasattr(config, 'num_attention_heads'):
        computed_head_dim = config.hidden_size // config.num_attention_heads
        print(f"   Computed head_dim (hidden_size/num_attention_heads): {computed_head_dim}")

    # Check KV cache configuration
    print(f"\n7. KV cache configuration:")

    # Try to access the KV cache manager
    kv_mgr = None
    if hasattr(model, 'context_encoding_model') and model.context_encoding_model:
        ce_model = model.context_encoding_model.model if hasattr(model.context_encoding_model, 'model') else None
        if ce_model and hasattr(ce_model, 'kv_mgr'):
            kv_mgr = ce_model.kv_mgr
    if kv_mgr is None and hasattr(model, 'token_generation_model') and model.token_generation_model:
        tg_model = model.token_generation_model.model if hasattr(model.token_generation_model, 'model') else None
        if tg_model and hasattr(tg_model, 'kv_mgr'):
            kv_mgr = tg_model.kv_mgr

    if kv_mgr:
        print(f"   KV cache manager found!")
        print(f"   k_shape: {getattr(kv_mgr, 'k_shape', 'N/A')}")
        print(f"   v_shape: {getattr(kv_mgr, 'v_shape', 'N/A')}")
        if hasattr(kv_mgr, 'past_key_values') and kv_mgr.past_key_values:
            # Show first layer's K and V shapes
            k_cache_shape = kv_mgr.past_key_values[0].shape if len(kv_mgr.past_key_values) > 0 else "N/A"
            v_cache_shape = kv_mgr.past_key_values[1].shape if len(kv_mgr.past_key_values) > 1 else "N/A"
            print(f"   Layer 0 K cache shape: {k_cache_shape}")
            print(f"   Layer 0 V cache shape: {v_cache_shape}")
    else:
        print("   Could not access KV cache manager")

    # The model wrapper uses 'models' which is a list of compiled models
    inner_model = None
    if hasattr(model, 'models') and model.models:
        # models is a list, get the first available model
        inner_model = model.models[0] if isinstance(model.models, list) else None

    if inner_model:
        print(f"   num_key_value_heads (model): {getattr(inner_model, 'num_key_value_heads', 'N/A')}")
        print(f"   n_positions: {getattr(inner_model, 'n_positions', 'N/A')}")
    else:
        print("   Could not access inner model from model.models")

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

    # Check embedding layer health
    print("\n9. Checking embedding layer...")
    try:
        # Try to access the embedding layer from context_encoding_model
        embed_tokens = None
        if hasattr(model, 'context_encoding_model') and model.context_encoding_model:
            ce_model = model.context_encoding_model.model if hasattr(model.context_encoding_model, 'model') else None
            if ce_model and hasattr(ce_model, 'embed_tokens'):
                embed_tokens = ce_model.embed_tokens

        if embed_tokens is not None:
            # Test embedding lookup
            test_ids = torch.tensor([[108386]])  # "你好" token
            with torch.no_grad():
                test_embed = embed_tokens(test_ids)
            print(f"   Embedding shape: {test_embed.shape}")
            print(f"   Embedding stats - mean: {test_embed.mean().item():.4f}, std: {test_embed.std().item():.4f}")
            print(f"   Embedding range: [{test_embed.min().item():.4f}, {test_embed.max().item():.4f}]")
            # Check if embeddings are reasonable (not all zeros or NaN)
            if torch.isnan(test_embed).any():
                print("   WARNING: Embedding contains NaN values!")
            elif test_embed.abs().max() < 1e-6:
                print("   WARNING: Embedding appears to be all zeros!")
            elif test_embed.abs().max() > 100:
                print("   WARNING: Embedding values seem unusually large!")
            else:
                print("   Embedding values look reasonable")
        else:
            print("   Could not access embedding layer")
    except Exception as e:
        print(f"   Error checking embeddings: {e}")

    # Try a single forward pass to check logits
    print("\n9a. Testing single forward pass (checking logits)...")
    try:
        # Call forward directly to see raw logits
        with torch.no_grad():
            # Need to prepare inputs properly
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
            seq_ids = torch.arange(input_ids.shape[0])

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                seq_ids=seq_ids,
            )

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            print(f"   Logits shape: {logits.shape}")
            print(f"   Logits stats - mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
            print(f"   Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

            # Get top-10 tokens from logits
            if logits.dim() == 3:
                last_logits = logits[0, -1, :]  # [vocab_size]
            else:
                last_logits = logits[0, :]
            top_values, top_indices = torch.topk(last_logits, k=10)
            print(f"   Top 10 predicted tokens (from forward):")
            for val, idx in zip(top_values.tolist(), top_indices.tolist()):
                token_text = tokenizer.decode([idx])
                print(f"     Token {idx}: '{token_text}' (logit={val:.4f})")

            # Also check if top prediction is reasonable for the input
            print(f"\n   Sanity check: What are reasonable responses to '你好'?")
            expected_tokens = ["！", "你好", ",", "，", "我", "有什么", "很高兴"]
            for token_text in expected_tokens:
                token_ids = tokenizer.encode(token_text, add_special_tokens=False)
                if token_ids:
                    token_id = token_ids[0]
                    if token_id < last_logits.shape[0]:
                        token_logit = last_logits[token_id].item()
                        print(f"     '{token_text}' (id={token_id}): logit={token_logit:.4f}")
    except Exception as e:
        import traceback
        print(f"   Error during forward pass: {e}")
        traceback.print_exc()

    # Try generation with chat template
    print("\n10. Testing generation WITH CHAT TEMPLATE...")
    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    generation_model = HuggingFaceGenerationAdapter(model)

    # Use chat template if available
    chat_inputs = None
    if hasattr(tokenizer, 'apply_chat_template'):
        try:
            messages = [{"role": "user", "content": args.prompt}]
            chat_input_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            # Truncate if longer than max_context_length
            max_ctx = config.neuron_config.max_context_length
            if chat_input_ids.shape[1] > max_ctx:
                print(f"   Warning: Chat template ({chat_input_ids.shape[1]} tokens) exceeds max_context_length ({max_ctx})")
                print(f"   Truncating to {max_ctx} tokens")
                chat_input_ids = chat_input_ids[:, :max_ctx]
            chat_attention_mask = torch.ones_like(chat_input_ids)
            chat_inputs = {"input_ids": chat_input_ids, "attention_mask": chat_attention_mask}
            print(f"   Chat template input length: {chat_input_ids.shape[1]} tokens")
        except Exception as e:
            print(f"   Could not apply chat template: {e}")

    if chat_inputs:
        print(f"   Generating 5 tokens with chat template...")
        outputs_chat = generation_model.generate(
            chat_inputs["input_ids"],
            attention_mask=chat_inputs["attention_mask"],
            max_new_tokens=5,
            do_sample=False,
        )
        print(f"   Output IDs: {outputs_chat[0].tolist()}")
        print(f"   Output text: '{tokenizer.decode(outputs_chat[0], skip_special_tokens=False)}'")
    else:
        print("   Skipping chat template test (not available)")

    # Also test with raw input for comparison
    print("\n11. Testing generation with RAW INPUT (no chat template)...")
    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=5,
        do_sample=False,
    )
    print(f"   Output IDs: {outputs[0].tolist()}")
    print(f"   Output text: '{tokenizer.decode(outputs[0], skip_special_tokens=True)}'")

    # Try with sampling enabled
    print("\n11. Testing with do_sample=True, temperature=0.7...")
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
