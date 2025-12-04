"""
Test MiniMax M2 on GPU as a reference to verify Neuron implementation.
This uses the HuggingFace transformers directly without Neuron.
"""
import torch
import sys
import os

# Add model path to sys.path for custom model loading
model_path = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
sys.path.insert(0, model_path)

from transformers import AutoTokenizer, AutoModelForCausalLM


def test_gpu_model():
    """Test MiniMax M2 on GPU."""
    print("\n" + "="*60)
    print("Testing MiniMax M2 on GPU (reference implementation)")
    print("="*60)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("\nLoading model (this may take a while for BF16 model)...")
    # Try to load with limited memory usage
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",  # Automatic device mapping
            trust_remote_code=True,
        )
        print(f"Model loaded successfully!")
        print(f"Model type: {type(model)}")
    except Exception as e:
        print(f"Failed to load full model: {e}")
        print("\nTrying to load only the first layer for debugging...")

        # Alternative: Just check weights directly
        from safetensors import safe_open
        import json

        index_path = os.path.join(model_path, 'model.safetensors.index.json')
        with open(index_path) as f:
            index = json.load(f)

        # Get embedding weight
        embed_key = 'model.embed_tokens.weight'
        shard_file = index['weight_map'].get(embed_key)
        with safe_open(os.path.join(model_path, shard_file), framework='pt', device='cpu') as f:
            embed_weight = f.get_tensor(embed_key)

        # Get lm_head weight
        lm_head_key = 'lm_head.weight'
        shard_file = index['weight_map'].get(lm_head_key)
        with safe_open(os.path.join(model_path, shard_file), framework='pt', device='cpu') as f:
            lm_head_weight = f.get_tensor(lm_head_key)

        print(f"\nEmbed weight shape: {embed_weight.shape}")
        print(f"LM head weight shape: {lm_head_weight.shape}")

        # Simple test: embed "Hello" and apply lm_head directly
        # This is NOT a proper forward pass but can help verify embedding
        text = "Hello"
        inputs = tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids

        print(f"\nInput: '{text}'")
        print(f"Token IDs: {input_ids[0].tolist()}")

        # Get embedding for input token
        token_id = input_ids[0, 0].item()
        embedding = embed_weight[token_id]
        print(f"\nEmbedding for token {token_id}:")
        print(f"  Shape: {embedding.shape}")
        print(f"  Mean: {embedding.float().mean().item():.6f}")
        print(f"  Std: {embedding.float().std().item():.6f}")

        return

    # If model loaded successfully, run inference
    text = "Hello"
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    print(f"\nInput: '{text}'")
    print(f"Token IDs: {input_ids[0].tolist()}")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :]  # Last token's logits

    # Get top predictions
    top_logits, top_indices = torch.topk(logits.float(), 20)
    print(f"\nTop 20 predictions for next token:")
    for i, (logit, idx) in enumerate(zip(top_logits.tolist(), top_indices.tolist())):
        token_str = tokenizer.decode([idx])
        print(f"  {i+1}. '{token_str}' (ID={idx}, logit={logit:.2f})")

    # Generate
    generated = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,
        do_sample=False,
    )
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"\nGenerated text: '{output_text}'")


if __name__ == "__main__":
    test_gpu_model()
