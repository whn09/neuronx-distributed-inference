"""
Get Neuron embedding output and compare with CPU.
Run this AFTER compare_cpu_neuron_tokens.py

This script loads the compiled Neuron model and extracts embedding output
to compare with CPU results.
"""
import torch
from transformers import AutoTokenizer

from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

MODEL_PATH = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
TRACED_MODEL_PATH = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights/"
TEST_TEXT = "The capital of France is"


def get_neuron_output():
    """Load compiled model and get output for comparison."""
    print("=" * 60)
    print("Loading Neuron Model")
    print("=" * 60)

    # Load compiled model
    print(f"Loading from: {TRACED_MODEL_PATH}")
    model = NeuronMiniMaxM2ForCausalLM(TRACED_MODEL_PATH)
    model.load(TRACED_MODEL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(TRACED_MODEL_PATH)

    print(f"Model loaded successfully")
    print(f"Model config vocab_size: {model.config.vocab_size}")
    print(f"Model config hidden_size: {model.config.hidden_size}")

    # Tokenize
    print("\n" + "=" * 60)
    print("Tokenization")
    print("=" * 60)

    inputs = tokenizer([TEST_TEXT], padding=True, return_tensors="pt")
    print(f"Input text: '{TEST_TEXT}'")
    print(f"Token IDs: {inputs.input_ids[0].tolist()}")

    # Generate just 1 token to see model behavior
    print("\n" + "=" * 60)
    print("Generation (1 token)")
    print("=" * 60)

    generation_model = HuggingFaceGenerationAdapter(model)

    # Generate with greedy decoding
    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1,  # Just 1 new token
        do_sample=False,
    )

    print(f"Output token IDs: {outputs[0].tolist()}")

    # The first new token after input
    new_token_id = outputs[0, inputs.input_ids.shape[1]].item()
    new_token_str = tokenizer.decode([new_token_id])
    print(f"First generated token: {new_token_id} -> '{new_token_str}'")

    # Expected: "Paris" or similar
    expected_tokens = ["Paris", " Paris", "paris", " paris"]
    if new_token_str.strip().lower() == "paris":
        print("✓ Generated token is correct!")
    else:
        print(f"✗ Generated token '{new_token_str}' is NOT 'Paris'!")

    # Compare with CPU
    print("\n" + "=" * 60)
    print("Comparison with CPU")
    print("=" * 60)

    try:
        cpu_data = torch.load('/tmp/cpu_embeddings.pt')
        print(f"CPU tokens: {cpu_data['tokens'].tolist()}")
        print(f"Neuron tokens: {inputs.input_ids[0].tolist()}")

        if cpu_data['tokens'].tolist() == inputs.input_ids[0].tolist():
            print("✓ Token IDs match!")
        else:
            print("✗ Token IDs MISMATCH!")
    except FileNotFoundError:
        print("CPU embeddings not found. Run compare_cpu_neuron_tokens.py first.")

    return outputs


if __name__ == "__main__":
    outputs = get_neuron_output()
