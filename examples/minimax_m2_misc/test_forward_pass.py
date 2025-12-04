"""
Test forward pass step by step to find where the issue is.
"""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

from transformers import AutoTokenizer
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

model_path = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights/"


def test_tokenizer():
    """Test that tokenizer produces expected results."""
    print("\n=== Testing Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    test_texts = [
        "Who are you?",
        "Hello",
        "The capital of France is",
    ]

    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"'{text}' -> {tokens} -> '{decoded}'")


def test_loaded_model_weights():
    """Check if model weights are loaded correctly after loading from traced checkpoint."""
    print("\n=== Testing Loaded Model Weights ===")

    # Load model
    print("Loading model from traced checkpoint...")
    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)

    print(f"Model type: {type(model)}")
    print(f"Model config vocab_size: {model.config.vocab_size}")
    print(f"Model config hidden_size: {model.config.hidden_size}")
    print(f"Model config num_hidden_layers: {model.config.num_hidden_layers}")

    # Try to access some model components
    print("\n=== Model Structure ===")

    # Check if we can access the underlying model
    if hasattr(model, 'models') and len(model.models) > 0:
        neuron_model = model.models[0]
        print(f"Neuron model type: {type(neuron_model)}")

        # Check embedding
        if hasattr(neuron_model, 'embed_tokens'):
            embed = neuron_model.embed_tokens
            print(f"embed_tokens type: {type(embed)}")
            if hasattr(embed, 'weight'):
                w = embed.weight
                print(f"embed_tokens weight: shape={w.shape}, dtype={w.dtype}")
                # Note: weight might be on Neuron device, so we can't inspect values directly

        # Check lm_head
        if hasattr(neuron_model, 'lm_head'):
            lm = neuron_model.lm_head
            print(f"lm_head type: {type(lm)}")
            if hasattr(lm, 'weight'):
                w = lm.weight
                print(f"lm_head weight: shape={w.shape}, dtype={w.dtype}")

        # Check first layer
        if hasattr(neuron_model, 'layers') and len(neuron_model.layers) > 0:
            layer0 = neuron_model.layers[0]
            print(f"layer0 type: {type(layer0)}")

            # Check attention
            if hasattr(layer0, 'self_attn'):
                attn = layer0.self_attn
                print(f"self_attn type: {type(attn)}")

                # Check qkv_proj
                qkv = attn.get_qkv_proj() if hasattr(attn, 'get_qkv_proj') else getattr(attn, 'qkv_proj', None)
                if qkv is not None:
                    print(f"qkv_proj type: {type(qkv)}")
                    if hasattr(qkv, 'q_proj') and hasattr(qkv.q_proj, 'weight'):
                        w = qkv.q_proj.weight
                        print(f"q_proj weight: shape={w.shape}, dtype={w.dtype}")
    else:
        print("Cannot access model.models - model structure may be different")
        print(f"Model attributes: {[a for a in dir(model) if not a.startswith('_')]}")


def test_simple_generation():
    """Test a very simple forward pass."""
    print("\n=== Testing Simple Generation ===")

    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Load model
    print("Loading model...")
    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)

    from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter
    generation_model = HuggingFaceGenerationAdapter(model)

    # Test with a simple prompt
    text = "1 + 1 ="
    print(f"Input: '{text}'")

    inputs = tokenizer([text], return_tensors="pt")
    print(f"Token IDs: {inputs.input_ids[0].tolist()}")

    # Generate just 10 tokens
    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=10,
        do_sample=False,
    )

    print(f"Output token IDs: {outputs[0].tolist()}")
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: '{decoded}'")


if __name__ == "__main__":
    test_tokenizer()
    # test_loaded_model_weights()
    # test_simple_generation()
