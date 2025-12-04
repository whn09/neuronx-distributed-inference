"""
Debug position_ids handling in MiniMax M2 model.
"""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

from transformers import AutoTokenizer
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights/"


def debug_position_handling():
    """Debug how position_ids are computed and passed to the model."""
    print("\n" + "="*60)
    print("Position ID handling debug")
    print("="*60)

    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Test input
    text = "Hello"
    inputs = tokenizer([text], return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    print(f"Input text: '{text}'")
    print(f"Input IDs: {input_ids}")
    print(f"Attention mask: {attention_mask}")

    # Check model config
    print(f"\nModel config:")
    print(f"  n_positions: {model.config.neuron_config.n_positions}")
    print(f"  seq_len: {model.config.neuron_config.seq_len}")
    print(f"  max_context_length: {model.config.neuron_config.max_context_length}")
    print(f"  buckets: {model.config.neuron_config.buckets}")
    print(f"  enable_bucketing: {model.config.neuron_config.enable_bucketing}")
    print(f"  padding_side: {model.config.neuron_config.padding_side}")

    # Compute position_ids as the HF adapter does
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    print(f"\nComputed position_ids: {position_ids}")

    # Check what the generation adapter does
    generation_model = HuggingFaceGenerationAdapter(model)
    print(f"\nGeneration adapter:")
    print(f"  padding_side: {generation_model.padding_side}")
    print(f"  kv_cache_populated (before forward): {model.kv_cache_populated}")

    # Simulate prepare_inputs_for_generation
    model_inputs = generation_model.prepare_inputs_for_generation(
        input_ids,
        attention_mask=attention_mask,
    )

    print(f"\nPrepared model inputs:")
    print(f"  input_ids shape: {model_inputs['input_ids'].shape}")
    print(f"  input_ids: {model_inputs['input_ids']}")
    print(f"  position_ids: {model_inputs['position_ids']}")
    print(f"  attention_mask shape: {model_inputs['attention_mask'].shape if model_inputs['attention_mask'] is not None else 'None'}")

    # Check if n_positions is much larger than input
    n_positions = model.config.neuron_config.n_positions
    input_len = input_ids.shape[1]
    print(f"\n--- Padding analysis ---")
    print(f"  n_positions (context length bucket): {n_positions}")
    print(f"  input_len: {input_len}")
    print(f"  padding ratio: {n_positions / input_len}x")

    # This is critical - if n_positions is 1024 and input is 1 token,
    # the attention mask should handle this correctly


def check_attention_mask_creation():
    """Check how attention mask is created in the model."""
    print("\n" + "="*60)
    print("Attention mask creation debug")
    print("="*60)

    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)

    # Access the underlying neuron model
    if hasattr(model, 'models') and len(model.models) > 0:
        neuron_model = model.models[0]
        print(f"Neuron model type: {type(neuron_model)}")

        # Check n_positions
        if hasattr(neuron_model, 'n_positions'):
            print(f"neuron_model.n_positions: {neuron_model.n_positions}")
        if hasattr(neuron_model, 'batch_size'):
            print(f"neuron_model.batch_size: {neuron_model.batch_size}")
        if hasattr(neuron_model, 'padding_side'):
            print(f"neuron_model.padding_side: {neuron_model.padding_side}")

        # Try to understand how create_attn_mask works
        if hasattr(neuron_model, 'create_attn_mask'):
            print("\nneuron_model.create_attn_mask exists")
            # We can't easily call it without proper inputs


if __name__ == "__main__":
    debug_position_handling()
    check_attention_mask_creation()
