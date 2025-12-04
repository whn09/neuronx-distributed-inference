"""
MiniMax M2 generation demo using the v2 implementation.
Clean implementation following Qwen3 MoE pattern.
"""
import torch

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_v2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

# Use BF16 checkpoint (converted from FP8 on GPU)
model_path = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-BF16-v2/"

torch.manual_seed(0)


def generate(skip_compile=False):
    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)

    if not skip_compile:
        # MiniMax M2 is a very large MoE model (62 layers, 256 experts)
        # Model uses GQA: num_attention_heads=48, num_key_value_heads=8
        # tp_degree must be a multiple of num_key_value_heads (8) for proper KV head distribution
        neuron_config = MoENeuronConfig(
            tp_degree=32,  # Must be multiple of num_key_value_heads=8
            batch_size=1,
            max_context_length=128,
            seq_len=1024,
            on_device_sampling_config=OnDeviceSamplingConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95),
            enable_bucketing=False,
            flash_decoding_enabled=False,
            # MiniMax M2 uses sigmoid activation for router (not softmax)
            router_config={
                'act_fn': 'sigmoid',
            },
        )
        config = MiniMaxM2InferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        # Compile and save model.
        print("\nCompiling and saving model (v2 implementation)...")
        model = NeuronMiniMaxM2ForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\n=== Generating outputs ===")

    # Simple text completion test
    text = "Hello"
    print(f"Input: '{text}'")

    inputs = tokenizer([text], padding=True, return_tensors="pt")
    print(f"Input token IDs: {inputs.input_ids[0].tolist()}")

    generation_model = HuggingFaceGenerationAdapter(model)
    print(f"\nGenerating with max_length={model.config.neuron_config.max_length}...")

    # Use greedy decoding for deterministic test
    print("Using GREEDY decoding (do_sample=False)")
    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        do_sample=False,
    )
    print(f"Output token IDs shape: {outputs.shape}")
    print(f"Output token IDs (first 20): {outputs[0, :20].tolist()}")

    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("\n=== Generated outputs ===")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}:\n{output_token}")
        print(f"(total length: {len(output_token)} chars)")


if __name__ == "__main__":
    # Compile and run
    generate(skip_compile=False)
