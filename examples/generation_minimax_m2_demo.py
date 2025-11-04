import torch

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

model_path = "/home/ubuntu/model_hf/MiniMax-M2/"
traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2/"

torch.manual_seed(0)


def generate(skip_compile=False):
    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)

    if not skip_compile:
        # MiniMax M2 is a very large MoE model (62 layers, 256 experts)
        # Model uses GQA: num_attention_heads=48, num_key_value_heads=8
        # tp_degree must be a multiple of num_key_value_heads (8) for proper KV head distribution
        neuron_config = MoENeuronConfig(
            tp_degree=8,  # Must be multiple of num_key_value_heads=8
            moe_ep_degree=4,
            batch_size=1,
            max_context_length=128,
            seq_len=1024,
            on_device_sampling_config=OnDeviceSamplingConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95),
            enable_bucketing=False,
            flash_decoding_enabled=False
        )
        config = MiniMaxM2InferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )        
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token
        # Compile and save model.
        print("\nCompiling and saving model...")
        model = NeuronMiniMaxM2ForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt}
    ]
    # Apply chat template (enable_thinking may not be supported by all tokenizers)
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes if supported.
        )
    except TypeError:
        # Fallback if enable_thinking is not supported
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


if __name__ == "__main__":
    generate()
