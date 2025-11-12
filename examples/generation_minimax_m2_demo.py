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
            tp_degree=64,  # Must be multiple of num_key_value_heads=8
            # ep_degree=64,
            # moe_tp_degree=16,
            # moe_ep_degree=4,
            batch_size=1,
            max_context_length=128,
            seq_len=1024,
            on_device_sampling_config=OnDeviceSamplingConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95),
            enable_bucketing=False,
            flash_decoding_enabled=False,
            save_sharded_checkpoint=True,  # ← 启用！保存分片权重，加载时快很多
            # Use torch implementation to bypass NKI kernel's DGE limitation
            # (intermediate_size=1536 / tp_degree=64 = 24 < 32 required by DGE)
            blockwise_matmul_config={
                'use_torch_block_wise': True,
            },
            # Enable FP8 quantization for MLP layers
            # This automatically sets quantization_dtype="f8e4m3" and quantization_type="per_channel_symmetric"
            quantized_mlp_kernel_enabled=True,
            # Specify modules that should NOT be quantized (matching HF config's quantization_config)
            # These modules don't have FP8 weights and scale parameters
            modules_to_not_convert=[
                "lm_head",
                # Note: "gate" and "e_score_correction_bias" are already excluded in the model architecture
            ],
            # Disable fused_qkv when using FP8 quantization
            # FP8 quantization requires separate scale parameters for Q, K, V
            fused_qkv=False,
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
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes if supported.
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
    # Step 1: Compile and save sharded checkpoint (run once, takes time)
    generate(skip_compile=False)

    # Step 2: After compilation, use this for fast loading
    # generate(skip_compile=True)
