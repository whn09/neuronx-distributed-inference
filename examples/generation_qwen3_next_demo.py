"""
Qwen3 Next generation demo for Trainium2.

Qwen3 Next is a hybrid attention model with:
- Full softmax attention (every full_attention_interval layers, default=4)
- Gated Delta Net linear attention (other layers)
- MoE with 512 experts and shared experts
- Partial RoPE (25% of head dimensions)

Model: Qwen3-Next-80B-A3B-Instruct
- 80B total parameters, 3B active parameters
- 48 hidden layers
- 512 experts, 10 experts per token
- head_dim=256, hidden_size=2048
"""
import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextInferenceConfig,
    NeuronQwen3NextForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling


# Model paths - update these to your local paths
model_path = "/home/ubuntu/model_hf/Qwen3-Next-80B-A3B-Instruct/"
traced_model_path = "/home/ubuntu/traced_model/Qwen3-Next-80B-A3B-Instruct/"

torch.manual_seed(0)

# Use bfloat16 for Qwen3 Next
DTYPE = torch.bfloat16


def generate(skip_compile=False):
    """Compile and run Qwen3 Next model for text generation."""

    # Initialize generation config from pretrained
    generation_config = GenerationConfig.from_pretrained(model_path)

    if not skip_compile:
        # Configure Neuron settings for Qwen3 Next
        # This model has 512 experts
        # NOTE: moe_ep_degree must be 1 because EP is not supported with token generation
        neuron_config = MoENeuronConfig(
            # Tensor parallelism configuration
            tp_degree=64,  # Total tensor parallel degree
            moe_tp_degree=1,  # MoE uses TP only (no intermediate dim sharding)
            # moe_ep_degree defaults to 1 (no expert parallelism - required for token gen)

            # Batch and sequence configuration
            batch_size=8,
            ctx_batch_size=1,
            tkg_batch_size=8,
            seq_len=8192,  # Max sequence length
            max_context_length=4096,

            # Memory optimization
            scratchpad_page_size=1024,

            # Data type
            torch_dtype=DTYPE,

            # Sampling configuration
            on_device_sampling_config=OnDeviceSamplingConfig(
                do_sample=True,
                temperature=0.6,
                top_k=20,
                top_p=0.95,
            ),

            # Attention and parallelism settings
            enable_bucketing=False,
            flash_decoding_enabled=False,
            attention_dp_degree=8,
            cp_degree=8,  # Context parallelism

            # Fused operations
            # NOTE: fused_qkv disabled because HF Qwen3 Next has different Q size (32 heads vs config's 16)
            fused_qkv=False,
            is_continuous_batching=True,
            logical_nc_config=2,
            sequence_parallel_enabled=True,

            # Kernel optimizations
            # Note: Linear attention layers will not use these kernels
            # IMPORTANT: Both kernels disabled because Qwen3 Next has head_dim=256,
            # but the NKI kernels only support head_dim <= 128
            qkv_kernel_enabled=False,  # head_dim=256 not supported
            attn_kernel_enabled=False,  # head_dim=256 not supported

            # MoE optimizations
            blockwise_matmul_config={
                "use_shard_on_intermediate_dynamic_while": True,
                "skip_dma_token": True,
            },
        )

        # Create inference config
        config = Qwen3NextInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )
        # Set the original model path for weight loading
        config._name_or_path = model_path

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        # Compile and save model
        print("\n" + "=" * 60)
        print("Compiling Qwen3 Next model for Trainium2...")
        print("=" * 60)
        print(f"Model path: {model_path}")
        print(f"Traced model path: {traced_model_path}")
        print(f"TP degree: {neuron_config.tp_degree}")
        print(f"MoE TP degree: {neuron_config.moe_tp_degree}")
        print(f"MoE EP degree: {neuron_config.moe_ep_degree}")
        print(f"Batch size: {neuron_config.batch_size}")
        print(f"Sequence length: {neuron_config.seq_len}")
        print(f"Attention type pattern: {config.layer_types[:8]}... (repeats)")
        print("=" * 60)

        model = NeuronQwen3NextForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)
        print("Compilation complete!")

    # Load from compiled checkpoint
    print("\n" + "=" * 60)
    print("Loading model from compiled checkpoint...")
    print("=" * 60)
    # Initialize from traced path (has neuron config with _name_or_path pointing to HF weights)
    model = NeuronQwen3NextForCausalLM(traced_model_path)
    model.load(traced_model_path)  # Weights loaded from config._name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Generate outputs
    print("\n" + "=" * 60)
    print("Generating outputs...")
    print("=" * 60)

    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Enable thinking mode for Qwen3 Next
    )

    inputs = tokenizer([text], padding=True, return_tensors="pt")

    # Create generation adapter
    generation_model = HuggingFaceGenerationAdapter(model)

    # Generate
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
    )

    # Decode and print outputs
    output_tokens = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\nGenerated outputs:")
    print("-" * 40)
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")
    print("-" * 40)

    # Performance benchmarking
    print("\n" + "=" * 60)
    print("Performance Benchmarking")
    print("=" * 60)
    benchmark_sampling(
        model=model,
        draft_model=None,
        generation_config=generation_config,
        target="all",
        benchmark_report_path="qwen3_next_benchmark_report.json",
        num_runs=5,
    )


def generate_simple(skip_compile=True):
    """Simplified generation for testing with smaller configuration."""

    generation_config = GenerationConfig.from_pretrained(model_path)

    if not skip_compile:
        # Smaller configuration for testing
        neuron_config = MoENeuronConfig(
            tp_degree=32,
            moe_tp_degree=4,
            moe_ep_degree=8,
            batch_size=1,
            ctx_batch_size=1,
            tkg_batch_size=1,
            seq_len=2048,
            max_context_length=1024,
            torch_dtype=DTYPE,
            on_device_sampling_config=OnDeviceSamplingConfig(
                do_sample=True,
                temperature=0.6,
                top_k=20,
            ),
            enable_bucketing=False,
            flash_decoding_enabled=False,
            fused_qkv=True,
            is_continuous_batching=False,
            sequence_parallel_enabled=True,
            qkv_kernel_enabled=False,  # Disable for initial testing
            attn_kernel_enabled=False,
        )

        config = Qwen3NextInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        print("Compiling Qwen3 Next model (simple config)...")
        model = NeuronQwen3NextForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load and generate
    print("Loading model...")
    model = NeuronQwen3NextForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    prompt = "What is machine learning?"
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")

    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=512,
    )

    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output_text[0]}")


def generate_minimal_test(skip_compile=False):
    """Minimal configuration that has been tested to compile successfully.

    Uses tp_degree=8 which works on smaller instances (trn2.8xlarge or similar).
    This configuration bypasses several compiler issues found during development.
    """

    if not skip_compile:
        # Minimal working configuration - tested successfully
        neuron_config = MoENeuronConfig(
            tp_degree=8,  # Reduced TP to avoid compiler bugs
            moe_tp_degree=1,  # No intermediate dimension sharding
            # moe_ep_degree defaults to 1 (no expert parallelism)
            batch_size=1,
            ctx_batch_size=1,
            tkg_batch_size=1,
            seq_len=512,
            max_context_length=256,
            torch_dtype=DTYPE,
            on_device_sampling_config=None,  # Disable on-device sampling for testing
            enable_bucketing=False,
            flash_decoding_enabled=False,
            fused_qkv=True,
            is_continuous_batching=False,
            sequence_parallel_enabled=False,
            # IMPORTANT: Disable kernels due to head_dim=256 limitation
            qkv_kernel_enabled=False,
            attn_kernel_enabled=False,
        )

        config = Qwen3NextInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        print("\n" + "=" * 60)
        print("Compiling Qwen3 Next model (minimal test config)...")
        print("=" * 60)
        print(f"TP degree: {neuron_config.tp_degree}")
        print(f"Batch size: {neuron_config.batch_size}")
        print(f"Sequence length: {neuron_config.seq_len}")
        print(f"Full attention layers: {sum(config.attn_type_list)}")
        print(f"Linear attention layers: {len(config.attn_type_list) - sum(config.attn_type_list)}")
        print("=" * 60)

        model = NeuronQwen3NextForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)
        print("Compilation complete!")

    # Load and test
    print("\nLoading model...")
    model = NeuronQwen3NextForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    prompt = "What is AI?"
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")

    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_model = HuggingFaceGenerationAdapter(model)

    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=256,
    )

    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output_text[0]}")


if __name__ == "__main__":
    # Full generation with tp_degree=64 (requires trn2.48xlarge)
    # Use skip_compile=True to load existing compiled model
    generate(skip_compile=True)  # Load from compiled checkpoint

    # Use the minimal test config (tp_degree=8, for smaller instances)
    # generate_minimal_test(skip_compile=False)

    # Skip compilation and load from checkpoint
    # generate(skip_compile=True)

    # Or use simplified version for testing
    # generate_simple(skip_compile=False)
