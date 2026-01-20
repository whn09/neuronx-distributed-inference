"""
MiMo-V2-Flash generation demo for NXD inference on Trainium2.

MiMo-V2-Flash is a large MoE model from Xiaomi with:
- 48 layers
- 256 routed experts, top-8 routing
- Hybrid attention (full + sliding window)
- Different Q/K dim (192) and V dim (128)
- Partial RoPE (34% of dimensions)

Reference: https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash

IMPORTANT CONSTRAINTS:
1. TP Degree: Must divide the minimum num_kv_heads (4). Valid values: 1, 2, 4.
   - Full attention uses num_kv_heads=4
   - Sliding window attention uses num_kv_heads=8
   - Common divisor is 4, so TP must be 1, 2, or 4.

2. Memory Requirements:
   - With TP=4, the model requires ~143GB per compilation unit
   - A single Trainium2 chip has ~24GB HBM
   - Full compilation requires distributed inference across multiple nodes,
     expert parallelism (EP), or model quantization.

3. Hybrid Attention:
   - Full attention layers (pattern=0): num_kv_heads=4, head_dim=192, v_head_dim=128
   - Sliding window layers (pattern=1): num_kv_heads=8, head_dim=192, v_head_dim=128
   - The KV cache uses the maximum num_kv_heads (8) for compatibility.
"""

import argparse
import os
import torch

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.mimo_v2.modeling_mimo_v2 import (
    MiMoV2InferenceConfig,
    NeuronMiMoV2ForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)

torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="MiMo-V2-Flash generation demo")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/ubuntu/models/MiMo-V2-Flash/",
        help="Path to HuggingFace model checkpoint",
    )
    parser.add_argument(
        "--compiled-model-path",
        type=str,
        default="/home/ubuntu/traced_model/MiMo-V2-Flash/",
        help="Path to save/load compiled model",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=4,
        help="Tensor parallelism degree. Must divide num_kv_heads=4. Valid: 1, 2, 4.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=128,
        help="Maximum context length for prefill",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Total sequence length (context + generation)",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip compilation and load from compiled checkpoint",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the model, don't run generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you today?",
        help="Prompt for text generation",
    )
    args = parser.parse_args()

    # Validate TP degree
    MIN_NUM_KV_HEADS = 4  # Full attention uses 4 KV heads
    if MIN_NUM_KV_HEADS % args.tp_degree != 0:
        raise ValueError(
            f"tp_degree ({args.tp_degree}) must divide num_kv_heads (4). "
            f"Valid values: 1, 2, 4."
        )

    return args


def create_neuron_config(args):
    """Create NeuronConfig for MiMo-V2-Flash."""

    # MiMo-V2-Flash uses:
    # - num_key_value_heads = 4 (full attention)
    # - num_key_value_heads = 8 (sliding window attention)
    # tp_degree must be divisible by the minimum (4)
    assert args.tp_degree % 4 == 0, f"tp_degree must be divisible by 4, got {args.tp_degree}"

    neuron_config = MoENeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        max_context_length=args.max_context_length,
        seq_len=args.seq_len,

        # Sampling configuration
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=False,  # Greedy for deterministic results
        ),

        # Disable bucketing for simpler testing
        enable_bucketing=False,

        # Flash decoding
        flash_decoding_enabled=False,

        # MoE configuration
        # MiMo uses sigmoid activation for router
        router_config={
            'act_fn': 'sigmoid',
            'dtype': torch.float32,
        },

        # GLU MLP
        glu_mlp=True,

        # Normalize top-k affinities
        normalize_top_k_affinities=True,
    )

    return neuron_config


def compile_model(args):
    """Compile MiMo-V2-Flash model for Neuron."""
    from transformers import AutoConfig

    print(f"\n{'='*60}")
    print("Compiling MiMo-V2-Flash model")
    print(f"{'='*60}")
    print(f"Model path: {args.model_path}")
    print(f"Compiled model path: {args.compiled_model_path}")
    print(f"TP degree: {args.tp_degree}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max context length: {args.max_context_length}")
    print(f"Sequence length: {args.seq_len}")

    # Create neuron config
    neuron_config = create_neuron_config(args)

    # Load HuggingFace config for display
    print("\nLoading HuggingFace config...")
    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"  hidden_size: {hf_config.hidden_size}")
    print(f"  num_hidden_layers: {hf_config.num_hidden_layers}")
    print(f"  num_attention_heads: {hf_config.num_attention_heads}")
    print(f"  num_key_value_heads: {hf_config.num_key_value_heads}")
    print(f"  n_routed_experts: {hf_config.n_routed_experts}")
    print(f"  num_experts_per_tok: {hf_config.num_experts_per_tok}")

    # Create inference config
    config = MiMoV2InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(args.model_path),
    )

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create and compile model
    print("\nCreating NeuronMiMoV2ForCausalLM...")
    model = NeuronMiMoV2ForCausalLM(args.model_path, config)

    print("\nCompiling model (this may take a while)...")
    model.compile(args.compiled_model_path)

    # Save tokenizer
    print("\nSaving tokenizer...")
    tokenizer.save_pretrained(args.compiled_model_path)

    print(f"\nCompilation complete! Model saved to: {args.compiled_model_path}")

    return model, tokenizer


def load_model(args):
    """Load compiled MiMo-V2-Flash model."""
    print(f"\n{'='*60}")
    print("Loading compiled MiMo-V2-Flash model")
    print(f"{'='*60}")
    print(f"Compiled model path: {args.compiled_model_path}")

    # Load model
    print("\nLoading model...")
    model = NeuronMiMoV2ForCausalLM(args.compiled_model_path)
    model.load(args.compiled_model_path)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.compiled_model_path,
        trust_remote_code=True,
    )

    print("Model loaded successfully!")

    return model, tokenizer


def generate(model, tokenizer, prompt, max_length):
    """Generate text using the model."""
    print(f"\n{'='*60}")
    print("Generating text")
    print(f"{'='*60}")
    print(f"Prompt: '{prompt}'")

    # Tokenize input
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    print(f"Input token IDs: {inputs.input_ids[0].tolist()}")
    print(f"Input length: {inputs.input_ids.shape[1]}")

    # Create generation adapter
    generation_model = HuggingFaceGenerationAdapter(model)

    # Generate
    print(f"\nGenerating with max_length={max_length}...")
    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        do_sample=False,  # Greedy decoding
    )

    print(f"Output token IDs shape: {outputs.shape}")
    print(f"Output token IDs (first 30): {outputs[0, :30].tolist()}")

    # Decode output
    output_text = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print(f"\n{'='*60}")
    print("Generated output")
    print(f"{'='*60}")
    for i, text in enumerate(output_text):
        print(f"\nOutput {i}:")
        print(text)
        print(f"\n(total length: {len(text)} chars, {outputs.shape[1]} tokens)")

    return output_text


def main():
    args = parse_args()

    # Compile or load model
    if args.skip_compile:
        if not os.path.exists(args.compiled_model_path):
            raise ValueError(
                f"Compiled model not found at {args.compiled_model_path}. "
                "Run without --skip-compile first."
            )
        model, tokenizer = load_model(args)
    else:
        model, tokenizer = compile_model(args)

        if args.compile_only:
            print("\nCompile-only mode, skipping generation.")
            return

        # Reload for generation
        model, tokenizer = load_model(args)

    # Generate text
    max_length = model.config.neuron_config.max_length
    generate(model, tokenizer, args.prompt, max_length)


if __name__ == "__main__":
    main()
