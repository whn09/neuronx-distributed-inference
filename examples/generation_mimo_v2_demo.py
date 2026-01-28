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
1. TP Degree:
   - For small TP (<=4): Must divide the minimum num_kv_heads (4). Valid: 1, 2, 4.
   - For large TP (>4): Uses GQA CONVERT_TO_MHA mode where K/V are replicated
     to match num_attention_heads (64). Valid: 8, 16, 32, 64.
   - Recommended: TP=32 for this large model (fits on trn2.48xlarge)

2. Memory Requirements:
   - With TP=4, the model requires ~143GB per compilation unit
   - A single Trainium2 chip has ~24GB HBM
   - With TP=32, memory per chip is ~4.5GB which fits comfortably

3. Hybrid Attention:
   - Full attention layers (pattern=0): num_kv_heads=4, head_dim=192, v_head_dim=128
   - Sliding window layers (pattern=1): num_kv_heads=8, head_dim=192, v_head_dim=128
   - With CONVERT_TO_MHA (TP>4): K/V are replicated to 64 heads for proper TP splitting

4. Model Weights:
   This script expects BF16 weights. The original HuggingFace checkpoint is FP8 quantized,
   which should be converted to BF16 format first. See README_MiMo_V2_Flash.md for details.

   Usage:
   - Compile: python generation_mimo_v2_demo.py --compile-only
   - Generate: python generation_mimo_v2_demo.py --skip-compile --prompt "Hello!"
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
        default="/opt/dlami/nvme/models/MiMo-V2-Flash-BF16/",
        help="Path to BF16 model checkpoint (converted from FP8)",
    )
    parser.add_argument(
        "--compiled-model-path",
        type=str,
        default="/opt/dlami/nvme/traced_model/MiMo-V2-Flash-BF16/",
        help="Path to save/load compiled model",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=64,
        help="Tensor parallelism degree. For TP<=4, must divide num_kv_heads=4. "
             "For TP>4, uses CONVERT_TO_MHA mode. Valid: 1, 2, 4, 8, 16, 32, 64. "
             "Recommended: 64 for this large model (with logical_nc_config=2).",
    )
    parser.add_argument(
        "--moe-tp-degree",
        type=int,
        default=None,
        help="MoE tensor parallelism degree (default: tp_degree).",
    )
    parser.add_argument(
        "--moe-ep-degree",
        type=int,
        default=1,
        help="MoE expert parallelism degree. Note: EP>1 may not be supported in token generation.",
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
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1, use 1.0 to disable)",
    )

    args = parser.parse_args()

    # Validate TP degree
    MIN_NUM_KV_HEADS = 4  # Full attention uses 4 KV heads
    NUM_ATTENTION_HEADS = 64  # MiMo-V2-Flash has 64 attention heads

    if args.tp_degree <= MIN_NUM_KV_HEADS:
        # Standard GQA mode: TP must divide num_kv_heads
        if MIN_NUM_KV_HEADS % args.tp_degree != 0:
            raise ValueError(
                f"tp_degree ({args.tp_degree}) must divide num_kv_heads (4). "
                f"Valid values for small TP: 1, 2, 4."
            )
    else:
        # CONVERT_TO_MHA mode: TP must divide num_attention_heads
        if NUM_ATTENTION_HEADS % args.tp_degree != 0:
            raise ValueError(
                f"tp_degree ({args.tp_degree}) must divide num_attention_heads (64). "
                f"Valid values for large TP: 8, 16, 32, 64."
            )

    return args


def create_neuron_config(args):
    """Create NeuronConfig for MiMo-V2-Flash."""

    # Determine MoE TP/EP degrees
    moe_ep_degree = args.moe_ep_degree
    moe_tp_degree = args.moe_tp_degree if args.moe_tp_degree else args.tp_degree

    # Check if CONVERT_TO_MHA mode will be used
    MIN_NUM_KV_HEADS = 4
    use_convert_to_mha = args.tp_degree > MIN_NUM_KV_HEADS

    print(f"\nParallelism configuration:")
    print(f"  TP Degree (attention): {args.tp_degree}")
    print(f"  MoE TP Degree: {moe_tp_degree}")
    print(f"  MoE EP Degree: {moe_ep_degree}")
    print(f"  Experts per EP rank: {256 // moe_ep_degree}")
    print(f"  GQA Mode: {'CONVERT_TO_MHA (K/V replicated to 64 heads)' if use_convert_to_mha else 'Standard GQA'}")

    # Build config kwargs (following MiniMax M2 pattern)
    config_kwargs = dict(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        ctx_batch_size=args.batch_size,
        tkg_batch_size=args.batch_size,
        max_context_length=args.max_context_length,
        seq_len=args.seq_len,

        # Data type - important for numerical stability
        torch_dtype=torch.bfloat16,

        # MoE parallelism
        moe_tp_degree=moe_tp_degree,
        moe_ep_degree=moe_ep_degree,

        # Sampling configuration
        # NOTE: on_device_sampling causes gather_output=False for lm_head,
        # which may cause issues with HuggingFace generation adapter.
        # Disable for now to ensure proper logits gathering.
        # on_device_sampling_config=OnDeviceSamplingConfig(
        #     do_sample=False,  # Greedy for deterministic results
        # ),

        # Disable bucketing for simpler testing
        enable_bucketing=False,

        # Flash decoding
        flash_decoding_enabled=False,

        # Sequence parallel - enabled following MiniMax M2
        sequence_parallel_enabled=True,

        # Logical NC config - splits 32 physical cores into 64 logical cores
        # Required for tp_degree=64 on trn2.48xlarge
        logical_nc_config=2,

        # Fused QKV projection - disabled because MiMo-V2 has different head_dim
        # for Q/K (192) vs V (128), which is incompatible with standard fused_qkv
        fused_qkv=False,

        # Disable continuous batching for simpler testing
        is_continuous_batching=False,

        # Disable kernel optimizations for debugging
        qkv_kernel_enabled=False,
        attn_kernel_enabled=False,
        strided_context_parallel_kernel_enabled=False,

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

        # Enable pre-sharded checkpoints for faster loading
        save_sharded_checkpoint=True,
    )

    neuron_config = MoENeuronConfig(**config_kwargs)

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


def generate(model, tokenizer, prompt, max_new_tokens=128, repetition_penalty=1.1):
    """Generate text using the model."""
    print(f"\n{'='*60}")
    print("Generating text")
    print(f"{'='*60}")
    print(f"Prompt: '{prompt}'")

    # Tokenize input
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    print(f"Input token IDs: {inputs.input_ids[0].tolist()}")
    print(f"Input length: {input_length}")

    # Create generation adapter
    generation_model = HuggingFaceGenerationAdapter(model)

    # Get EOS token ID
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    # Calculate max_length from max_new_tokens
    max_length = min(input_length + max_new_tokens, model.config.neuron_config.max_length)

    print(f"\nGeneration parameters:")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  max_length: {max_length}")
    print(f"  repetition_penalty: {repetition_penalty}")
    print(f"  eos_token_id: {eos_token_id}")

    # Generate
    print(f"\nGenerating...")
    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # Greedy decoding
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        repetition_penalty=repetition_penalty,
    )

    # Get only newly generated tokens for display
    new_tokens = outputs.shape[1] - input_length
    print(f"Output token IDs shape: {outputs.shape}")
    print(f"New tokens generated: {new_tokens}")

    # Decode output
    output_text = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Also decode just the generated part (excluding prompt)
    generated_only = tokenizer.batch_decode(
        outputs[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print(f"\n{'='*60}")
    print("Generated output")
    print(f"{'='*60}")
    for i, (full_text, gen_text) in enumerate(zip(output_text, generated_only)):
        print(f"\nOutput {i}:")
        print(f"[Full]: {full_text}")
        print(f"\n[Generated only]: {gen_text}")
        print(f"\n(generated {new_tokens} tokens, {len(gen_text)} chars)")

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
    generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()
