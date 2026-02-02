"""
MiniMax M2 generation demo v4 for NXD inference on Trainium2.

This implementation borrows from MiMo-V2-Flash demo to support:
- Expert parallelism (EP) with hybrid sharding
- Long context support
- Command-line arguments for flexible configuration

MiniMax M2 is a large MoE model with:
- 62 layers
- 256 routed experts, top-8 routing
- GQA: num_attention_heads=48, num_key_value_heads=8
- Partial RoPE: rotary_dim=64, head_dim=128
- Sigmoid router with e_score_correction_bias

IMPORTANT CONSTRAINTS:
1. TP Degree:
   - Must be a multiple of num_key_value_heads (8) for proper KV head distribution
   - Valid: 8, 16, 32, 64
   - Recommended: 64 for this large model (with logical_nc_config=2)

2. Expert Parallelism (EP):
   EP distributes experts across ranks to reduce memory per rank. With 256 experts and
   world_size=64, valid configurations are:
   - EP=64, MoE_TP=1: 4 experts per rank (maximum EP)
   - EP=32, MoE_TP=2: 8 experts per rank
   - EP=16, MoE_TP=4: 16 experts per rank
   - EP=8, MoE_TP=8: 32 experts per rank
   - EP=1, MoE_TP=64: 256 experts per rank (default, no EP)

   NOTE: EP is only supported during prefill (context encoding). When EP>1, hybrid
   sharding is automatically enabled to use EP=1 for token generation.

   Usage:
   - Compile: python generation_minimax_m2_v4_demo.py --compile-only
   - Compile with EP: python generation_minimax_m2_v4_demo.py --compile-only --moe-ep-degree 64 --moe-tp-degree 1
   - Generate: python generation_minimax_m2_v4_demo.py --skip-compile --prompt "Hello!"
"""

import argparse
import os
import torch

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_v3 import (
    MiniMaxM2InferenceConfigV3,
    NeuronMiniMaxM2ForCausalLMV3,
)
from neuronx_distributed_inference.utils.hf_adapter import (
    HuggingFaceGenerationAdapter,
    load_pretrained_config,
)

torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMax M2 generation demo v4")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/opt/dlami/nvme/model_hf/MiniMax-M2-BF16/",
        help="Path to BF16 model checkpoint (converted from FP8)",
    )
    parser.add_argument(
        "--compiled-model-path",
        type=str,
        default="/opt/dlami/nvme/traced_model/MiniMax-M2-BF16-v4/",
        help="Path to save/load compiled model",
    )
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=64,
        help="Tensor parallelism degree. Must be a multiple of num_kv_heads=8. "
             "Valid: 8, 16, 32, 64. Recommended: 64 for this large model (with logical_nc_config=2).",
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
        help="MoE expert parallelism degree for prefill (context encoding). "
             "When EP>1, hybrid sharding is automatically enabled with EP=1 for token generation.",
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
        default=256,
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
        default="Give me a short introduction to large language models.",
        help="Prompt for text generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=True,
        help="Enable thinking mode for MiniMax M2 (default: True)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking mode",
    )
    parser.add_argument(
        "--enable-bucketing",
        action="store_true",
        help="Enable bucketing for variable sequence lengths",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark after generation",
    )

    args = parser.parse_args()

    # Handle thinking mode
    if args.no_thinking:
        args.enable_thinking = False

    # Validate TP degree
    NUM_KV_HEADS = 8  # MiniMax M2 has 8 KV heads

    if args.tp_degree % NUM_KV_HEADS != 0:
        raise ValueError(
            f"tp_degree ({args.tp_degree}) must be a multiple of num_kv_heads ({NUM_KV_HEADS}). "
            f"Valid values: 8, 16, 32, 64."
        )

    return args


def create_neuron_config(args):
    """Create NeuronConfig for MiniMax M2."""

    # MiniMax M2 model constants
    NUM_EXPERTS = 256
    TOP_K = 8

    # Determine MoE TP/EP degrees
    moe_ep_degree = args.moe_ep_degree
    moe_tp_degree = args.moe_tp_degree if args.moe_tp_degree else args.tp_degree

    # EP support in token generation depends on batch size
    # The threshold is: batch_size * top_k / num_experts >= 1.0
    # For MiniMax M2: batch_size >= 256 / 8 = 32
    min_batch_for_ep = NUM_EXPERTS // TOP_K  # = 32

    # Determine if we can use EP directly or need hybrid sharding
    hybrid_sharding_config = None
    if moe_ep_degree > 1:
        # Validate that moe_tp_degree * moe_ep_degree = tp_degree (world_size)
        if moe_tp_degree * moe_ep_degree != args.tp_degree:
            raise ValueError(
                f"With EP>1, moe_tp_degree ({moe_tp_degree}) x moe_ep_degree ({moe_ep_degree}) "
                f"must equal tp_degree ({args.tp_degree}). "
                f"Try: --moe-tp-degree {args.tp_degree // moe_ep_degree}"
            )

        if args.batch_size >= min_batch_for_ep:
            # Batch size is large enough to use EP directly without hybrid sharding
            use_hybrid_sharding = False
            print(f"\n  NOTE: batch_size ({args.batch_size}) >= {min_batch_for_ep}, "
                  f"EP will work in token generation without hybrid sharding")
        else:
            # Batch size too small, need hybrid sharding (EP only in prefill)
            use_hybrid_sharding = True
            print(f"\n  NOTE: batch_size ({args.batch_size}) < {min_batch_for_ep}, "
                  f"using hybrid sharding (EP only in prefill, EP=1 in token generation)")
            print(f"  TIP: Use --batch-size {min_batch_for_ep} or higher for full EP support")
    else:
        use_hybrid_sharding = False

    print(f"\nParallelism configuration:")
    print(f"  TP Degree (attention): {args.tp_degree}")
    print(f"  MoE TP Degree: {moe_tp_degree}")
    print(f"  MoE EP Degree: {moe_ep_degree}")
    print(f"  Experts per EP rank: {NUM_EXPERTS // moe_ep_degree}")
    if use_hybrid_sharding:
        print(f"  Hybrid Sharding: ENABLED (CTE: EP={moe_ep_degree}/TP={moe_tp_degree}, TKG: EP=1/TP={args.tp_degree})")

    # Memory estimation for token generation
    # MiniMax M2 has intermediate_size=1536 (small per expert)
    # With 256 experts and EP=1, all experts are on each rank
    INTERMEDIATE_SIZE = 1536  # MiniMax M2's actual intermediate_size per expert
    HIDDEN_SIZE = 3072  # MiniMax M2's hidden_size
    intermediate_per_tp = INTERMEDIATE_SIZE // moe_tp_degree
    tkg_ep_degree = 1 if use_hybrid_sharding else moe_ep_degree
    experts_per_rank = NUM_EXPERTS // tkg_ep_degree

    print(f"\n  Memory estimate for token generation:")
    print(f"    Experts per TKG rank: {experts_per_rank}")
    print(f"    Intermediate size: {INTERMEDIATE_SIZE}")
    print(f"    Hidden size: {HIDDEN_SIZE}")
    print(f"    Seq len: {args.seq_len}")

    # Build config kwargs
    config_kwargs = dict(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        ctx_batch_size=args.batch_size,
        tkg_batch_size=args.batch_size,
        max_context_length=args.max_context_length,
        seq_len=args.seq_len,

        # Data type
        torch_dtype=torch.bfloat16,

        # Sampling configuration
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=True,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
        ),

        # Bucketing
        enable_bucketing=args.enable_bucketing,

        # Flash decoding
        flash_decoding_enabled=False,

        # Sequence parallel
        sequence_parallel_enabled=True,

        # Logical NC config - splits 32 physical cores into 64 logical cores
        # Required for tp_degree=64 on trn2.48xlarge
        logical_nc_config=2,

        # Fused QKV - enabled for MiniMax M2
        fused_qkv=True,

        # Continuous batching
        is_continuous_batching=False,

        # Async mode
        async_mode=False,

        # === MoE Optimizations ===
        # Index calculation kernel
        use_index_calc_kernel=True,

        # Mask padded tokens in MoE computation
        moe_mask_padded_tokens=True,

        # === Kernel optimizations ===
        qkv_kernel_enabled=False,
        qkv_nki_kernel_enabled=False,
        attn_kernel_enabled=False,
        strided_context_parallel_kernel_enabled=False,

        # MoE configuration - MiniMax M2 uses sigmoid activation for router
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

        # Blockwise matmul configuration
        # NOTE: MiniMax M2 has intermediate_size=1536, which when divided by tp_degree=64
        # gives I_TP=24. This is NOT divisible by SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP=256,
        # so enabling use_shard_on_intermediate_dynamic_while would pad intermediate_size
        # from 1536 to 16384 (10x increase!), causing OOM.
        # Keep this disabled for MiniMax M2.
        blockwise_matmul_config={
            "use_shard_on_intermediate_dynamic_while": False,
            "skip_dma_token": True,
        },

        # Workaround for extra add/multiply in all-gather/reduce-scatter CC ops
        disable_numeric_cc_token=True,

        # Scratchpad page size for large tensors
        scratchpad_page_size=1024,
    )

    # Add bucketing configuration if enabled
    if args.enable_bucketing:
        config_kwargs['context_encoding_buckets'] = [args.max_context_length]
        config_kwargs['token_generation_buckets'] = [args.seq_len]

    # Configure MoE parallelism
    if use_hybrid_sharding:
        # Hybrid sharding: different EP/TP for prefill vs decode
        config_kwargs['hybrid_sharding_config'] = {
            'moe_cte_tp_degree': moe_tp_degree,
            'moe_cte_ep_degree': moe_ep_degree,
            'moe_tkg_tp_degree': args.tp_degree,  # Full TP for token generation
            'moe_tkg_ep_degree': 1,  # No EP for token generation (not supported)
        }
        # When using hybrid sharding, these are not used directly
        config_kwargs['moe_tp_degree'] = 1
        config_kwargs['moe_ep_degree'] = 1
    else:
        # Standard configuration: same parallelism for both phases
        config_kwargs['moe_tp_degree'] = moe_tp_degree
        config_kwargs['moe_ep_degree'] = moe_ep_degree

    neuron_config = MoENeuronConfig(**config_kwargs)

    return neuron_config


def compile_model(args):
    """Compile MiniMax M2 model for Neuron."""
    from transformers import AutoConfig

    print(f"\n{'='*60}")
    print("Compiling MiniMax M2 model (v4)")
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
    print(f"  num_local_experts: {hf_config.num_local_experts}")
    print(f"  num_experts_per_tok: {hf_config.num_experts_per_tok}")
    print(f"  head_dim: {hf_config.head_dim}")
    print(f"  rotary_dim: {getattr(hf_config, 'rotary_dim', hf_config.head_dim)}")

    # Create inference config
    config = MiniMaxM2InferenceConfigV3(
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
    print("\nCreating NeuronMiniMaxM2ForCausalLMV3...")
    model = NeuronMiniMaxM2ForCausalLMV3(args.model_path, config)

    print("\nCompiling model (this may take a while)...")
    model.compile(args.compiled_model_path)

    # Save tokenizer
    print("\nSaving tokenizer...")
    tokenizer.save_pretrained(args.compiled_model_path)

    print(f"\nCompilation complete! Model saved to: {args.compiled_model_path}")

    return model, tokenizer


def load_model(args):
    """Load compiled MiniMax M2 model."""
    print(f"\n{'='*60}")
    print("Loading compiled MiniMax M2 model")
    print(f"{'='*60}")
    print(f"Compiled model path: {args.compiled_model_path}")

    # Load model
    print("\nLoading model...")
    model = NeuronMiniMaxM2ForCausalLMV3(args.compiled_model_path)
    model.load(args.compiled_model_path)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.compiled_model_path,
        trust_remote_code=True,
    )

    print("Model loaded successfully!")

    return model, tokenizer


def generate(model, tokenizer, prompt, args):
    """Generate text using the model."""
    print(f"\n{'='*60}")
    print("Generating text")
    print(f"{'='*60}")
    print(f"Prompt: '{prompt}'")
    print(f"Thinking mode: {'enabled' if args.enable_thinking else 'disabled'}")

    # Apply chat template
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )

    # Tokenize input
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    print(f"Input length: {input_length} tokens")

    # Create generation adapter
    generation_model = HuggingFaceGenerationAdapter(model)

    # Get EOS token ID
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    # Calculate max_length from max_new_tokens
    max_length = min(input_length + args.max_new_tokens, model.config.neuron_config.max_length)

    print(f"\nGeneration parameters:")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  max_length: {max_length}")
    print(f"  eos_token_id: {eos_token_id}")

    # Load generation config if available
    try:
        generation_config = GenerationConfig.from_pretrained(args.model_path)
    except Exception:
        generation_config = None

    # Generate
    print(f"\nGenerating...")
    if generation_config:
        outputs = generation_model.generate(
            inputs.input_ids,
            generation_config=generation_config,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
        )
    else:
        outputs = generation_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
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


def run_benchmark(model, args):
    """Run performance benchmark."""
    from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

    print(f"\n{'='*60}")
    print("Performance Benchmarking")
    print(f"{'='*60}")

    try:
        generation_config = GenerationConfig.from_pretrained(args.model_path)
    except Exception:
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
        )

    benchmark_sampling(
        model=model,
        draft_model=None,
        generation_config=generation_config,
        target="all",
        benchmark_report_path="benchmark_report_minimax_m2_v4.json",
        num_runs=5,
    )


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
    generate(model, tokenizer, args.prompt, args)

    # Run benchmark if requested
    if args.benchmark:
        run_benchmark(model, args)


if __name__ == "__main__":
    main()
