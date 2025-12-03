"""
MiniMax M2 Demo with Expert Parallelism (EP)

Expert Parallelism distributes the 256 experts across multiple cores,
while keeping attention computation within TP groups.

For MiniMax M2 (256 experts):
- EP=64: 4 experts per core (current TP=64 default)
- EP=32: 8 experts per core
- EP=16: 16 experts per core
- EP=8:  32 experts per core

EP + TP Configuration:
- Total cores = EP × TP (or more complex with hybrid sharding)
- EP handles expert distribution
- TP handles attention/embedding sharding

Benefits of EP over pure TP:
- Experts are naturally sharded (no complex interleaving)
- Attention can use smaller TP (less all-reduce overhead)
- qk_norm becomes simpler with smaller TP
"""

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

# Model paths
model_path = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
traced_model_path_template = "/home/ubuntu/traced_model/MiniMax-M2-EP{ep}-TP{tp}/"

torch.manual_seed(0)


def calculate_configuration(tp_degree, moe_ep_degree, moe_tp_degree=None):
    """Calculate and validate configuration."""
    if moe_tp_degree is None:
        moe_tp_degree = tp_degree

    num_experts = 256
    num_layers = 62
    hidden_size = 3072
    expert_intermediate = 1536

    # Expert memory per core
    experts_per_core = num_experts // moe_ep_degree
    expert_params = 3 * hidden_size * expert_intermediate  # gate + up + down
    expert_memory_gb = experts_per_core * expert_params * 2 / 1e9  # BF16

    # Attention memory per core (with TP)
    attn_params = (3072 * 6144 + 3072 * 1024 * 2 + 6144 * 3072)  # Q, K, V, O
    attn_memory_per_rank = attn_params / tp_degree * 2 / 1e9

    # Total per core
    total_per_core = expert_memory_gb + attn_memory_per_rank

    print(f"""
Configuration Analysis:
=======================
TP degree (attention): {tp_degree}
MoE EP degree: {moe_ep_degree}
MoE TP degree: {moe_tp_degree}

Expert Distribution:
- Total experts: {num_experts}
- Experts per EP rank: {experts_per_core}
- Expert memory per core: {expert_memory_gb:.2f} GB

Attention Distribution:
- Attention TP: {tp_degree}
- Attention memory per rank: {attn_memory_per_rank:.2f} GB

Estimated total per core: {total_per_core:.2f} GB
Trainium2 HBM: 24 GB
Fits in memory: {'✓' if total_per_core < 24 else '✗'}

Total cores needed: {max(tp_degree, moe_ep_degree * moe_tp_degree)}
""")

    return total_per_core < 24


def generate_with_ep(tp_degree=8, moe_ep_degree=32, moe_tp_degree=1):
    """
    Run MiniMax M2 with Expert Parallelism.

    Args:
        tp_degree: Tensor parallelism for attention
        moe_ep_degree: Expert parallelism degree
        moe_tp_degree: Tensor parallelism within each expert
    """
    traced_model_path = traced_model_path_template.format(ep=moe_ep_degree, tp=tp_degree)

    print(f"\n{'='*60}")
    print(f"MiniMax M2 with TP={tp_degree}, MoE EP={moe_ep_degree}")
    print(f"{'='*60}")

    if not calculate_configuration(tp_degree, moe_ep_degree, moe_tp_degree):
        print("ERROR: Configuration may not fit in memory!")
        return None

    generation_config = GenerationConfig.from_pretrained(model_path)

    # Configure with EP
    neuron_config = MoENeuronConfig(
        tp_degree=tp_degree,
        # EP configuration
        moe_ep_degree=moe_ep_degree,
        moe_tp_degree=moe_tp_degree,
        batch_size=1,
        max_context_length=1024,
        seq_len=1024,
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=True,
            temperature=0.6,
            top_k=20,
            top_p=0.95
        ),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        blockwise_matmul_config={
            'use_torch_block_wise': True,
        },
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

    # Compile
    print("\nCompiling model...")
    model = NeuronMiniMaxM2ForCausalLM(model_path, config)
    model.compile(traced_model_path)
    tokenizer.save_pretrained(traced_model_path)

    # Load
    print("\nLoading model...")
    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate
    print("\nGenerating...")
    text = "The capital of France is"
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=50,
        do_sample=False,
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nInput: {text}")
    print(f"Output: {output_text}")

    return output_text


def print_ep_options():
    """Print available EP configurations."""
    print("""
Available Expert Parallelism Configurations for MiniMax M2:
============================================================

MiniMax M2 has 256 experts. Here are some EP configurations:

Option 1: EP=64, TP=1 (64 cores)
  - 4 experts per core (same as current TP=64)
  - No attention TP overhead
  - qk_norm is standard (no distributed)

Option 2: EP=32, TP=2 (64 cores)
  - 8 experts per core
  - Attention split across 2 cores
  - qk_norm with TP=2 (simpler distributed)

Option 3: EP=16, TP=4 (64 cores)
  - 16 experts per core
  - Attention split across 4 cores

Option 4: EP=8, TP=8 (64 cores)
  - 32 experts per core
  - Attention split across 8 cores

Option 5: EP=64, TP=8 (hybrid, 64+ cores)
  - 4 experts per EP rank
  - Full attention TP
  - Most complex but highest parallelism

Recommended for debugging: EP=64, TP=1
  - Simplifies qk_norm (no DistributedRMSNorm needed)
  - Each core handles complete attention for 4 experts
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniMax M2 with Expert Parallelism")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism degree for attention")
    parser.add_argument("--ep", type=int, default=32, help="Expert parallelism degree")
    parser.add_argument("--moe-tp", type=int, default=1, help="Tensor parallelism within experts")
    parser.add_argument("--list", action="store_true", help="List available configurations")
    args = parser.parse_args()

    if args.list:
        print_ep_options()
    else:
        generate_with_ep(
            tp_degree=args.tp,
            moe_ep_degree=args.ep,
            moe_tp_degree=args.moe_tp
        )
