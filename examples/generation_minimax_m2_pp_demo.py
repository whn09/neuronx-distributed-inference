"""
MiniMax M2 Demo with Pipeline Parallelism (PP)

This demo uses PP instead of pure TP to simplify the implementation
and avoid complex distributed operations like DistributedRMSNorm.

Memory calculation:
- Each layer: ~7.34 GB
- Trainium2 core: 24 GB
- Layers per core: 3
- 62 layers / 3 = ~21 PP stages

Option 1: PP=21 (or 22 for safety), TP=1
  - Simplest, no TP complexity
  - Each stage handles ~3 layers completely

Option 2: PP=8, TP=8 (hybrid)
  - Total 64 cores (same as current)
  - Each PP stage handles ~8 layers
  - TP=8 splits each layer across 8 cores
"""

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

# Model paths
model_path = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-PP/"

torch.manual_seed(0)


def generate_with_pp(pp_degree=8, tp_degree=8):
    """
    Run MiniMax M2 with Pipeline Parallelism.

    Args:
        pp_degree: Number of pipeline stages
        tp_degree: Tensor parallelism within each stage
    """
    print(f"\n{'='*60}")
    print(f"MiniMax M2 with PP={pp_degree}, TP={tp_degree}")
    print(f"Total cores: {pp_degree * tp_degree}")
    print(f"{'='*60}")

    generation_config = GenerationConfig.from_pretrained(model_path)

    # Configure with PP
    neuron_config = MoENeuronConfig(
        tp_degree=tp_degree,
        pp_degree=pp_degree,  # Enable Pipeline Parallelism!
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pp", type=int, default=8, help="Pipeline parallelism degree")
    parser.add_argument("--tp", type=int, default=8, help="Tensor parallelism degree")
    args = parser.parse_args()

    # Calculate expected configuration
    num_layers = 62
    layer_memory_gb = 7.34
    hbm_per_core = 24

    layers_per_stage = num_layers // args.pp
    memory_per_stage = layers_per_stage * layer_memory_gb / args.tp

    print(f"""
Configuration Analysis:
- PP degree: {args.pp}
- TP degree: {args.tp}
- Total cores: {args.pp * args.tp}
- Layers per PP stage: {layers_per_stage}
- Memory per stage per TP rank: {memory_per_stage:.2f} GB
- Trainium2 HBM per core: {hbm_per_core} GB
- Fits in memory: {'✓' if memory_per_stage < hbm_per_core else '✗'}
""")

    if memory_per_stage >= hbm_per_core:
        print("WARNING: Configuration may not fit in memory!")
        print("Try increasing PP or TP degree.")
    else:
        generate_with_pp(pp_degree=args.pp, tp_degree=args.tp)
