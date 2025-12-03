"""
Calculate memory requirements for MiniMax M2 model.
Determine if Pipeline Parallelism (PP) is feasible on Trainium2.
"""

def calculate_memory():
    # MiniMax M2 configuration
    hidden_size = 3072
    num_attention_heads = 48
    num_kv_heads = 8
    head_dim = 128
    intermediate_size = 1536  # per expert MLP
    mlp_intermediate_size = 8192  # unused?
    num_experts = 256
    num_hidden_layers = 62
    vocab_size = 200064

    # Trainium2 specs
    hbm_per_core = 24  # GB

    # Bytes per parameter (bfloat16)
    bytes_per_param = 2

    print("=" * 70)
    print("MiniMax M2 Memory Calculation")
    print("=" * 70)

    # Per-layer memory calculation
    print("\n### Per Layer Memory ###\n")

    # 1. Attention weights
    q_proj = hidden_size * (num_attention_heads * head_dim)  # 3072 * 6144
    k_proj = hidden_size * (num_kv_heads * head_dim)         # 3072 * 1024
    v_proj = hidden_size * (num_kv_heads * head_dim)         # 3072 * 1024
    o_proj = (num_attention_heads * head_dim) * hidden_size  # 6144 * 3072

    attn_params = q_proj + k_proj + v_proj + o_proj
    print(f"Attention:")
    print(f"  Q proj: {hidden_size} x {num_attention_heads * head_dim} = {q_proj:,} params")
    print(f"  K proj: {hidden_size} x {num_kv_heads * head_dim} = {k_proj:,} params")
    print(f"  V proj: {hidden_size} x {num_kv_heads * head_dim} = {v_proj:,} params")
    print(f"  O proj: {num_attention_heads * head_dim} x {hidden_size} = {o_proj:,} params")
    print(f"  Total attention: {attn_params:,} params = {attn_params * bytes_per_param / 1e6:.2f} MB")

    # 2. QK Norm weights
    q_norm = num_attention_heads * head_dim  # 6144
    k_norm = num_kv_heads * head_dim         # 1024
    qk_norm_params = q_norm + k_norm
    print(f"\nQK Norm:")
    print(f"  q_norm: {q_norm:,} params")
    print(f"  k_norm: {k_norm:,} params")
    print(f"  Total QK norm: {qk_norm_params:,} params = {qk_norm_params * bytes_per_param / 1e6:.4f} MB")

    # 3. LayerNorm weights (input + post-attention)
    layernorm_params = hidden_size * 2  # input_layernorm + post_attention_layernorm
    print(f"\nLayerNorm:")
    print(f"  input_layernorm: {hidden_size:,} params")
    print(f"  post_attention_layernorm: {hidden_size:,} params")
    print(f"  Total layernorm: {layernorm_params:,} params = {layernorm_params * bytes_per_param / 1e6:.4f} MB")

    # 4. MoE FFN weights
    # Each expert: gate_proj + up_proj + down_proj (GLU)
    gate_proj = hidden_size * intermediate_size  # 3072 * 1536
    up_proj = hidden_size * intermediate_size    # 3072 * 1536
    down_proj = intermediate_size * hidden_size  # 1536 * 3072

    expert_params = gate_proj + up_proj + down_proj
    all_experts_params = expert_params * num_experts

    print(f"\nMoE FFN ({num_experts} experts):")
    print(f"  Per expert:")
    print(f"    gate_proj: {hidden_size} x {intermediate_size} = {gate_proj:,} params")
    print(f"    up_proj:   {hidden_size} x {intermediate_size} = {up_proj:,} params")
    print(f"    down_proj: {intermediate_size} x {hidden_size} = {down_proj:,} params")
    print(f"    Expert total: {expert_params:,} params = {expert_params * bytes_per_param / 1e6:.2f} MB")
    print(f"  All {num_experts} experts: {all_experts_params:,} params = {all_experts_params * bytes_per_param / 1e9:.2f} GB")

    # 5. Router weights
    router_params = hidden_size * num_experts + num_experts  # linear + bias
    print(f"\nRouter:")
    print(f"  linear: {hidden_size} x {num_experts} = {hidden_size * num_experts:,} params")
    print(f"  e_score_correction_bias: {num_experts:,} params")
    print(f"  Total router: {router_params:,} params = {router_params * bytes_per_param / 1e6:.4f} MB")

    # Total per layer
    layer_params = attn_params + qk_norm_params + layernorm_params + all_experts_params + router_params
    layer_memory_gb = layer_params * bytes_per_param / 1e9

    print(f"\n### Total Per Layer ###")
    print(f"  Parameters: {layer_params:,}")
    print(f"  Memory: {layer_memory_gb:.2f} GB")

    # Total model
    print(f"\n### Total Model ({num_hidden_layers} layers) ###")
    total_layer_params = layer_params * num_hidden_layers

    # Embedding + LM head
    embed_params = vocab_size * hidden_size
    lm_head_params = hidden_size * vocab_size
    final_norm_params = hidden_size

    print(f"  Embedding: {embed_params:,} params = {embed_params * bytes_per_param / 1e9:.2f} GB")
    print(f"  LM head: {lm_head_params:,} params = {lm_head_params * bytes_per_param / 1e9:.2f} GB")
    print(f"  Final norm: {final_norm_params:,} params")

    total_params = total_layer_params + embed_params + lm_head_params + final_norm_params
    total_memory_gb = total_params * bytes_per_param / 1e9

    print(f"\n  Total parameters: {total_params:,} ({total_params/1e9:.1f}B)")
    print(f"  Total memory (BF16): {total_memory_gb:.2f} GB")

    # Pipeline Parallelism analysis
    print("\n" + "=" * 70)
    print("Pipeline Parallelism Analysis (Trainium2, 24GB per core)")
    print("=" * 70)

    print(f"\n### Option 1: PP only (no TP within layers) ###")
    print(f"  Each layer needs: {layer_memory_gb:.2f} GB")
    print(f"  Trainium2 core: {hbm_per_core} GB")

    if layer_memory_gb <= hbm_per_core:
        print(f"  ✓ One layer fits in one core!")
        layers_per_core = int(hbm_per_core / layer_memory_gb)
        cores_needed = (num_hidden_layers + layers_per_core - 1) // layers_per_core
        print(f"  Layers per core: {layers_per_core}")
        print(f"  Cores needed for {num_hidden_layers} layers: {cores_needed}")
    else:
        print(f"  ✗ One layer ({layer_memory_gb:.2f} GB) exceeds core memory ({hbm_per_core} GB)")
        print(f"  Need to use TP to split each layer")

    print(f"\n### Option 2: PP + TP hybrid ###")

    for tp in [2, 4, 8, 16, 32, 64]:
        # With TP, MoE experts are distributed
        # Attention is sharded by heads
        # Per-rank memory per layer:
        expert_per_rank = num_experts // tp if num_experts >= tp else 1
        attn_per_rank = attn_params // tp
        moe_per_rank = expert_params * expert_per_rank

        layer_per_rank = (attn_per_rank + qk_norm_params + layernorm_params + moe_per_rank + router_params)
        layer_per_rank_gb = layer_per_rank * bytes_per_param / 1e9

        layers_per_core = int(hbm_per_core / layer_per_rank_gb) if layer_per_rank_gb > 0 else num_hidden_layers
        layers_per_core = max(1, min(layers_per_core, num_hidden_layers))

        # PP degree needed
        pp_needed = (num_hidden_layers + layers_per_core - 1) // layers_per_core

        total_cores = tp * pp_needed

        print(f"\n  TP={tp}:")
        print(f"    MoE experts per rank: {expert_per_rank}")
        print(f"    Layer memory per rank: {layer_per_rank_gb:.2f} GB")
        print(f"    Layers per PP stage: {layers_per_core}")
        print(f"    PP degree needed: {pp_needed}")
        print(f"    Total cores: {total_cores}")

    print("\n" + "=" * 70)
    print("Recommendation")
    print("=" * 70)
    print("""
    For MiniMax M2 on Trainium2 (24GB HBM per core):

    Current approach: TP=64, PP=1
    - All 256 experts distributed across 64 cores (4 experts per core)
    - Each layer's attention distributed across 64 cores
    - High communication overhead (all-reduce every layer)

    Alternative: TP=8 or TP=16 with PP
    - TP=8, PP=8: Use 64 cores total
      - 8 experts per core (256/8/4 = 8)
      - ~8 layers per PP stage (62/8 ≈ 8)
      - Less all-reduce within layers

    The main benefit of PP:
    - Less communication between cores
    - Simpler weight distribution (no complex sharding)
    - But requires careful handling of layer boundaries
    """)


if __name__ == "__main__":
    calculate_memory()
