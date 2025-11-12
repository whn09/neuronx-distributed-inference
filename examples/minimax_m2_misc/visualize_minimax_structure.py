#!/usr/bin/env python3
"""
Visualize MiniMax-M2 Model Architecture
"""

from transformers import AutoConfig
import json

model_path = "/home/ubuntu/model_hf/MiniMax-M2/"

# Load config
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

def print_section(title):
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)

def print_subsection(title):
    print(f"\n{'─' * 100}")
    print(f"  {title}")
    print('─' * 100)

print_section("MiniMax-M2 Architecture Visualization")

print("\n📊 Model Configuration:")
print(f"  • Architecture: {config.architectures[0]}")
print(f"  • Model Type: {config.model_type}")
print(f"  • Total Layers: {config.num_hidden_layers}")
print(f"  • Hidden Dimension: {config.hidden_size}")
print(f"  • Vocabulary Size: {config.vocab_size:,}")

print_section("Layer-by-Layer Structure")

print("""
MiniMaxM2ForCausalLM
│
├── model (MiniMaxM2Model)
│   │
│   ├── embed_tokens (Embedding)
│   │   └── [vocab_size={}, hidden_size={}]
│   │
│   ├── layers (ModuleList[{}])
│   │   │
""".format(config.vocab_size, config.hidden_size, config.num_hidden_layers))

# Detailed layer structure
print(f"│   │   ├── Layer[0...{config.num_hidden_layers-1}] (MiniMaxM2DecoderLayer)")
print( "│   │   │   │")
print( "│   │   │   ├── input_layernorm (RMSNorm)")
print(f"│   │   │   │   └── normalized_shape: [{config.hidden_size}], eps={config.rms_norm_eps}")
print( "│   │   │   │")
print( "│   │   │   ├── self_attn (MiniMaxM2Attention - Group Query Attention)")
print( "│   │   │   │   │")
print(f"│   │   │   │   ├── q_proj (Linear): [{config.hidden_size} → {config.num_attention_heads * (config.hidden_size // config.num_attention_heads)}]")
print(f"│   │   │   │   │   • Output: {config.num_attention_heads} attention heads")
print(f"│   │   │   │   │   • Head dim: {config.hidden_size // config.num_attention_heads}")

if config.use_qk_norm:
    print(f"│   │   │   │   │   └── q_norm (RMSNorm): per-head normalization")
    print(f"│   │   │   │   │       • Shape: [{config.num_attention_heads}, {config.hidden_size // config.num_attention_heads}]")

print(f"│   │   │   │   │")
print(f"│   │   │   │   ├── k_proj (Linear): [{config.hidden_size} → {config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)}]")
print(f"│   │   │   │   │   • Output: {config.num_key_value_heads} KV heads (GQA)")
print(f"│   │   │   │   │   • Head dim: {config.hidden_size // config.num_attention_heads}")
print(f"│   │   │   │   │   • Ratio: {config.num_attention_heads // config.num_key_value_heads}:1 (Q:KV)")

if config.use_qk_norm:
    print(f"│   │   │   │   │   └── k_norm (RMSNorm): per-head normalization")
    print(f"│   │   │   │   │       • Shape: [{config.num_key_value_heads}, {config.hidden_size // config.num_attention_heads}]")

print(f"│   │   │   │   │")
print(f"│   │   │   │   ├── v_proj (Linear): [{config.hidden_size} → {config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)}]")
print(f"│   │   │   │   │   • Output: {config.num_key_value_heads} KV heads")
print( "│   │   │   │   │")
print(f"│   │   │   │   ├── o_proj (Linear): [{config.num_attention_heads * (config.hidden_size // config.num_attention_heads)} → {config.hidden_size}]")
print( "│   │   │   │   │   • Merges all attention heads back")
print( "│   │   │   │   │")
print(f"│   │   │   │   └── rotary_emb (RotaryEmbedding)")
print(f"│   │   │   │       • Theta: {config.rope_theta:,}")
print(f"│   │   │   │       • Max positions: {config.max_position_embeddings:,}")
print( "│   │   │   │")
print( "│   │   │   ├── post_attention_layernorm (RMSNorm)")
print(f"│   │   │   │   └── normalized_shape: [{config.hidden_size}], eps={config.rms_norm_eps}")
print( "│   │   │   │")
print( "│   │   │   └── block_sparse_moe (MoE - Mixture of Experts)")
print( "│   │   │       │")
print(f"│   │   │       ├── gate (Linear): [{config.hidden_size} → {config.num_local_experts}]")
print(f"│   │   │       │   • Routes input to top-{config.num_experts_per_tok} experts")
print( "│   │   │       │")
print(f"│   │   │       ├── experts (ModuleList[{config.num_local_experts}])")
print(f"│   │   │       │   │   Each expert is an FFN:")
print( "│   │   │       │   │")
print(f"│   │   │       │   ├── Expert[0...{config.num_local_experts-1}] (MiniMaxM2MLP)")
print( "│   │   │       │   │   │")
print(f"│   │   │       │   │   ├── w1 (gate_proj): [{config.hidden_size} → {config.intermediate_size}]")
print( "│   │   │       │   │   │   • For SwiGLU activation (gate path)")
print( "│   │   │       │   │   │")
print(f"│   │   │       │   │   ├── w3 (up_proj): [{config.hidden_size} → {config.intermediate_size}]")
print( "│   │   │       │   │   │   • For SwiGLU activation (up path)")
print( "│   │   │       │   │   │")
print(f"│   │   │       │   │   ├── activation: SwiGLU")
print( "│   │   │       │   │   │   • Combines: silu(w1(x)) * w3(x)")
print( "│   │   │       │   │   │")
print(f"│   │   │       │   │   └── w2 (down_proj): [{config.intermediate_size} → {config.hidden_size}]")
print( "│   │   │       │   │       • Projects back to hidden dimension")
print( "│   │   │       │   │")
print( "│   │   │       │   └── ... (repeated for all experts)")
print( "│   │   │       │")
if hasattr(config, 'e_score_correction_bias') and config.e_score_correction_bias is not None:
    print( "│   │   │       └── e_score_correction_bias (Parameter)")
    print(f"│   │   │           • Shape: [{config.num_local_experts}]")
    print( "│   │   │           • Adjusts expert selection scores")
else:
    print( "│   │   │       └── (no e_score_correction_bias)")
print( "│   │   │")
print(f"│   │   └── ... (repeated for {config.num_hidden_layers} layers)")
print( "│   │")
print( "│   └── norm (RMSNorm - Final)")
print(f"│       └── normalized_shape: [{config.hidden_size}], eps={config.rms_norm_eps}")
print( "│")
print( "└── lm_head (Linear)")
print(f"    └── [{config.hidden_size} → {config.vocab_size:,}]")
print( "        • Generates vocabulary logits for next token prediction")

print_section("Parameter Count Breakdown")

# Calculate parameters
embed_params = config.vocab_size * config.hidden_size
lm_head_params = config.hidden_size * config.vocab_size  # Usually tied with embedding

# Per layer
attn_params_per_layer = (
    config.hidden_size * config.num_attention_heads * (config.hidden_size // config.num_attention_heads) +  # q_proj
    config.hidden_size * config.num_key_value_heads * (config.hidden_size // config.num_attention_heads) * 2 +  # k_proj + v_proj
    config.num_attention_heads * (config.hidden_size // config.num_attention_heads) * config.hidden_size  # o_proj
)

if config.use_qk_norm:
    qk_norm_params = config.num_attention_heads * (config.hidden_size // config.num_attention_heads) + \
                     config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
    attn_params_per_layer += qk_norm_params

# MoE params per layer
expert_params = config.num_local_experts * (
    config.hidden_size * config.intermediate_size +  # w1
    config.hidden_size * config.intermediate_size +  # w3
    config.intermediate_size * config.hidden_size    # w2
)
gate_params = config.hidden_size * config.num_local_experts

moe_params_per_layer = expert_params + gate_params

# LayerNorm params
layernorm_params_per_layer = config.hidden_size * 2  # input_layernorm + post_attention_layernorm
final_layernorm_params = config.hidden_size

# Total
total_layer_params = config.num_hidden_layers * (
    attn_params_per_layer + moe_params_per_layer + layernorm_params_per_layer
)

total_params = embed_params + total_layer_params + final_layernorm_params  # lm_head usually tied

print(f"\n📊 Embedding Layer:")
print(f"  • Parameters: {embed_params:,}")

print(f"\n📊 Per Decoder Layer ({config.num_hidden_layers} total):")
print(f"  • Attention: {attn_params_per_layer:,} params")
print(f"    - Q/K/V/O projections: {attn_params_per_layer:,} params")
if config.use_qk_norm:
    print(f"    - Q/K norms: {qk_norm_params:,} params")
print(f"  • MoE MLP: {moe_params_per_layer:,} params")
print(f"    - Router (gate): {gate_params:,} params")
print(f"    - {config.num_local_experts} Experts: {expert_params:,} params")
print(f"    - Per expert: {expert_params // config.num_local_experts:,} params")
print(f"  • LayerNorms: {layernorm_params_per_layer:,} params")
print(f"  ─────────────────────────────")
print(f"  • Total per layer: {attn_params_per_layer + moe_params_per_layer + layernorm_params_per_layer:,} params")

print(f"\n📊 All {config.num_hidden_layers} Layers:")
print(f"  • Total: {total_layer_params:,} params")

print(f"\n📊 Final LayerNorm:")
print(f"  • Parameters: {final_layernorm_params:,}")

print(f"\n📊 LM Head (usually tied with embedding):")
print(f"  • Parameters: {lm_head_params:,} (if not tied)")

print(f"\n{'=' * 100}")
print(f"  🎯 TOTAL MODEL PARAMETERS: {total_params:,}")
print(f"{'=' * 100}")

print(f"\n📈 Model Characteristics:")
print(f"  • Sparsity: {config.num_experts_per_tok}/{config.num_local_experts} experts activated")
print(f"  • Sparsity ratio: {config.num_experts_per_tok/config.num_local_experts:.1%}")
print(f"  • Effective MoE params per token: {(expert_params // config.num_local_experts) * config.num_experts_per_tok + gate_params:,}")
print(f"  • Attention type: Group Query Attention (GQA)")
print(f"  • GQA ratio: {config.num_attention_heads // config.num_key_value_heads}:1 (Query:KV)")
print(f"  • Use QK normalization: {config.use_qk_norm}")

print_section("Key Features")

print("""
✨ Architecture Highlights:

1. 🎯 Group Query Attention (GQA)
   • Reduces KV cache size while maintaining model quality
   • Q heads: 48, KV heads: 8 (6:1 ratio)
   • Per-head Q/K normalization for training stability

2. 🔀 Mixture of Experts (MoE)
   • 256 experts per layer (massive expert pool)
   • Top-8 expert routing (sparse activation)
   • Only ~3.125% of experts active per token
   • Enables scaling model capacity without proportional compute increase

3. 🎨 SwiGLU Activation
   • silu(w1(x)) ⊗ w3(x) → w2(x)
   • Gated activation for better expressiveness
   • Separate gate and up projections

4. 🔄 RoPE (Rotary Position Embedding)
   • Theta: 5,000,000 (supports very long contexts)
   • Max positions: 196,608 tokens (~200K context length!)
   • Relative positional encoding

5. 📐 RMSNorm
   • Lightweight normalization (no bias, no mean centering)
   • Lower computational cost than LayerNorm
   • eps=1e-6 for numerical stability
""")

print("=" * 100)
print("Visualization complete!")
print("=" * 100)
