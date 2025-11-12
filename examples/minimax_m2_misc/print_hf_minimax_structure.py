from transformers import AutoConfig, AutoModelForCausalLM
import torch

model_path = "/home/ubuntu/model_hf/MiniMax-M2/"

print("=" * 80)
print("Loading MiniMax-M2 Configuration")
print("=" * 80)

# Load config
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

print(f"Architecture: {config.architectures}")
print(f"Model type: {config.model_type}")
print(f"Number of layers: {config.num_hidden_layers}")
print(f"Hidden size: {config.hidden_size}")
print(f"Intermediate size: {config.intermediate_size}")
print(f"Number of attention heads: {config.num_attention_heads}")
print(f"Number of KV heads: {config.num_key_value_heads}")
print(f"Vocab size: {config.vocab_size}")
print(f"Max position embeddings: {config.max_position_embeddings}")
print(f"RMS norm eps: {config.rms_norm_eps}")
print(f"Rope theta: {config.rope_theta}")
print(f"Use QK norm: {config.use_qk_norm if hasattr(config, 'use_qk_norm') else 'N/A'}")
print(f"Number of experts: {config.num_local_experts}")
print(f"Number of experts per token: {config.num_experts_per_tok}")
print()

print("=" * 80)
print("Loading Model Structure (may take a few minutes, loading to CPU)...")
print("=" * 80)

# Load model structure only (without weights to save memory)
# Set device_map to "meta" to avoid loading actual weights
try:
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    print("Model structure loaded successfully!\n")

    print("=" * 80)
    print("Full Model Structure")
    print("=" * 80)
    print(model)
    print()

    print("=" * 80)
    print("First Decoder Layer Structure")
    print("=" * 80)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        print(model.model.layers[0])
    print()

    print("=" * 80)
    print("Module Hierarchy (Top-level)")
    print("=" * 80)
    for name, module in model.named_children():
        print(f"\n{name}:")
        for subname, submodule in module.named_children():
            print(f"  └─ {subname}: {type(submodule).__name__}")
            # Show structure of first layer
            if subname == "layers":
                first_layer = list(submodule.children())[0]
                for layer_name, layer_module in first_layer.named_children():
                    print(f"      └─ {layer_name}: {type(layer_module).__name__}")
                print(f"      └─ ... (total {len(list(submodule.children()))} layers)")
                break

    print()
    print("=" * 80)
    print("Attention Module Structure (Layer 0)")
    print("=" * 80)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        attn = model.model.layers[0].self_attn
        print(attn)
        print("\nAttention submodules:")
        for name, module in attn.named_children():
            print(f"  {name}: {type(module).__name__}")

    print()
    print("=" * 80)
    print("MoE MLP Module Structure (Layer 0)")
    print("=" * 80)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        mlp = model.model.layers[0].block_sparse_moe
        print(mlp)
        print("\nMLP submodules:")
        for name, module in mlp.named_children():
            print(f"  {name}: {type(module).__name__}")
            if name == "experts":
                print(f"    Total experts: {len(list(module.children()))}")
                if len(list(module.children())) > 0:
                    expert_0 = list(module.children())[0]
                    print(f"    Expert 0 structure:")
                    for exp_name, exp_module in expert_0.named_children():
                        print(f"      └─ {exp_name}: {type(exp_module).__name__}")

except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTrying alternative method...")
    print("\nBasic structure from config:")
    print(f"- Embedding layer: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
    print(f"- {config.num_hidden_layers} Decoder Layers, each containing:")
    print(f"  - Input LayerNorm")
    print(f"  - Self-Attention (GQA):")
    print(f"    - q_proj: [{config.hidden_size} -> {config.num_attention_heads * (config.hidden_size // config.num_attention_heads)}]")
    print(f"    - k_proj: [{config.hidden_size} -> {config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)}]")
    print(f"    - v_proj: [{config.hidden_size} -> {config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)}]")
    print(f"    - o_proj: [{config.num_attention_heads * (config.hidden_size // config.num_attention_heads)} -> {config.hidden_size}]")
    if config.use_qk_norm:
        print(f"    - q_norm: RMSNorm")
        print(f"    - k_norm: RMSNorm")
    print(f"  - Post-attention LayerNorm")
    print(f"  - MoE MLP:")
    print(f"    - Router (gate): [{config.hidden_size} -> {config.num_local_experts}]")
    print(f"    - {config.num_local_experts} Experts, each with:")
    print(f"      - w1 (gate_proj): [{config.hidden_size} -> {config.intermediate_size}]")
    print(f"      - w3 (up_proj): [{config.hidden_size} -> {config.intermediate_size}]")
    print(f"      - w2 (down_proj): [{config.intermediate_size} -> {config.hidden_size}]")
    print(f"- Final LayerNorm")
    print(f"- LM Head: [{config.hidden_size} -> {config.vocab_size}]")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
print(f"Total parameters (estimated):")
print(f"- Embedding: {config.vocab_size * config.hidden_size:,}")
print(f"- Per layer attention: ~{4 * config.hidden_size * config.hidden_size:,}")
print(f"- Per layer MoE: ~{config.num_local_experts * 3 * config.hidden_size * config.intermediate_size:,}")
print(f"- Total layers: {config.num_hidden_layers}")
print("=" * 80)
