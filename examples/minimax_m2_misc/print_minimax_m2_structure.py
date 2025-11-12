import torch
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

model_path = "/home/ubuntu/model_hf/MiniMax-M2/"

# Initialize config with minimal settings (no compilation)
neuron_config = MoENeuronConfig(
    tp_degree=1,  # Use 1 for simplicity
    batch_size=1,
    max_context_length=128,
    seq_len=1024,
    enable_bucketing=False,
    flash_decoding_enabled=False,
)

config = MiniMaxM2InferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path),
)

print("=" * 80)
print("MiniMax-M2 Model Configuration")
print("=" * 80)
print(f"Number of layers: {config.num_hidden_layers}")
print(f"Hidden size: {config.hidden_size}")
print(f"Intermediate size: {config.intermediate_size}")
print(f"Number of attention heads: {config.num_attention_heads}")
print(f"Number of KV heads: {config.num_key_value_heads}")
print(f"Head dimension: {config.head_dim}")
print(f"Vocab size: {config.vocab_size}")
print(f"Number of experts: {config.num_local_experts}")
print(f"Number of experts per token: {config.num_experts_per_tok}")
print(f"Use QK norm: {config.use_qk_norm}")
print()

print("=" * 80)
print("Creating Model Structure (this may take a few minutes)...")
print("=" * 80)

# Create model on CPU mode to avoid Neuron device requirements
import os
os.environ['NEURON_COMPILE_CACHE_URL'] = '/tmp/neuron_cache'

try:
    from neuronx_distributed.utils import cpu_mode
    # Force CPU mode
    os.environ['NEURON_RT_VISIBLE_CORES'] = ''
except:
    pass

# Create a lightweight model structure
model = NeuronMiniMaxM2ForCausalLM(model_path, config)

print()
print("=" * 80)
print("Model Structure")
print("=" * 80)
print(model.model)

print()
print("=" * 80)
print("Detailed Layer Structure (First Decoder Layer)")
print("=" * 80)
if hasattr(model.model, 'layers') and len(model.model.layers) > 0:
    print(model.model.layers[0])

print()
print("=" * 80)
print("All Module Names")
print("=" * 80)
for name, module in model.model.named_modules():
    if name:  # Skip empty names
        print(f"{name}: {type(module).__name__}")
