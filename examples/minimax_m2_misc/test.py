import sys
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

# 检查 traced model 的 config
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig
config = MiniMaxM2InferenceConfig.load('/home/ubuntu/traced_model/MiniMax-M2-BF16-weights/')
print('=== Loaded config ===')
print(f'use_qk_norm: {getattr(config, "use_qk_norm", "NOT FOUND")}')
print(f'rotary_dim: {getattr(config, "rotary_dim", "NOT FOUND")}')
print(f'head_dim: {config.head_dim}')
print(f'num_attention_heads: {config.num_attention_heads}')
