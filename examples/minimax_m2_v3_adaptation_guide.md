# MiniMax M2 v3 模型适配指南

本文档总结了 MiniMax M2 模型在 Neuronx Distributed Inference 框架上的适配方案，特别是 v3 版本如何解决乱码和答非所问的问题。

## 一、MiniMax M2 的独特架构特性

MiniMax M2 与 Qwen3 MoE 相比有几个关键差异：

| 特性 | MiniMax M2 | Qwen3 MoE |
|------|-----------|-----------|
| QK Norm 位置 | reshape **之前** (全投影输出) | reshape **之后** (per-head) |
| RoPE | 部分旋转 (rotary_dim=64, head_dim=128) | 全旋转 (rotary_dim=head_dim) |
| Router 激活函数 | Sigmoid + e_score_correction_bias | Softmax |
| MoE 结构 | block_sparse_moe (w1/w2/w3) | mlp (gate_proj/up_proj/down_proj) |
| KV Heads | 8 个 KV heads | 根据模型变化 |
| Attention Heads | 48 个 attention heads | 根据模型变化 |

## 二、解决乱码问题的关键修复

### 1. QK Norm 权重切片问题 (根本原因)

**问题描述**：原实现在 `MiniMaxM2QKNorm.forward()` 中使用 `parallel_state.get_tensor_model_parallel_rank()` 获取 TP rank，但这在 XLA trace 时返回的是 Python 常量，而不是 Tensor。这导致所有 SPMD 设备使用相同的权重切片，产生完全错误的 QK norm 结果。

```python
# ❌ 错误实现 - trace 时返回 Python 常量，所有 SPMD 设备用同一份权重
def forward(self, hidden_states):
    tp_rank = parallel_state.get_tensor_model_parallel_rank()  # 常量!
    local_weight = self.weight[tp_rank * size : (tp_rank+1) * size]
    ...

# ✅ 正确实现 - 使用 rank_util tensor，SPMD 各设备用自己的权重
def forward(self, hidden_states, rank_util=None):
    if rank_util is not None and self.tp_degree > 1:
        weight_reshaped = self.weight.view(self.tp_degree, self.hidden_size)
        rank_index = rank_util.rank[:1]  # Tensor，非常量
        local_weight = torch.index_select(weight_reshaped, 0, rank_index).squeeze(0)
    ...
```

**原理**：`parallel_state.get_tensor_model_parallel_rank()` 在 XLA trace 时被固定为常量（例如 rank 0），导致所有 64 个 TP rank 都使用 rank 0 的权重切片。

### 2. QK Norm RMS 除数计算

**问题描述**：QK Norm 需要计算全局 RMS，但 Q 和 K 的 padding/replication 策略不同，需要使用不同的除数。

```python
# Q norm: 使用原始大小 6144，因为 padding zeros 不贡献 sum_sq
full_q_hidden_size = config.num_attention_heads * config.head_dim  # 48 * 128 = 6144

# K norm: 使用 padded 大小 8192，因为 KV heads 被复制了 8 倍
# all_reduce 后 sum_sq 放大 8 倍，除以 8192 正好抵消
full_k_hidden_size = padded_num_kv_heads * config.head_dim  # 64 * 128 = 8192
```

**原理**：
- **Q heads**：使用 interleaved padding (48→64)，padding 部分是 0，不贡献 sum of squares
- **K heads**：使用 replication (8→64)，每个原始值被复制 8 次，all-reduce 后 sum_sq 放大 8 倍

### 3. 传递 rank_util 到 QK Norm

```python
# 在 prep_qkv_tensors 中传递 rank_util
def prep_qkv_tensors(self, ...):
    Q, K, V, residual = self.get_qkv_proj()(...)

    # Apply qk_norm BEFORE reshape (MiniMax M2 specific)
    # Pass rank_util for proper SPMD weight slicing
    if self.use_minimax_qk_norm:
        Q = self.q_norm(Q, self.rank_util)  # ← 传递 rank_util
        K = self.k_norm(K, self.rank_util)  # ← 传递 rank_util
    ...
```

## 三、MiniMax M2 QK Norm vs Qwen3 MoE QK Norm

### MiniMax M2 的 QK Norm

MiniMax M2 的 QK Norm 应用在 **reshape 之前**，即在完整的投影输出上：

```
Q 投影输出: [batch, seq, num_attention_heads * head_dim] = [batch, seq, 6144]
K 投影输出: [batch, seq, num_key_value_heads * head_dim] = [batch, seq, 1024]

↓ QK Norm (在全投影输出上计算 RMS)

Q normalized: [batch, seq, 6144]
K normalized: [batch, seq, 1024]

↓ Reshape

Q: [batch, num_heads, seq, head_dim]
K: [batch, num_kv_heads, seq, head_dim]
```

### Qwen3 MoE 的 QK Norm

Qwen3 MoE 的 QK Norm 应用在 **reshape 之后**，即 per-head：

```
Q 投影输出: [batch, seq, num_attention_heads * head_dim]
K 投影输出: [batch, seq, num_key_value_heads * head_dim]

↓ Reshape

Q: [batch, num_heads, seq, head_dim]
K: [batch, num_kv_heads, seq, head_dim]

↓ Per-head LayerNorm (在每个 head 的 head_dim 上计算)

Q normalized: [batch, num_heads, seq, head_dim]
K normalized: [batch, num_kv_heads, seq, head_dim]
```

### 关键差异

| 方面 | MiniMax M2 | Qwen3 MoE |
|-----|-----------|-----------|
| Norm 位置 | reshape 之前 | reshape 之后 |
| Norm 维度 | 全投影维度 (6144/1024) | per-head (head_dim=128) |
| 实现类 | `MiniMaxM2QKNorm` | `q_layernorm`/`k_layernorm` |
| 权重大小 | `[num_heads * head_dim]` | `[head_dim]` |
| TP 处理 | 需要分布式 all-reduce | 本地计算即可 |

## 四、分布式 QK Norm 实现细节

### MiniMaxM2QKNorm 类

```python
class MiniMaxM2QKNorm(nn.Module):
    """
    分布式 QK Norm for MiniMax M2 - 在 reshape 之前应用（在全投影输出上）。

    For tensor parallel:
    - 每个 rank 计算本地 sum(x²)
    - All-reduce 获取全局 sum
    - 计算全局 RMS normalization
    - 存储完整的 padded 权重 [tp_degree * per_rank_size]
    - 在 forward() 中使用 rank_util tensor 动态切片权重
    """
    def __init__(self, hidden_size, eps=1e-6, tp_degree=1,
                 full_hidden_size=None, padded_hidden_size=None):
        super().__init__()
        self.hidden_size = hidden_size  # Per-rank hidden size
        self.variance_epsilon = eps
        self.tp_degree = tp_degree
        # full_hidden_size: 用于 RMS 除数计算
        # - Q: 6144 (原始大小，padding zeros 不贡献)
        # - K: 8192 (复制后的大小，抵消 all-reduce 放大)
        self.full_hidden_size = full_hidden_size or (hidden_size * tp_degree)
        # padded_hidden_size: 权重存储大小
        self.padded_hidden_size = padded_hidden_size or (hidden_size * tp_degree)
        # 存储完整权重 - 在 forward() 中动态切片
        self.weight = nn.Parameter(torch.ones(self.padded_hidden_size))

    def forward(self, hidden_states, rank_util=None):
        from neuronx_distributed.parallel_layers.mappings import (
            reduce_from_tensor_model_parallel_region
        )

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # 计算本地 sum of squares
        local_sum_sq = hidden_states.pow(2).sum(-1, keepdim=True)

        # All-reduce 获取全局 sum of squares
        if self.tp_degree > 1:
            global_sum_sq = reduce_from_tensor_model_parallel_region(local_sum_sq)
        else:
            global_sum_sq = local_sum_sq

        # 计算全局 RMS
        global_variance = global_sum_sq / self.full_hidden_size
        hidden_states = hidden_states * torch.rsqrt(global_variance + self.variance_epsilon)

        # 使用 rank_util tensor 动态切片权重
        # 重要：不能使用 parallel_state.get_tensor_model_parallel_rank()
        # 因为它在 trace 时返回常量
        if rank_util is not None and self.tp_degree > 1:
            weight_reshaped = self.weight.view(self.tp_degree, self.hidden_size)
            rank_index = rank_util.rank[:1]
            local_weight = torch.index_select(weight_reshaped, 0, rank_index).squeeze(0)
        else:
            local_weight = self.weight[:self.hidden_size]

        return (local_weight * hidden_states).to(input_dtype)
```

## 五、部分 RoPE 实现

MiniMax M2 使用部分旋转位置编码，只旋转前 64 维（head_dim=128 的一半）：

```python
def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, ...):
    if self.rotary_dim < self.head_dim:
        # 分割为 [旋转部分, 直通部分]
        Q_rot = Q[..., :self.rotary_dim]      # 前 64 维
        Q_pass = Q[..., self.rotary_dim:]     # 后 64 维
        K_rot = K[..., :self.rotary_dim]
        K_pass = K[..., self.rotary_dim:]

        # 只对旋转部分应用 RoPE
        Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos_cache, sin_cache)

        # 拼接回去
        Q = torch.cat([Q_rot, Q_pass], dim=-1)
        K = torch.cat([K_rot, K_pass], dim=-1)
    else:
        # 全旋转
        Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

    return Q, K, cos_cache, sin_cache
```

## 六、Sigmoid Router with Bias

MiniMax M2 使用 Sigmoid 激活函数和 `e_score_correction_bias` 来选择 experts：

```python
class RouterTopKWithBias(RouterTopK):
    """
    MiniMax M2 使用 sigmoid 激活函数，并在 expert 选择时添加 bias，
    但最终传递给 experts 的权重不包含 bias。
    """
    def __init__(self, num_experts: int, *args, **kwargs):
        super().__init__(num_experts=num_experts, *args, **kwargs)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states):
        router_logits = self.get_router_logits(hidden_states)
        expert_affinities = self.apply_activation_fn(router_logits)  # sigmoid

        # 选择时加 bias
        scores_for_choice = expert_affinities + self.e_score_correction_bias.unsqueeze(0)
        _, expert_index = torch.topk(scores_for_choice, self.top_k, dim=-1)

        # 返回的 affinities 不含 bias（用于加权求和）
        return router_logits, expert_affinities, expert_index
```

## 七、权重转换流程

### State Dict 转换关键步骤

```python
def convert_minimax_m2_hf_to_neuron_state_dict(neuron_state_dict, config):
    # 1. 添加 rank_util tensor
    neuron_state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

    for layer_idx in range(num_layers):
        # 2. 为每层 attention 添加 rank_util
        neuron_state_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = ...

        # 3. QK Norm 权重处理
        # Q norm: interleaved padding (48 heads → 64 heads)
        q_norm_padded = _maybe_pad_interleaved(q_norm_full, ...)
        # K norm: replication (8 KV heads → 64 KV heads)
        k_norm_replicated = k_norm_full.repeat_interleave(repeats, dim=0)

        # 4. Router 权重重命名
        # gate.weight → router.linear_router.weight
        # e_score_correction_bias → router.e_score_correction_bias

        # 5. MoE 权重重组
        # w1 (gate_proj) + w3 (up_proj) → gate_up_proj
        # w2 (down_proj) → down_proj

    # 6. Fused QKV (如果启用)
    if config.neuron_config.fused_qkv:
        # q_proj + k_proj + v_proj → Wqkv
        neuron_state_dict = convert_state_dict_to_fused_qkv(...)

    return neuron_state_dict
```

## 八、完整的 v3 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    MiniMax M2 v3 架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  NeuronMiniMaxM2ForCausalLMV3                               │
│  ├── get_state_dict() - 加载/转换 HF 权重                    │
│  │   ├── 移除 "model." 前缀                                  │
│  │   ├── QK norm 权重 padding/replication                   │
│  │   ├── MoE 权重重组 (w1+w3→gate_up_proj, w2→down_proj)    │
│  │   └── Router 权重重命名 + e_score_correction_bias        │
│  │                                                          │
│  └── NeuronMiniMaxM2ModelV3                                 │
│      ├── embed_tokens (ParallelEmbedding)                   │
│      ├── layers[] (NeuronMiniMaxM2DecoderLayerV3)           │
│      │   ├── ModuleMarkerStartWrapper                       │
│      │   ├── input_layernorm                                │
│      │   ├── self_attn (NeuronMiniMaxM2AttentionV3)         │
│      │   │   ├── MiniMaxM2QKNorm (q_norm, k_norm)           │
│      │   │   │   └── 使用 rank_util 进行 SPMD 权重切片       │
│      │   │   ├── 部分 RoPE (只旋转前 64 维)                  │
│      │   │   └── GQA attention                              │
│      │   ├── post_attention_layernorm                       │
│      │   ├── block_sparse_moe (RouterTopKWithBias + MoE)    │
│      │   │   └── Sigmoid router + e_score_correction_bias   │
│      │   └── ModuleMarkerEndWrapper                         │
│      ├── norm                                               │
│      └── lm_head                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 九、关键代码位置

| 功能 | 文件位置 | 行号 |
|-----|---------|-----|
| QK Norm 实现 | `modeling_minimax_m2_v3.py` | `MiniMaxM2QKNorm` class |
| 权重切片逻辑 | `MiniMaxM2QKNorm.forward()` | rank_util 处理部分 |
| Attention | `NeuronMiniMaxM2AttentionV3` | `__init__` 和 `prep_qkv_tensors` |
| 部分 RoPE | `apply_rotary_embedding()` | Q/K 分割和拼接 |
| Router | `RouterTopKWithBias` | sigmoid + bias 逻辑 |
| 权重转换 | `convert_minimax_m2_hf_to_neuron_state_dict()` | 完整转换流程 |
| Decoder Layer | `NeuronMiniMaxM2DecoderLayerV3` | ModuleMarker + forward |

## 十、经验教训

### 1. SPMD 编程要点

在 XLA trace 环境下，任何需要按 rank 变化的操作都必须使用 **Tensor**（如 `rank_util.rank`），而不是 Python 变量。常见错误：

```python
# ❌ 错误：trace 时变成常量
rank = parallel_state.get_tensor_model_parallel_rank()
weight_slice = self.weight[rank * size : (rank+1) * size]

# ✅ 正确：使用 Tensor 操作
rank_tensor = rank_util.rank[:1]
weight_slice = torch.index_select(weight_reshaped, 0, rank_tensor)
```

### 2. 分布式 RMSNorm

当 norm 作用于被 TP 分片的数据时：
- 需要 **all-reduce** 来获取全局统计量
- 除数必须正确计算（考虑 padding vs replication）

### 3. 权重映射验证

复杂模型的权重转换容易出错，建议：
- 添加 shape 检查和调试日志
- 逐层比较 GPU 和 Neuron 的输出
- 验证特殊权重（如 qk_norm, router bias）的转换

### 4. 采样参数

- 过高的 temperature (如 1.0) 可能导致输出不连贯
- 建议使用较低的 temperature (如 0.6) 以获得更稳定的输出

## 十一、运行指南

```python
# 首次运行需要编译
generate(skip_compile=False)

# 编译完成后可跳过编译直接加载
generate(skip_compile=True)
```

**注意**：修改模型代码后必须重新编译，因为修复影响模型的图结构。
