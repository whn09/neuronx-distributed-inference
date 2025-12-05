# coding=utf-8
# Copyright 2025 MiniMax and the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MiniMax M2 model for NXD inference.
Clean implementation following Qwen3 MoE pattern.

Key differences from standard models:
1. QK norm applied on full Q/K projection output BEFORE reshape (not per-head)
2. Partial RoPE: rotary_dim=64, head_dim=128 (only first half gets rotation)
3. Sigmoid router with e_score_correction_bias for expert selection
"""
import gc
import warnings
from typing import Dict, Any, List, Optional, Tuple

import torch
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.expert_mlps import RoutedExpertsMLPOpsConfig

from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_process_group


GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


def get_rmsnorm_cls():
    """Get appropriate RMSNorm class based on execution environment."""
    if cpu_mode():
        # Use a simple RMSNorm for CPU mode
        class SimpleRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return self.weight * hidden_states.to(input_dtype)
        return SimpleRMSNorm
    return CustomRMSNorm


class RouterTopKWithBias(RouterTopK):
    """
    RouterTopK with e_score_correction_bias support for MiniMax M2.

    MiniMax M2 uses sigmoid activation with a bias term added to scores for expert selection,
    but the final weights (affinities without bias) are passed to experts.
    """

    def __init__(self, num_experts: int, *args, **kwargs):
        super().__init__(num_experts=num_experts, *args, **kwargs)
        # Register e_score_correction_bias buffer (will be loaded from checkpoint)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states):
        # Save original dtype before get_router_logits modifies hidden_states
        original_dtype = hidden_states.dtype

        # Get router_logits using base class method (handles flattening from (B,S,H) to (T,E))
        router_logits = self.get_router_logits(hidden_states)

        # Apply activation (sigmoid for MiniMax M2) - returns (T, E) in float64
        expert_affinities = self.apply_activation_fn(router_logits)

        # For expert selection, add bias to affinities (MiniMax M2 specific)
        scores_for_choice = expert_affinities + self.e_score_correction_bias.unsqueeze(0)

        # Select top-k experts based on biased scores
        # expert_index: (T, top_k)
        _, expert_index = torch.topk(scores_for_choice, self.top_k, dim=-1)

        # Cast to required dtypes (must match base RouterTopK behavior)
        expert_affinities = expert_affinities.to(dtype=original_dtype)
        expert_index = expert_index.detach().to(dtype=torch.long)

        # Return unbiased affinities for expert weighting (normalization happens in ExpertMLPs)
        return router_logits, expert_affinities, expert_index


def convert_minimax_m2_hf_to_neuron_state_dict(neuron_state_dict: Dict[str, Any], config: "MiniMaxM2InferenceConfig") -> Dict[str, Any]:
    """
    Convert HuggingFace MiniMax M2 state dict to Neuron-compatible format.
    """
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported for MiniMax M2"

    # Add rank utility tensor for TP parallel operations
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    # Calculate padded head counts for qk_norm sharding
    tp_degree = config.neuron_config.tp_degree
    from neuronx_distributed_inference.modules.attention.gqa import get_shardable_head_counts, _maybe_pad_interleaved
    padded_num_attention_heads, padded_num_kv_heads = get_shardable_head_counts(
        tp_degree, config.num_attention_heads, config.num_key_value_heads, GQA_SHARDING_STRATEGY
    )

    # Add rank_util.rank for each attention layer
    rank_tensor = torch.arange(0, tp_degree, dtype=torch.int32)
    for layer_idx in range(config.num_hidden_layers):
        neuron_state_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = rank_tensor.clone()

    for layer_idx in range(config.num_hidden_layers):
        # Handle QK norm weights: apply interleaved padding to match Q/K projection sharding
        # MiniMax M2 qk_norm applies RMSNorm on the full Q/K output BEFORE reshape.
        # The weights are organized as [num_heads * head_dim], e.g., [6144] for Q.
        has_qk_norm = getattr(config, 'use_qk_norm', False)
        if has_qk_norm:
            # q_norm: apply interleaved padding [48 heads -> 64 heads]
            q_norm_key = f"layers.{layer_idx}.self_attn.q_norm.weight"
            if q_norm_key in neuron_state_dict:
                q_norm_full = neuron_state_dict[q_norm_key]  # [num_attention_heads * head_dim] = [6144]
                # Apply the same interleaved padding as Q projection weights
                source_group_size = config.num_attention_heads // config.num_key_value_heads
                q_norm_padded = _maybe_pad_interleaved(
                    q_norm_full.unsqueeze(0),  # Add batch dim for the function: [1, 6144]
                    pad_dim=1,
                    source_heads=config.num_attention_heads,  # 48
                    target_heads=padded_num_attention_heads,  # 64
                    source_group_size=source_group_size,  # 6
                ).squeeze(0)  # [8192]
                neuron_state_dict[q_norm_key] = q_norm_padded
                if layer_idx == 0:
                    print(f"  q_norm: {q_norm_full.shape} -> {q_norm_padded.shape} (interleaved padding)")

            # k_norm: replicate from 8 to 64 heads
            k_norm_key = f"layers.{layer_idx}.self_attn.k_norm.weight"
            if k_norm_key in neuron_state_dict:
                k_norm_full = neuron_state_dict[k_norm_key]  # [num_kv_heads * head_dim] = [1024]
                # KV heads are replicated: each of the 8 original heads is replicated 8 times (64/8=8)
                k_norm_reshaped = k_norm_full.reshape(config.num_key_value_heads, config.head_dim)  # [8, 128]
                repeats = padded_num_kv_heads // config.num_key_value_heads  # 64 / 8 = 8
                k_norm_replicated = k_norm_reshaped.repeat_interleave(repeats, dim=0)  # [64, 128]
                k_norm_padded = k_norm_replicated.reshape(-1)  # [8192]
                neuron_state_dict[k_norm_key] = k_norm_padded
                if layer_idx == 0:
                    print(f"  k_norm: {k_norm_full.shape} -> {k_norm_padded.shape} (replicated {repeats}x)")

        # Copy router weights: gate -> router.linear_router
        neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.router.linear_router.weight"] = (
            neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.gate.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.gate.weight"]

        # Handle e_score_correction_bias: move to router path
        bias_key = f"layers.{layer_idx}.block_sparse_moe.e_score_correction_bias"
        if bias_key in neuron_state_dict:
            neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.router.e_score_correction_bias"] = (
                neuron_state_dict[bias_key].detach().clone()
            )
            del neuron_state_dict[bias_key]

        # Get expert weight dimensions
        # MiniMax M2: w1 (gate), w2 (down), w3 (up)
        w1_key = f"layers.{layer_idx}.block_sparse_moe.experts.0.w1.weight"
        intermediate_size, hidden_size = neuron_state_dict[w1_key].shape
        device = neuron_state_dict[w1_key].device
        dtype = neuron_state_dict[w1_key].dtype

        # Merge gate_proj (w1) and up_proj (w3) into gate_up_proj
        # Shape: [num_experts, hidden_size, 2 * intermediate_size]
        gate_up_proj = torch.empty(
            config.num_local_experts,
            hidden_size,
            2 * intermediate_size,
            dtype=dtype,
            device=device,
        )

        for expert_idx in range(config.num_local_experts):
            # w1 (gate_proj): [intermediate_size, hidden_size] -> transpose to [hidden_size, intermediate_size]
            gate_proj_weights = (
                neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight"]
                .T.detach().clone()
            )
            # w3 (up_proj): [intermediate_size, hidden_size] -> transpose to [hidden_size, intermediate_size]
            up_proj_weights = (
                neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight"]
                .T.detach().clone()
            )

            # Concatenate: [hidden_size, 2 * intermediate_size]
            gate_up_proj_slice = torch.narrow(gate_up_proj, 0, expert_idx, 1)
            gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
            gate_proj_slice.copy_(gate_proj_weights)
            up_proj_slice = torch.narrow(gate_up_proj_slice, 2, intermediate_size, intermediate_size)
            up_proj_slice.copy_(up_proj_weights)

            del neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w1.weight"]
            del neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w3.weight"]

        neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        # Handle down_proj (w2)
        # Shape: [num_experts, intermediate_size, hidden_size]
        down_proj = torch.empty(
            config.num_local_experts,
            intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )

        for expert_idx in range(config.num_local_experts):
            # w2 (down_proj): [hidden_size, intermediate_size] -> transpose to [intermediate_size, hidden_size]
            down_proj_weights = (
                neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight"]
                .T.detach().clone()
            )
            down_proj_slice = torch.narrow(down_proj, 0, expert_idx, 1)
            down_proj_slice.copy_(down_proj_weights)
            del neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}.w2.weight"]

        neuron_state_dict[f"layers.{layer_idx}.block_sparse_moe.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    return neuron_state_dict


class MiniMaxM2InferenceConfig(InferenceConfig):
    """Inference configuration for MiniMax M2."""

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "intermediate_size",
            "max_position_embeddings",
            "num_attention_heads",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "num_local_experts",
            "rms_norm_eps",
            "rope_theta",
            "tie_word_embeddings",
            "vocab_size",
            "use_qk_norm",
            "rotary_dim",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


class NeuronMiniMaxM2Attention(NeuronAttentionBase):
    """
    MiniMax M2 Attention with:
    1. QK norm on full projection output (BEFORE reshape)
    2. Partial RoPE (rotary_dim=64, head_dim=128)
    """

    def __init__(self, config: MiniMaxM2InferenceConfig):
        # Get rotary_dim for partial RoPE
        self.rotary_dim = getattr(config, 'rotary_dim', config.head_dim)

        # Create RotaryEmbedding with rotary_dim (not head_dim)
        rotary_emb = RotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            # Disable base class qk_norm - we handle it differently
            use_qk_norm=False,
        )

        # MiniMax M2 QK norm: applied on full projection output BEFORE reshape
        # This is different from Qwen3 which applies per-head after reshape
        self.use_minimax_qk_norm = getattr(config, 'use_qk_norm', False)

        if self.use_minimax_qk_norm:
            # Full Q dimension: num_attention_heads * head_dim (per TP rank after sharding)
            # Full K dimension: num_key_value_heads * head_dim (per TP rank after sharding)
            # Note: self.num_heads and self.num_key_value_heads are already divided by tp_degree
            q_norm_dim = self.num_heads * self.head_dim
            k_norm_dim = self.num_key_value_heads * self.head_dim

            # Use q_norm/k_norm names to match HF checkpoint naming
            self.q_norm = get_rmsnorm_cls()(q_norm_dim, config.rms_norm_eps)
            self.k_norm = get_rmsnorm_cls()(k_norm_dim, config.rms_norm_eps)

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronMiniMaxM2Attention requires initialized distributed environment."
            )

    def get_qkv_with_rope(
        self,
        hidden_states,
        position_ids,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        skip_rope=False,
        residual=None,
        use_polar_compatible_rope=False,
    ):
        """Override to apply MiniMax M2 qk_norm BEFORE reshape and handle partial RoPE."""
        from neuronx_distributed_inference.modules.attention.utils import move_heads_front, apply_rotary_pos_emb

        # Get Q, K, V projections
        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids, residual=residual
        )

        # MiniMax M2 qk_norm: apply on full projection output BEFORE reshape
        if self.use_minimax_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        # Don't pass layernorm to move_heads_front since we already applied qk_norm
        Q = move_heads_front(Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=None)
        K = move_heads_front(K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        if not skip_rope:
            # Apply RoPE with partial rotary support
            Q, K, cos_cache, sin_cache = self.apply_rotary_embedding(
                Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
            )

        return Q, K, V, cos_cache, sin_cache, residual

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """Apply partial rotary embeddings (rotary_dim=64, head_dim=128)."""
        from neuronx_distributed_inference.modules.attention.utils import apply_rotary_pos_emb

        if not use_polar_compatible_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            # For partial rotary: only rotate first rotary_dim dimensions
            if self.rotary_dim < self.head_dim:
                # Split into [rotary_part, pass_through_part]
                Q_rot = Q[..., :self.rotary_dim]
                Q_pass = Q[..., self.rotary_dim:]
                K_rot = K[..., :self.rotary_dim]
                K_pass = K[..., self.rotary_dim:]

                # Apply RoPE only to rotary part
                Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos_cache, sin_cache)

                # Concatenate back
                Q = torch.cat([Q_rot, Q_pass], dim=-1)
                K = torch.cat([K_rot, K_pass], dim=-1)
            else:
                # Full rotary (when rotary_dim == head_dim)
                Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

        return Q, K, cos_cache, sin_cache


def initialize_minimax_m2_moe_module(config: MiniMaxM2InferenceConfig):
    """Initialize MoE module for MiniMax M2 with sigmoid router and bias support."""
    enabled_hybrid_sharding = config.neuron_config.hybrid_sharding_config is not None
    moe_tkg_tensor_model_parallel_group, moe_tkg_expert_model_parallel_group, \
        moe_cte_tensor_model_parallel_group, moe_cte_expert_model_parallel_group = \
        initialize_moe_process_group(config, enabled_hybrid_sharding)

    # Use custom router with e_score_correction_bias support
    router = RouterTopKWithBias(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        dtype=config.neuron_config.router_config.dtype,
        act_fn=config.neuron_config.router_config.act_fn,  # 'sigmoid' for MiniMax M2
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        sequence_dimension=1,
        bias=False,
        apply_act_fn_over_topk=False,
        store_transposed_weights=False,
    )

    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            top_k=config.num_experts_per_tok,
            hidden_act=config.hidden_act,
            bias=False,
            glu_mlp=config.neuron_config.glu_mlp,
            glu_type=config.neuron_config.glu_type,
            hidden_act_scaling_factor=config.neuron_config.hidden_act_scaling_factor,
            hidden_act_bias=config.neuron_config.hidden_act_bias,
            early_expert_affinity_modulation=config.neuron_config.early_expert_affinity_modulation,
            normalize_top_k_affinities=config.neuron_config.normalize_top_k_affinities,
            enable_spmd_rank=config.neuron_config.blockwise_matmul_config.parallelize_token_to_block_mapping
        ),
        blockwise_matmul_config=config.neuron_config.blockwise_matmul_config,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        dtype=config.neuron_config.torch_dtype,
        is_prefill=config.neuron_config.is_prefill_stage,
        enabled_hybrid_sharding=enabled_hybrid_sharding,
        tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        expert_model_parallel_group=parallel_state.get_expert_model_parallel_group(),
        cte_tensor_model_parallel_group=moe_cte_tensor_model_parallel_group,
        cte_expert_model_parallel_group=moe_cte_expert_model_parallel_group,
        tkg_tensor_model_parallel_group=moe_tkg_tensor_model_parallel_group,
        tkg_expert_model_parallel_group=moe_tkg_expert_model_parallel_group,
    )

    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        shared_experts=None,
        rmsnorm=None,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        return_expert_index=config.neuron_config.return_expert_index,
        sequence_dimension=1,
        init_tkg_module=False,
        tkg_config=None,
    )

    moe.eval()
    return moe


class NeuronMiniMaxM2DecoderLayer(nn.Module):
    """MiniMax M2 decoder layer with attention and MoE."""

    def __init__(self, config: MiniMaxM2InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronMiniMaxM2Attention(config=config)
        self.block_sparse_moe = initialize_minimax_m2_moe_module(config=config)
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)[0]
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache, None)


class NeuronMiniMaxM2Model(NeuronBaseModel):
    """NeuronMiniMaxM2Model for tracing."""

    def setup_attr_for_model(self, config: MiniMaxM2InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: MiniMaxM2InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [NeuronMiniMaxM2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


class NeuronMiniMaxM2ForCausalLM(NeuronBaseForCausalLM):
    """MiniMax M2 for causal language modeling."""

    _model_cls = NeuronMiniMaxM2Model

    @classmethod
    def get_config_cls(cls):
        return MiniMaxM2InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: MiniMaxM2InferenceConfig) -> dict:
        return convert_minimax_m2_hf_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        compiler_args += " --auto-cast=none"
        return compiler_args

    @classmethod
    def get_state_dict(cls, model_name_or_path: str, config: InferenceConfig) -> dict:
        """Load state dict from HuggingFace checkpoint."""
        import os
        from safetensors import safe_open
        import json

        # Override to use the actual HF model path
        model_name_or_path = '/home/ubuntu/model_hf/MiniMax-M2-BF16/'

        if os.path.isdir(model_name_or_path):
            index_path = os.path.join(model_name_or_path, 'model.safetensors.index.json')
            if os.path.exists(index_path):
                with open(index_path, 'r') as f:
                    index = json.load(f)

                model_sd = {}
                shard_files = set(index['weight_map'].values())
                for i, shard_file in enumerate(sorted(shard_files)):
                    if i % 20 == 0:
                        print(f"  Loading shard {i+1}/{len(shard_files)}: {shard_file}")
                    shard_path = os.path.join(model_name_or_path, shard_file)
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            model_sd[key] = f.get_tensor(key)

                print(f"  Loaded {len(model_sd)} parameters from {len(shard_files)} shards")

                # Remove model. prefix
                param_name_list = list(model_sd.keys())
                for param_name in param_name_list:
                    if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                        updated_param_name = param_name.replace(
                            cls._STATE_DICT_MODEL_PREFIX, cls._NEW_STATE_DICT_MODEL_PREFIX, 1
                        )
                        model_sd[updated_param_name] = model_sd[param_name]
                        del model_sd[param_name]

                # Convert HF state dict to Neuron format (router, MoE weights, qk_norm, etc.)
                model_sd = cls.convert_hf_to_neuron_state_dict(model_sd, config)

                # Handle tied word embeddings if configured
                if getattr(config, "tie_word_embeddings", False):
                    cls.update_state_dict_for_tied_weights(model_sd)

                return model_sd
            else:
                from neuronx_distributed_inference.modules.checkpoint import load_state_dict
                return load_state_dict(model_name_or_path)
        else:
            return super().get_state_dict(model_name_or_path, config)
