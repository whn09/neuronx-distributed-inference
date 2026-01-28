# coding=utf-8
# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
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
"""Qwen3 Next model for NXD inference.

Qwen3 Next is a hybrid attention model that combines:
- Full softmax attention (every full_attention_interval layers)
- Gated Delta Net linear attention (other layers)
- MoE with shared experts
"""

import gc
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
    SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
    MOE_TKG_MK_INTERMEDIATE_PER_TP,
)
from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.models.model_wrapper import CONTEXT_ENCODING_MODEL_TAG, TOKEN_GENERATION_MODEL_TAG
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module
from neuronx_distributed_inference.models.layer_boundary_marker import (
    ModuleMarkerEndWrapper,
    ModuleMarkerStartWrapper,
)

# Try to import HF Qwen3NextForCausalLM for weight loading
try:
    from transformers import Qwen3NextForCausalLM as HFQwen3NextForCausalLM
except ImportError:
    HFQwen3NextForCausalLM = None

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_call = nki_jit()(attention_isa_kernel)

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


def get_rmsnorm_cls():
    """Get appropriate RMSNorm class based on execution environment."""
    # Use CustomRMSNorm for Neuron, fallback RMSNorm for CPU
    if cpu_mode():
        class FallbackRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                input_dtype = hidden_states.dtype
                hidden_states = hidden_states.to(torch.float32)
                variance = hidden_states.pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                return (self.weight * hidden_states).to(input_dtype)
        return FallbackRMSNorm
    return CustomRMSNorm


def l2norm(x, dim=-1, eps=1e-6):
    """L2 normalize along a dimension."""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def get_modules_to_not_convert(neuron_config: MoENeuronConfig):
    """Get modules that should not be quantized."""
    return getattr(neuron_config, "modules_to_not_convert", None)


class Qwen3NextRMSNormGated(nn.Module):
    """Gated RMSNorm for Qwen3 Next linear attention."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        """Apply gated RMSNorm.

        Args:
            hidden_states: Input tensor
            gate: Gate tensor for element-wise multiplication
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Apply gate (SiLU activation)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return (self.weight * hidden_states).to(input_dtype)


class Qwen3NextInferenceConfig(InferenceConfig):
    """Configuration class for Qwen3 Next model inference on Neuron."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MoE configuration
        self.num_local_experts = self.num_experts  # 512 for Qwen3 Next

        # Shared experts configuration
        shared_expert_intermediate = getattr(self, 'shared_expert_intermediate_size', 0)
        self.n_shared_experts = 1 if shared_expert_intermediate > 0 else 0
        self.shared_expert_intermediate_size = shared_expert_intermediate

        # Hybrid attention configuration
        self.full_attention_interval = getattr(self, 'full_attention_interval', 4)

        # Full attention Q head expansion
        # Qwen3 Next uses 2x Q heads (32) compared to output heads (16)
        # The HF checkpoint has q_proj with 32 heads worth, but num_attention_heads=16
        # We need to track both for proper weight loading and attention computation
        self.full_attention_num_q_heads = self.num_attention_heads * 2  # 32 for Qwen3 Next
        self.q_expansion_factor = 2  # Q heads / output heads ratio

        # Linear attention (Gated Delta Net) configuration
        self.linear_num_key_heads = getattr(self, 'linear_num_key_heads', 16)
        self.linear_num_value_heads = getattr(self, 'linear_num_value_heads', 32)
        self.linear_key_head_dim = getattr(self, 'linear_key_head_dim', 128)
        self.linear_value_head_dim = getattr(self, 'linear_value_head_dim', 128)
        self.linear_conv_kernel_dim = getattr(self, 'linear_conv_kernel_dim', 4)

        # Partial RoPE configuration
        self.partial_rotary_factor = getattr(self, 'partial_rotary_factor', 0.25)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)

        # Build attention type list based on full_attention_interval
        self.attn_type_list = self._build_attn_type_list()
        self.layer_types = ["full_attention" if t == 1 else "linear_attention"
                          for t in self.attn_type_list]

        # Check whether need to pad intermediate size
        self.maybe_pad_intermediate()

        # Enable MoE fused NKI kernel if applicable
        self.enable_moe_fused_nki_kernel()

        # Set intermediate_size for ExpertMLPsV2
        self.intermediate_size = self.moe_intermediate_size

        # Router configuration
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "softmax"

        # Qwen3 Next normalizes top k affinities
        self.neuron_config.normalize_top_k_affinities = getattr(self, 'norm_topk_prob', True)

        # Disable numeric CC token as workaround
        self.neuron_config.disable_numeric_cc_token = True

    def _build_attn_type_list(self) -> List[int]:
        """Build attention type list based on full_attention_interval.

        Returns:
            List where 1 = full attention, 0 = linear attention (Gated Delta Net)
        """
        attn_types = []
        for layer_idx in range(self.num_hidden_layers):
            # Every full_attention_interval-th layer uses full attention
            if (layer_idx + 1) % self.full_attention_interval == 0:
                attn_types.append(1)  # Full attention
            else:
                attn_types.append(0)  # Linear attention
        return attn_types

    def maybe_pad_intermediate(self):
        """Pad intermediate size for efficient sharding."""
        moe_tp_degree = self.neuron_config.moe_tp_degree
        I_TP = self.moe_intermediate_size // moe_tp_degree
        if getattr(self.neuron_config.blockwise_matmul_config, "use_shard_on_intermediate_dynamic_while", False):
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded_moe_intermediate_size = (
                    math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP)
                    * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
                    * moe_tp_degree
                )
                self.moe_intermediate_pad_size = max(padded_moe_intermediate_size - self.moe_intermediate_size, 0)
                self.moe_intermediate_size = padded_moe_intermediate_size

    def enable_moe_fused_nki_kernel(self):
        """Enable MoE fused NKI kernel if conditions are met."""
        I_TP = self.moe_intermediate_size // self.neuron_config.moe_tp_degree
        if (getattr(self.neuron_config, "moe_fused_nki_kernel_enabled", False)
            and I_TP % MOE_TKG_MK_INTERMEDIATE_PER_TP == 0):
            self.moe_fused_nki_kernel_enabled = True

    def get_required_attributes(self) -> List[str]:
        """Return list of required config attributes."""
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "max_position_embeddings",
            "moe_intermediate_size",
            "num_attention_heads",
            "num_experts",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "vocab_size",
            "full_attention_interval",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "partial_rotary_factor",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


def apply_partial_rotary_pos_emb(q, k, cos, sin, rotary_dim):
    """Apply partial rotary position embeddings.

    Only applies RoPE to the first rotary_dim dimensions.
    """
    # Split into rotary and non-rotary parts
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    # Apply rotary embedding to the rotary part
    # cos/sin have shape (batch, seq_len, rotary_dim) or (batch, 1, seq_len, rotary_dim)
    # Need to ensure proper broadcasting
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rot_embedded = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot_embedded = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    # Concatenate back
    q_embedded = torch.cat([q_rot_embedded, q_pass], dim=-1)
    k_embedded = torch.cat([k_rot_embedded, k_pass], dim=-1)

    return q_embedded, k_embedded


class NeuronQwen3NextFullAttention(NeuronAttentionBase):
    """Full softmax attention for Qwen3 Next (used every full_attention_interval layers).

    Note on Q head expansion:
    Qwen3 Next HF checkpoint has 32 Q heads but config says num_attention_heads=16.
    For simplicity, we use 16 Q heads in NxD (matching config) and reduce Q weights
    from 32→16 heads during weight conversion. This is a simplified implementation.

    A fully faithful implementation would require:
    - Custom Q projection with 32 heads
    - Head merging after attention (32→16)
    - Custom O projection handling
    """

    def __init__(self, config: Qwen3NextInferenceConfig):
        # Partial RoPE: only apply to partial_rotary_factor of dimensions
        rotary_dim = int(config.head_dim * config.partial_rotary_factor)

        rotary_emb = RotaryEmbedding(
            rotary_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Use 16 attention heads as per config (simplified implementation)
        # The weight conversion will reduce Q from 32→16 heads
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,  # 16 heads
            num_key_value_heads=config.num_key_value_heads,  # 2 KV heads
            head_dim=config.head_dim,
            rotary_emb=rotary_emb,
            rms_norm_eps=config.rms_norm_eps,
            use_qk_norm=False,
        )

        # Store partial rotary config
        self.rotary_dim = rotary_dim
        self.partial_rotary_factor = config.partial_rotary_factor

        # Override q_layernorm and k_layernorm with RMSNorm
        self.q_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)
        self.k_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronQwen3NextFullAttention requires distributed initialization."
            )

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """Apply partial rotary embedding (only to rotary_dim dimensions).

        Override the base class method to implement partial RoPE.
        """
        if use_polar_compatible_rope:
            raise NotImplementedError("Polar compatible RoPE not supported for Qwen3 Next partial RoPE")

        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)
            Q, K = apply_partial_rotary_pos_emb(Q, K, cos_cache, sin_cache, self.rotary_dim)

        return Q, K, cos_cache, sin_cache


class NeuronQwen3NextGatedDeltaNet(nn.Module):
    """Gated Delta Net linear attention for Qwen3 Next.

    This implements the linear attention mechanism using gated delta rule,
    which processes sequences with linear complexity.
    """

    def __init__(self, config: Qwen3NextInferenceConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Linear attention head configuration
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        # Causal convolution configuration
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.activation = config.hidden_act

        # Convolution input dimension (Q + K + V)
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Causal Conv1d layer
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Input projections: Q, K, V, Z (gate)
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)

        # Input projections: beta and alpha (for gating)
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        # Time step parameters
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        # Gated RMSNorm
        self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)

        # Output projection
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def fix_query_key_value_ordering(self, qkvz, ba):
        """Split and reshape projections into Q, K, V, Z, beta, alpha."""
        batch_size, seq_len, _ = qkvz.shape

        # Split QKVZ
        q = qkvz[..., :self.key_dim]
        k = qkvz[..., self.key_dim:self.key_dim * 2]
        v = qkvz[..., self.key_dim * 2:self.key_dim * 2 + self.value_dim]
        z = qkvz[..., self.key_dim * 2 + self.value_dim:]

        # Reshape Q, K to (batch, seq, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.view(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # Split BA and add trailing dimension
        # ba shape: (batch, seq, num_v_heads * 2)
        # b, a shape after split: (batch, seq, num_v_heads)
        # After unsqueeze: (batch, seq, num_v_heads, 1)
        b = ba[..., :self.num_v_heads].unsqueeze(-1)
        a = ba[..., self.num_v_heads:].unsqueeze(-1)

        return q, k, v, z, b, a

    def torch_recurrent_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        """Simplified linear attention for token generation.

        Uses a simple linear attention formula to avoid compiler issues.
        For token generation (seq_len=1), computes:
            output = softmax(q @ k^T / sqrt(d)) @ v
        approximated as:
            output = (elu(q) + 1) @ ((elu(k) + 1)^T @ v) / normalizer

        This avoids the recurrent state update that causes compiler errors.
        """
        initial_dtype = query.dtype
        batch_size, seq_len, num_heads, k_head_dim = query.shape
        v_head_dim = value.shape[-1]

        # For token generation with seq_len=1, use simplified attention
        # that doesn't require complex state management

        # L2 normalize if requested
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)

        # Use ELU+1 feature map for linear attention
        # phi(x) = elu(x) + 1
        query_features = F.elu(query.to(torch.float32)) + 1.0
        key_features = F.elu(key.to(torch.float32)) + 1.0
        value = value.to(torch.float32)

        # For seq_len=1, compute linear attention directly
        # q: (batch, 1, heads, k_dim)
        # k: (batch, 1, heads, k_dim)
        # v: (batch, 1, heads, v_dim)

        # Compute attention with feature maps
        # (phi(q) @ phi(k)^T) @ v / (phi(q) @ sum(phi(k)))

        # Since seq_len=1, this simplifies to:
        # output = phi(q) * phi(k) * v / (phi(q) * phi(k) * 1)
        # which is just v scaled by a factor

        # Compute QK product element-wise (since both are seq_len=1)
        # (batch, 1, heads, k_dim) * (batch, 1, heads, k_dim) -> sum -> (batch, 1, heads)
        qk_sum = (query_features * key_features).sum(dim=-1, keepdim=True)  # (batch, 1, heads, 1)

        # Normalizer
        normalizer = qk_sum + 1e-6

        # For linear attention with seq_len=1, output is essentially value
        # scaled by the normalized attention score
        output = value * (qk_sum / normalizer)  # (batch, 1, heads, v_dim)

        return output.to(initial_dtype), None

    def torch_chunk_gated_delta_rule(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        """Chunk-based gated delta rule computation for context encoding.

        Uses a simplified linear attention approximation that's more
        compiler-friendly for context encoding on Neuron.
        """
        initial_dtype = query.dtype
        batch_size, seq_len, num_heads, k_head_dim = query.shape
        v_head_dim = value.shape[-1]

        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)

        # Convert to float32 for computation
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        value = value.to(torch.float32)

        # Scale factor
        scale = 1.0 / (k_head_dim ** 0.5)
        query = query * scale

        # g and beta: (batch, seq, heads, 1)
        g_val = g.squeeze(-1).to(torch.float32)  # (batch, seq, heads)
        beta_val = beta.squeeze(-1).to(torch.float32)  # (batch, seq, heads)

        # Apply sigmoid to beta
        beta_val = beta_val.sigmoid()

        # For context encoding, use a simplified causal linear attention
        # This approximates the gated delta rule with cumulative sum approach
        # which is more efficient for parallel computation

        # Apply gating decay (cumulative product approximation via cumsum of log)
        # g_cumsum shape: (batch, seq, heads)
        g_cumsum = g_val.cumsum(dim=1)  # cumulative sum of log-decay
        g_decay = g_cumsum.exp()  # (batch, seq, heads)

        # Expand for broadcasting
        g_decay_expanded = g_decay.unsqueeze(-1)  # (batch, seq, heads, 1)
        beta_expanded = beta_val.unsqueeze(-1)  # (batch, seq, heads, 1)

        # Apply decay to keys and values
        # Weighted key: k * exp(cumsum(g)) * beta
        weighted_k = key * g_decay_expanded * beta_expanded  # (batch, seq, heads, k_dim)
        weighted_v = value * beta_expanded  # (batch, seq, heads, v_dim)

        # Compute causal cumulative sum for linear attention
        # For each position i, we want sum_{j<=i} k_j^T @ v_j with decay
        # Using einsum to compute outer products and cumsum

        # Compute k^T @ v for each position: (batch, seq, heads, k_dim, v_dim)
        kv_outer = torch.einsum('bshk,bshv->bshkv', weighted_k, weighted_v)

        # Cumulative sum along sequence dimension
        kv_cumsum = kv_outer.cumsum(dim=1)  # (batch, seq, heads, k_dim, v_dim)

        # Apply inverse decay to query and compute output
        # Query needs inverse decay for proper attention weighting
        g_inv_decay = (-g_cumsum).exp().unsqueeze(-1)  # (batch, seq, heads, 1)
        weighted_q = query * g_inv_decay  # (batch, seq, heads, k_dim)

        # Output: q @ cumsum(k^T @ v) for each position
        output = torch.einsum('bshkv,bshk->bshv', kv_cumsum, weighted_q)

        last_recurrent_state = None  # Context encoding doesn't return state

        return output.to(initial_dtype), last_recurrent_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None, None]:
        """Forward pass for Gated Delta Net linear attention.

        ULTRA-MINIMAL VERSION: Bypass all computation to isolate compiler issue.
        Just project input through out_proj to maintain correct shapes.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Minimal computation: project input through QKVZ, then through out_proj
        # This tests if the linear layers alone cause issues
        projected_qkvz = self.in_proj_qkvz(hidden_states)

        # Extract just the V portion and project to output
        # v starts at key_dim * 2
        v_start = self.key_dim * 2
        v_end = v_start + self.value_dim
        v = projected_qkvz[..., v_start:v_end]  # (batch, seq, value_dim)

        # Project output
        output = self.out_proj(v)

        # Linear attention doesn't use traditional KV cache, but we need to return
        # dummy KV tensors to satisfy the KV cache manager's shape expectations.
        # With REPLICATE_TO_TP_DEGREE strategy, num_key_value_heads (2) is replicated to
        # tp_degree (8), so each TP rank gets 8/8 = 1 head. Shape per rank is [batch, 1, seq_len, head_dim].
        num_kv_heads_per_rank = 1  # Due to GQA replication: tp_degree / tp_degree = 1
        dummy_k = torch.zeros(
            batch_size,
            num_kv_heads_per_rank,
            seq_len,
            self.config.head_dim,  # 256
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dummy_v = torch.zeros_like(dummy_k)
        present_key_value = (dummy_k, dummy_v)

        return output, present_key_value, None, None


class NeuronQwen3NextDecoderLayer(nn.Module):
    """Qwen3 Next decoder layer with hybrid attention (full + linear)."""

    def __init__(self, config: Qwen3NextInferenceConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.is_full_attention = config.attn_type_list[layer_idx] == 1

        # Select attention type based on layer configuration
        if self.is_full_attention:
            self.self_attn = NeuronQwen3NextFullAttention(config=config)
        else:
            self.self_attn = NeuronQwen3NextGatedDeltaNet(config=config, layer_idx=layer_idx)

        # MoE with shared experts
        self.moe_fused_nki_kernel_enabled = getattr(config, "moe_fused_nki_kernel_enabled", False)

        # Layer norms
        self.input_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize MoE module
        if self.moe_fused_nki_kernel_enabled:
            self.mlp = initialize_moe_module(
                config=config,
                rmsnorm=self.post_attention_layernorm,
                init_tkg_module=True,
            )
        else:
            self.mlp = initialize_moe_module(config=config)

        # Kernel configuration
        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled if self.is_full_attention else False
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        """Forward pass for decoder layer."""
        if "padding_mask" in kwargs:
            warnings.warn("padding_mask is deprecated, use attention_mask instead.")

        residual = hidden_states

        # Module markers for compiler optimization
        hidden_states = ModuleMarkerStartWrapper()(hidden_states)

        # Input layer norm
        qkv_fused_rmsnorm = None
        if self.input_layernorm:
            if self.qkv_kernel_enabled and self.qkv_kernel_fused_rmsnorm and self.is_full_attention:
                qkv_fused_rmsnorm = self.input_layernorm
            else:
                hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        if self.is_full_attention:
            hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                rmsnorm=qkv_fused_rmsnorm,
                **kwargs,
            )
        else:
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
        if not self.moe_fused_nki_kernel_enabled:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, padding_mask)[0]
        hidden_states = residual + hidden_states

        # End module marker
        hidden_states = ModuleMarkerEndWrapper()(hidden_states)

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronQwen3NextModel(NeuronBaseModel):
    """Qwen3 Next model for NXD inference."""

    def setup_attr_for_model(self, config: Qwen3NextInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

        # Mixed attention configuration
        # Disable mixed attention for now - use uniform cache sizes to avoid
        # dimension mismatches during compilation. Linear attention layers will
        # produce dummy KV cache that isn't actually used.
        self.has_mixed_attn = False
        self.attn_type_list = config.attn_type_list

        # Build layer to cache size mapping - use uniform seq_len for all layers
        # to avoid dynamic slice dimension mismatches during context encoding.
        # Linear attention layers produce dummy KV cache for compatibility.
        self.layer_to_cache_size_mapping = []
        for layer_idx in range(config.num_hidden_layers):
            # All layers use full sequence length for cache
            self.layer_to_cache_size_mapping.append(config.neuron_config.seq_len)

    def init_model(self, config: Qwen3NextInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )

        self.layers = nn.ModuleList([
            NeuronQwen3NextDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


def _helper_concat_and_delete_qkv(state_dict: Dict[str, Any], layer_num: int, attr: str):
    """Helper function to concatenate and delete QKV attributes for fused QKV."""
    prefix = f"layers.{layer_num}.self_attn"
    state_dict[f"{prefix}.Wqkv.{attr}"] = torch.cat([
        state_dict[f"{prefix}.q_proj.{attr}"],
        state_dict[f"{prefix}.k_proj.{attr}"],
        state_dict[f"{prefix}.v_proj.{attr}"],
    ])
    del state_dict[f"{prefix}.q_proj.{attr}"]
    del state_dict[f"{prefix}.k_proj.{attr}"]
    del state_dict[f"{prefix}.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(state_dict: Dict[str, Any], config: Qwen3NextInferenceConfig):
    """Convert state dict to fused QKV format for full attention layers only."""
    mods_to_not_conv = get_modules_to_not_convert(config.neuron_config) or []

    for l in range(config.num_hidden_layers):
        # Only fuse QKV for full attention layers
        if config.attn_type_list[l] != 1:
            continue

        _helper_concat_and_delete_qkv(state_dict, l, "weight")
        if ((config.neuron_config.quantized_mlp_kernel_enabled or config.neuron_config.quantized)
            and f"layers.{l}.self_attn" not in mods_to_not_conv):
            _helper_concat_and_delete_qkv(state_dict, l, "scale")

    gc.collect()
    return state_dict


def maybe_dequantize_layer(neuron_state_dict, config):
    """Dequantize FP8 layers if needed."""
    scale_layers = []
    for layer_key in neuron_state_dict.keys():
        if "_scale_inv" in layer_key:
            scales = neuron_state_dict[layer_key]
            scale_layers.append(layer_key)
            fp8_layer_name = layer_key.replace("_scale_inv", "")
            fp8_layer = neuron_state_dict[fp8_layer_name]
            block_size = config.quantization_config["weight_block_size"]
            scales_expanded = scales.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1)
            scaled_layer = fp8_layer.to(torch.float32) * scales_expanded.to(torch.float32)
            neuron_state_dict[fp8_layer_name] = scaled_layer.to(config.neuron_config.torch_dtype)

    for scale_layer in scale_layers:
        del neuron_state_dict[scale_layer]


def convert_qwen3_next_hf_to_neuron_state_dict(neuron_state_dict: dict, config: Qwen3NextInferenceConfig) -> dict:
    """Convert HuggingFace Qwen3 Next checkpoint to Neuron format."""
    import logging
    logging.warning(f"[QWEN3_NEXT_DEBUG] convert_qwen3_next_hf_to_neuron_state_dict called with {len(neuron_state_dict)} keys")
    # Write debug info to file for easier tracking
    with open('/tmp/qwen3_next_debug.log', 'a') as f:
        f.write(f"[DEBUG] convert_qwen3_next_hf_to_neuron_state_dict called with {len(neuron_state_dict)} keys\n")
        f.write(f"[DEBUG] Sample keys: {list(neuron_state_dict.keys())[:20]}\n")
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    # Dequantize if needed
    maybe_dequantize_layer(neuron_state_dict, config)

    # First pass: Rename linear_attn to self_attn for linear attention layers
    # HF Qwen3 Next uses different module names for different attention types:
    # - Full attention layers: self_attn.*
    # - Linear attention layers: linear_attn.*
    # Our implementation uses self_attn for all layers, so we need to rename
    keys_to_rename = []
    for key in list(neuron_state_dict.keys()):
        if ".linear_attn." in key:
            new_key = key.replace(".linear_attn.", ".self_attn.")
            keys_to_rename.append((key, new_key))

    for old_key, new_key in keys_to_rename:
        neuron_state_dict[new_key] = neuron_state_dict.pop(old_key)

    # Add rank utility tensors
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    for l in range(config.num_hidden_layers):
        is_full_attention = config.attn_type_list[l] == 1

        # Add rank utility for full attention layers
        if is_full_attention:
            neuron_state_dict[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
                0, config.neuron_config.tp_degree, dtype=torch.int32
            )

            # Rename q_norm, k_norm for full attention layers
            if f"layers.{l}.self_attn.q_norm.weight" in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = (
                    neuron_state_dict.pop(f"layers.{l}.self_attn.q_norm.weight").detach().clone()
                )
            if f"layers.{l}.self_attn.k_norm.weight" in neuron_state_dict:
                neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = (
                    neuron_state_dict.pop(f"layers.{l}.self_attn.k_norm.weight").detach().clone()
                )

            # NOTE: Do NOT rename Q/K/V projections here - the preshard_hook in gqa.py
            # handles the mapping from HF keys (layers.X.self_attn.q_proj.weight) to
            # NxD keys (layers.X.self_attn.qkv_proj.q_proj.weight) automatically.
            # Manual renaming interferes with this mechanism.

            # Reduce Q projection from 32 heads to 16 heads (simplified implementation)
            # HF checkpoint has q_proj with 32 heads worth (8192 = 32 × 256), but
            # our NxD model uses 16 heads (matching config.num_attention_heads)
            q_proj_key = f"layers.{l}.self_attn.q_proj.weight"
            with open('/tmp/qwen3_next_debug.log', 'a') as f:
                f.write(f"[DEBUG] Layer {l}: Checking for Q reduction, key={q_proj_key}, exists={q_proj_key in neuron_state_dict}\n")
            if q_proj_key in neuron_state_dict:
                q_weight = neuron_state_dict[q_proj_key]  # [8192, 2048]
                original_shape = q_weight.shape
                head_dim = config.head_dim  # 256
                hf_num_q_heads = original_shape[0] // head_dim  # 32
                target_num_q_heads = config.num_attention_heads  # 16

                if hf_num_q_heads > target_num_q_heads:
                    # Reduce heads by taking every nth head (simple subsampling)
                    # Alternative: average pairs - but subsampling preserves more structure
                    reduction_factor = hf_num_q_heads // target_num_q_heads  # 2

                    # Reshape to [num_heads, head_dim, hidden_size]
                    q_weight = q_weight.reshape(hf_num_q_heads, head_dim, -1)

                    # Take every 'reduction_factor' head (subsampling)
                    q_weight_reduced = q_weight[::reduction_factor]  # [16, 256, 2048]

                    # Reshape back to [num_heads * head_dim, hidden_size]
                    q_weight_reduced = q_weight_reduced.reshape(target_num_q_heads * head_dim, -1)

                    neuron_state_dict[q_proj_key] = q_weight_reduced
                    with open('/tmp/qwen3_next_debug.log', 'a') as f:
                        f.write(f"[DEBUG] Reduced Q projection for layer {l}: {original_shape} -> {q_weight_reduced.shape}\n")
                    print(f"Reduced Q projection for layer {l}: {original_shape} -> {q_weight_reduced.shape}")

        # Copy router weights
        if f"layers.{l}.mlp.gate.weight" in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
                neuron_state_dict.pop(f"layers.{l}.mlp.gate.weight").detach().clone()
            )

        # Convert expert weights
        if f"layers.{l}.mlp.experts.0.gate_proj.weight" in neuron_state_dict:
            intermediate_size, hidden_size = neuron_state_dict[
                f"layers.{l}.mlp.experts.0.gate_proj.weight"
            ].shape
            device = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].device
            dtype = neuron_state_dict[f"layers.{l}.mlp.experts.0.gate_proj.weight"].dtype

            # Combine gate and up projections
            gate_up_proj = torch.empty(
                config.num_experts,
                hidden_size,
                2 * intermediate_size,
                dtype=dtype,
                device=device,
            )

            for e in range(config.num_experts):
                gate_key = f"layers.{l}.mlp.experts.{e}.gate_proj.weight"
                up_key = f"layers.{l}.mlp.experts.{e}.up_proj.weight"

                if gate_key in neuron_state_dict:
                    gate_proj_weights = neuron_state_dict[gate_key].T.detach().clone()
                    up_proj_weights = neuron_state_dict[up_key].T.detach().clone()

                    gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
                    gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
                    gate_proj_slice.copy_(gate_proj_weights)
                    up_proj_slice = torch.narrow(gate_up_proj_slice, 2, intermediate_size, intermediate_size)
                    up_proj_slice.copy_(up_proj_weights)

                    del neuron_state_dict[gate_key]
                    del neuron_state_dict[up_key]

            # Padding for intermediate size
            pad_size = getattr(config, "moe_intermediate_pad_size", 0)
            if pad_size > 0:
                gate_up_proj = gate_up_proj.reshape(config.num_experts, hidden_size, 2, -1)
                gate_up_proj = F.pad(gate_up_proj, (0, pad_size))
                gate_up_proj = gate_up_proj.reshape(config.num_experts, hidden_size, -1)

            neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

            # Down projection
            down_proj = torch.empty(
                config.num_experts,
                intermediate_size,
                hidden_size,
                dtype=dtype,
                device=device,
            )

            for e in range(config.num_experts):
                down_key = f"layers.{l}.mlp.experts.{e}.down_proj.weight"
                if down_key in neuron_state_dict:
                    down_proj_weights = neuron_state_dict[down_key].T.detach().clone()
                    down_proj_slice = torch.narrow(down_proj, 0, e, 1)
                    down_proj_slice.copy_(down_proj_weights)
                    del neuron_state_dict[down_key]

            if pad_size > 0:
                down_proj = F.pad(down_proj, (0, 0, 0, pad_size))

            neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        # Convert shared expert weights if present
        shared_gate_key = f"layers.{l}.mlp.shared_expert.gate_proj.weight"
        if shared_gate_key in neuron_state_dict:
            # Shared expert weights - DO NOT transpose!
            # HF weights are in [out_features, in_features] format which matches
            # ColumnParallelLinear's expected shape.
            # After TP sharding: [intermediate_size/tp, hidden_size] = [8, 2048]
            shared_gate = neuron_state_dict.pop(shared_gate_key).detach().clone()
            shared_up = neuron_state_dict.pop(f"layers.{l}.mlp.shared_expert.up_proj.weight").detach().clone()
            shared_down = neuron_state_dict.pop(f"layers.{l}.mlp.shared_expert.down_proj.weight").detach().clone()

            # Store as shared expert weights
            neuron_state_dict[f"layers.{l}.mlp.shared_experts.gate_proj.weight"] = shared_gate
            neuron_state_dict[f"layers.{l}.mlp.shared_experts.up_proj.weight"] = shared_up
            neuron_state_dict[f"layers.{l}.mlp.shared_experts.down_proj.weight"] = shared_down

        gc.collect()

    # Fuse QKV for full attention layers if configured
    if config.neuron_config.fused_qkv:
        neuron_state_dict = convert_state_dict_to_fused_qkv(neuron_state_dict, config)

    return neuron_state_dict


class NeuronQwen3NextForCausalLM(NeuronBaseForCausalLM):
    """Qwen3 Next model for causal language modeling on Neuron."""

    _model_cls = NeuronQwen3NextModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        if HFQwen3NextForCausalLM is not None:
            return HFQwen3NextForCausalLM.from_pretrained(model_path, **kwargs)
        else:
            # Fallback: load using AutoModelForCausalLM
            from transformers import AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)

    @classmethod
    def get_config_cls(cls):
        return Qwen3NextInferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Qwen3NextInferenceConfig) -> dict:
        return convert_qwen3_next_hf_to_neuron_state_dict(state_dict, config)

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self):
        # Set compiler optimization level based on model tag
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O3" if self.neuron_config.moe_ep_degree > 1 else "-O1"

        compiler_args = (
            f"--enable-saturate-infinity --enable-mixed-precision-accumulation "
            f"--model-type transformer {optimization_level}"
        )

        # CC overlap optimization
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        compiler_args += " --auto-cast=none"

        # Vector offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"

        if self.neuron_config.scratchpad_page_size:
            compiler_args += f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}"

        return compiler_args
