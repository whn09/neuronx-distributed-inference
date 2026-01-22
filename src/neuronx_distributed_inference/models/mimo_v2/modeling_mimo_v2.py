# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This implementation is based on the MiMo-V2-Flash model from Xiaomi.
# Reference: https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash

"""MiMo-V2-Flash model for NXD inference."""

import gc
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    MoENeuronConfig,
)
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.models.model_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from neuronx_distributed_inference.modules.attention.attention_base import (
    NeuronAttentionBase,
)
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from torch_neuronx.xla_impl.ops import nki_jit

_flash_fwd_call = nki_jit()(attention_isa_kernel)


def get_rmsnorm_cls():
    """Get appropriate RMSNorm class based on execution environment."""
    return MiMoV2RMSNorm if cpu_mode() else CustomRMSNorm


class MiMoV2RMSNorm(nn.Module):
    """RMSNorm implementation for CPU mode."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MiMoV2RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for MiMo-V2-Flash.

    Supports partial rotary embedding where only a fraction of dimensions
    use rotary position encoding.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 262144,
        base: float = 5000000.0,
        partial_rotary_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        # Calculate the actual dimension used for rotary embedding
        self.rope_dim = int(dim * partial_rotary_factor)
        # Ensure rope_dim is even
        self.rope_dim = self.rope_dim - (self.rope_dim % 2)

        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32) / self.rope_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            position_ids: Position indices of shape (batch_size, seq_len)

        Returns:
            Tuple of (cos, sin) tensors for rotary embedding
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MiMoV2InferenceConfig(InferenceConfig):
    """Configuration class for MiMo-V2-Flash inference on Neuron."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # MoE configuration
        self.num_local_experts = self.n_routed_experts
        self.n_shared_experts = 0  # MiMo-V2-Flash has no shared experts

        # Set intermediate_size for MoE layers
        self.intermediate_size = self.moe_intermediate_size

        # Check and pad intermediate size if needed
        self.maybe_pad_intermediate()

        # Router configuration
        self.neuron_config.router_config.dtype = torch.float32
        self.neuron_config.router_config.act_fn = "sigmoid"  # MiMo uses sigmoid

        # Disable numeric CC token as workaround
        self.neuron_config.disable_numeric_cc_token = True

        # MiMo normalizes top-k affinities
        self.neuron_config.normalize_top_k_affinities = True

        # Parse hybrid layer pattern
        self._parse_hybrid_pattern()

    def _parse_hybrid_pattern(self):
        """Parse hybrid layer pattern to determine attention types."""
        if hasattr(self, 'hybrid_layer_pattern') and self.hybrid_layer_pattern:
            self.layer_attention_types = [
                "sliding_window" if p == 1 else "full"
                for p in self.hybrid_layer_pattern
            ]
        else:
            self.layer_attention_types = ["full"] * self.num_hidden_layers

        # Parse MoE layer frequency
        if hasattr(self, 'moe_layer_freq') and self.moe_layer_freq:
            self.layer_uses_moe = [bool(f) for f in self.moe_layer_freq]
        else:
            self.layer_uses_moe = [True] * self.num_hidden_layers

    def maybe_pad_intermediate(self):
        """Pad intermediate size if required for efficient computation."""
        from neuronx_distributed_inference.models.config import (
            SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP,
        )

        moe_tp_degree = self.neuron_config.moe_tp_degree
        I_TP = self.moe_intermediate_size // moe_tp_degree

        if getattr(
            self.neuron_config.blockwise_matmul_config,
            "use_shard_on_intermediate_dynamic_while",
            False,
        ):
            if I_TP % SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP != 0:
                padded_size = (
                    math.ceil(I_TP / SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP)
                    * SHARD_ON_INTERMEDIATE_DIMENSION_PER_TP
                    * moe_tp_degree
                )
                self.moe_intermediate_pad_size = max(
                    padded_size - self.moe_intermediate_size, 0
                )
                self.moe_intermediate_size = padded_size

    def get_required_attributes(self) -> List[str]:
        return [
            "attention_bias",
            "head_dim",
            "hidden_act",
            "hidden_size",
            "hybrid_layer_pattern",
            "layernorm_epsilon",
            "max_position_embeddings",
            "moe_intermediate_size",
            "moe_layer_freq",
            "n_routed_experts",
            "norm_topk_prob",
            "num_attention_heads",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "partial_rotary_factor",
            "rope_theta",
            "scoring_func",
            "sliding_window",
            "swa_head_dim",
            "swa_num_attention_heads",
            "swa_num_key_value_heads",
            "swa_rope_theta",
            "swa_v_head_dim",
            "tie_word_embeddings",
            "v_head_dim",
            "vocab_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[MoENeuronConfig]:
        return MoENeuronConfig


class NeuronMiMoV2Attention(NeuronAttentionBase):
    """MiMo-V2-Flash Attention implementation supporting hybrid attention patterns.

    Supports both full attention and sliding window attention with different
    head dimensions for Q/K vs V.

    NOTE: MiMo-V2 has different K head_dim (192) and V head_dim (128), which is not
    supported by the base class. Therefore, this class implements a custom forward()
    method to handle this case properly.

    The MiMo-V2 model uses an "attention sink" mechanism for SWA layers. This adds
    a learnable bias that acts as a virtual token to absorb excess attention probability,
    preventing attention from being too concentrated on specific tokens.
    """

    def __init__(
        self,
        config: MiMoV2InferenceConfig,
        layer_idx: int,
        is_sliding_window: bool = False,
    ):
        self.layer_idx = layer_idx
        self.is_sliding_window = is_sliding_window

        # Determine if this layer uses attention sink
        # SWA layers use attention sink if add_swa_attention_sink_bias=True
        # Full attention layers use attention sink if add_full_attention_sink_bias=True
        add_swa_sink = getattr(config, 'add_swa_attention_sink_bias', False)
        add_full_sink = getattr(config, 'add_full_attention_sink_bias', False)
        self.use_attention_sink = (is_sliding_window and add_swa_sink) or (not is_sliding_window and add_full_sink)

        # Select parameters based on attention type
        if is_sliding_window:
            self.attn_head_dim = config.swa_head_dim
            self.attn_v_head_dim = config.swa_v_head_dim
            self.attn_num_heads = config.swa_num_attention_heads
            self.attn_num_kv_heads = config.swa_num_key_value_heads
            rope_theta = getattr(config, 'swa_rope_theta', 10000.0)
            self.sliding_window_size = config.sliding_window
        else:
            self.attn_head_dim = config.head_dim
            self.attn_v_head_dim = config.v_head_dim
            self.attn_num_heads = config.num_attention_heads
            self.attn_num_kv_heads = config.num_key_value_heads
            rope_theta = config.rope_theta
            self.sliding_window_size = None

        # Calculate partial rotary dimensions
        self.partial_rotary_factor = config.partial_rotary_factor
        self.rope_dim = int(self.attn_head_dim * self.partial_rotary_factor)
        self.rope_dim = self.rope_dim - (self.rope_dim % 2)  # Ensure even
        self.nope_dim = self.attn_head_dim - self.rope_dim

        # Create rotary embedding
        rotary_emb = MiMoV2RotaryEmbedding(
            dim=self.attn_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
        )

        # Initialize base attention
        # Note: We pass head_dim for scaling calculation
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=self.attn_num_heads,
            num_key_value_heads=self.attn_num_kv_heads,
            head_dim=self.attn_head_dim,  # Use Q/K head_dim for base class
            rotary_emb=rotary_emb,
            rms_norm_eps=config.layernorm_epsilon,
            use_qk_norm=False,
        )

        # Override projections with correct dimensions for different K/V head dims
        self._init_projections(config)

        # Scaling factor for attention
        self.scaling = self.attn_head_dim ** -0.5

        # Store cache KV heads for cache compatibility
        # With CONVERT_TO_MHA, all layers have num_attention_heads KV heads
        # Otherwise, use max of full and sliding window kv heads
        tp_degree = config.neuron_config.tp_degree
        if self.use_gqa_convert_to_mha:
            self.cache_num_kv_heads = self.attn_num_heads
            self.local_cache_kv_heads = self.local_num_heads
        else:
            self.cache_num_kv_heads = max(
                config.num_key_value_heads,
                getattr(config, 'swa_num_key_value_heads', config.num_key_value_heads)
            )
            self.local_cache_kv_heads = max(1, self.cache_num_kv_heads // tp_degree)

        # Attention sink bias for SWA layers
        # This is a learnable bias per attention head that acts as a virtual token
        # to absorb excess attention probability
        if self.use_attention_sink:
            # The attention_sink_bias is stored as [tp_degree, local_num_heads]
            # At runtime, we use rank_util.rank to select the correct shard
            # This allows dynamic per-rank selection that works with NXD's weight loading
            self.attention_sink_bias = nn.Parameter(
                torch.zeros(tp_degree, self.local_num_heads, dtype=config.neuron_config.torch_dtype),
                requires_grad=False
            )
        else:
            self.attention_sink_bias = None

    def _init_projections(self, config: MiMoV2InferenceConfig):
        """Initialize projection layers with correct dimensions for different K/V head dims."""
        dtype = config.neuron_config.torch_dtype
        tp_degree = config.neuron_config.tp_degree

        # Check if we need GQA CONVERT_TO_MHA (when tp_degree > num_kv_heads)
        self.use_gqa_convert_to_mha = tp_degree > self.attn_num_kv_heads

        if self.use_gqa_convert_to_mha:
            k_num_heads = self.attn_num_heads
            v_num_heads = self.attn_num_heads
        else:
            k_num_heads = self.attn_num_kv_heads
            v_num_heads = self.attn_num_kv_heads

        # Q/K use head_dim, V uses v_head_dim
        q_hidden_size = self.attn_num_heads * self.attn_head_dim
        k_hidden_size = k_num_heads * self.attn_head_dim
        v_hidden_size = v_num_heads * self.attn_v_head_dim
        o_hidden_size = self.attn_num_heads * self.attn_v_head_dim

        if parallel_state.model_parallel_is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()

            self.q_proj = ColumnParallelLinear(
                config.hidden_size, q_hidden_size,
                bias=config.attention_bias, gather_output=False,
                dtype=dtype, tensor_model_parallel_group=tp_group,
            )
            self.k_proj = ColumnParallelLinear(
                config.hidden_size, k_hidden_size,
                bias=config.attention_bias, gather_output=False,
                dtype=dtype, tensor_model_parallel_group=tp_group,
            )
            self.v_proj = ColumnParallelLinear(
                config.hidden_size, v_hidden_size,
                bias=config.attention_bias, gather_output=False,
                dtype=dtype, tensor_model_parallel_group=tp_group,
            )
            self.o_proj = RowParallelLinear(
                o_hidden_size, config.hidden_size,
                bias=False, input_is_parallel=True,
                dtype=dtype, tensor_model_parallel_group=tp_group,
            )

            self.local_num_heads = self.attn_num_heads // tp_degree
            if self.use_gqa_convert_to_mha:
                self.local_num_kv_heads = self.local_num_heads
            else:
                self.local_num_kv_heads = max(1, self.attn_num_kv_heads // tp_degree)
        else:
            self.q_proj = nn.Linear(config.hidden_size, q_hidden_size, bias=config.attention_bias)
            self.k_proj = nn.Linear(config.hidden_size, k_hidden_size, bias=config.attention_bias)
            self.v_proj = nn.Linear(config.hidden_size, v_hidden_size, bias=config.attention_bias)
            self.o_proj = nn.Linear(o_hidden_size, config.hidden_size, bias=False)

            self.local_num_heads = self.attn_num_heads
            self.local_num_kv_heads = k_num_heads

        # Remove qkv_proj from base class
        if hasattr(self, 'qkv_proj'):
            self.qkv_proj = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for MiMo-V2-Flash attention with different K/V head dims."""

        # Handle sequence parallel
        if self.sequence_parallel_enabled and parallel_state.model_parallel_is_initialized():
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
                process_group=parallel_state.get_tensor_model_parallel_group(),
            )

        bsz, q_len, _ = hidden_states.size()
        is_token_gen = past_key_value is not None

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape: [bsz, seq, hidden] -> [bsz, heads, seq, head_dim]
        query_states = query_states.view(bsz, q_len, self.local_num_heads, self.attn_head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.local_num_kv_heads, self.attn_head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.local_num_kv_heads, self.attn_v_head_dim).transpose(1, 2)

        # Apply partial RoPE
        query_rope = query_states[..., :self.rope_dim]
        query_nope = query_states[..., self.rope_dim:]
        key_rope = key_states[..., :self.rope_dim]
        key_nope = key_states[..., self.rope_dim:]

        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(value_states, position_ids)

        query_rope, key_rope = apply_rotary_pos_emb(
            query_rope, key_rope, cos_cache, sin_cache, position_ids
        )

        query_states = torch.cat([query_rope, query_nope], dim=-1)
        key_states = torch.cat([key_rope, key_nope], dim=-1)

        # Prepare KV for cache (BEFORE GQA repeat)
        key_states_for_cache = key_states
        value_states_for_cache = value_states

        # Pad V from v_head_dim to head_dim for cache compatibility
        if self.attn_v_head_dim < self.attn_head_dim:
            pad_size = self.attn_head_dim - self.attn_v_head_dim
            value_states_for_cache = F.pad(value_states_for_cache, (0, pad_size), value=0.0)

        # Pad KV heads if needed (when layer has fewer KV heads than cache expects)
        if not self.use_gqa_convert_to_mha and self.local_num_kv_heads < self.local_cache_kv_heads:
            repeat_factor = self.local_cache_kv_heads // self.local_num_kv_heads
            key_states_for_cache = key_states_for_cache.repeat(1, repeat_factor, 1, 1)
            value_states_for_cache = value_states_for_cache.repeat(1, repeat_factor, 1, 1)

        # GQA: repeat KV heads to match Q heads
        num_key_value_groups = self.local_num_heads // self.local_num_kv_heads
        if num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

        if is_token_gen:
            # Token generation with decomposed attention
            K_prior = past_key_value[0]
            V_prior = past_key_value[1]

            # Slice KV heads if cache has more than layer needs
            if not self.use_gqa_convert_to_mha and self.local_num_kv_heads < self.local_cache_kv_heads:
                K_prior = K_prior[:, :self.local_num_kv_heads, :, :]
                V_prior = V_prior[:, :self.local_num_kv_heads, :, :]

            # Slice V back to v_head_dim
            if self.attn_v_head_dim < self.attn_head_dim:
                V_prior = V_prior[..., :self.attn_v_head_dim]

            # GQA repeat for cached KV
            if num_key_value_groups > 1:
                K_prior = K_prior.repeat_interleave(num_key_value_groups, dim=1)
                V_prior = V_prior.repeat_interleave(num_key_value_groups, dim=1)

            # Compute prior scores: Q @ K_prior^T
            prior_scores = torch.matmul(query_states, K_prior.transpose(-2, -1)) * self.scaling

            # Pad attention mask if needed
            if attention_mask is not None and prior_scores.shape[-1] > attention_mask.shape[-1]:
                pad_size = prior_scores.shape[-1] - attention_mask.shape[-1]
                attention_mask = F.pad(attention_mask, (0, pad_size), "constant", False)

            # Apply mask (True = attend, False = mask)
            if attention_mask is not None:
                prior_scores = torch.where(
                    attention_mask.to(torch.bool),
                    prior_scores,
                    torch.finfo(prior_scores.dtype).min
                )
            prior_scores = prior_scores.to(torch.float32)

            # Compute active scores: Q @ K_active^T
            active_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
            active_scores = active_scores.to(torch.float32)

            # Combined softmax with optional attention sink
            all_scores = torch.cat([prior_scores, active_scores], dim=-1)

            # Apply attention sink bias if enabled
            if self.use_attention_sink and self.attention_sink_bias is not None:
                # attention_sink_bias has shape [tp_degree, local_num_heads]
                # Use rank_util.rank to dynamically select the correct shard at runtime
                # rank_util.rank is loaded per-rank and contains that rank's index
                rank_idx = self.rank_util.rank[0]  # scalar tensor with current rank
                local_sink_bias = self.attention_sink_bias[rank_idx]  # [local_num_heads]
                # Reshape: [local_num_heads] -> [1, local_num_heads, 1, 1]
                sink_bias = local_sink_bias.view(1, -1, 1, 1).expand(bsz, -1, q_len, 1).to(torch.float32)
                # Concatenate sink as extra column
                all_scores = torch.cat([all_scores, sink_bias], dim=-1)
                # Subtract max for numerical stability
                all_scores = all_scores - all_scores.max(dim=-1, keepdim=True).values
                # Softmax
                attn_weights = F.softmax(all_scores, dim=-1, dtype=torch.float32)
                # Drop the sink column
                attn_weights = attn_weights[..., :-1]
            else:
                attn_weights = F.softmax(all_scores, dim=-1, dtype=torch.float32)

            prior_weights = attn_weights[..., :-q_len].to(V_prior.dtype)
            active_weights = attn_weights[..., -q_len:].to(value_states.dtype)

            attn_output = torch.matmul(prior_weights, V_prior) + torch.matmul(active_weights, value_states)
        else:
            # Context encoding: standard attention
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling

            if attention_mask is not None:
                attn_weights = torch.where(
                    attention_mask.to(torch.bool),
                    attn_weights,
                    torch.finfo(attn_weights.dtype).min
                )

            # Sliding window mask
            if self.is_sliding_window and self.sliding_window_size is not None:
                seq_len = attn_weights.size(-1)
                sliding_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=attn_weights.device)
                sliding_mask = torch.triu(sliding_mask, diagonal=-self.sliding_window_size + 1)
                sliding_mask = torch.tril(sliding_mask)
                attn_weights = attn_weights.masked_fill(~sliding_mask, float('-inf'))

            # Apply attention sink bias if enabled
            # The sink acts as a virtual token that absorbs excess attention probability
            if self.use_attention_sink and self.attention_sink_bias is not None:
                # attention_sink_bias has shape [tp_degree, local_num_heads]
                # Use rank_util.rank to dynamically select the correct shard at runtime
                rank_idx = self.rank_util.rank[0]  # scalar tensor with current rank
                local_sink_bias = self.attention_sink_bias[rank_idx]  # [local_num_heads]
                # Reshape: [local_num_heads] -> [1, local_num_heads, 1, 1]
                # Then expand to [bsz, local_num_heads, q_len, 1]
                sink_bias = local_sink_bias.view(1, -1, 1, 1).expand(bsz, -1, q_len, 1)
                # Concatenate sink as extra column
                attn_weights = torch.cat([attn_weights, sink_bias], dim=-1)
                # Subtract max for numerical stability (as in HF implementation)
                attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
                # Softmax
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
                # Drop the sink column
                attn_weights = attn_weights[..., :-1].to(value_states.dtype)
            else:
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.local_num_heads * self.attn_v_head_dim)
        attn_output = self.o_proj(attn_output)

        new_key_value = (key_states_for_cache, value_states_for_cache)
        return attn_output, new_key_value, cos_cache, sin_cache


class MiMoV2MLP(nn.Module):
    """Standard MLP for non-MoE layers in MiMo-V2-Flash."""

    def __init__(self, config: MiMoV2InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Use the dense intermediate size for non-MoE layers
        self.intermediate_size = getattr(config, 'dense_intermediate_size', config.intermediate_size * 8)

        dtype = config.neuron_config.torch_dtype

        if parallel_state.model_parallel_is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()

            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                gather_output=False,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                input_is_parallel=True,
                dtype=dtype,
                tensor_model_parallel_group=tp_group,
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.act_fn = F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class NeuronMiMoV2DecoderLayer(nn.Module):
    """MiMo-V2-Flash Decoder Layer with hybrid attention and conditional MoE."""

    def __init__(self, config: MiMoV2InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Determine attention type for this layer
        is_sliding_window = config.layer_attention_types[layer_idx] == "sliding_window"
        self.attention_type = "sliding_window" if is_sliding_window else "full"

        # Create attention module
        self.self_attn = NeuronMiMoV2Attention(
            config=config,
            layer_idx=layer_idx,
            is_sliding_window=is_sliding_window,
        )

        # Determine if this layer uses MoE
        self.uses_moe = config.layer_uses_moe[layer_idx]

        # Create MLP/MoE module
        if self.uses_moe:
            self.mlp = initialize_moe_module(config=config)
        else:
            self.mlp = MiMoV2MLP(config)

        # Layer norms
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )

        # Config flags
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        """Forward pass for decoder layer.

        Each layer uses the same attention_mask from the base class. Sliding window
        attention is handled internally by the NeuronMiMoV2Attention layer, which
        applies the sliding window mask on top of the causal mask.
        """

        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MLP/MoE with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.uses_moe:
            hidden_states = self.mlp(hidden_states, padding_mask)[0]
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs


class NeuronMiMoV2Model(NeuronBaseModel):
    """MiMo-V2-Flash Model for NXD inference."""

    def setup_attr_for_model(self, config: MiMoV2InferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads

        # Check if we need GQA CONVERT_TO_MHA mode
        # When tp_degree > num_kv_heads, we replicate K/V to match num_attention_heads
        min_kv_heads = min(
            config.num_key_value_heads,
            getattr(config, 'swa_num_key_value_heads', config.num_key_value_heads)
        )
        self.use_gqa_convert_to_mha = self.tp_degree > min_kv_heads

        if self.use_gqa_convert_to_mha:
            # With CONVERT_TO_MHA, KV cache stores num_attention_heads (same as Q)
            self.num_key_value_heads = config.num_attention_heads
        else:
            # Standard GQA: use the maximum num_kv_heads for KV cache
            # (handles hybrid full/sliding window attention)
            self.num_key_value_heads = max(
                config.num_key_value_heads,
                getattr(config, 'swa_num_key_value_heads', config.num_key_value_heads)
            )

        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

        # MiMo-V2 has hybrid attention: some layers use full attention, others use sliding window
        # Each attention layer applies its own sliding window mask internally, so we don't need
        # has_mixed_attn or sliding_window at the model level. The attention layers handle
        # the masking themselves.
        self.has_mixed_attn = False
        self.sliding_window = None
        self.attention_chunk_size = None

    def init_model(self, config: MiMoV2InferenceConfig):
        self.padding_idx = getattr(config, 'pad_token_id', None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )

        self.layers = nn.ModuleList([
            NeuronMiMoV2DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.layernorm_epsilon,
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
        )


def _replicate_kv_projections_for_convert_to_mha(
    neuron_state_dict: Dict[str, Any],
    config: MiMoV2InferenceConfig,
    layer_idx: int,
) -> None:
    """Replicate K/V projection weights for CONVERT_TO_MHA mode.

    When TP > num_kv_heads, we use CONVERT_TO_MHA strategy where K/V projections
    are expanded to match num_attention_heads. This function replicates the
    original K/V weights to create the expanded projections.

    Args:
        neuron_state_dict: State dict to modify in-place
        config: Model configuration
        layer_idx: Layer index to process
    """
    # Determine if this is a sliding window layer
    is_sliding_window = config.layer_attention_types[layer_idx] == "sliding_window"

    if is_sliding_window:
        num_kv_heads = config.swa_num_key_value_heads
        head_dim = config.swa_head_dim
        v_head_dim = config.swa_v_head_dim
    else:
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        v_head_dim = config.v_head_dim

    num_attention_heads = config.num_attention_heads
    repeat_factor = num_attention_heads // num_kv_heads

    # K projection: [num_kv_heads * head_dim, hidden_size] -> [num_attention_heads * head_dim, hidden_size]
    k_proj_key = f"layers.{layer_idx}.self_attn.k_proj.weight"
    if k_proj_key in neuron_state_dict:
        k_proj = neuron_state_dict[k_proj_key]
        # Reshape to [num_kv_heads, head_dim, hidden_size]
        k_proj = k_proj.view(num_kv_heads, head_dim, -1)
        # Use repeat_interleave to repeat each head consecutively
        # This matches the GQA CONVERT_TO_MHA pattern where each KV head serves multiple Q heads
        k_proj = k_proj.repeat_interleave(repeat_factor, dim=0)
        # Reshape back to [num_attention_heads * head_dim, hidden_size]
        neuron_state_dict[k_proj_key] = k_proj.view(-1, k_proj.shape[-1])

    # V projection: [num_kv_heads * v_head_dim, hidden_size] -> [num_attention_heads * v_head_dim, hidden_size]
    v_proj_key = f"layers.{layer_idx}.self_attn.v_proj.weight"
    if v_proj_key in neuron_state_dict:
        v_proj = neuron_state_dict[v_proj_key]
        # Reshape to [num_kv_heads, v_head_dim, hidden_size]
        v_proj = v_proj.view(num_kv_heads, v_head_dim, -1)
        # Use repeat_interleave to repeat each head consecutively
        v_proj = v_proj.repeat_interleave(repeat_factor, dim=0)
        # Reshape back to [num_attention_heads * v_head_dim, hidden_size]
        neuron_state_dict[v_proj_key] = v_proj.view(-1, v_proj.shape[-1])


def convert_mimo_v2_hf_to_neuron_state_dict(
    neuron_state_dict: Dict[str, Any],
    config: MiMoV2InferenceConfig,
) -> Dict[str, Any]:
    """Convert HuggingFace MiMo-V2-Flash weights to Neuron format.

    This handles:
    1. Router weight renaming
    2. Expert weight concatenation and transposition
    3. FP8 dequantization or native FP8 scale conversion
    4. K/V projection replication for CONVERT_TO_MHA mode
    """

    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    # Check if using native FP8
    use_native_fp8 = (
        config.neuron_config.quantized or
        config.neuron_config.quantized_mlp_kernel_enabled
    )

    # Handle FP8 weights (dequantize or convert scale format)
    _maybe_dequantize_layer(neuron_state_dict, config)

    # Check if we need CONVERT_TO_MHA mode
    tp_degree = config.neuron_config.tp_degree
    min_num_kv_heads = min(
        config.num_key_value_heads,
        getattr(config, 'swa_num_key_value_heads', config.num_key_value_heads)
    )
    use_convert_to_mha = tp_degree > min_num_kv_heads

    # Add rank utility tensors
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    for layer_idx in range(config.num_hidden_layers):
        # Replicate K/V projections if CONVERT_TO_MHA mode
        if use_convert_to_mha:
            _replicate_kv_projections_for_convert_to_mha(
                neuron_state_dict, config, layer_idx
            )
            # Also replicate scales if using native FP8
            if use_native_fp8:
                _replicate_kv_scales_for_convert_to_mha(
                    neuron_state_dict, config, layer_idx
                )

        # Shard attention_sink_bias for TP
        # The HF checkpoint has shape [num_attention_heads], we reshape to [tp_degree, local_num_heads]
        # so that NXD's checkpoint sharding picks the correct shard for each TP rank
        sink_bias_key = f"layers.{layer_idx}.self_attn.attention_sink_bias"
        if sink_bias_key in neuron_state_dict:
            sink_bias = neuron_state_dict[sink_bias_key]
            num_heads = sink_bias.shape[0]
            local_num_heads = num_heads // tp_degree
            # Reshape to [tp_degree, local_num_heads] for automatic TP sharding
            # NXD will pick dimension 0 for each rank
            neuron_state_dict[sink_bias_key] = sink_bias.view(tp_degree, local_num_heads)

        # Add rank utility for attention
        neuron_state_dict[f"layers.{layer_idx}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        # Only convert MoE layers
        if not config.layer_uses_moe[layer_idx]:
            continue

        # Check if this layer has MoE weights
        gate_key = f"layers.{layer_idx}.mlp.gate.weight"
        if gate_key not in neuron_state_dict:
            continue

        # Rename router weights
        neuron_state_dict[f"layers.{layer_idx}.mlp.router.linear_router.weight"] = (
            neuron_state_dict[gate_key].detach().clone()
        )
        del neuron_state_dict[gate_key]

        # Get dimensions from first expert
        expert_0_gate = f"layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
        if expert_0_gate not in neuron_state_dict:
            continue

        intermediate_size, hidden_size = neuron_state_dict[expert_0_gate].shape
        device = neuron_state_dict[expert_0_gate].device
        dtype = neuron_state_dict[expert_0_gate].dtype

        num_experts = config.n_routed_experts

        # Concatenate gate and up projections
        gate_up_proj = torch.empty(
            num_experts,
            hidden_size,
            2 * intermediate_size,
            dtype=dtype,
            device=device,
        )

        for e in range(num_experts):
            gate_proj_weights = neuron_state_dict[
                f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"
            ].T.detach().clone()
            up_proj_weights = neuron_state_dict[
                f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"
            ].T.detach().clone()

            gate_up_proj[e, :, :intermediate_size] = gate_proj_weights
            gate_up_proj[e, :, intermediate_size:] = up_proj_weights

            del neuron_state_dict[f"layers.{layer_idx}.mlp.experts.{e}.gate_proj.weight"]
            del neuron_state_dict[f"layers.{layer_idx}.mlp.experts.{e}.up_proj.weight"]

        # Pad if needed
        pad_size = getattr(config, "moe_intermediate_pad_size", 0)
        if pad_size > 0:
            gate_up_proj = gate_up_proj.reshape(num_experts, hidden_size, 2, -1)
            gate_up_proj = F.pad(gate_up_proj, (0, pad_size))
            gate_up_proj = gate_up_proj.reshape(num_experts, hidden_size, -1)

        neuron_state_dict[f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        # Convert down projections
        down_proj = torch.empty(
            num_experts,
            intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )

        for e in range(num_experts):
            down_proj_weights = neuron_state_dict[
                f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"
            ].T.detach().clone()
            down_proj[e] = down_proj_weights
            del neuron_state_dict[f"layers.{layer_idx}.mlp.experts.{e}.down_proj.weight"]

        # Pad if needed
        if pad_size > 0:
            down_proj = F.pad(down_proj, (0, 0, 0, pad_size))

        neuron_state_dict[f"layers.{layer_idx}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    return neuron_state_dict


def _maybe_dequantize_layer(
    neuron_state_dict: Dict[str, Any],
    config: MiMoV2InferenceConfig,
):
    """Handle FP8 weights - either dequantize to BF16 or convert for native FP8.

    If config.neuron_config.quantized or config.neuron_config.quantized_mlp_kernel_enabled
    is True, this function converts weight_scale_inv to .scale format for native FP8.
    Otherwise, it dequantizes FP8 weights to BF16.
    """
    use_native_fp8 = (
        config.neuron_config.quantized or
        config.neuron_config.quantized_mlp_kernel_enabled
    )

    scale_layers = []

    for layer_key in list(neuron_state_dict.keys()):
        if "_scale_inv" in layer_key:
            scales = neuron_state_dict[layer_key]
            scale_layers.append(layer_key)

            fp8_layer_name = layer_key.replace("_scale_inv", "")
            if fp8_layer_name not in neuron_state_dict:
                continue

            fp8_layer = neuron_state_dict[fp8_layer_name]

            if use_native_fp8:
                # Convert scale format for native FP8 inference
                # HuggingFace: weight_scale_inv = 1/scale (used as: original = weight * weight_scale_inv)
                # Neuron: scale (used as: original = weight * scale)
                # So: neuron_scale = weight_scale_inv (same semantics, just rename)
                # Note: The FP8 rescaling (OCP to Neuron format) should be done in preprocessing
                new_scale_key = fp8_layer_name.replace(".weight", ".scale")
                neuron_state_dict[new_scale_key] = scales.to(torch.float32)
            else:
                # Dequantize FP8 to BF16
                # Get block size from config if available
                if hasattr(config, 'quantization_config') and config.quantization_config:
                    block_size = config.quantization_config.get("weight_block_size", [128, 128])
                else:
                    block_size = [128, 128]

                # Expand scales and dequantize
                scales_expanded = scales.repeat_interleave(block_size[0], dim=0)
                scales_expanded = scales_expanded.repeat_interleave(block_size[1], dim=1)

                # Ensure shapes match
                if scales_expanded.shape != fp8_layer.shape:
                    scales_expanded = scales_expanded[:fp8_layer.shape[0], :fp8_layer.shape[1]]

                scaled_layer = fp8_layer.to(torch.float32) * scales_expanded.to(torch.float32)
                neuron_state_dict[fp8_layer_name] = scaled_layer.to(config.neuron_config.torch_dtype)

    # Remove original scale layers (renamed or no longer needed after dequantization)
    for scale_layer in scale_layers:
        del neuron_state_dict[scale_layer]


def _replicate_kv_scales_for_convert_to_mha(
    neuron_state_dict: Dict[str, Any],
    config: MiMoV2InferenceConfig,
    layer_idx: int,
    block_size: List[int] = [128, 128],
) -> None:
    """Replicate K/V projection scales for CONVERT_TO_MHA mode (native FP8 only).

    When using native FP8 with CONVERT_TO_MHA, scales must also be replicated
    alongside weights.

    Args:
        neuron_state_dict: State dict to modify in-place
        config: Model configuration
        layer_idx: Layer index to process
        block_size: Block size for quantization
    """
    is_sliding_window = config.layer_attention_types[layer_idx] == "sliding_window"

    if is_sliding_window:
        num_kv_heads = config.swa_num_key_value_heads
        head_dim = config.swa_head_dim
        v_head_dim = config.swa_v_head_dim
    else:
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        v_head_dim = config.v_head_dim

    num_attention_heads = config.num_attention_heads
    repeat_factor = num_attention_heads // num_kv_heads

    for proj, proj_head_dim in [("k_proj", head_dim), ("v_proj", v_head_dim)]:
        scale_key = f"layers.{layer_idx}.self_attn.{proj}.scale"

        if scale_key not in neuron_state_dict:
            continue

        scale = neuron_state_dict[scale_key]

        # Scale shape: [scale_h, scale_w] where scale_h = ceil(weight_h / block_size[0])
        # weight_h = num_kv_heads * head_dim
        # After replication: weight_h = num_attention_heads * head_dim
        # So scale_h should increase proportionally

        # Number of scale rows per KV head
        scales_per_head = scale.shape[0] // num_kv_heads

        # Reshape to [num_kv_heads, scales_per_head, scale_w]
        scale_reshaped = scale.view(num_kv_heads, scales_per_head, -1)

        # Replicate using repeat_interleave
        scale_replicated = scale_reshaped.repeat_interleave(repeat_factor, dim=0)

        # Reshape back to [new_scale_h, scale_w]
        neuron_state_dict[scale_key] = scale_replicated.view(-1, scale.shape[-1])


class NeuronMiMoV2ForCausalLM(NeuronBaseForCausalLM):
    """MiMo-V2-Flash for Causal Language Modeling on Neuron."""

    _model_cls = NeuronMiMoV2Model

    @staticmethod
    def load_hf_model(model_path: str, **kwargs):
        """Load HuggingFace model.

        Note: MiMo-V2-Flash uses custom code, so we need trust_remote_code=True
        """
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs,
        )

    @classmethod
    def get_config_cls(cls) -> Type[MiMoV2InferenceConfig]:
        return MiMoV2InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: Dict[str, Any],
        config: MiMoV2InferenceConfig,
    ) -> Dict[str, Any]:
        return convert_mimo_v2_hf_to_neuron_state_dict(state_dict, config)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict: Dict[str, Any]):
        """Copy embed_tokens weights to lm_head when tie_word_embeddings=True.

        MiMo-V2-Flash uses tie_word_embeddings=True, meaning the embedding
        and lm_head share weights. This method ensures the lm_head gets
        the correct weights from embed_tokens.
        """
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    def enable_context_encoding(self):
        self.compile_tag = CONTEXT_ENCODING_MODEL_TAG
        super().enable_context_encoding()

    def enable_token_generation(self):
        self.compile_tag = TOKEN_GENERATION_MODEL_TAG
        super().enable_token_generation()

    def get_compiler_args(self) -> str:
        """Get compiler arguments optimized for MiMo-V2-Flash."""
        if self.compile_tag == CONTEXT_ENCODING_MODEL_TAG:
            optimization_level = "-O1"
        elif self.compile_tag == TOKEN_GENERATION_MODEL_TAG:
            optimization_level = "-O3" if self.neuron_config.moe_ep_degree > 1 else "-O1"
        else:
            optimization_level = "-O1"

        compiler_args = (
            f"--enable-saturate-infinity "
            f"--enable-mixed-precision-accumulation "
            f"--model-type transformer "
            f"{optimization_level}"
        )

        # Add CC overlap optimization
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap "
            "--cc-pipeline-tiling-factor=2'"
        )

        compiler_args += " --auto-cast=none"

        # Enable vector-offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"

        if self.neuron_config.scratchpad_page_size:
            compiler_args += f" --hbm-scratchpad-page-size={self.neuron_config.scratchpad_page_size}"

        return compiler_args
