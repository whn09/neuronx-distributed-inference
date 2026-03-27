# Copyright 2025 ByteDance and Amazon Web Services. All rights reserved.
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
PyTorch Seed-OSS model for NXD inference.

Based on the Qwen3 NXD implementation. Key differences from Qwen3:
- No QK normalization (no q_norm / k_norm)
- attention_bias=True on Q/K/V projections
- attention_out_bias=False on output projection
"""

from typing import List, Optional, Tuple, Type

import gc
import torch
from torch import nn
from transformers import SeedOssForCausalLM
from transformers.models.seed_oss.modeling_seed_oss import SeedOssRMSNorm

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.utils import cpu_mode

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaMLP
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm


def get_rmsnorm_cls():
    return SeedOssRMSNorm if cpu_mode() else CustomRMSNorm


class SeedOssNeuronConfig(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attn_cls = NeuronSeedOssAttention


class SeedOssInferenceConfig(InferenceConfig):

    def add_derived_config(self):
        self.num_cores_per_group = 1

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "head_dim",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[SeedOssNeuronConfig]:
        return SeedOssNeuronConfig


class NeuronSeedOssAttention(NeuronAttentionBase):

    def __init__(self, config: SeedOssInferenceConfig):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        rotary_emb = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Seed-OSS has attention_bias=True on QKV, attention_out_bias=False on O
        # No QK normalization (unlike Qwen3)
        attention_bias = getattr(config, "attention_bias", True)
        attention_out_bias = getattr(config, "attention_out_bias", False)

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            qkv_bias=attention_bias,
            o_bias=attention_out_bias,
        )


class NeuronSeedOssDecoderLayer(nn.Module):

    def __init__(self, config: SeedOssInferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronSeedOssAttention(config)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronSeedOssModel(NeuronBaseModel):

    def setup_attr_for_model(self, config: SeedOssInferenceConfig):
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: SeedOssInferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
            sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        )
        self.layers = nn.ModuleList(
            [NeuronSeedOssDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            pad=True,
            gather_output=not self.on_device_sampling,
            dtype=config.neuron_config.torch_dtype,
        )



def _helper_concat_and_delete_qkv(state_dict, layer_num, attr):
    """Helper to concatenate Q/K/V into fused Wqkv and delete originals."""
    q_key = f"layers.{layer_num}.self_attn.q_proj.{attr}"
    k_key = f"layers.{layer_num}.self_attn.k_proj.{attr}"
    v_key = f"layers.{layer_num}.self_attn.v_proj.{attr}"
    if q_key in state_dict and k_key in state_dict and v_key in state_dict:
        state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
            [state_dict[q_key], state_dict[k_key], state_dict[v_key]],
        )
        del state_dict[q_key]
        del state_dict[k_key]
        del state_dict[v_key]


def convert_state_dict_to_fused_qkv(state_dict, cfg: InferenceConfig):
    """Concatenate separate Q/K/V weights and biases into fused Wqkv for all layers."""
    mods_to_not_conv = getattr(cfg.neuron_config, "modules_to_not_convert", None)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):
        _helper_concat_and_delete_qkv(state_dict, l, "weight")
        _helper_concat_and_delete_qkv(state_dict, l, "bias")  # Seed-OSS has QKV bias
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
        ) and f"layers.{l}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(state_dict, l, "scale")

    gc.collect()
    return state_dict


class NeuronSeedOssForCausalLM(NeuronBaseForCausalLM):
    """
    NXD inference implementation for Seed-OSS (ByteDance).
    """

    _model_cls = NeuronSeedOssModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return SeedOssForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        neuron_config = config.neuron_config

        if neuron_config.vocab_parallel:
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            # To facilitate rank usage in attention
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )

        # To facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        return state_dict

    @classmethod
    def get_config_cls(cls):
        return SeedOssInferenceConfig
