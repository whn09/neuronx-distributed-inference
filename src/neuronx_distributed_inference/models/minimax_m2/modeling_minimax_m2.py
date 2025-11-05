# coding=utf-8
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
""" MiniMax M2 MoE model for NXD inference."""
import gc
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any

import torch

from neuronx_distributed_inference.models.model_base import NeuronBaseForCausalLM, NeuronBaseModel
from neuronx_distributed_inference.modules.attention.gqa import GQA
from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.utils import cpu_mode
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
# from neuronx_distributed_inference.modules.moe import initialize_moe_module
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


# Get the modules_to_not_convert from the neuron configs
def get_modules_to_not_convert(neuron_config: MoENeuronConfig):
    return getattr(neuron_config, "modules_to_not_convert", None)


def _helper_concat_and_delete_qkv(minimax_state_dict: Dict[str, Any], layer_num: int, attr: str):
    """
    Helper function to concatenate and delete QKV attributes for fusedqkv (weight or scale).
    Args:
        minimax_state_dict: The state dictionary containing model weights
        layer_num: The index of the layer to process
        attr: The attribute to process ('weight' or 'scale')
    """
    minimax_state_dict[f"layers.{layer_num}.self_attn.Wqkv.{attr}"] = torch.cat(
        [
            minimax_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"],
            minimax_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"],
            minimax_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"],
        ],
    )
    del minimax_state_dict[f"layers.{layer_num}.self_attn.q_proj.{attr}"]
    del minimax_state_dict[f"layers.{layer_num}.self_attn.k_proj.{attr}"]
    del minimax_state_dict[f"layers.{layer_num}.self_attn.v_proj.{attr}"]


def convert_state_dict_to_fused_qkv(minimax_state_dict: Dict[str, Any], cfg: InferenceConfig):
    """
    This function concats the qkv weights and scales to a Wqkv weight and scale for fusedqkv, and deletes the qkv weights.
    """
    mods_to_not_conv = get_modules_to_not_convert(cfg.neuron_config)
    if mods_to_not_conv is None:
        mods_to_not_conv = []

    for l in range(cfg.num_hidden_layers):  # noqa: E741
        _helper_concat_and_delete_qkv(minimax_state_dict, l, "weight")
        if (
            cfg.neuron_config.quantized_mlp_kernel_enabled or cfg.neuron_config.quantized
        ) and f"layers.{l}.self_attn" not in mods_to_not_conv:
            _helper_concat_and_delete_qkv(minimax_state_dict, l, "scale")

    gc.collect()

    return minimax_state_dict


def convert_minimax_m2_hf_to_neuron_state_dict(neuron_state_dict, config):
    """
    Helper function which converts the huggingface checkpoints to state dictionary compatible with the structure of the neuron MoE model.
    """
    assert config.neuron_config.glu_mlp is True, "Only GLU MLP is supported"

    # to facilitate rank usage in base model
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    for l in range(config.num_hidden_layers):  # noqa: E741
        # Handle QK norm if present
        if hasattr(config, 'use_qk_norm') and config.use_qk_norm:
            # MiniMax M2 uses per-head QK norm: [num_heads, head_dim]
            # Neuron expects shared QK norm: [head_dim]
            # We average across all heads to convert per-head to shared

            # k_norm: [num_kv_heads * head_dim] -> [head_dim]
            k_norm_full = neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"]
            k_norm_reshaped = k_norm_full.reshape(config.num_key_value_heads, config.head_dim)
            neuron_state_dict[f"layers.{l}.self_attn.k_layernorm.weight"] = (
                k_norm_reshaped.mean(dim=0).detach().clone()
            )
            del neuron_state_dict[f"layers.{l}.self_attn.k_norm.weight"]

            # q_norm: [num_attention_heads * head_dim] -> [head_dim]
            q_norm_full = neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"]
            q_norm_reshaped = q_norm_full.reshape(config.num_attention_heads, config.head_dim)
            neuron_state_dict[f"layers.{l}.self_attn.q_layernorm.weight"] = (
                q_norm_reshaped.mean(dim=0).detach().clone()
            )
            del neuron_state_dict[f"layers.{l}.self_attn.q_norm.weight"]

        # Copy router weights from block_sparse_moe
        neuron_state_dict[f"layers.{l}.mlp.router.linear_router.weight"] = (
            neuron_state_dict[f"layers.{l}.block_sparse_moe.gate.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.block_sparse_moe.gate.weight"]

        # Handle e_score_correction_bias if present
        if f"layers.{l}.block_sparse_moe.e_score_correction_bias" in neuron_state_dict:
            del neuron_state_dict[f"layers.{l}.block_sparse_moe.e_score_correction_bias"]

        intermediate_size, hidden_size = neuron_state_dict[
            f"layers.{l}.block_sparse_moe.experts.0.w1.weight"
        ].shape
        device = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w1.weight"].device
        dtype = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w1.weight"].dtype

        # copy the MLP parameters (w1 is gate_proj, w3 is up_proj, w2 is down_proj in MiniMax M2)
        gate_up_proj = torch.empty(
            config.num_local_experts,
            hidden_size,
            2 * intermediate_size,
            dtype=dtype,
            device=device,
        )
        for e in range(config.num_local_experts):
            # Copy w1 (gate_proj) and w3 (up_proj) after concatenation
            gate_proj_weights = (
                neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w1.weight"]
                .T.detach()
                .clone()
            )
            up_proj_weights = (
                neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w3.weight"]
                .T.detach()
                .clone()
            )

            gate_up_proj_slice = torch.narrow(gate_up_proj, 0, e, 1)
            gate_proj_slice = torch.narrow(gate_up_proj_slice, 2, 0, intermediate_size)
            gate_proj_slice.copy_(gate_proj_weights)
            up_proj_slice = torch.narrow(
                gate_up_proj_slice, 2, intermediate_size, intermediate_size
            )
            up_proj_slice.copy_(up_proj_weights)

            del neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w1.weight"]
            del neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w3.weight"]
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        down_proj = torch.empty(
            config.num_local_experts,
            intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        for e in range(config.num_local_experts):
            # Copy w2 (down_proj)
            down_proj_weights = (
                neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w2.weight"]
                .T.detach()
                .clone()
            )
            down_proj_slice = torch.narrow(down_proj, 0, e, 1)
            down_proj_slice.copy_(down_proj_weights)
            del neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w2.weight"]
        neuron_state_dict[f"layers.{l}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        gc.collect()

    if config.neuron_config.fused_qkv:
        neuron_state_dict = convert_state_dict_to_fused_qkv(neuron_state_dict, config)

    return neuron_state_dict


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    if cpu_mode():
        from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_gpu import MiniMaxM2RMSNorm
        return MiniMaxM2RMSNorm
    else:
        return CustomRMSNorm


class MiniMaxM2InferenceConfig(InferenceConfig):
    def __init__(self, neuron_config, fused_spec_config=None, load_config=None, metadata=None, **kwargs):
        super().__init__(neuron_config, fused_spec_config, load_config, metadata, **kwargs)
        # MiniMax M2 doesn't have shared experts
        self.n_shared_experts = 0

    def get_required_attributes(self) -> List[str]:
        return [
            "head_dim",
            "hidden_act",
            "hidden_size",
            "max_position_embeddings",
            "intermediate_size",  # MiniMaxM2 uses intermediate_size instead of moe_intermediate_size
            "num_attention_heads",
            "num_local_experts",  # MiniMaxM2 uses num_local_experts
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "rms_norm_eps",
            "rope_theta",
            "tie_word_embeddings",
            "vocab_size",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return MoENeuronConfig


class NeuronMiniMaxM2Attention(NeuronAttentionBase):
    def __init__(self, config: MiniMaxM2InferenceConfig):
        # MiniMax M2 uses partial rotary embeddings (rotary_dim=64, head_dim=128)
        # Calculate rotary_dim from config
        rotary_dim = getattr(config, 'rotary_dim', config.head_dim)
        partial_rotary_factor = getattr(config, 'partial_rotary_factor', rotary_dim / config.head_dim if rotary_dim != config.head_dim else 1.0)

        # Store for later use
        self.rotary_dim = rotary_dim
        self.partial_rotary_factor = partial_rotary_factor

        # Create RotaryEmbedding with the actual rotary_dim (not head_dim)
        # This generates cos/sin for only the dimensions that need rotation
        rotary_emb = RotaryEmbedding(
            rotary_dim,  # Use rotary_dim (64) instead of head_dim (128)
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
            # qk_norm in the base class is different from MiniMaxM2RMSNorm
            use_qk_norm=False,
        )

        # Override q_layernorm and k_layernorm with RMSNorm if use_qk_norm is enabled
        use_qk_norm = getattr(config, 'use_qk_norm', False)
        if use_qk_norm:
            self.q_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)
            self.k_layernorm = get_rmsnorm_cls()(self.head_dim, self.rms_norm_eps)

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronMiniMaxM2Attention has to be initialized in a distributed env. Please use neuronx_distributed"
                " module to initialize a distributed env."
            )

    def apply_rotary_embedding(self, Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope):
        """Override to handle partial rotary embeddings (rotary_dim=64, head_dim=128)."""
        if not use_polar_compatible_rope and self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            # For partial rotary: split Q/K, apply RoPE to first rotary_dim dimensions only
            if self.rotary_dim < self.head_dim:
                # Split into [rotary_part, pass_through_part]
                Q_rot = Q[..., :self.rotary_dim]
                Q_pass = Q[..., self.rotary_dim:]
                K_rot = K[..., :self.rotary_dim]
                K_pass = K[..., self.rotary_dim:]

                # Apply RoPE only to rotary part
                from neuronx_distributed_inference.modules.attention.utils import apply_rotary_pos_emb
                Q_rot, K_rot = apply_rotary_pos_emb(Q_rot, K_rot, cos_cache, sin_cache)

                # Concatenate back
                Q = torch.cat([Q_rot, Q_pass], dim=-1)
                K = torch.cat([K_rot, K_pass], dim=-1)
            else:
                # Full rotary (when rotary_dim == head_dim)
                from neuronx_distributed_inference.modules.attention.utils import apply_rotary_pos_emb
                Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

        elif use_polar_compatible_rope:
            # Fallback to polar-compatible RoPE if needed
            from neuronx_distributed_inference.modules.attention.utils import precompute_freqs_cis, apply_rotary_polar_compatible
            rotary_freqs = precompute_freqs_cis(
                self.rotary_dim,  # Use rotary_dim here too
                self.neuron_config.max_context_length * 2,
                self.rope_theta,
                self.use_scaled_rope,
                device=Q.device
            )
            rotary_freqs = rotary_freqs[position_ids]

            # For partial rotary with polar-compatible
            if self.rotary_dim < self.head_dim:
                Q_rot = Q[:, :, :, :self.rotary_dim].transpose(1, 2)
                K_rot = K[:, :, :, :self.rotary_dim].transpose(1, 2)
                Q_pass = Q[..., self.rotary_dim:]
                K_pass = K[..., self.rotary_dim:]

                Q_rot, K_rot = apply_rotary_polar_compatible(Q_rot, K_rot, rotary_freqs)
                Q_rot, K_rot = Q_rot.transpose(1, 2), K_rot.transpose(1, 2)

                Q = torch.cat([Q_rot, Q_pass], dim=-1)
                K = torch.cat([K_rot, K_pass], dim=-1)
            else:
                Q, K = apply_rotary_polar_compatible(Q.transpose(1, 2), K.transpose(1, 2), rotary_freqs)
                Q, K = Q.transpose(1, 2), K.transpose(1, 2)

        return Q, K, cos_cache, sin_cache


class NeuronMiniMaxM2DecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: MiniMaxM2InferenceConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronMiniMaxM2Attention(config=config)

        self.mlp = initialize_moe_module(config=config)

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
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.FloatTensor`, *optional*):
                position ids of size `(batch_size, sequence_length)`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

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
        hidden_states = self.mlp(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronMiniMaxM2Model(NeuronBaseModel):
    """
    NeuronMiniMaxM2Model extends the MiniMaxM2Model to be traceable.
    The forward function of this class is traced.
    """

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
            [
                NeuronMiniMaxM2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


class NeuronMiniMaxM2ForCausalLM(NeuronBaseForCausalLM):
    """
    This class can be used as MiniMaxM2ForCausalLM
    """

    _model_cls = NeuronMiniMaxM2Model

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_gpu import MiniMaxM2ForCausalLM
        print('model_path:', model_path)
        print('kwargs:', kwargs)
        model_path = '/home/ubuntu/model_hf/MiniMax-M2/'  # TODO Set to a fixed path
        return MiniMaxM2ForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16, **kwargs)  # TODO: No GPU or XPU found. A GPU or XPU is needed for FP8 quantization.

    @classmethod
    def get_config_cls(cls):
        return MiniMaxM2InferenceConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: MiniMaxM2InferenceConfig) -> dict:
        return convert_minimax_m2_hf_to_neuron_state_dict(state_dict, config)

    def get_compiler_args(self):
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        # Add flags for cc-overlap
        compiler_args += (
            " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        )
        compiler_args += " --auto-cast=none"
        # Enable vector-offset DGE (try with lower tp_degree first, e.g., 8 or 16)
        # Disable this if you get "Invalid Shape for Scalar DGE" error with high tp_degree
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        compiler_args += " --internal-hlo2tensorizer-options='--verify-hlo=true'"
        return compiler_args
