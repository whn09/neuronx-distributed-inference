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
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_process_group

# Import MoE components for custom router
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig, MoEFusedTKGConfig

_flash_fwd_call = nki_jit()(attention_isa_kernel)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

GQA_SHARDING_STRATEGY = GQA.REPLICATE_TO_TP_DEGREE


class RouterTopKWithBias(RouterTopK):
    """
    RouterTopK with e_score_correction_bias support for MiniMax M2.

    MiniMax M2 uses sigmoid activation with a bias term added to scores for expert selection,
    but the final weights do not include the bias.
    """

    def __init__(self, num_experts: int, *args, **kwargs):
        super().__init__(num_experts=num_experts, *args, **kwargs)
        # Register e_score_correction_bias buffer (will be loaded from checkpoint)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states):
        # Get router_logits
        router_logits = self.get_router_logits(hidden_states)

        # Apply activation function to get expert affinities
        expert_affinities = self.apply_activation_fn(router_logits)

        # For expert selection, add bias to affinities (MiniMax M2 specific)
        # This affects which experts are selected, but not the final weights
        scores_for_choice = expert_affinities + self.e_score_correction_bias.unsqueeze(0)

        # Select top-k experts based on biased scores
        _, expert_index = torch.topk(scores_for_choice, self.top_k, dim=-1)

        # Cast to required dtype (affinities without bias are used as final weights)
        expert_affinities = expert_affinities.to(dtype=hidden_states.dtype)
        expert_index = expert_index.detach().to(dtype=torch.long)

        return router_logits, expert_affinities, expert_index


def initialize_minimax_m2_moe_module(config: InferenceConfig):
    """
    Initialize MoE module for MiniMax M2 with custom router that supports e_score_correction_bias.
    """
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
        act_fn=config.neuron_config.router_config.act_fn,  # Should be 'sigmoid' for MiniMax M2
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
        shared_experts=None,  # MiniMax M2 doesn't have shared experts
        rmsnorm=None,
        sequence_parallel_enabled=config.neuron_config.sequence_parallel_enabled,
        return_expert_index=config.neuron_config.return_expert_index,
        sequence_dimension=1,
        init_tkg_module=False,
        tkg_config=None,
    )

    moe.eval()
    return moe


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

    # # Debug: Check what keys are in the state_dict
    # print("\n=== DEBUG: Checking state_dict keys ===")
    # all_keys = list(neuron_state_dict.keys())
    # print(f"  Total keys in state_dict: {len(all_keys)}")

    # # Check for weight_scale_inv keys
    # scale_inv_keys = [k for k in all_keys if 'weight_scale_inv' in k]
    # print(f"  Keys with 'weight_scale_inv': {len(scale_inv_keys)}")
    # if scale_inv_keys:
    #     print(f"  First 5 weight_scale_inv keys:")
    #     for k in scale_inv_keys[:5]:
    #         print(f"    {k}")

    # # Check layer 0 self_attn keys
    # layer0_attn = [k for k in all_keys if k.startswith('layers.0.self_attn.')]
    # print(f"  Layer 0 self_attn keys: {len(layer0_attn)}")
    # for k in sorted(layer0_attn)[:10]:
    #     print(f"    {k}")

    # # Handle FP8 quantized weights
    # # MiniMax M2 uses FP8 (float8_e4m3fn) weights with weight_scale_inv parameters
    # print("\n=== Processing FP8 quantized weights ===")

    # # Check if we should use FP8 compute or dequantize to bfloat16
    # use_fp8_compute = config.neuron_config.quantized_mlp_kernel_enabled

    # if use_fp8_compute:
    #     # Keep FP8 weights and convert scale parameters
    #     print("  Mode: FP8 compute (quantized_mlp_kernel_enabled=True)")
    #     param_name_list = list(neuron_state_dict.keys())
    #     scale_count = 0
    #     for param_name in param_name_list:
    #         if param_name.endswith(".weight_scale_inv"):
    #             # Convert weight_scale_inv to scale
    #             # Note: Despite the name, weight_scale_inv is actually the scale itself!
    #             new_param_name = param_name.replace(".weight_scale_inv", ".scale")
    #             neuron_state_dict[new_param_name] = neuron_state_dict[param_name]
    #             del neuron_state_dict[param_name]
    #             scale_count += 1
    #             if scale_count <= 3:
    #                 print(f"    Converted: {param_name} -> {new_param_name}")
    #     print(f"  Total converted: {scale_count} FP8 scale parameters")
    # else:
    #     # Dequantize FP8 weights to bfloat16
    #     print("  Mode: Dequantize to bfloat16 (quantized_mlp_kernel_enabled=False)")
    #     param_name_list = list(neuron_state_dict.keys())
    #     dequant_count = 0

    #     for param_name in param_name_list:
    #         if param_name.endswith(".weight") and param_name.replace(".weight", ".weight_scale_inv") in neuron_state_dict:
    #             weight = neuron_state_dict[param_name]
    #             scale_inv_name = param_name.replace(".weight", ".weight_scale_inv")

    #             if weight.dtype == torch.float8_e4m3fn:
    #                 # Get scale parameter
    #                 # Note: Despite the name "weight_scale_inv", it's actually the scale itself!
    #                 scale = neuron_state_dict[scale_inv_name]

    #                 # Dequantize: FP8 -> float32 (with scale) -> bfloat16
    #                 weight_float = weight.float()

    #                 # Block-wise dequantization
    #                 # Scale shape: [scale_h, scale_w], Weight shape: [weight_h, weight_w]
    #                 # Each block is [weight_h/scale_h, weight_w/scale_w]
    #                 scale_h, scale_w = scale.shape
    #                 weight_h, weight_w = weight.shape
    #                 block_h = weight_h // scale_h
    #                 block_w = weight_w // scale_w

    #                 dequantized_weight = torch.empty_like(weight_float)
    #                 for i in range(scale_h):
    #                     for j in range(scale_w):
    #                         block_scale = scale[i, j]
    #                         h_start, h_end = i * block_h, (i + 1) * block_h
    #                         w_start, w_end = j * block_w, (j + 1) * block_w
    #                         dequantized_weight[h_start:h_end, w_start:w_end] = weight_float[h_start:h_end, w_start:w_end] * block_scale

    #                 # Convert to bfloat16 and update state dict
    #                 neuron_state_dict[param_name] = dequantized_weight.to(torch.bfloat16)

    #                 # Remove scale parameter
    #                 del neuron_state_dict[scale_inv_name]

    #                 dequant_count += 1
    #                 if dequant_count <= 3:
    #                     print(f"    Dequantized: {param_name} (FP8 -> bfloat16)")

    #     print(f"  Total dequantized: {dequant_count} FP8 weights")

    # to facilitate rank usage in base model
    neuron_state_dict["rank_util.rank"] = torch.arange(
        0, config.neuron_config.tp_degree, dtype=torch.int32
    )

    # Rename attention projection keys to match Neuron's GQA module expectations
    # Neuron GQA expects: layers.X.self_attn.qkv_proj.{q,k,v}_proj.{weight,scale}
    # HF model has: layers.X.self_attn.{q,k,v}_proj.{weight,scale}
    # NOTE: This must run AFTER FP8 scale conversion to capture both .weight and .scale
    print("\n=== Renaming attention projections to add qkv_proj prefix ===")
    param_name_list = list(neuron_state_dict.keys())
    renamed_count = 0
    weight_count = 0
    scale_count = 0
    for param_name in param_name_list:
        new_param_name = None
        # Add qkv_proj prefix to attention projections (handles both .weight and .scale)
        if '.self_attn.q_proj.' in param_name:
            new_param_name = param_name.replace('.self_attn.q_proj.', '.self_attn.qkv_proj.q_proj.')
        elif '.self_attn.k_proj.' in param_name:
            new_param_name = param_name.replace('.self_attn.k_proj.', '.self_attn.qkv_proj.k_proj.')
        elif '.self_attn.v_proj.' in param_name:
            new_param_name = param_name.replace('.self_attn.v_proj.', '.self_attn.qkv_proj.v_proj.')
        # NOTE: Do NOT rename o_proj here - GroupQueryAttention_O.preshard_hook will handle it

        if new_param_name:
            neuron_state_dict[new_param_name] = neuron_state_dict.pop(param_name)
            renamed_count += 1
            if param_name.endswith('.weight'):
                weight_count += 1
            elif param_name.endswith('.scale'):
                scale_count += 1
            if renamed_count <= 5:  # Print first 5
                print(f"  Renamed: {param_name} -> {new_param_name}")
    print(f"  Total renamed: {renamed_count} parameters ({weight_count} weights, {scale_count} scales)")

    # Debug: Check if layer 0 qkv_proj keys exist
    print("\n=== Checking layer 0 attention keys ===")
    layer0_attn_keys = [k for k in neuron_state_dict.keys() if k.startswith('layers.0.self_attn.')]
    for key in sorted(layer0_attn_keys):
        print(f"  {key}")

    # Calculate sharded num_heads for q_norm/k_norm
    # Use same logic as GQA sharding: with REPLICATE_TO_TP_DEGREE, heads are padded to tp_degree
    tp_degree = config.neuron_config.tp_degree
    # For REPLICATE_TO_TP_DEGREE: num_attention_heads is padded to tp_degree, then divided by tp_degree
    # So each rank has tp_degree/tp_degree = 1 head for Q
    # And KV heads are replicated to tp_degree, so each rank also has 1 KV head
    from neuronx_distributed_inference.modules.attention.gqa import get_shardable_head_counts, GQA
    sharding_strategy = GQA.REPLICATE_TO_TP_DEGREE  # Same as GQA_SHARDING_STRATEGY
    padded_num_attention_heads, padded_num_kv_heads = get_shardable_head_counts(
        tp_degree, config.num_attention_heads, config.num_key_value_heads, sharding_strategy
    )
    num_heads_per_rank = padded_num_attention_heads // tp_degree
    num_kv_heads_per_rank = padded_num_kv_heads // tp_degree
    print(f"\n=== QK norm sharding info ===")
    print(f"  tp_degree: {tp_degree}")
    print(f"  padded_num_attention_heads: {padded_num_attention_heads}")
    print(f"  padded_num_kv_heads: {padded_num_kv_heads}")
    print(f"  num_heads_per_rank: {num_heads_per_rank}")
    print(f"  num_kv_heads_per_rank: {num_kv_heads_per_rank}")

    # Import padding function for qk_norm
    from neuronx_distributed_inference.modules.attention.gqa import _maybe_pad_interleaved

    for l in range(config.num_hidden_layers):  # noqa: E741
        # Handle QK norm weights: apply interleaved padding to match Q/K projection sharding
        # MiniMax M2 qk_norm applies RMSNorm on the full Q/K output BEFORE reshape.
        # The weights are organized as [num_heads * head_dim], e.g., [6144] for Q.
        #
        # With GQA REPLICATE_TO_TP_DEGREE sharding:
        # - Q heads are padded from 48 to 64 using interleaved padding
        # - The padding is done in groups: [6 real heads, 2 padding heads] x 8 groups
        # - KV heads are replicated from 8 to 64
        #
        # The qk_norm weights need the same interleaved padding to match the sharded Q/K data.
        if hasattr(config, 'use_qk_norm') and config.use_qk_norm:
            # q_norm: apply interleaved padding [48 heads -> 64 heads]
            q_norm_key = f"layers.{l}.self_attn.q_norm.weight"
            if q_norm_key in neuron_state_dict:
                q_norm_full = neuron_state_dict[q_norm_key]  # [num_attention_heads * head_dim] = [6144]
                # Apply the same interleaved padding as Q projection weights
                # source_group_size = num_attention_heads / num_kv_heads = 48 / 8 = 6
                source_group_size = config.num_attention_heads // config.num_key_value_heads
                q_norm_padded = _maybe_pad_interleaved(
                    q_norm_full.unsqueeze(0),  # Add batch dim for the function: [1, 6144]
                    pad_dim=1,  # Pad along the second dimension
                    source_heads=config.num_attention_heads,  # 48
                    target_heads=padded_num_attention_heads,  # 64
                    source_group_size=source_group_size,  # 6
                ).squeeze(0)  # Remove batch dim: [8192]
                neuron_state_dict[q_norm_key] = q_norm_padded
                if l == 0:
                    print(f"  q_norm: {q_norm_full.shape} -> {q_norm_padded.shape} (interleaved padding)")

            # k_norm: replicate from 8 to 64 heads
            k_norm_key = f"layers.{l}.self_attn.k_norm.weight"
            if k_norm_key in neuron_state_dict:
                k_norm_full = neuron_state_dict[k_norm_key]  # [num_kv_heads * head_dim] = [1024]
                # KV heads are replicated: each of the 8 original heads is replicated 8 times (64/8=8)
                # Reshape to [num_kv_heads, head_dim] then repeat
                k_norm_reshaped = k_norm_full.reshape(config.num_key_value_heads, config.head_dim)  # [8, 128]
                repeats = padded_num_kv_heads // config.num_key_value_heads  # 64 / 8 = 8
                k_norm_replicated = k_norm_reshaped.repeat_interleave(repeats, dim=0)  # [64, 128]
                k_norm_padded = k_norm_replicated.reshape(-1)  # [8192]
                neuron_state_dict[k_norm_key] = k_norm_padded
                if l == 0:
                    print(f"  k_norm: {k_norm_full.shape} -> {k_norm_padded.shape} (replicated {repeats}x)")

        # Copy router weights from block_sparse_moe
        neuron_state_dict[f"layers.{l}.block_sparse_moe.router.linear_router.weight"] = (
            neuron_state_dict[f"layers.{l}.block_sparse_moe.gate.weight"].detach().clone()
        )
        del neuron_state_dict[f"layers.{l}.block_sparse_moe.gate.weight"]

        # Handle e_score_correction_bias: rename to router path for RouterTopKWithBias
        if f"layers.{l}.block_sparse_moe.e_score_correction_bias" in neuron_state_dict:
            neuron_state_dict[f"layers.{l}.block_sparse_moe.router.e_score_correction_bias"] = (
                neuron_state_dict[f"layers.{l}.block_sparse_moe.e_score_correction_bias"].detach().clone()
            )
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
        neuron_state_dict[f"layers.{l}.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj

        # # Merge scale parameters for FP8 quantization (if present)
        # # Check if scale parameters exist (after weight_scale_inv -> scale conversion)
        # if l == 0:  # Debug: print for first layer
        #     print(f"\n=== DEBUG: Checking scale parameters for layer {l} ===")
        #     has_w1_scale = f"layers.{l}.block_sparse_moe.experts.0.w1.scale" in neuron_state_dict
        #     print(f"  Has w1.scale: {has_w1_scale}")
        #     if has_w1_scale:
        #         scale_shape = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w1.scale"].shape
        #         print(f"  w1.scale shape: {scale_shape}")

        # if f"layers.{l}.block_sparse_moe.experts.0.w1.scale" in neuron_state_dict:
        #     # Get scale shape from first expert
        #     scale_h, scale_w = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w1.scale"].shape
        #     scale_device = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w1.scale"].device
        #     scale_dtype = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w1.scale"].dtype

        #     # Merge gate_up_proj scales (concatenate w1 and w3 scales)
        #     gate_up_proj_scale = torch.empty(
        #         config.num_local_experts,
        #         scale_h,
        #         2 * scale_w,
        #         dtype=scale_dtype,
        #         device=scale_device,
        #     )
        #     for e in range(config.num_local_experts):
        #         w1_scale = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w1.scale"]
        #         w3_scale = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w3.scale"]

        #         gate_up_proj_scale_slice = torch.narrow(gate_up_proj_scale, 0, e, 1)
        #         gate_scale_slice = torch.narrow(gate_up_proj_scale_slice, 2, 0, scale_w)
        #         gate_scale_slice.copy_(w1_scale)
        #         up_scale_slice = torch.narrow(gate_up_proj_scale_slice, 2, scale_w, scale_w)
        #         up_scale_slice.copy_(w3_scale)

        #         del neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w1.scale"]
        #         del neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w3.scale"]
        #     neuron_state_dict[f"layers.{l}.block_sparse_moe.expert_mlps.mlp_op.gate_up_proj.scale"] = gate_up_proj_scale

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
        neuron_state_dict[f"layers.{l}.block_sparse_moe.expert_mlps.mlp_op.down_proj.weight"] = down_proj

        # # Merge down_proj scale parameters for FP8 quantization (if present)
        # if f"layers.{l}.block_sparse_moe.experts.0.w2.scale" in neuron_state_dict:
        #     # Get scale shape from first expert
        #     scale_h, scale_w = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w2.scale"].shape
        #     scale_device = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w2.scale"].device
        #     scale_dtype = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.0.w2.scale"].dtype

        #     down_proj_scale = torch.empty(
        #         config.num_local_experts,
        #         scale_h,
        #         scale_w,
        #         dtype=scale_dtype,
        #         device=scale_device,
        #     )
        #     for e in range(config.num_local_experts):
        #         w2_scale = neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w2.scale"]
        #         down_proj_scale_slice = torch.narrow(down_proj_scale, 0, e, 1)
        #         down_proj_scale_slice.copy_(w2_scale)
        #         del neuron_state_dict[f"layers.{l}.block_sparse_moe.experts.{e}.w2.scale"]
        #     neuron_state_dict[f"layers.{l}.block_sparse_moe.expert_mlps.mlp_op.down_proj.scale"] = down_proj_scale

        gc.collect()

    if config.neuron_config.fused_qkv:
        neuron_state_dict = convert_state_dict_to_fused_qkv(neuron_state_dict, config)

    return neuron_state_dict


class DistributedRMSNorm(nn.Module):
    """
    Distributed RMSNorm for MiniMax M2 qk_norm.

    MiniMax M2's qk_norm applies RMSNorm on the full Q/K output (e.g., [6144] dims for Q)
    BEFORE reshape. In TP parallel, Q/K are sharded, so we need distributed RMS computation:
    1. Each rank computes local sum(x²)
    2. All-reduce to get global sum
    3. Compute global RMS
    4. Each rank normalizes with global RMS and local weights

    This implementation uses all-reduce to compute the correct global RMS, matching
    the GPU behavior exactly.
    """
    def __init__(self, hidden_size, eps=1e-6, tp_degree=1, full_hidden_size=None):
        super().__init__()
        self.hidden_size = hidden_size  # Per-rank hidden size (e.g., 128)
        self.full_hidden_size = full_hidden_size or hidden_size  # Full hidden size (e.g., 6144)
        self.variance_epsilon = eps
        self.tp_degree = tp_degree
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Compute local sum of squares (not mean yet)
        # hidden_states shape: [batch, seq, hidden_size]
        local_sum_sq = hidden_states.pow(2).sum(-1, keepdim=True)

        # All-reduce to get global sum of squares across all TP ranks
        # This is necessary because GPU RMSNorm computes RMS over all 6144 elements
        if self.tp_degree > 1 and parallel_state.model_parallel_is_initialized():
            # All-reduce sum across TP group
            global_sum_sq = local_sum_sq.clone()
            torch.distributed.all_reduce(
                global_sum_sq,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_tensor_model_parallel_group()
            )
        else:
            global_sum_sq = local_sum_sq

        # Compute global RMS: sqrt(sum(x²) / total_elements)
        # total_elements = full_hidden_size (e.g., 6144 for Q, 1024 for K)
        global_variance = global_sum_sq / self.full_hidden_size
        rsqrt_variance = torch.rsqrt(global_variance + self.variance_epsilon)

        # Normalize with global RMS and apply local weights
        hidden_states = hidden_states * rsqrt_variance
        result = self.weight * hidden_states

        return result.to(original_dtype)


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    if cpu_mode():
        from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_gpu import MiniMaxM2RMSNorm
        return MiniMaxM2RMSNorm
    else:
        return CustomRMSNorm


def get_distributed_rmsnorm_cls():
    """Get the distributed RMSNorm class for qk_norm."""
    if cpu_mode():
        from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_gpu import MiniMaxM2RMSNorm

        # Wrapper to make MiniMaxM2RMSNorm accept the same args as DistributedRMSNorm
        class CPUCompatibleRMSNorm(MiniMaxM2RMSNorm):
            def __init__(self, hidden_size, eps=1e-6, tp_degree=1, full_hidden_size=None):
                # CPU mode doesn't need distributed computation, just use standard RMSNorm
                super().__init__(hidden_size, eps=eps)

        return CPUCompatibleRMSNorm
    else:
        return DistributedRMSNorm


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
            "use_qk_norm",  # MiniMax M2 uses per-head QK normalization
            "rotary_dim",  # MiniMax M2 uses partial rotary (rotary_dim=64, head_dim=128)
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
            # Don't use base class qk_norm - MiniMax M2 has different qk_norm implementation
            use_qk_norm=False,
        )

        # MiniMax M2 qk_norm: RMSNorm applied on QKV projection output BEFORE reshape
        # In TP parallel, Q/K are already sharded, so we need to use sharded dimensions
        # self.num_heads and self.num_key_value_heads are already divided by tp_degree in base class
        self.use_minimax_qk_norm = getattr(config, 'use_qk_norm', False)
        # Debug: Print qk_norm config (only for first layer to avoid spam)
        if not hasattr(NeuronMiniMaxM2Attention, '_qk_norm_debug_printed'):
            print(f"\n=== DEBUG: NeuronMiniMaxM2Attention config ===")
            print(f"  use_qk_norm from config: {getattr(config, 'use_qk_norm', 'NOT FOUND')}")
            print(f"  self.use_minimax_qk_norm: {self.use_minimax_qk_norm}")
            print(f"  self.num_heads (per TP rank): {self.num_heads}")
            print(f"  self.num_key_value_heads (per TP rank): {self.num_key_value_heads}")
            print(f"  config.num_attention_heads (total): {config.num_attention_heads}")
            print(f"  config.num_key_value_heads (total): {config.num_key_value_heads}")
            print(f"  head_dim: {config.head_dim}")
            print(f"  rotary_dim: {self.rotary_dim}")
            print(f"  tp_degree: {self.tp_degree}")
            NeuronMiniMaxM2Attention._qk_norm_debug_printed = True
        if self.use_minimax_qk_norm:
            # q_norm dimension: num_heads_per_rank * head_dim (sharded)
            # self.num_heads is already padded_num_attention_heads / tp_degree from base class
            # After interleaved padding: [48 heads * 128] -> [64 heads * 128]
            q_norm_dim = self.num_heads * config.head_dim  # = 1 * 128 = 128 per rank
            k_norm_dim = self.num_key_value_heads * config.head_dim  # = 1 * 128 = 128 per rank

            # Full dimensions for GPU-equivalent RMS computation
            # IMPORTANT: Use ORIGINAL (unpadded) dimensions for the RMS normalization!
            # The global RMS should be computed over the original 48 heads (6144 elements),
            # NOT the padded 64 heads (8192 elements), because:
            # 1. Padding heads contain zeros that shouldn't contribute to the RMS
            # 2. GPU computes RMS over original 6144 elements
            # 3. After all-reduce, we need to divide by original full_hidden_size
            full_q_norm_dim = config.num_attention_heads * config.head_dim  # 48 * 128 = 6144
            full_k_norm_dim = config.num_key_value_heads * config.head_dim  # 8 * 128 = 1024

            print(f"  Creating q_norm with dim={q_norm_dim} (full: {full_q_norm_dim})")
            print(f"  Creating k_norm with dim={k_norm_dim} (full: {full_k_norm_dim})")

            # Use DistributedRMSNorm with all-reduce for correct global RMS
            self.q_norm = get_distributed_rmsnorm_cls()(
                q_norm_dim,
                eps=self.rms_norm_eps,
                tp_degree=self.tp_degree,
                full_hidden_size=full_q_norm_dim
            )
            self.k_norm = get_distributed_rmsnorm_cls()(
                k_norm_dim,
                eps=self.rms_norm_eps,
                tp_degree=self.tp_degree,
                full_hidden_size=full_k_norm_dim
            )

        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "NeuronMiniMaxM2Attention has to be initialized in a distributed env. Please use neuronx_distributed"
                " module to initialize a distributed env."
            )

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
        skip_rope=False,
        residual=None,
        use_polar_compatible_rope=False,
    ):
        """Override to apply MiniMax M2 qk_norm on full projection output BEFORE reshape."""
        from neuronx_distributed_inference.modules.attention.utils import move_heads_front

        Q, K, V, residual = self.get_qkv_proj()(
            hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids, residual=residual
        )

        # MiniMax M2 qk_norm: apply RMSNorm on full [B, S, num_heads * head_dim] BEFORE reshape
        # This is different from typical per-head qk_norm which applies after reshape
        if self.use_minimax_qk_norm:
            Q = self.q_norm(Q)
            K = self.k_norm(K)

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        # Don't use q_layernorm/k_layernorm here since we already applied qk_norm above
        Q = move_heads_front(Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=None)
        K = move_heads_front(K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        if not skip_rope:
            # Rotate Q and K
            Q, K, cos_cache, sin_cache = self.apply_rotary_embedding(
                Q, K, V, position_ids, cos_cache, sin_cache, use_polar_compatible_rope
            )

        return Q, K, V, cos_cache, sin_cache, residual

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

        # Use custom MoE module with e_score_correction_bias support
        self.block_sparse_moe = initialize_minimax_m2_moe_module(config=config)

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
        hidden_states = self.block_sparse_moe(hidden_states)[0]
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

    @classmethod
    def get_state_dict(cls, model_name_or_path: str, config: InferenceConfig) -> dict:
        """
        Override get_state_dict to handle FP8 quantized models.
        The default load_state_dict doesn't load weight_scale_inv parameters when quantization_config is missing.
        """
        import os
        from safetensors import safe_open
        import json
        
        # model_name_or_path = '/home/ubuntu/model_hf/MiniMax-M2/'  # TODO Set to a fixed path
        model_name_or_path = '/home/ubuntu/model_hf/MiniMax-M2-BF16/'  # TODO Set to a fixed path

        print(f"\n=== CUSTOM get_state_dict CALLED ===")
        print(f"  model_name_or_path: {model_name_or_path}")
        print(f"  Is directory: {os.path.isdir(model_name_or_path)}")

        if os.path.isdir(model_name_or_path):
            # For FP8 quantized models, we need to load ALL parameters from safetensors
            # The standard load_state_dict filters out weight_scale_inv when quantization_config is missing
            print(f"\n=== Loading state_dict from {model_name_or_path} ===")

            # Check if this is a sharded safetensors model
            index_path = os.path.join(model_name_or_path, 'model.safetensors.index.json')
            if os.path.exists(index_path):
                print("  Detected sharded safetensors model, loading all shards...")
                with open(index_path, 'r') as f:
                    index = json.load(f)

                model_sd = {}
                # Load all shard files
                shard_files = set(index['weight_map'].values())
                for i, shard_file in enumerate(sorted(shard_files)):
                    if i % 20 == 0:  # Progress every 20 files
                        print(f"  Loading shard {i+1}/{len(shard_files)}: {shard_file}")
                    shard_path = os.path.join(model_name_or_path, shard_file)
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            model_sd[key] = f.get_tensor(key)

                print(f"  Loaded {len(model_sd)} parameters from {len(shard_files)} shards")

                # # Check for scale parameters BEFORE any conversion
                # weight_scale_inv_keys = [k for k in model_sd.keys() if 'weight_scale_inv' in k]
                # print(f"  Found {len(weight_scale_inv_keys)} parameters with 'weight_scale_inv'")
                # if weight_scale_inv_keys:
                #     print(f"  First 3 weight_scale_inv keys:")
                #     for k in weight_scale_inv_keys[:3]:
                #         print(f"    {k}")

                # scale_keys = [k for k in model_sd.keys() if 'scale' in k and 'weight_scale_inv' not in k]
                # print(f"  Found {len(scale_keys)} parameters with 'scale' (excluding weight_scale_inv)")
            else:
                # Fall back to standard loading for non-sharded models
                from neuronx_distributed_inference.modules.checkpoint import load_state_dict
                model_sd = load_state_dict(model_name_or_path)

            # Remove model. prefix and handle other transformations
            param_name_list = list(model_sd.keys())
            for param_name in param_name_list:
                updated_param_name = param_name
                if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                    updated_param_name = param_name.replace(
                        cls._STATE_DICT_MODEL_PREFIX, cls._NEW_STATE_DICT_MODEL_PREFIX, 1
                    )
                # if param_name.endswith(".weight_scale"):
                #     updated_param_name = updated_param_name.replace(".weight_scale", ".scale")
                if updated_param_name != param_name:
                    model_sd[updated_param_name] = model_sd[param_name]
                    del model_sd[param_name]
        else:
            # Use parent class implementation for non-directory paths
            return super().get_state_dict(model_name_or_path, config)

        model_sd = cls.convert_hf_to_neuron_state_dict(model_sd, config)
        if getattr(config, "tie_word_embeddings", False):
            cls.update_state_dict_for_tied_weights(model_sd)

        param_name_list = list(model_sd.keys())
        if cls._FUSED_PREFIX != "":
            for param_name in param_name_list:
                model_sd[f"{cls._FUSED_PREFIX}.{param_name}"] = model_sd[param_name]
                del model_sd[param_name]
        return model_sd

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        import json
        import os
        import shutil
        from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_gpu import MiniMaxM2ForCausalLM

        print('model_path:', model_path)
        print('kwargs:', kwargs)
        # model_path = '/home/ubuntu/model_hf/MiniMax-M2/'  # TODO Set to a fixed path
        model_path = '/home/ubuntu/model_hf/MiniMax-M2-BF16/'  # TODO Set to a fixed path

        # For FP8 quantized models, transformers will check for GPU/XPU even with device_map="cpu"
        # We need to temporarily remove quantization_config from config.json to bypass the check
        # The actual FP8 weights will be loaded correctly from safetensors by Neuron
        config_path = os.path.join(model_path, 'config.json')
        config_backup_path = os.path.join(model_path, 'config.json.backup')

        # Backup original config and create a version without quantization_config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Check if quantization_config exists
            if 'quantization_config' in config_data:
                print("Temporarily removing quantization_config to bypass transformers FP8 GPU check...")
                # Backup original config
                shutil.copy2(config_path, config_backup_path)

                # Remove quantization_config
                quantization_config = config_data.pop('quantization_config')

                # Write modified config
                with open(config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)

                try:
                    # Load model without quantization_config check
                    model = MiniMaxM2ForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        device_map="cpu",
                        **kwargs
                    )
                finally:
                    # Restore original config
                    if os.path.exists(config_backup_path):
                        shutil.move(config_backup_path, config_path)
                        print("Restored original config.json with quantization_config")

                return model

        # Fallback if no config manipulation needed
        return MiniMaxM2ForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            **kwargs
        )

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
