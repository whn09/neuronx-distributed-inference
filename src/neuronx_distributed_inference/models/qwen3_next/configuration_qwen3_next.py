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
"""Configuration class for Qwen3 Next model."""

from transformers import PretrainedConfig


class Qwen3NextConfig(PretrainedConfig):
    """Configuration class for Qwen3 Next model.

    Qwen3 Next is a hybrid attention model that combines:
    - Full softmax attention (every full_attention_interval layers)
    - Gated Delta Net linear attention (other layers)
    - MoE with shared experts
    """

    model_type = "qwen3_next"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=5120,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_hidden_layers=48,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=256,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000000.0,
        rope_scaling=None,
        attention_dropout=0.0,
        # MoE configuration
        num_experts=512,
        num_experts_per_tok=10,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        decoder_sparse_step=1,
        # Hybrid attention configuration
        full_attention_interval=4,
        mlp_only_layers=None,
        # Linear attention configuration
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        # Partial RoPE configuration
        partial_rotary_factor=0.25,
        use_sliding_window=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout

        # MoE configuration
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.decoder_sparse_step = decoder_sparse_step

        # Hybrid attention configuration
        self.full_attention_interval = full_attention_interval
        self.mlp_only_layers = mlp_only_layers if mlp_only_layers is not None else []

        # Linear attention configuration
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim

        # Partial RoPE configuration
        self.partial_rotary_factor = partial_rotary_factor
        self.use_sliding_window = use_sliding_window

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
