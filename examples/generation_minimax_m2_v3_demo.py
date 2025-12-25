"""
MiniMax M2 generation demo using the v3 implementation.
Updated implementation following latest Qwen3 MoE pattern.

v3 updates:
- fused_qkv support
- moe_fused_nki_kernel support
- maybe_pad_intermediate for shard-on-I
- ModuleMarker wrappers for compiler optimization
- Enhanced compiler args
- qkv_kernel_enabled support
"""
import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_v3 import MiniMaxM2InferenceConfigV3, NeuronMiniMaxM2ForCausalLMV3
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.utils.benchmark import benchmark_sampling

# Use BF16 checkpoint (converted from FP8 on GPU)
model_path = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-BF16-v3/"

torch.manual_seed(0)

DTYPE = torch.bfloat16

def generate(skip_compile=False):
    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)

    if not skip_compile:
        # MiniMax M2 is a very large MoE model (62 layers, 256 experts)
        # Model uses GQA: num_attention_heads=48, num_key_value_heads=8
        # tp_degree must be a multiple of num_key_value_heads (8) for proper KV head distribution
        #
        # NOTE: Context parallel (cp_degree) and attention_dp_degree are disabled for now
        # because they cause tensor shape incompatibility with MiniMax M2's attention architecture.
        # These can be re-enabled once compatibility is verified.
        # Simplified config for testing - no EP to avoid TKG compatibility issues
        # Using smaller batch_size and seq_len to fit in memory with pure TP
        neuron_config = MoENeuronConfig(
            tp_degree=64,  # Must be multiple of num_key_value_heads=8
            moe_tp_degree=64,  # Full TP for MoE (no EP)
            moe_ep_degree=1,  # moe_ep_degree not set - defaults to 1
            batch_size=1,  # Reduced for memory, 16
            ctx_batch_size=1,
            tkg_batch_size=1,  # 16
            seq_len=512,  # Reduced for memory, 10240
            max_context_length=256,  # 1024
            # scratchpad_page_size=1024,
            torch_dtype=DTYPE,
            # Lower temperature for more coherent output (Qwen3 MoE uses 0.6)
            on_device_sampling_config=OnDeviceSamplingConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95),
            enable_bucketing=False,
            flash_decoding_enabled=False,
            # attention_dp_degree=8,
            # cp_degree=16,
            fused_qkv=True,
            is_continuous_batching=False,  # Simpler for testing, True
            logical_nc_config=2,
            sequence_parallel_enabled=True,
            # Disable kernel optimizations
            qkv_kernel_enabled=False,  # True
            attn_kernel_enabled=False,  # True
            strided_context_parallel_kernel_enabled=False,  # True
            # blockwise_matmul_config={"use_shard_on_intermediate_dynamic_while": True, "skip_dma_token": True},  
            # MiniMax M2 uses sigmoid activation for router (not softmax)
            router_config={
                'act_fn': 'sigmoid',
            },
        )
        config = MiniMaxM2InferenceConfigV3(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token
        # Compile and save model.
        print("\nCompiling and saving model (v3 implementation)...")
        model = NeuronMiniMaxM2ForCausalLMV3(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronMiniMaxM2ForCausalLMV3(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")

    print("\nPerformance Benchmarking!")
    benchmark_sampling(model=model, draft_model=None, generation_config=generation_config, target="all",benchmark_report_path="benchmark_report.json", num_runs=5)

if __name__ == "__main__":
    # 首次运行需要编译，设置skip_compile=False
    # 编译完成后可以改为skip_compile=True跳过编译直接加载
    generate(skip_compile=False)
    # generate(skip_compile=True)
