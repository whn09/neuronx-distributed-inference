import torch

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

# Use BF16 checkpoint (converted from FP8 on GPU)
model_path = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
# Original FP8 checkpoint:
# model_path = "/home/ubuntu/model_hf/MiniMax-M2/"
traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights/"

torch.manual_seed(0)


def generate(skip_compile=False):
    # Initialize configs and tokenizer.
    generation_config = GenerationConfig.from_pretrained(model_path)

    if not skip_compile:
        # MiniMax M2 is a very large MoE model (62 layers, 256 experts)
        # Model uses GQA: num_attention_heads=48, num_key_value_heads=8
        # tp_degree must be a multiple of num_key_value_heads (8) for proper KV head distribution
        neuron_config = MoENeuronConfig(
            tp_degree=64,  # Must be multiple of num_key_value_heads=8
            # ep_degree=64,
            # moe_tp_degree=1,
            # moe_ep_degree=64,
            batch_size=1,
            max_context_length=1024,  # default: 128
            seq_len=1024,
            on_device_sampling_config=OnDeviceSamplingConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95),
            enable_bucketing=False,
            flash_decoding_enabled=False,
            save_sharded_checkpoint=True,  # ← 启用！保存分片权重，加载时快很多
            # Use torch implementation to bypass NKI kernel's DGE limitation
            # (intermediate_size=1536 / tp_degree=64 = 24 < 32 required by DGE)
            blockwise_matmul_config={
                'use_torch_block_wise': True,
            },
            # MiniMax M2 uses sigmoid activation for router (not softmax)
            router_config={
                'act_fn': 'sigmoid',
            },
            # Dequantize FP8 weights to bfloat16 (quantized_mlp_kernel_enabled=False)
            # This avoids scale sharding issues with block-wise quantization
            # quantized_mlp_kernel_enabled=False,
            # Disable fused_qkv when using FP8 quantization
            # FP8 quantization requires separate scale parameters for Q, K, V
            # fused_qkv=False,  # default: False
        )
        config = MiniMaxM2InferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(model_path),
        )      
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token
        # Compile and save model.
        print("\nCompiling and saving model...")
        model = NeuronMiniMaxM2ForCausalLM(model_path, config)
        model.compile(traced_model_path)
        tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Debug: Check if model weights are loaded correctly
    print("\n=== Debug: Checking loaded model ===")
    print(f"Model type: {type(model)}")
    print(f"Model config neuron_config.quantized: {model.config.neuron_config.quantized}")
    print(f"Model config neuron_config.quantized_mlp_kernel_enabled: {model.config.neuron_config.quantized_mlp_kernel_enabled}")
    print(f"Model config torch_dtype: {model.config.neuron_config.torch_dtype}")
    print(f"Model config use_qk_norm: {getattr(model.config, 'use_qk_norm', 'NOT FOUND')}")
    print(f"Model config rotary_dim: {getattr(model.config, 'rotary_dim', 'NOT FOUND')}")

    # Check if qk_norm modules exist and have correct shapes
    print("\n=== Debug: Checking qk_norm modules ===")
    try:
        # Try to access model internals
        if hasattr(model, 'models') and len(model.models) > 0:
            # During inference, the model structure might be different
            first_model = model.models[0]
            if hasattr(first_model, 'layers'):
                first_layer = first_model.layers[0]
                if hasattr(first_layer, 'self_attn'):
                    attn = first_layer.self_attn
                    print(f"  Attention module type: {type(attn)}")
                    print(f"  Has q_norm: {hasattr(attn, 'q_norm')}")
                    print(f"  Has k_norm: {hasattr(attn, 'k_norm')}")
                    print(f"  use_minimax_qk_norm: {getattr(attn, 'use_minimax_qk_norm', 'NOT FOUND')}")
                    if hasattr(attn, 'q_norm'):
                        print(f"  q_norm weight shape: {attn.q_norm.weight.shape if hasattr(attn.q_norm, 'weight') else 'no weight attr'}")
                    if hasattr(attn, 'k_norm'):
                        print(f"  k_norm weight shape: {attn.k_norm.weight.shape if hasattr(attn.k_norm, 'weight') else 'no weight attr'}")
    except Exception as e:
        print(f"  Error accessing model internals: {e}")

    # Check a sample weight - try different possible structures
    try:
        if hasattr(model, 'model'):
            sample_weight = model.model.layers[0].self_attn.qkv_proj.q_proj.weight
        elif hasattr(model, 'models'):
            print(f"Model has 'models' attribute with {len(model.models)} model(s)")
            sample_weight = model.models[0].layers[0].self_attn.qkv_proj.q_proj.weight
        else:
            print("Could not access model layers - skipping weight check")
            sample_weight = None

        if sample_weight is not None:
            print(f"\nSample weight (layers.0.self_attn.qkv_proj.q_proj.weight):")
            print(f"  dtype: {sample_weight.dtype}")
            print(f"  shape: {sample_weight.shape}")
            print(f"  device: {sample_weight.device}")
            if sample_weight.device.type == 'cpu':
                print(f"  value range: [{sample_weight.min():.6f}, {sample_weight.max():.6f}]")
                print(f"  mean: {sample_weight.float().mean():.6f}")
                print(f"  std: {sample_weight.float().std():.6f}")
            else:
                print(f"  (weight is on Neuron device, cannot inspect values)")
    except Exception as e:
        print(f"Error checking weights: {e}")

    # Generate outputs.
    print("\n=== Generating outputs ===")
    prompt = "Give me a short introduction to large language models."
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes if supported.
    )
    print(f"Formatted prompt (first 200 chars): {text[:200]}...")

    inputs = tokenizer([text], padding=True, return_tensors="pt")
    print(f"Input token IDs shape: {inputs.input_ids.shape}")
    print(f"Input token IDs (first 10): {inputs.input_ids[0, :10].tolist()}")

    generation_model = HuggingFaceGenerationAdapter(model)
    print(f"\nGenerating with max_length={model.config.neuron_config.max_length}...")

    # Try greedy decoding first to check if model works correctly
    # If greedy works but sampling doesn't, it's a sampling parameter issue
    use_greedy = True  # Set to False to use sampling

    if use_greedy:
        print("Using GREEDY decoding (do_sample=False)")
        outputs = generation_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=model.config.neuron_config.max_length,
            do_sample=False,
        )
    else:
        print("Using SAMPLING with temperature=0.6, top_k=20, top_p=0.95")
        outputs = generation_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=model.config.neuron_config.max_length,
            do_sample=True,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
        )
    print(f"Output token IDs shape: {outputs.shape}")
    print(f"Output token IDs (first 20): {outputs[0, :20].tolist()}")

    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("\n=== Generated outputs ===")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i} (first 500 chars):\n{output_token[:500]}...")
        if len(output_token) > 500:
            print(f"...(total length: {len(output_token)} chars)")


if __name__ == "__main__":
    # Step 1: Compile and save sharded checkpoint (run once, takes time)
    generate(skip_compile=False)

    # Step 2: After compilation, use this for fast loading
    # generate(skip_compile=True)
