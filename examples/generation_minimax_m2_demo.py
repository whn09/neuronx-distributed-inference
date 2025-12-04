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
            # on_device_sampling_config=OnDeviceSamplingConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95),
            enable_bucketing=False,
            flash_decoding_enabled=False,
            # save_sharded_checkpoint=True,  # ← 启用！保存分片权重，加载时快很多
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

    # # Debug: Check if model weights are loaded correctly
    # print("\n=== Debug: Checking loaded model ===")
    # print(f"Model type: {type(model)}")
    # print(f"Model config neuron_config.quantized: {model.config.neuron_config.quantized}")
    # print(f"Model config neuron_config.quantized_mlp_kernel_enabled: {model.config.neuron_config.quantized_mlp_kernel_enabled}")
    # print(f"Model config torch_dtype: {model.config.neuron_config.torch_dtype}")
    # print(f"Model config use_qk_norm: {getattr(model.config, 'use_qk_norm', 'NOT FOUND')}")
    # print(f"Model config rotary_dim: {getattr(model.config, 'rotary_dim', 'NOT FOUND')}")

    # # Check if qk_norm modules exist and have correct shapes
    # print("\n=== Debug: Checking qk_norm modules ===")
    # print(f"  Model type: {type(model)}")
    # print(f"  Model attributes: {[a for a in dir(model) if not a.startswith('_')][:20]}")

    # # Explore model structure more thoroughly
    # def explore_model(obj, path="model", depth=0, max_depth=5):
    #     if depth > max_depth:
    #         return
    #     indent = "  " * depth
    #     if hasattr(obj, 'q_norm'):
    #         print(f"{indent}FOUND q_norm at {path}")
    #         q_norm = obj.q_norm
    #         if hasattr(q_norm, 'weight'):
    #             w = q_norm.weight
    #             print(f"{indent}  q_norm weight shape: {w.shape}, dtype: {w.dtype}")
    #             try:
    #                 w_cpu = w.detach().cpu().float()
    #                 print(f"{indent}  q_norm weight stats: mean={w_cpu.mean():.6f}, std={w_cpu.std():.6f}")
    #             except:
    #                 print(f"{indent}  (cannot get weight stats)")
    #     if hasattr(obj, 'k_norm'):
    #         print(f"{indent}FOUND k_norm at {path}")
    #         k_norm = obj.k_norm
    #         if hasattr(k_norm, 'weight'):
    #             w = k_norm.weight
    #             print(f"{indent}  k_norm weight shape: {w.shape}, dtype: {w.dtype}")
    #             try:
    #                 w_cpu = w.detach().cpu().float()
    #                 print(f"{indent}  k_norm weight stats: mean={w_cpu.mean():.6f}, std={w_cpu.std():.6f}")
    #             except:
    #                 print(f"{indent}  (cannot get weight stats)")

    #     # Recurse into common container attributes
    #     for attr_name in ['model', 'models', 'layers', 'decoder', 'encoder']:
    #         if hasattr(obj, attr_name):
    #             attr = getattr(obj, attr_name)
    #             if isinstance(attr, (list, tuple)) and len(attr) > 0:
    #                 explore_model(attr[0], f"{path}.{attr_name}[0]", depth + 1, max_depth)
    #             elif attr is not None and not isinstance(attr, (str, int, float, bool)):
    #                 explore_model(attr, f"{path}.{attr_name}", depth + 1, max_depth)

    #     # Also check self_attn
    #     if hasattr(obj, 'self_attn'):
    #         explore_model(obj.self_attn, f"{path}.self_attn", depth + 1, max_depth)

    # try:
    #     explore_model(model)
    # except Exception as e:
    #     import traceback
    #     print(f"  Error exploring model: {e}")
    #     traceback.print_exc()

    # # Also print named_modules to find qk_norm
    # print("\n=== Searching for qk_norm in named_modules ===")
    # try:
    #     qk_norm_found = False
    #     for name, module in model.named_modules():
    #         if 'q_norm' in name or 'k_norm' in name:
    #             print(f"  Found: {name} -> {type(module)}")
    #             if hasattr(module, 'weight'):
    #                 w = module.weight
    #                 print(f"    weight shape: {w.shape}, dtype: {w.dtype}")
    #                 try:
    #                     w_cpu = w.detach().cpu().float()
    #                     print(f"    weight stats: mean={w_cpu.mean():.6f}, std={w_cpu.std():.6f}, min={w_cpu.min():.6f}, max={w_cpu.max():.6f}")
    #                 except Exception as e:
    #                     print(f"    (cannot get weight stats: {e})")
    #             qk_norm_found = True
    #     if not qk_norm_found:
    #         print("  No qk_norm modules found in named_modules")
    # except Exception as e:
    #     print(f"  Error searching named_modules: {e}")

    # # Check a sample weight - try different possible structures
    # try:
    #     if hasattr(model, 'model'):
    #         sample_weight = model.model.layers[0].self_attn.qkv_proj.q_proj.weight
    #     elif hasattr(model, 'models'):
    #         print(f"Model has 'models' attribute with {len(model.models)} model(s)")
    #         sample_weight = model.models[0].layers[0].self_attn.qkv_proj.q_proj.weight
    #     else:
    #         print("Could not access model layers - skipping weight check")
    #         sample_weight = None

    #     if sample_weight is not None:
    #         print(f"\nSample weight (layers.0.self_attn.qkv_proj.q_proj.weight):")
    #         print(f"  dtype: {sample_weight.dtype}")
    #         print(f"  shape: {sample_weight.shape}")
    #         print(f"  device: {sample_weight.device}")
    #         if sample_weight.device.type == 'cpu':
    #             print(f"  value range: [{sample_weight.min():.6f}, {sample_weight.max():.6f}]")
    #             print(f"  mean: {sample_weight.float().mean():.6f}")
    #             print(f"  std: {sample_weight.float().std():.6f}")
    #         else:
    #             print(f"  (weight is on Neuron device, cannot inspect values)")
    # except Exception as e:
    #     print(f"Error checking weights: {e}")

    # Generate outputs.
    print("\n=== Generating outputs ===")

    # Test 1: Simple completion (no chat template) to verify model works
    use_simple_completion = True  # Set to True for simple test

    if use_simple_completion:
        # Simple text completion - easier to verify model behavior
        # text = "The capital of France is"
        text = "Who are you?"
        print(f"Using SIMPLE COMPLETION test: '{text}'")
    else:
        # Full chat template
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
    print(f"Innputs: {inputs}")

    # Debug: Verify tokenization is correct by decoding back
    decoded_text = tokenizer.decode(inputs.input_ids[0])
    print(f"Decoded back from tokens: '{decoded_text}'")

    # # Debug: Check if token IDs are in valid range
    # vocab_size = tokenizer.vocab_size
    # max_token_id = inputs.input_ids.max().item()
    # min_token_id = inputs.input_ids.min().item()
    # print(f"Tokenizer vocab_size: {vocab_size}")
    # print(f"Model config vocab_size: {model.config.vocab_size}")
    # print(f"Token ID range in input: [{min_token_id}, {max_token_id}]")
    # if max_token_id >= vocab_size:
    #     print(f"WARNING: Token ID {max_token_id} exceeds tokenizer vocab_size {vocab_size}!")
    # if max_token_id >= model.config.vocab_size:
    #     print(f"WARNING: Token ID {max_token_id} exceeds model config vocab_size {model.config.vocab_size}!")

    # # Debug: Check vocab_size alignment with TP
    # tp_degree = model.config.neuron_config.tp_degree
    # print(f"TP degree: {tp_degree}")
    # print(f"vocab_size % tp_degree = {model.config.vocab_size % tp_degree}")
    # if model.config.vocab_size % tp_degree != 0:
    #     print(f"WARNING: vocab_size {model.config.vocab_size} is not divisible by tp_degree {tp_degree}!")

    # # Debug: Check what per-rank vocab size would be
    # per_rank_vocab = model.config.vocab_size // tp_degree
    # print(f"Per-rank vocab size: {per_rank_vocab}")
    # print(f"Token 758 would be on rank: {758 // per_rank_vocab} (local idx: {758 % per_rank_vocab})")
    # print(f"Token 5969 would be on rank: {5969 // per_rank_vocab} (local idx: {5969 % per_rank_vocab})")

    generation_model = HuggingFaceGenerationAdapter(model)
    print(f"\nGenerating with max_length={model.config.neuron_config.max_length}...")

    # Try greedy decoding first to check if model works correctly
    # If greedy works but sampling doesn't, it's a sampling parameter issue
    use_greedy = True  # Set to True for deterministic test

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
        # print(f"Output {i} (first 500 chars):\n{output_token[:500]}...")
        # if len(output_token) > 500:
        #     print(f"...(total length: {len(output_token)} chars)")
        print(f"Output {i}:\n{output_token}...")
        print(f"...(total length: {len(output_token)} chars)")


if __name__ == "__main__":
    # Step 1: Compile and save sharded checkpoint (run once, takes time)
    generate(skip_compile=False)

    # Step 2: After compilation, use this for fast loading
    # generate(skip_compile=True)
