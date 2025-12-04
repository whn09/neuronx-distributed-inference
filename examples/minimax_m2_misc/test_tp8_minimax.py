"""
Test MiniMax M2 with tp_degree=8 to simplify sharding.
With tp_degree=8 (same as num_kv_heads), there's no padding needed for KV heads.
"""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config

model_path = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
traced_model_path_tp8 = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights-tp8/"

torch.manual_seed(0)


def test_tp8():
    """Test with tp_degree=8."""
    print("\n" + "="*60)
    print("Testing MiniMax M2 with tp_degree=8")
    print("="*60)

    # tp_degree=8 means:
    # - Q heads: 48 / 8 = 6 heads per rank (no padding needed)
    # - KV heads: 8 / 8 = 1 head per rank (no replication needed)
    # This is the simplest sharding scenario

    neuron_config = MoENeuronConfig(
        tp_degree=8,  # Same as num_key_value_heads=8, simplest case
        batch_size=1,
        max_context_length=1024,
        seq_len=1024,
        enable_bucketing=False,
        flash_decoding_enabled=False,
        blockwise_matmul_config={
            'use_torch_block_wise': True,
        },
        router_config={
            'act_fn': 'sigmoid',
        },
    )

    config = MiniMaxM2InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    print(f"Config settings:")
    print(f"  tp_degree: {config.neuron_config.tp_degree}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  heads_per_rank (Q): {config.num_attention_heads // config.neuron_config.tp_degree}")
    print(f"  heads_per_rank (KV): {config.num_key_value_heads // config.neuron_config.tp_degree}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    # Compile model
    print("\nCompiling model with tp_degree=8...")
    model = NeuronMiniMaxM2ForCausalLM(model_path, config)
    model.compile(traced_model_path_tp8)
    tokenizer.save_pretrained(traced_model_path_tp8)

    # Load and test
    print("\nLoading compiled model...")
    model = NeuronMiniMaxM2ForCausalLM(traced_model_path_tp8)
    model.load(traced_model_path_tp8)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path_tp8)

    # Test generation
    text = "Hello"
    inputs = tokenizer([text], return_tensors="pt")
    print(f"\nInput text: '{text}'")
    print(f"Input token IDs: {inputs.input_ids[0].tolist()}")

    generation_model = HuggingFaceGenerationAdapter(model)

    # Generate with return_dict_in_generate to get logits
    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=10,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
    )

    print(f"\nGenerated token IDs: {outputs.sequences[0].tolist()}")
    print(f"Generated text: '{tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)}'")

    # Analyze first step logits
    if hasattr(outputs, 'logits') and outputs.logits is not None:
        first_logits = outputs.logits[0][0]
        top_logits, top_indices = torch.topk(first_logits, 10)
        print(f"\nFirst step top-10 predictions:")
        for i, (logit, idx) in enumerate(zip(top_logits.tolist(), top_indices.tolist())):
            token_str = tokenizer.decode([idx])
            print(f"  {i+1}. '{token_str}' (ID={idx}, logit={logit:.2f})")


if __name__ == "__main__":
    test_tp8()
