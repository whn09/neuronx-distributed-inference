"""
Debug script to check layer-by-layer output statistics.
This helps identify which layer/component is causing issues.
"""
import torch
import sys
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

from transformers import AutoTokenizer
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2InferenceConfig, NeuronMiniMaxM2ForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter

traced_model_path = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights/"


def check_generation_step_by_step():
    """Check generation output step by step."""
    print("\n" + "="*60)
    print("Step-by-step generation analysis")
    print("="*60)

    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Test multiple inputs to see if the issue is input-specific
    test_inputs = [
        "Hello",
        "The capital of France is",
        "1 + 1 =",
        "What is",
    ]

    generation_model = HuggingFaceGenerationAdapter(model)

    for text in test_inputs:
        print(f"\n{'='*60}")
        print(f"Input: '{text}'")
        print("="*60)

        inputs = tokenizer([text], return_tensors="pt")
        input_ids = inputs.input_ids
        print(f"  Token IDs: {input_ids[0].tolist()}")

        # Reset model state
        generation_model.neuron_model.reset()

        # Generate 5 tokens
        outputs = generation_model.generate(
            input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=5,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        print(f"  Generated: '{generated_text}'")

        # Check first token prediction
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            first_logits = outputs.logits[0][0]
            top_logits, top_indices = torch.topk(first_logits, 5)
            print(f"  First token predictions:")
            for i, (logit, idx) in enumerate(zip(top_logits.tolist(), top_indices.tolist())):
                token_str = tokenizer.decode([idx])
                print(f"    {i+1}. '{token_str}' (ID={idx}, logit={logit:.2f})")

            # Check logit distribution
            print(f"  Logit stats: mean={first_logits.float().mean():.2f}, "
                  f"std={first_logits.float().std():.2f}, "
                  f"min={first_logits.float().min():.2f}, "
                  f"max={first_logits.float().max():.2f}")


def check_specific_tokens():
    """Check if specific known tokens have reasonable logits."""
    print("\n" + "="*60)
    print("Specific token logit analysis")
    print("="*60)

    model = NeuronMiniMaxM2ForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)
    generation_model = HuggingFaceGenerationAdapter(model)

    # For "Hello", expected first tokens might be things like " there", " world", "!", " everyone"
    text = "Hello"
    inputs = tokenizer([text], return_tensors="pt")

    outputs = generation_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1,
        do_sample=False,
        return_dict_in_generate=True,
        output_logits=True,
    )

    first_logits = outputs.logits[0][0]

    # Check logits for expected tokens
    expected_tokens = [
        " there", " world", "!", " everyone", " how",
        ",", " ", "\n", ".", " I"
    ]

    print(f"Input: '{text}'")
    print(f"Logits for expected continuation tokens:")
    for tok_str in expected_tokens:
        tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if tok_ids:
            tok_id = tok_ids[0]
            logit = first_logits[tok_id].item()
            print(f"  '{tok_str}' (ID={tok_id}): logit={logit:.2f}")

    # Check logits for code-like tokens that were incorrectly predicted
    print(f"\nLogits for incorrectly predicted code tokens:")
    code_tokens = ["()", "(", ")", "\n\n", ".org", "{\n", "//"]
    for tok_str in code_tokens:
        tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if tok_ids:
            tok_id = tok_ids[0]
            logit = first_logits[tok_id].item()
            print(f"  '{tok_str}' (ID={tok_id}): logit={logit:.2f}")


if __name__ == "__main__":
    check_generation_step_by_step()
    check_specific_tokens()
