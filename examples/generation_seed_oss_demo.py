"""
Seed-OSS-36B-Instruct generation demo on Trainium2 (trn2.48xlarge).

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
    python generation_seed_oss_demo.py
"""

import torch
from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.seed_oss.modeling_seed_oss import (
    SeedOssInferenceConfig,
    NeuronSeedOssForCausalLM,
)
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params

model_path = "/opt/dlami/nvme/Seed-OSS-36B-Instruct"
traced_model_path = "/opt/dlami/nvme/traced_model/Seed-OSS-36B-Instruct"

torch.manual_seed(0)


def run_seed_oss_generate():
    # Initialize configs and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config_kwargs = {
        "do_sample": True,
        "top_k": 1,
        "pad_token_id": tokenizer.pad_token_id,
    }
    generation_config.update(**generation_config_kwargs)

    # Seed-OSS-36B: 64 layers, 80 Q heads, 8 KV heads, hidden_size=5120
    # trn2.48xlarge has 64 NeuronCores → tp_degree=8 works well
    neuron_config = NeuronConfig(
        tp_degree=8,
        batch_size=1,
        max_context_length=2048,
        seq_len=4096,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=True,
        flash_decoding_enabled=False,
    )

    config = SeedOssInferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(model_path),
    )

    # Compile and save model.
    print("\nCompiling and saving model...")
    model = NeuronSeedOssForCausalLM(model_path, config)
    model.compile(traced_model_path)
    tokenizer.save_pretrained(traced_model_path)

    # Load from compiled checkpoint.
    print("\nLoading model from compiled checkpoint...")
    model = NeuronSeedOssForCausalLM(traced_model_path)
    model.load(traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(traced_model_path)

    # Generate outputs.
    print("\nGenerating outputs...")
    prompts = ["Hello, tell me about yourself."]
    sampling_params = prepare_sampling_params(
        batch_size=neuron_config.batch_size,
        top_k=[10],
        top_p=[0.95],
        temperature=[1.1],
    )
    print(f"Prompts: {prompts}")
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    generation_model = HuggingFaceGenerationAdapter(model)
    outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
    )
    output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")


if __name__ == "__main__":
    run_seed_oss_generate()
