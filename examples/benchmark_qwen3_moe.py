"""
Qwen3 MoE Benchmark Script

Measures:
- TTFT (Time To First Token): Prefill latency
- TPOT (Time Per Output Token): Decode latency per token
- Throughput: Tokens per second
"""
import torch
import time
import argparse
import numpy as np

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeInferenceConfig, NeuronQwen3MoeForCausalLM
from neuronx_distributed_inference.utils.hf_adapter import HuggingFaceGenerationAdapter, load_pretrained_config


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 MoE Benchmark")
    parser.add_argument("--model-path", type=str, default="/home/ubuntu/model_hf/Qwen3-235B-A22B/",
                        help="Path to HuggingFace model")
    parser.add_argument("--traced-model-path", type=str, default="/home/ubuntu/traced_model/Qwen3-235B-A22B-benchmark/",
                        help="Path to save/load traced model")
    parser.add_argument("--tp-degree", type=int, default=32,
                        help="Tensor parallel degree")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--input-length", type=int, default=10240,
                        help="Input sequence length (context length for prefill)")
    parser.add_argument("--output-length", type=int, default=256,
                        help="Number of tokens to generate")
    parser.add_argument("--warmup-runs", type=int, default=3,
                        help="Number of warmup runs")
    parser.add_argument("--benchmark-runs", type=int, default=5,
                        help="Number of benchmark runs")
    parser.add_argument("--compile", action="store_true",
                        help="Compile model (required for new input lengths)")
    parser.add_argument("--use-sample", action="store_true",
                        help="Use sampling instead of greedy decoding")
    return parser.parse_args()


def create_dummy_input(tokenizer, input_length, batch_size=1):
    """Create dummy input of specified length."""
    # Use a repeated pattern to reach desired length
    base_text = "This is a test sentence for benchmarking the model performance. "
    repeated_text = base_text * (input_length // 10 + 1)

    # Tokenize and truncate to exact length
    tokens = tokenizer.encode(repeated_text, add_special_tokens=False)
    tokens = tokens[:input_length]

    # Pad if necessary
    if len(tokens) < input_length:
        tokens = tokens + [tokenizer.pad_token_id] * (input_length - len(tokens))

    # Create batch
    input_ids = torch.tensor([tokens] * batch_size, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    return input_ids, attention_mask


def benchmark_generation(model, tokenizer, args):
    """
    Benchmark generation performance.

    Measures TTFT and TPOT by:
    1. Running context encoding (prefill) separately
    2. Running token generation loop
    """
    print(f"\n{'='*60}")
    print(f"Benchmark Configuration:")
    print(f"  Input Length: {args.input_length}")
    print(f"  Output Length: {args.output_length}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Warmup Runs: {args.warmup_runs}")
    print(f"  Benchmark Runs: {args.benchmark_runs}")
    print(f"{'='*60}\n")

    # Create dummy input
    input_ids, attention_mask = create_dummy_input(tokenizer, args.input_length, args.batch_size)
    print(f"Input shape: {input_ids.shape}")

    generation_model = HuggingFaceGenerationAdapter(model)

    # Generation config
    if args.use_sample:
        gen_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            max_new_tokens=args.output_length,
        )
    else:
        gen_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=args.output_length,
        )

    max_length = args.input_length + args.output_length

    ttft_times = []
    tpot_times = []
    total_times = []

    # Warmup runs
    print(f"Running {args.warmup_runs} warmup iterations...")
    for i in range(args.warmup_runs):
        _ = generation_model.generate(
            input_ids,
            generation_config=gen_config,
            attention_mask=attention_mask,
            max_length=max_length,
        )
        print(f"  Warmup {i+1}/{args.warmup_runs} completed")

    # Benchmark runs
    print(f"\nRunning {args.benchmark_runs} benchmark iterations...")
    for i in range(args.benchmark_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start_time = time.perf_counter()

        outputs = generation_model.generate(
            input_ids,
            generation_config=gen_config,
            attention_mask=attention_mask,
            max_length=max_length,
        )

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()

        total_time = end_time - start_time
        output_tokens = outputs.shape[1] - args.input_length

        # Estimate TTFT and TPOT
        # Note: For accurate TTFT/TPOT, we'd need hooks into the model
        # Here we estimate based on total time
        # Typically TTFT takes a significant portion for long inputs
        estimated_ttft = total_time * 0.3  # Rough estimate
        estimated_decode_time = total_time * 0.7
        estimated_tpot = estimated_decode_time / max(output_tokens, 1)

        ttft_times.append(estimated_ttft)
        tpot_times.append(estimated_tpot)
        total_times.append(total_time)

        print(f"  Run {i+1}/{args.benchmark_runs}: Total={total_time:.3f}s, "
              f"Output tokens={output_tokens}, "
              f"Throughput={output_tokens/total_time:.2f} tok/s")

    # Calculate statistics
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")

    mean_total = np.mean(total_times)
    std_total = np.std(total_times)
    mean_ttft = np.mean(ttft_times)
    mean_tpot = np.mean(tpot_times)

    actual_output_tokens = outputs.shape[1] - args.input_length
    throughput = actual_output_tokens / mean_total

    print(f"\nTotal Generation Time:")
    print(f"  Mean: {mean_total*1000:.2f} ms")
    print(f"  Std:  {std_total*1000:.2f} ms")

    print(f"\nEstimated TTFT (Time To First Token):")
    print(f"  Mean: {mean_ttft*1000:.2f} ms")

    print(f"\nEstimated TPOT (Time Per Output Token):")
    print(f"  Mean: {mean_tpot*1000:.2f} ms")

    print(f"\nThroughput:")
    print(f"  {throughput:.2f} tokens/second")
    print(f"  {throughput * args.batch_size:.2f} tokens/second (batch adjusted)")

    print(f"\n{'='*60}")

    return {
        "input_length": args.input_length,
        "output_length": actual_output_tokens,
        "batch_size": args.batch_size,
        "mean_total_ms": mean_total * 1000,
        "std_total_ms": std_total * 1000,
        "estimated_ttft_ms": mean_ttft * 1000,
        "estimated_tpot_ms": mean_tpot * 1000,
        "throughput_tok_per_sec": throughput,
    }


def benchmark_prefill_only(model, tokenizer, args):
    """
    Benchmark prefill (context encoding) only.
    This gives accurate TTFT measurement.
    """
    print(f"\n{'='*60}")
    print(f"PREFILL BENCHMARK (TTFT)")
    print(f"{'='*60}")
    print(f"  Input Length: {args.input_length}")
    print(f"  Batch Size: {args.batch_size}")

    # Create dummy input
    input_ids, attention_mask = create_dummy_input(tokenizer, args.input_length, args.batch_size)
    print(f"  Input shape: {input_ids.shape}")

    generation_model = HuggingFaceGenerationAdapter(model)

    # Generate only 1 token to measure prefill time
    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=1,
    )

    max_length = args.input_length + 1

    # Warmup
    print(f"\nWarmup runs...")
    for i in range(args.warmup_runs):
        _ = generation_model.generate(
            input_ids,
            generation_config=gen_config,
            attention_mask=attention_mask,
            max_length=max_length,
        )
        print(f"  Warmup {i+1}/{args.warmup_runs}")

    # Benchmark
    ttft_times = []
    print(f"\nBenchmark runs...")
    for i in range(args.benchmark_runs):
        start_time = time.perf_counter()

        _ = generation_model.generate(
            input_ids,
            generation_config=gen_config,
            attention_mask=attention_mask,
            max_length=max_length,
        )

        end_time = time.perf_counter()
        ttft = end_time - start_time
        ttft_times.append(ttft)
        print(f"  Run {i+1}: TTFT = {ttft*1000:.2f} ms")

    mean_ttft = np.mean(ttft_times)
    std_ttft = np.std(ttft_times)

    print(f"\n{'='*60}")
    print(f"TTFT Results (Input Length = {args.input_length}):")
    print(f"  Mean: {mean_ttft*1000:.2f} ms")
    print(f"  Std:  {std_ttft*1000:.2f} ms")
    print(f"  Min:  {min(ttft_times)*1000:.2f} ms")
    print(f"  Max:  {max(ttft_times)*1000:.2f} ms")
    print(f"{'='*60}")

    return mean_ttft * 1000, std_ttft * 1000


def benchmark_decode_only(model, tokenizer, args, short_context=128):
    """
    Benchmark decode (token generation) with short context.
    This gives more accurate TPOT measurement.
    """
    print(f"\n{'='*60}")
    print(f"DECODE BENCHMARK (TPOT)")
    print(f"{'='*60}")
    print(f"  Context Length: {short_context}")
    print(f"  Output Length: {args.output_length}")
    print(f"  Batch Size: {args.batch_size}")

    # Create short input to minimize prefill time
    input_ids, attention_mask = create_dummy_input(tokenizer, short_context, args.batch_size)
    print(f"  Input shape: {input_ids.shape}")

    generation_model = HuggingFaceGenerationAdapter(model)

    gen_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.output_length,
    )

    max_length = short_context + args.output_length

    # Warmup
    print(f"\nWarmup runs...")
    for i in range(args.warmup_runs):
        _ = generation_model.generate(
            input_ids,
            generation_config=gen_config,
            attention_mask=attention_mask,
            max_length=max_length,
        )
        print(f"  Warmup {i+1}/{args.warmup_runs}")

    # Benchmark
    decode_times = []
    tpot_times = []
    print(f"\nBenchmark runs...")
    for i in range(args.benchmark_runs):
        start_time = time.perf_counter()

        outputs = generation_model.generate(
            input_ids,
            generation_config=gen_config,
            attention_mask=attention_mask,
            max_length=max_length,
        )

        end_time = time.perf_counter()
        total_time = end_time - start_time
        output_tokens = outputs.shape[1] - short_context

        # Subtract estimated prefill time (for short context, it's small)
        estimated_prefill = total_time * 0.05  # ~5% for short context
        decode_time = total_time - estimated_prefill
        tpot = decode_time / max(output_tokens, 1)

        decode_times.append(decode_time)
        tpot_times.append(tpot)
        print(f"  Run {i+1}: Decode={decode_time*1000:.2f}ms, "
              f"TPOT={tpot*1000:.2f}ms, Tokens={output_tokens}")

    mean_tpot = np.mean(tpot_times)
    std_tpot = np.std(tpot_times)

    print(f"\n{'='*60}")
    print(f"TPOT Results (Output Length = {args.output_length}):")
    print(f"  Mean: {mean_tpot*1000:.2f} ms")
    print(f"  Std:  {std_tpot*1000:.2f} ms")
    print(f"  Min:  {min(tpot_times)*1000:.2f} ms")
    print(f"  Max:  {max(tpot_times)*1000:.2f} ms")
    print(f"  Throughput: {1000/mean_tpot:.2f} tokens/s")
    print(f"{'='*60}")

    return mean_tpot * 1000, std_tpot * 1000


def main():
    args = parse_args()

    torch.manual_seed(0)

    if args.compile:
        print(f"\n{'='*60}")
        print("COMPILING MODEL")
        print(f"{'='*60}")

        # Calculate max_length for the benchmark
        max_length = args.input_length + args.output_length

        neuron_config = MoENeuronConfig(
            tp_degree=args.tp_degree,
            batch_size=args.batch_size,
            max_context_length=args.input_length,  # Support full input length for prefill
            seq_len=max_length,  # Total sequence length including generation
            on_device_sampling_config=OnDeviceSamplingConfig(
                do_sample=args.use_sample,
                temperature=0.6,
                top_k=20,
                top_p=0.95
            ) if args.use_sample else OnDeviceSamplingConfig(),
            enable_bucketing=False,
            flash_decoding_enabled=False,
            save_sharded_checkpoint=False,
        )

        config = Qwen3MoeInferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(args.model_path),
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        print(f"\nCompiling with:")
        print(f"  TP Degree: {args.tp_degree}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Max Context Length: {args.input_length}")
        print(f"  Max Sequence Length: {max_length}")

        print("\nCompiling and saving model...")
        model = NeuronQwen3MoeForCausalLM(args.model_path, config)
        model.compile(args.traced_model_path)
        tokenizer.save_pretrained(args.traced_model_path)
        print("Compilation complete!")

    # Load model
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    print(f"Loading from: {args.traced_model_path}")

    model = NeuronQwen3MoeForCausalLM(args.traced_model_path)
    model.load(args.traced_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.traced_model_path)

    print("Model loaded successfully!")
    print(f"  Max context length: {model.config.neuron_config.max_context_length}")
    print(f"  Max sequence length: {model.config.neuron_config.seq_len}")

    # Run benchmarks
    print("\n" + "="*60)
    print("STARTING BENCHMARKS")
    print("="*60)

    # 1. Prefill benchmark (TTFT)
    ttft_mean, ttft_std = benchmark_prefill_only(model, tokenizer, args)

    # 2. Decode benchmark (TPOT)
    tpot_mean, tpot_std = benchmark_decode_only(model, tokenizer, args)

    # 3. Full generation benchmark
    results = benchmark_generation(model, tokenizer, args)

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"TP Degree: {args.tp_degree}")
    print(f"Batch Size: {args.batch_size}")
    print(f"\nPrefill (Input Length = {args.input_length}):")
    print(f"  TTFT: {ttft_mean:.2f} ms (+/- {ttft_std:.2f} ms)")
    print(f"\nDecode (Output Length = {args.output_length}):")
    print(f"  TPOT: {tpot_mean:.2f} ms (+/- {tpot_std:.2f} ms)")
    print(f"  Decode Throughput: {1000/tpot_mean:.2f} tokens/s")
    print(f"\nEnd-to-End:")
    print(f"  Total Time: {results['mean_total_ms']:.2f} ms")
    print(f"  Overall Throughput: {results['throughput_tok_per_sec']:.2f} tokens/s")
    print("="*60)


if __name__ == "__main__":
    main()
