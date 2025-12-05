# Qwen3-235B-A22B MoE Benchmark Results

**Date**: 2025-12-05
**Platform**: AWS Trainium2 (trn2)
**Log File**: `/tmp/qwen3_moe_benchmark_10240.log`

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-235B-A22B |
| Model Type | Mixture of Experts (MoE) |
| TP Degree | 32 |
| Batch Size | 1 |
| Input Length | 10240 tokens |
| Output Length | 256 tokens |
| Max Sequence Length | 10496 tokens |
| Warmup Runs | 3 |
| Benchmark Runs | 5 |

## Compilation Time

| Phase | Time |
|-------|------|
| HLO Generation | 44.3 s |
| token_generation_model Compilation | 360.5 s |
| context_encoding_model Compilation | 190.8 s |
| Weight Layout Optimization | ~1.1 s |
| **Total Build Time** | **669.5 s (~11 min)** |

## Benchmark Results

### 1. Prefill Benchmark (TTFT - Time To First Token)

Measures the latency to process input context and generate the first token.

| Metric | Value |
|--------|-------|
| **TTFT Mean** | **2678.97 ms** |
| TTFT Std | 0.59 ms |
| TTFT Min | 2678.45 ms |
| TTFT Max | 2680.04 ms |

**Raw Results:**
```
Run 1: TTFT = 2680.04 ms
Run 2: TTFT = 2678.45 ms
Run 3: TTFT = 2679.10 ms
Run 4: TTFT = 2678.45 ms
Run 5: TTFT = 2678.79 ms
```

### 2. Decode Benchmark (TPOT - Time Per Output Token)

Measures the latency per token during auto-regressive decoding (with short 128-token context).

| Metric | Value |
|--------|-------|
| **TPOT Mean** | **27.90 ms** |
| TPOT Std | 0.10 ms |
| TPOT Min | 27.77 ms |
| TPOT Max | 28.05 ms |
| **Decode Throughput** | **35.85 tokens/s** |

**Raw Results:**
```
Run 1: Decode=7179.57ms, TPOT=28.05ms, Tokens=256
Run 2: Decode=7125.33ms, TPOT=27.83ms, Tokens=256
Run 3: Decode=7110.39ms, TPOT=27.77ms, Tokens=256
Run 4: Decode=7161.65ms, TPOT=27.98ms, Tokens=256
Run 5: Decode=7130.40ms, TPOT=27.85ms, Tokens=256
```

### 3. End-to-End Generation Benchmark

Full generation with 10240 input tokens and 256 output tokens.

| Metric | Value |
|--------|-------|
| **Total Time Mean** | **7535.11 ms** |
| Total Time Std | 20.92 ms |
| **Overall Throughput** | **33.97 tokens/s** |

**Raw Results:**
```
Run 1/5: Total=7.501s, Output tokens=256, Throughput=34.13 tok/s
Run 2/5: Total=7.532s, Output tokens=256, Throughput=33.99 tok/s
Run 3/5: Total=7.532s, Output tokens=256, Throughput=33.99 tok/s
Run 4/5: Total=7.546s, Output tokens=256, Throughput=33.92 tok/s
Run 5/5: Total=7.565s, Output tokens=256, Throughput=33.84 tok/s
```

## Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **TTFT** | 2.68 s | For 10240 token context |
| **TPOT** | 27.9 ms | Per output token |
| **Decode Throughput** | 35.85 tok/s | Pure decode phase |
| **E2E Throughput** | 33.97 tok/s | Including prefill overhead |

## Analysis

1. **TTFT (2.68s)**: Reasonable for processing 10240 tokens of context. The prefill phase processes ~3820 tokens/second.

2. **TPOT (27.9ms)**: Each token generation takes approximately 28ms, corresponding to a decode throughput of ~36 tokens/second.

3. **End-to-End Throughput (~34 tok/s)**: Slightly lower than decode-only throughput due to prefill overhead (2.68s out of 7.5s total time).

4. **Consistency**: Very low standard deviation across runs indicates stable and predictable performance.

## Benchmark Script

The benchmark was run using:
```bash
python3 benchmark_qwen3_moe.py \
    --compile \
    --input-length 10240 \
    --output-length 256 \
    --tp-degree 32 \
    --batch-size 1 \
    --warmup-runs 3 \
    --benchmark-runs 5
```

Script location: `/home/ubuntu/neuronx-distributed-inference/examples/benchmark_qwen3_moe.py`
