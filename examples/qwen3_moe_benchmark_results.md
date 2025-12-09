# Qwen3-235B-A22B MoE Benchmark Results

**Date**: 2025-12-05
**Platform**: AWS Trainium2 (trn2)

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-235B-A22B |
| Model Type | Mixture of Experts (MoE) |
| TP Degree | 32 |
| Input Length | 10240 tokens |
| Output Length | 256 tokens |
| Max Sequence Length | 10496 tokens |
| Warmup Runs | 3 |
| Benchmark Runs | 5 |

---

## Batch Size Comparison

### Summary Table

| Metric | Batch Size 1 | Batch Size 4 | Notes |
|--------|-------------|--------------|-------|
| **TTFT (Prefill)** | 2678.97 ms | 6947.41 ms | Time to first token |
| **TPOT (Decode)** | 27.90 ms | 67.20 ms | Per token decode time |
| **Decode Throughput** | 35.85 tok/s | 14.88 tok/s | Per-sample throughput |
| **Batch Throughput** | 35.85 tok/s | 59.52 tok/s | Total tokens/s |
| **E2E Throughput** | 33.97 tok/s | 52.32 tok/s | Batch adjusted |

**Note**: `batch_size=16` failed due to HBM memory limit (21 GB needed vs 16 GB available per Neuron Core).

---

## Batch Size 1 Results

**Log File**: `/tmp/qwen3_moe_benchmark_10240.log`

### Compilation Time

| Phase | Time |
|-------|------|
| HLO Generation | 44.3 s |
| token_generation_model Compilation | 360.5 s |
| context_encoding_model Compilation | 190.8 s |
| Weight Layout Optimization | ~1.1 s |
| **Total Build Time** | **669.5 s (~11 min)** |

### Prefill Benchmark (TTFT)

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

### Decode Benchmark (TPOT)

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

### End-to-End Generation

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

---

## Batch Size 4 Results

**Log File**: `/tmp/qwen3_moe_benchmark_10240_bs4.log`

### Compilation Time

| Phase | Time |
|-------|------|
| HLO Generation | 50.5 s |
| token_generation_model Compilation | 688.8 s (~11.5 min) |
| context_encoding_model Compilation | ~620 s (~10 min) |
| Weight Layout Optimization | ~1.7 s |
| **Total Build Time** | **~1450 s (~24 min)** |

### Prefill Benchmark (TTFT)

| Metric | Value |
|--------|-------|
| **TTFT Mean** | **6947.41 ms** |
| TTFT Std | 2.07 ms |

### Decode Benchmark (TPOT)

| Metric | Value |
|--------|-------|
| **TPOT Mean** | **67.20 ms** |
| TPOT Std | 0.02 ms |
| TPOT Min | 67.18 ms |
| TPOT Max | 67.25 ms |
| **Decode Throughput** | **14.88 tokens/s** (per sample) |

**Raw Results:**
```
Run 1: Decode=17201.45ms, TPOT=67.19ms, Tokens=256
Run 2: Decode=17215.34ms, TPOT=67.25ms, Tokens=256
Run 3: Decode=17202.62ms, TPOT=67.20ms, Tokens=256
Run 4: Decode=17203.19ms, TPOT=67.20ms, Tokens=256
Run 5: Decode=17198.00ms, TPOT=67.18ms, Tokens=256
```

### End-to-End Generation

| Metric | Value |
|--------|-------|
| **Total Time Mean** | **19570.59 ms** |
| Total Time Std | 25.05 ms |
| **Per-Sample Throughput** | **13.08 tokens/s** |
| **Batch Throughput** | **52.32 tokens/s** |

**Raw Results:**
```
Run 1/5: Total=19.608s, Output tokens=256, Throughput=13.06 tok/s
Run 2/5: Total=19.546s, Output tokens=256, Throughput=13.10 tok/s
Run 3/5: Total=19.545s, Output tokens=256, Throughput=13.10 tok/s
Run 4/5: Total=19.565s, Output tokens=256, Throughput=13.08 tok/s
Run 5/5: Total=19.590s, Output tokens=256, Throughput=13.07 tok/s
```

---

## Batch Size 16 Results

**Status**: FAILED - OOM Error

**Error Message:**
```
Memory requirement exceeds target architecture's HBM limit.
Needed 23045971136 bytes (21 GB) vs. available 17179869184 bytes (16 GB).
```

The HBM limit per Neuron Core (16 GB) prevents batch_size=16 from fitting with 10240 token context length.

---

## Analysis

### 1. Latency Scaling with Batch Size

| Batch Size | TTFT | TPOT | Scaling Factor (vs bs=1) |
|------------|------|------|--------------------------|
| 1 | 2.68 s | 27.9 ms | 1.0x |
| 4 | 6.95 s | 67.2 ms | 2.4-2.6x |

The latency scales sub-linearly with batch size, indicating good batching efficiency.

### 2. Throughput Improvements

- **Batch Size 1**: 33.97 tok/s total
- **Batch Size 4**: 52.32 tok/s total (batch adjusted)
- **Throughput Gain**: **~1.54x** improvement with 4x batch size

### 3. Memory Constraints

- Each Neuron Core has 16 GB HBM
- Batch Size 16 requires ~21 GB, exceeding the limit
- Maximum practical batch size for 10240 context: **batch_size=4 to 8**

### 4. Prefill Throughput

| Batch Size | TTFT | Tokens Processed | Prefill Throughput |
|------------|------|------------------|-------------------|
| 1 | 2.68 s | 10,240 | 3,821 tok/s |
| 4 | 6.95 s | 40,960 | 5,893 tok/s |

Batch size 4 achieves ~1.54x better prefill throughput.

### 5. Consistency

Very low standard deviation across all runs indicates stable and predictable performance on Trainium2.

---

## Benchmark Commands

**Batch Size 1:**
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

**Batch Size 4:**
```bash
python3 benchmark_qwen3_moe.py \
    --compile \
    --input-length 10240 \
    --output-length 256 \
    --tp-degree 32 \
    --batch-size 4 \
    --warmup-runs 3 \
    --benchmark-runs 5 \
    --traced-model-path /home/ubuntu/traced_model/Qwen3-235B-A22B-benchmark-bs4/
```

**Batch Size 8 ([ERROR] [NCC_VRF009] Memory requirement exceeds target architecture's HBM limit. Needed 19004423264 bytes (17 GB) vs. available 17179869184 bytes (16 GB). TIP: Consider using smaller batches or applying model parallelism):**
```bash
python3 benchmark_qwen3_moe.py \
    --compile \
    --input-length 10240 \
    --output-length 256 \
    --tp-degree 32 \
    --batch-size 8 \
    --warmup-runs 3 \
    --benchmark-runs 5 \
    --traced-model-path /home/ubuntu/traced_model/Qwen3-235B-A22B-benchmark-bs8/
```

**Batch Size 16:**
```bash
python3 benchmark_qwen3_moe.py \
    --compile \
    --input-length 4096 \
    --output-length 256 \
    --tp-degree 64 \
    --moe-tp-degree 2 \
    --moe-ep-degree 32 \
    --batch-size 16 \
    --warmup-runs 3 \
    --benchmark-runs 5 \
    --traced-model-path /home/ubuntu/traced_model/Qwen3-235B-A22B-benchmark-bs16/
```

**Batch Size 16 (Qwen3-30B-A3B):**
```bash
python3 benchmark_qwen3_moe.py \
    --model-path /home/ubuntu/model_hf/Qwen3-30B-A3B/ \
    --compile \
    --input-length 10240 \
    --output-length 256 \
    --tp-degree 8 \
    --moe-tp-degree 2 \
    --moe-ep-degree 4 \
    --batch-size 16 \
    --warmup-runs 3 \
    --benchmark-runs 5 \
    --traced-model-path /home/ubuntu/traced_model/Qwen3-30B-A3B-benchmark-bs16/
```

Script location: `/home/ubuntu/neuronx-distributed-inference/examples/benchmark_qwen3_moe.py`
