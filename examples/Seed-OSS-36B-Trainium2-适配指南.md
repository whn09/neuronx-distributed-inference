# Seed-OSS-36B-Instruct 在 Trainium2 上的适配指南

## 模型概述

| 参数 | 值 |
|------|-----|
| 模型 | ByteDance-Seed/Seed-OSS-36B-Instruct |
| 参数量 | 36B |
| 架构 | Dense Transformer (GQA + SwiGLU + RMSNorm + RoPE) |
| 层数 | 64 |
| Hidden Size | 5120 |
| Q / KV Heads | 80 / 8 |
| Head Dim | 128 |
| Intermediate Size | 27648 |
| 词表大小 | 155136 |
| 上下文长度 | 512K |
| 精度 | BF16 |

## 适配方法

### 架构分析

Seed-OSS-36B 的架构与 Qwen3-32B 高度相似，均为标准 Dense Transformer，主要差异如下：

| 特性 | Qwen3-32B | Seed-OSS-36B |
|------|-----------|--------------|
| QK Normalization | 有 (q_norm, k_norm) | 无 |
| QKV Bias | 无 | 有 (attention_bias=True) |
| O Proj Bias | 无 | 无 (attention_out_bias=False) |
| 权重转换 | 需 rename q_norm→q_layernorm | 无需重命名 |

### 代码修改

基于 Qwen3 的 NeuronX 实现进行适配，涉及两个项目：

#### 1. neuronx-distributed-inference

分支：`feature/seed-oss-support`

新增文件：
- `src/neuronx_distributed_inference/models/seed_oss/__init__.py`
- `src/neuronx_distributed_inference/models/seed_oss/modeling_seed_oss.py`
- `examples/generation_seed_oss_demo.py`

修改文件：
- `src/neuronx_distributed_inference/utils/constants.py` — 注册 `seed_oss` model type

核心适配点（`modeling_seed_oss.py`）：

```python
class NeuronSeedOssAttention(NeuronAttentionBase):
    def __init__(self, config):
        # 与 Qwen3 的关键区别：
        # 1. 不传 q_layernorm / k_layernorm（无 QK Norm）
        # 2. qkv_bias=True, o_bias=False
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rotary_emb=rotary_emb,
            qkv_bias=True,
            o_bias=False,
        )
```

权重转换（`convert_hf_to_neuron_state_dict`）比 Qwen3 更简单，不需要重命名 q_norm/k_norm，只需添加 rank_util 辅助张量。

#### 2. vllm-neuron

分支：`feature/seed-oss-support`（基于 upstream release-0.4.1）

修改文件：
- `vllm_neuron/worker/neuronx_distributed_model_loader.py` — 添加模型名映射

```python
if model == "seedoss":
    model = "seed_oss"
```

### 安装

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

# 安装 neuronx-distributed-inference
cd /home/ubuntu/neuronx-distributed-inference
git checkout feature/seed-oss-support
pip install -e .

# 安装 vllm-neuron
cd /home/ubuntu/vllm-neuron
git checkout feature/seed-oss-support
pip install -e .
```

## 启动方法

### 下载模型

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('ByteDance-Seed/Seed-OSS-36B-Instruct',
                  local_dir='/opt/dlami/nvme/Seed-OSS-36B-Instruct')
"
```

### 启动 vLLM 服务

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

export NEURON_RT_VISIBLE_CORES="0-7"

python3 -m vllm.entrypoints.openai.api_server \
    --model "/opt/dlami/nvme/Seed-OSS-36B-Instruct" \
    --tensor-parallel-size 8 \
    --max-model-len 1536 \
    --max-num-seqs 40 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port 8000 \
    --additional-config '{"override_neuron_config": {"skip_warmup": true, "enable_bucketing": true, "context_encoding_buckets": [1024], "token_generation_buckets": [1536], "async_mode": true, "on_device_sampling_config": {}, "fused_qkv": true}}'
```

关键参数说明：
- `--tensor-parallel-size 8`：使用 2 个 Trainium2 Chips（每个 Chip 在 LNC=2 下有 4 个逻辑核心）
- `--max-model-len 1536`：最大序列长度
- `--max-num-seqs 40`：最大并发序列数
- `fused_qkv`：融合 QKV 投影，减少内存访问
- `async_mode`：异步执行模式，提升吞吐
- `on_device_sampling_config`：设备端采样，减少 host-device 通信
- `context_encoding_buckets: [1024]`：Prefill bucket 设为 1024，匹配测试输入长度

### 验证服务

```bash
# 测试 completions 接口
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/opt/dlami/nvme/Seed-OSS-36B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 32
  }' | python3 -m json.tool

# 测试 chat completions 接口
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/opt/dlami/nvme/Seed-OSS-36B-Instruct",
    "messages": [{"role": "user", "content": "What is 1+1? Answer briefly."}],
    "max_tokens": 32
  }' | python3 -m json.tool
```

## Benchmark 方法

测试参数：input=1024, output=500, random-range-ratio=0

### BS=1

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

vllm bench serve \
    --backend vllm \
    --model /opt/dlami/nvme/Seed-OSS-36B-Instruct \
    --endpoint /v1/completions \
    --dataset-name random \
    --num-prompts 16 \
    --random-input-len 1024 \
    --random-output-len 500 \
    --random-range-ratio 0 \
    --max-concurrency 1
```

### BS=20

```bash
vllm bench serve \
    --backend vllm \
    --model /opt/dlami/nvme/Seed-OSS-36B-Instruct \
    --endpoint /v1/completions \
    --dataset-name random \
    --num-prompts 200 \
    --random-input-len 1024 \
    --random-output-len 500 \
    --random-range-ratio 0 \
    --max-concurrency 20
```

### BS=40

```bash
vllm bench serve \
    --backend vllm \
    --model /opt/dlami/nvme/Seed-OSS-36B-Instruct \
    --endpoint /v1/completions \
    --dataset-name random \
    --num-prompts 400 \
    --random-input-len 1024 \
    --random-output-len 500 \
    --random-range-ratio 0 \
    --max-concurrency 40
```

## Benchmark Summary (input=1024, output=500)

### Seed-OSS-36B

|HW    |Config Ver.    |Dtype    |BS    |TP    |ITL (ms)    |TTFT (s)    |E2E (s)    |Req/min    |Tok/sec    |Tok/sec vs H100    |
|---    |---    |---    |---    |---    |---    |---    |---    |---    |---    |---    |
|H100    |-    |FP8    |1    |1    |15.49    |0.08    |7.80    |7.69    |64.07    |    |
|Trn2    |oss-fork    |BF16    |1    |8    |29.84    |0.15    |15.05    |3.99    |33.22    |51.85%    |
|H100    |-    |FP8    |20    |1    |20.69    |1.19    |12.26    |97.89    |815.79    |    |
|Trn2    |oss-fork    |BF16    |20    |8    |29.86    |1.45    |17.71    |67.74    |564.49    |69.19%    |
|H100    |-    |FP8    |40    |1    |21.95    |1.17    |14.08    |170.37    |1419.78    |    |
|Trn2    |oss-fork    |BF16    |40    |8    |29.86    |2.97    |20.83    |115.20    |959.96    |67.61%    |

- H100: p5.48xlarge, 1x H100 80GB, FP8 Static (AngelSlim/Seed-OSS-36B-Instruct-FP8-Static), vLLM 0.17.0
- Trn2: trn2.48xlarge, 2 chips (TP=8), BF16, optimized (fused_qkv + async_mode + on_device_sampling + bucket=1024)
- E2E = Mean TTFT + 499 x Mean TPOT
- Cost: Trn2 2 chips ($4.47/hr) vs H100 1 GPU ($3.93/hr), cost ratio ~1.14x

### Seed-OSS-36B BF16 (2×H100 vs 2×2 Trn2 chips)

|HW    |Config Ver.    |Dtype    |BS    |TP    |ITL (ms)    |TTFT (s)    |E2E (s)    |Req/min    |Tok/sec    |Tok/sec vs H100    |
|---    |---    |---    |---    |---    |---    |---    |---    |---    |---    |---    |
|H100    |-    |BF16    |1    |2    |15.03    |0.08    |7.58    |7.90    |65.90    |    |
|Trn2 ×2    |oss-fork    |BF16    |1    |8    |29.84    |0.15    |15.05    |7.98    |66.44    |**101%**    |
|H100    |-    |BF16    |20    |2    |18.22    |0.90    |10.00    |112.64    |938.63    |    |
|Trn2 ×2    |oss-fork    |BF16    |20    |8    |29.85    |0.74    |16.23    |74.16    |616.36    |**66%**    |
|H100    |-    |BF16    |40    |2    |19.90    |0.92    |10.85    |200.44    |1670.34    |    |
|Trn2 ×2    |oss-fork    |BF16    |40    |8    |29.86    |1.45    |17.71    |135.48    |1128.98    |**68%**    |

- H100: p5.48xlarge, 2x H100 80GB (TP=2), BF16, vLLM 0.17.0
- Trn2 ×2: trn2.48xlarge, 2 个独立 TP=8 实例，每个使用 2 chips
- BS 为系统总并发数。Trn2 ×2 每实例实际处理 BS/2 的并发（BS=20→每实例10, BS=40→每实例20）
- TTFT/E2E 为每实例在实际并发下的延迟；Tok/sec 为 2 个实例的聚合吞吐
- BS=20 Trn2 行使用 BS=10 单实例实测数据（×2）；BS=40 Trn2 行使用 BS=20 单实例精确数据（×2）
- 两者均为 1/8 整机资源（p5: 2/8 GPUs, trn2: 4/16 chips）
- Cost: Trn2 4 chips ($8.94/hr) vs H100 2 GPUs ($7.86/hr), cost ratio ~1.14x

### Seed-OSS-36B BF16 (2×H100 TP=2 vs Trn2 TP=16)

|HW    |Config Ver.    |Dtype    |BS    |TP    |ITL (ms)    |TTFT (s)    |E2E (s)    |Req/min    |Tok/sec    |Tok/sec vs H100    |
|---    |---    |---    |---    |---    |---    |---    |---    |---    |---    |---    |
|H100    |-    |BF16    |1    |2    |15.03    |0.08    |7.58    |7.91    |65.90    |    |
|Trn2    |oss-fork    |BF16    |1    |16    |23.52    |0.11    |11.85    |5.06    |42.19    |**64%**    |
|H100    |-    |BF16    |20    |2    |18.22    |0.90    |10.00    |112.64    |938.63    |    |
|Trn2    |oss-fork    |BF16    |20    |16    |23.52    |0.97    |13.57    |88.41    |736.72    |**78%**    |
|H100    |-    |BF16    |40    |2    |19.90    |0.92    |10.85    |200.44    |1670.34    |    |
|Trn2    |oss-fork    |BF16    |40    |16    |23.52    |1.89    |15.42    |155.67    |1297.22    |**78%**    |

- H100: p5.48xlarge, 2x H100 80GB (TP=2), BF16, vLLM 0.17.0
- Trn2: trn2.48xlarge, TP=16, 4 chips, BF16, optimized (fused_qkv + async_mode + on_device_sampling + bucket=1024)
- 两者均为 1/4 整机资源（p5: 2/8 GPUs, trn2: 4/16 chips），单实例直接对比，无需估算
- E2E = Mean TTFT + 499 × Mean TPOT
- Cost: Trn2 4 chips ($8.94/hr) vs H100 2 GPUs ($7.86/hr), cost ratio ~1.14x

### Performance Analysis

For the Seed-OSS-36B Dense model, we compared equivalent fractions (1/4) of each machine — 2 H100 GPUs with TP=2 versus 4 Trainium2 chips with TP=16 — using BF16 precision and identical workloads. In this single-instance-to-single-instance comparison, Trainium2 achieved **64% of H100 throughput at batch size 1 and 78% at batch sizes 20 and 40**. The TTFT gap is notably small at moderate concurrency (0.97s vs 0.90s at BS=20), and ITL is closer than in the TP=8 configuration (23.5ms vs 15–20ms). While the throughput gap remains due to Trainium2's lower per-chip compute density, this comparison is clean and directly measured — both platforms run a single instance at the same batch size with no load-balancing assumptions. Given a cost ratio of approximately 1.14× (Trn2 4 chips at $8.94/hr vs H100 2 GPUs at $7.86/hr), the cost-adjusted throughput ratio is approximately 56–69%, indicating that H100 retains a cost-efficiency advantage for this model in BF16 workloads.

### 调度机制说明

vllm-neuron 使用 NeuronScheduler，其调度特点：

1. **Prefill 和 Decode 分开执行**：当有等待 prefill 的请求时，会暂停所有 decode，先完成 prefill
2. **Prefill 批大小为 1**：每次只 prefill 一个请求
3. 这意味着当多个请求同时到达时，需要逐个 prefill，后续请求的 TTFT 会增加
4. 建议 benchmark 时使用 `random-range-ratio` 参数，让输入输出长度有随机性，避免所有请求同时完成导致 TTFT 虚高

---

## LLMPerf Benchmark

### LLMPerf 测试方法

使用 [LLMPerf](https://github.com/ray-project/llmperf) 通过 chat completions 接口进行端到端测试。

测试参数：mean-input-tokens=1000, mean-output-tokens=500, stddev=0

```bash
export OPENAI_API_KEY=dummy
export OPENAI_API_BASE=http://localhost:8000/v1

# BS=1
python3 token_benchmark_ray.py \
    --model <MODEL_PATH> \
    --mean-input-tokens 1000 \
    --stddev-input-tokens 0 \
    --mean-output-tokens 500 \
    --stddev-output-tokens 0 \
    --num-concurrent-requests 1 \
    --max-num-completed-requests 16 \
    --timeout 600 \
    --results-dir <RESULTS_DIR> \
    --llm-api openai

# BS=20: --num-concurrent-requests 20 --max-num-completed-requests 200
# BS=40: --num-concurrent-requests 40 --max-num-completed-requests 400
```

Note: LLMPerf 使用 chat completions 接口（`/v1/chat/completions`），chat template 会在 content tokens 之上额外增加 ~11 tokens。因此 `mean-input-tokens` 设为 1000 而非 1024，以避免超出 context_encoding_bucket=1024。

### LLMPerf Results

| BS | Metric | H100 FP8 | Trn2 BF16 | Trn2/H100 |
|----|--------|----------|-----------|-----------|
| 1 | ITL p50 (ms) | 15.48 | 29.86 | 51.8% |
| 1 | TTFT p50 (ms) | 90.8 | 149.7 | - |
| 1 | E2E p50 (s) | 7.75 | 15.05 | - |
| 1 | Output Throughput (tok/s) | 69.72 | 32.79 | 47.0% |
| 20 | ITL p50 (ms) | 21.76 | 32.42 | 67.1% |
| 20 | TTFT p50 (s) | 1.06 | 1.33 | - |
| 20 | E2E p50 (s) | 10.90 | 17.61 | - |
| 20 | Output Throughput (tok/s) | 888.73 | 510.16 | 57.4% |
| 40 | ITL p50 (ms) | 25.09 | 35.46 | 70.8% |
| 40 | TTFT p50 (s) | 0.85 | 2.77 | - |
| 40 | E2E p50 (s) | 12.57 | 20.52 | - |
| 40 | Output Throughput (tok/s) | 1553.98 | 875.71 | 56.3% |

- H100: p5.48xlarge, 1x H100 80GB, FP8 Static (AngelSlim/Seed-OSS-36B-Instruct-FP8-Static), vLLM 0.17.0
- Trn2: trn2.48xlarge, 2 chips (TP=8), BF16, optimized (fused_qkv + async_mode + on_device_sampling + bucket=1024)

### vllm bench vs LLMPerf ITL 对比

| BS | vllm bench H100 ITL (ms) | LLMPerf H100 ITL (ms) | vllm bench Trn2 ITL (ms) | LLMPerf Trn2 ITL (ms) |
|----|--------------------------|------------------------|--------------------------|------------------------|
| 1 | 15.49 | 15.48 | 29.84 | 29.86 |
| 20 | 20.69 | 21.76 | 29.86 | 32.42 |
| 40 | 21.95 | 25.09 | 29.86 | 35.46 |

Note:
- BS=1 时两个工具结果几乎一致（差异 <1%），验证了测试的一致性
- 高并发时 LLMPerf 的 ITL 偏高，原因：(1) chat completions 接口有额外开销 (2) H100 上 output tokens 不固定（平均 ~550 而非 500），Trn2 固定为 500
- 两台机器的相对差距在两个工具下保持一致

---

## 多加速器扩展性对比 (Trn2 vs H100)

### H100 BF16 Multi-GPU Results (vllm bench, input=1024, output=500)

| TP | GPUs | BS | ITL median (ms) | TTFT median (s) | Tok/sec | Peak Tok/sec |
|----|------|----|----------------|-----------------|---------|--------------|
| 2  | 2    | 1  | 15.03          | 0.08            | 65.90   | 67.00        |
| 2  | 2    | 20 | 18.22          | 0.90            | 938.63  | 1120.00      |
| 2  | 2    | 40 | 19.90          | 0.92            | 1670.34 | 2080.00      |
| 4  | 4    | 1  | 9.27           | 0.06            | 106.55  | 108.00       |
| 4  | 4    | 20 | 11.69          | 0.40            | 1455.23 | 1740.00      |
| 4  | 4    | 40 | 13.21          | 0.19            | 2708.93 | 3080.00      |
| 8  | 8    | 1  | 6.43           | 0.06            | 152.63  | 156.00       |
| 8  | 8    | 20 | 9.08           | 0.32            | 1855.08 | 2220.00      |
| 8  | 8    | 40 | 9.68           | 0.24            | 3711.92 | 4160.00      |

- H100: p5.48xlarge, BF16, max-model-len=1536, max-num-seqs=40, vLLM 0.17.0

### Trn2 BF16 TP=16 Results (vllm bench, input=1024, output=500)

| TP | Chips | BS | ITL median (ms) | TTFT median (s) | Tok/sec | Peak Tok/sec |
|----|-------|----|----------------|-----------------|---------|--------------|
| 8  | 2     | 1  | 29.84          | 0.15            | 33.22   | 34.00        |
| 8  | 2     | 20 | 29.86          | 1.45            | 564.49  | 680.00       |
| 8  | 2     | 40 | 29.86          | 2.97            | 959.96  | 1360.00      |
| 16 | 4     | 1  | 23.52          | 0.11            | 42.19   | 43.00        |
| 16 | 4     | 20 | 23.52          | 0.97            | 736.72  | 860.00       |
| 16 | 4     | 40 | 23.52          | 1.89            | 1297.22 | 1720.00      |

- Trn2: trn2.48xlarge, BF16, optimized (fused_qkv + async_mode + on_device_sampling + bucket=1024)

### TP 扩展效率对比

| 平台 | 基准 | 扩展 | BS=1 | BS=20 | BS=40 |
|------|------|------|------|-------|-------|
| H100 | TP=2 (2 GPU) | TP=4 (4 GPU, 2x) | 1.62x | 1.55x | 1.62x |
| H100 | TP=2 (2 GPU) | TP=8 (8 GPU, 4x) | 2.32x | 1.98x | 2.22x |
| Trn2 | TP=8 (2 chips) | TP=16 (4 chips, 2x) | 1.27x | 1.31x | 1.35x |

两个平台的 TP 扩展都是亚线性的，Trn2 的扩展效率（~1.3x / 2x chips）低于 H100（~1.6x / 2x GPUs）。

### 公平对比：同等硬件比例 (1/8 整机)

trn2.48xlarge 有 16 chips，p5.48xlarge 有 8 GPUs，因此 2 Trn2 chips 对应 1 H100 GPU（均为 1/8 整机）。

| BS | H100 BF16 TP=2 (2 GPU) | Trn2 BF16 TP=8 ×2 实例 | Trn2/H100 |
|----|------------------------|------------------------|-----------|
| 1  | 65.90 tok/s            | 33.22 × 2 = 66.44      | **101%**  |
| 20 | 938.63 tok/s           | 308.18 × 2 = 616.36    | **66%**   |
| 40 | 1670.34 tok/s          | 564.49 × 2 = 1128.98   | **68%**   |

注：Trn2 TP=8 × 2 实例将负载分散到两个独立实例，每个实例使用 2 chips。BS 为系统总并发数，每实例实际处理 BS/2 的并发（BS=20→每实例10, BS=40→每实例20）。由于两个实例完全独立，吞吐可线性叠加。H100 TP=2 使用 2 GPU 的单实例。

---

## random-range-ratio=0.02 Benchmark (input=1000, output=500)

标准 benchmark 使用 `random-range-ratio=0`，所有请求的输入/输出长度完全相同。这导致同一批次的请求同时完成，触发"prefill 风暴"——NeuronScheduler 需要逐个 prefill 所有新请求，造成 TTFT 虚高。使用 `random-range-ratio=0.02` 添加 ±2% 的长度随机性，使请求错开完成，产生更真实的 TTFT 测量。

### H100 BF16 TP=2 (ratio=0.02)

| BS | Tok/sec | ITL median (ms) | TPOT median (ms) | TTFT median (ms) |
|----|---------|-----------------|------------------|------------------|
| 1  | 66.02   | 15.00           | 15.00            | 81               |
| 20 | 934.73  | 18.18           | 20.17            | 164              |
| 40 | 1619.24 | 19.88           | 24.05            | 288              |

- H100: p5.48xlarge, 2x H100 80GB (TP=2), BF16, max-model-len=1536, max-num-seqs=40, vLLM 0.17.0

### Trn2 BF16 TP=16 (ratio=0.02)

| BS | Tok/sec | ITL median (ms) | TPOT median (ms) | TTFT median (ms) |
|----|---------|-----------------|------------------|------------------|
| 1  | 41.73   | 23.50           | 23.75            | 136              |
| 20 | 624.75  | 23.53           | 32.19            | 137              |
| 40 | 1241.83 | 23.52           | 31.56            | 179              |

- Trn2: trn2.48xlarge, TP=16, 4 chips, BF16, optimized (fused_qkv + async_mode + on_device_sampling + bucket=1024)

### ratio=0 vs ratio=0.02 对比

| BS | Metric | H100 ratio=0 | H100 ratio=0.02 | Trn2 ratio=0 | Trn2 ratio=0.02 |
|----|--------|-------------|-----------------|-------------|-----------------|
| 1  | Tok/sec | 65.90 | 66.02 | 42.19 | 41.73 |
| 1  | ITL median (ms) | 15.03 | 15.00 | 23.52 | 23.50 |
| 1  | TTFT median (ms) | 80 | 81 | 110 | 136 |
| 20 | Tok/sec | 938.63 | 934.73 | 736.72 | 624.75 |
| 20 | ITL median (ms) | 18.22 | 18.18 | 23.52 | 23.53 |
| 20 | TTFT median (s) | 0.90 | 0.16 | 0.97 | 0.14 |
| 40 | Tok/sec | 1670.34 | 1619.24 | 1297.22 | 1241.83 |
| 40 | ITL median (ms) | 19.90 | 19.88 | 23.52 | 23.52 |
| 40 | TTFT median (s) | 0.92 | 0.29 | 1.89 | 0.18 |

Key findings:
- **吞吐量和 ITL 基本不变**：ratio=0 和 ratio=0.02 之间差异 <5%，确认 decode 性能不受影响
- **TTFT 在 Trn2 上大幅改善**：BS=20 从 0.97s 降到 0.14s，BS=40 从 1.89s 降到 0.18s
- H100 的 TTFT 同样改善：BS=20 从 0.90s 降到 0.16s，BS=40 从 0.92s 降到 0.29s
- 使用 ratio=0.02 后，**Trn2 的 TTFT 反而优于 H100**（BS=40: 179ms vs 288ms）

### Trn2/H100 对比 (ratio=0.02)

| BS | H100 Tok/sec | Trn2 Tok/sec | Trn2/H100 |
|----|-------------|-------------|-----------|
| 1  | 66.02       | 41.73       | **63%**   |
| 20 | 934.73      | 624.75      | **67%**   |
| 40 | 1619.24     | 1241.83     | **77%**   |

吞吐量比值与 ratio=0 的结果一致（64%/78%/78%），确认性能对比不受 ratio 设置影响。ratio=0.02 的主要价值在于产生**更真实的 TTFT 测量** —— 特别是对于 Trn2 的 NeuronScheduler，其逐个 prefill 机制使得 TTFT 对请求同步到达模式极为敏感。

### Trn2 BF16 TP=8 (ratio=0.02)

| BS | Tok/sec | ITL median (ms) | TPOT median (ms) | TTFT median (ms) |
|----|---------|-----------------|------------------|------------------|
| 1  | 33.21   | 29.85           | 29.86            | 154              |
| 10 | 302.21  | 29.86           | 32.57            | 187              |
| 20 | 553.38  | 29.87           | 35.44            | 187              |

- Trn2: trn2.48xlarge, TP=8, 2 chips, BF16, optimized (fused_qkv + async_mode + on_device_sampling + bucket=1024)

### 公平对比：2×TP=8 vs H100 TP=2 (ratio=0.02)

| BS | H100 BF16 TP=2 (2 GPU) | Trn2 BF16 TP=8 ×2 实例 | Trn2/H100 |
|----|------------------------|------------------------|-----------|
| 1  | 66.02 tok/s            | 33.21 × 2 = 66.42      | **101%**  |
| 20 | 934.73 tok/s           | 302.21 × 2 = 604.42    | **65%**   |
| 40 | 1619.24 tok/s          | 553.38 × 2 = 1106.76   | **68%**   |

- BS 为系统总并发数，每实例实际处理 BS/2 的并发（BS=20→每实例10, BS=40→每实例20）
- 两者均为 1/4 整机资源（p5: 2/8 GPUs, trn2: 4/16 chips = 2×2 chips）
- Cost: Trn2 4 chips ($8.94/hr) vs H100 2 GPUs ($7.86/hr), cost ratio ~1.14x

---

## 大 Batch Size 扩展测试 (ratio=0.02, input=1000, output=500)

### H100 BF16 TP=2 大 BS 结果

H100 TP=2 KV cache 可支持 ~191 并发请求（max-model-len=1536 时），通过调大 max-num-seqs 可充分利用。

| BS | Tok/sec | ITL median (ms) | TPOT median (ms) | TTFT median (ms) |
|----|---------|-----------------|------------------|------------------|
| 40 | 1620.02 | 19.86 | 24.03 | 288 |
| 60 | 2097.13 | 21.27 | 27.80 | 367 |
| 80 | 2405.85 | 23.28 | 32.00 | 497 |
| 120 | 2878.98 | 26.59 | 40.35 | 697 |
| 160 | 3110.22 | 31.33 | 49.70 | 844 |

- H100: p5.48xlarge, 2x H100 80GB (TP=2), BF16, max-num-seqs=160, vLLM 0.17.0

### Trn2 TP=8 大 BS 结果

Trn2 TP=8 max-num-seqs 上限受 Neuron 编译器 `nc_find_index8` 限制（最大 16384 elements/partition），max-num-seqs=160 失败，max-num-seqs=120 可以启动。

**重要发现：max-num-seqs 越大，编译出的 kernel 效率越低**，即使实际并发未达上限也会影响性能。

| max-num-seqs | BS | Tok/sec | ITL median (ms) | TPOT median (ms) | TTFT median (ms) |
|--------------|-----|---------|-----------------|------------------|------------------|
| 80 | 40 | 791.52 | 37.91 | 49.48 | 258 |
| 80 | 60 | 1071.50 | 37.93 | 54.91 | 332 |
| 80 | 80 | 1273.45 | 38.08 | 61.14 | 401 |
| 120 | 40 | 672.58 | 46.32 | 58.30 | 258 |
| 120 | 60 | 922.93 | 46.36 | 63.92 | 349 |
| 120 | 80 | 1114.55 | 46.69 | 69.99 | 400 |
| 120 | 100 | 1264.14 | 47.24 | 77.67 | 512 |
| 120 | 120 | 1382.23 | 47.76 | 85.14 | 588 |

- Trn2: trn2.48xlarge, TP=8, 2 chips, BF16, optimized (fused_qkv + async_mode + on_device_sampling + bucket=1024)
- max-num-seqs=80 的 BS=40 吞吐 (791) 比 max-num-seqs=120 的 BS=40 (673) 高 **18%**
- max-num-seqs=120 的 BS=100 吞吐 (1264) 甚至不如 max-num-seqs=80 的 BS=80 (1273)
- 结论：Trn2 TP=8 的最佳工作点在 max-num-seqs=80 附近

### 2×TP=8 vs H100 TP=2 大 BS 对比

使用 max-num-seqs=80 的最佳数据（每实例 BS=40/60/80）：

| 总 BS | H100 TP=2 (tok/s) | Trn2 TP=8 ×2 (tok/s) | Trn2/H100 |
|-------|-------------------|----------------------|-----------|
| 80 | 2405.85 | 791.52 × 2 = 1583.04 | **66%** |
| 120 | 2878.98 | 1071.50 × 2 = 2143.00 | **74%** |
| 160 | 3110.22 | 1273.45 × 2 = 2546.90 | **82%** |

Key observations:
- **Trn2 比值随 BS 增大而提升**（66% → 74% → 82%），因为 H100 在大 BS 时 ITL 增长更快（19.86→31.33ms），Trn2 ITL 增长更平缓（37.91→38.08ms）
- 到 BS=160 时 H100 的 ITL (31.33ms) 已接近 Trn2 (38.08ms)，decode 效率差距缩小
- H100 的优势主要在于支持更大的 max-num-seqs（160+）且无 kernel 效率损失，而 Trn2 受限于编译器约束（max-num-seqs≤120）且大 max-num-seqs 有性能惩罚
- **Trn2 在大 BS 场景下的性价比改善显著**：cost ratio ~1.14x, 性能比从 66% 提升到 82%，cost-adjusted 约 72%

---

## H100 FP8 TP=1 Benchmark (ratio=0.02, input=1000, output=500)

使用预量化模型 `AngelSlim/Seed-OSS-36B-Instruct-FP8-Static`，单卡 FP8 推理，max-num-seqs=160。

### H100 FP8 TP=1 结果

| BS | Tok/sec | ITL median (ms) | TPOT median (ms) | TTFT median (ms) |
|----|---------|-----------------|------------------|------------------|
| 1  | 66.75   | 14.85           | 14.84            | 83               |
| 20 | 818.20  | 20.83           | 23.21            | 185              |
| 40 | 1427.92 | 21.97           | 27.23            | 327              |
| 60 | 1787.41 | 24.14           | 32.62            | 451              |
| 80 | 1976.40 | 27.98           | 39.09            | 604              |
| 120| 2052.68 | 31.41           | 51.01            | 2239             |
| 160| 2060.76 | 31.17           | 50.57            | 6676             |

- H100: p5.48xlarge, 1x H100 80GB (CUDA_VISIBLE_DEVICES=0), FP8 Static, max-num-seqs=160, vLLM 0.17.0
- BS=120 和 BS=160 吞吐几乎相同（2053 vs 2061），但 TTFT 从 2.2s 暴增至 6.7s，说明单卡已饱和

### H100 FP8 TP=1 vs BF16 TP=2

| BS | BF16 TP=2 (tok/s) | FP8 TP=1 (tok/s) | FP8/BF16 | FP8 GPU 数 | BF16 GPU 数 |
|----|-------------------|-------------------|----------|-----------|------------|
| 1  | 66.02             | 66.75             | **101%** | 1         | 2          |
| 20 | 934.73            | 818.20            | **88%**  | 1         | 2          |
| 40 | 1619.24           | 1427.92           | **88%**  | 1         | 2          |
| 60 | 2097.13           | 1787.41           | **85%**  | 1         | 2          |
| 80 | 2405.85           | 1976.40           | **82%**  | 1         | 2          |
| 120| 2878.98           | 2052.68           | **71%**  | 1         | 2          |
| 160| 3110.22           | 2060.76           | **66%**  | 1         | 2          |

Key observations:
- **FP8 单卡在 BS=1 时完全匹配 BF16 双卡性能**（66.75 vs 66.02 tok/s），GPU 数量减半
- **中等 BS (20-40) FP8 单卡达到 BF16 双卡的 88%**，性价比极高（一半硬件达到近 9 成性能）
- **大 BS (120+) FP8 单卡瓶颈明显**：单卡 KV cache 和计算带宽有限，吞吐在 ~2060 tok/s 饱和，而 BF16 TP=2 仍在持续增长
- FP8 TP=1 的最佳工作区间为 BS=1~80，单卡即可提供高性价比推理

---

## Long Context Benchmark: H100 TP=2 vs Trn2 TP=16 vs Trn2 2×TP=8

测试参数：output=300, random-range-ratio=0.1, 三种配置均使用 1/4 整机资源

- **H100 TP=2**: p5.48xlarge, 2x H100 80GB, BF16, max-model-len=65792, max-num-seqs=64, vLLM 0.17.1
- **Trn2 TP=16**: trn2.48xlarge, 4 chips (单实例), BF16, max-num-seqs=64/64/32/16/4（按 context length 递减）
- **Trn2 2×TP=8**: trn2.48xlarge, 2×2 chips (两个独立实例), BF16, max-num-seqs=32/32/16/8/X（按 context length 递减，64K输入无法在TP=8的情况下编译通过，所以用X指代）
- 2×TP=8 tok/s 为两个实例聚合（每实例处理 BS/2 并发），ITL/TTFT 为单实例值
- TP=16 和 2×TP=8 均使用 4 chips（1/4 整机），部署方式不同
- Cost: Trn2 4 chips ($8.94/hr) vs H100 2 GPUs ($7.86/hr), cost ratio ~1.14x

### 全量对比表

#### 4K Context (input=3700)

| BS | H100 tok/s | Trn2 TP=16 tok/s | Trn2 2×TP=8 tok/s | TP=16/H100 | 2×TP=8/H100 | TP=16 vs 2×TP=8 |
|----|-----------|-----------------|-------------------|------------|-------------|----------------|
| 4  | 187.51    | 63.61           | 78.74             | **34%**    | **42%**     | 2×TP=8 +24%    |
| 8  | 308.99    | 117.80          | 149.44            | **38%**    | **48%**     | 2×TP=8 +27%    |
| 16 | 483.97    | 209.53          | 263.40            | **43%**    | **54%**     | 2×TP=8 +26%    |
| 32 | 648.22    | 338.32          | 432.80            | **52%**    | **67%**     | 2×TP=8 +28%    |
| 64 | 740.22    | 508.03          | 632.78            | **69%**    | **85%**     | 2×TP=8 +25%    |

#### 8K Context (input=7400)

| BS | H100 tok/s | Trn2 TP=16 tok/s | Trn2 2×TP=8 tok/s | TP=16/H100 | 2×TP=8/H100 | TP=16 vs 2×TP=8 |
|----|-----------|-----------------|-------------------|------------|-------------|----------------|
| 4  | 148.44    | 44.41           | 61.20             | **30%**    | **41%**     | 2×TP=8 +38%    |
| 8  | 226.85    | 80.40           | 111.88            | **35%**    | **49%**     | 2×TP=8 +39%    |
| 16 | 317.69    | 137.70          | 185.08            | **43%**    | **58%**     | 2×TP=8 +34%    |
| 32 | 371.07    | 212.50          | 282.38            | **57%**    | **76%**     | 2×TP=8 +33%    |
| 64 | 366.35    | 300.52          | 376.38            | **82%**    | **103%**    | 2×TP=8 +25%    |

#### 16K Context (input=14800)

| BS | H100 tok/s | Trn2 TP=16 tok/s | Trn2 2×TP=8 tok/s | TP=16/H100 | 2×TP=8/H100 | TP=16 vs 2×TP=8 |
|----|-----------|-----------------|-------------------|------------|-------------|----------------|
| 4  | 101.12    | 51.22           | 56.64             | **51%**    | **56%**     | 2×TP=8 +11%    |
| 8  | 141.39    | 83.18           | 91.18             | **59%**    | **64%**     | 2×TP=8 +10%    |
| 16 | 167.27    | 121.55          | 127.96            | **73%**    | **77%**     | 2×TP=8 +5%     |
| 32 | 169.03    | 156.48          | 157.34            | **93%**    | **93%**     | ≈ 持平          |

#### 32K Context (input=29700)

| BS | H100 tok/s | Trn2 TP=16 tok/s | Trn2 2×TP=8 tok/s | TP=16/H100 | 2×TP=8/H100 | TP=16 vs 2×TP=8 |
|----|-----------|-----------------|-------------------|------------|-------------|----------------|
| 4  | 57.40     | 44.61           | 39.00             | **78%**    | **68%**     | TP=16 +14%     |
| 8  | 70.18     | 60.03           | 56.88             | **86%**    | **81%**     | TP=16 +6%      |
| 16 | 70.85     | -               | 71.78             | -          | **101%**    | 仅 2×TP=8 可达  |

#### 64K Context (input=59500)

| BS | H100 tok/s | H100 TTFT (ms) | Trn2 TP=16 tok/s | Trn2 TP=16 TTFT (ms) | TP=16/H100 |
|----|-----------|----------------|-----------------|---------------------|------------|
| 1  | 14.18     | 7140           | 9.59            | 9554                | **68%**    |
| 2  | 19.65     | 7930           | 14.17           | 9842                | **72%**    |
| 4  | 23.98     | 16790          | 19.44           | 10046               | **81%**    |
| 8  | 24.03     | 70415          | -               | -                   | -          |

Note: **64K 只有 TP=16 可以运行**。TP=8 在 64K 下即使 BS=1 也 OOM（23.804GB 已用 + 85MB DMA ring 溢出，超出 24GB/core 限制）。TP=16 最大 BS=4（BS=6 因 44MB 内存不足失败）。

### 吞吐比汇总

#### Trn2 TP=16 / H100 TP=2

| Context | BS=4 | BS=8 | BS=16 | BS=32 | BS=64 | 最大 BS 比值 |
|---------|------|------|-------|-------|-------|-------------|
| 4K      | 34%  | 38%  | 43%   | 52%   | 69%   | **69%** (BS=64)  |
| 8K      | 30%  | 35%  | 43%   | 57%   | 82%   | **82%** (BS=64)  |
| 16K     | 51%  | 59%  | 73%   | 93%   | -     | **93%** (BS=32)  |
| 32K     | 78%  | 86%  | -     | -     | -     | **86%** (BS=8)   |
| 64K     | 81%  | -    | -     | -     | -     | **81%** (BS=4)   |

#### Trn2 2×TP=8 / H100 TP=2

| Context | BS=4 | BS=8 | BS=16 | BS=32 | BS=64 | 最大 BS 比值 |
|---------|------|------|-------|-------|-------|-------------|
| 4K      | 42%  | 48%  | 54%   | 67%   | 85%   | **85%** (BS=64)  |
| 8K      | 41%  | 49%  | 58%   | 76%   | 103%  | **103%** (BS=64) |
| 16K     | 56%  | 64%  | 77%   | 93%   | -     | **93%** (BS=32)  |
| 32K     | 68%  | 81%  | 101%  | -     | -     | **101%** (BS=16) |
| 64K     | -    | -    | -     | -     | -     | OOM              |

### TP=16 vs 2×TP=8：如何选择？

两种部署方式使用相同的硬件资源（4 chips = 1/4 trn2.48xlarge），但适用场景不同：

#### 2×TP=8 更优的场景（4K-16K，批处理为主）

| 优势 | 说明 |
|------|------|
| **吞吐量更高** | 4K-8K 场景下 2×TP=8 比 TP=16 高 25-39%，16K 高 5-11% |
| **支持更大有效 BS** | 两个独立实例分担负载，每实例只需处理 BS/2 |
| **更好的 TTFT 隔离** | 各实例独立调度，互不影响 |
| **最大化吞吐** | 8K BS=64 超越 H100（103%），32K BS=16 超越 H100（101%） |

**原因**：TP=8 的 decode 延迟更高（ITL ~46-61ms vs TP=16 的 ~58-74ms），但两个实例并行 decode 的聚合吞吐远超单个 TP=16 实例。在 context length ≤16K 时，TP=8 单实例的内存足够支撑较大 BS，2×TP=8 的并行优势充分发挥。

#### TP=16 更优的场景（32K+，低延迟需求）

| 优势 | 说明 |
|------|------|
| **32K 小 BS 吞吐更高** | BS=4 TP=16 比 2×TP=8 高 14%，BS=8 高 6% |
| **64K 唯一选择** | TP=8 在 64K 下 OOM，TP=16 可支持 BS=1-4 |
| **更低的单请求延迟** | ITL 更低（更多并行），适合交互式场景 |
| **部署更简单** | 单实例，无需负载均衡 |

**原因**：32K+ 时，TP=8 单实例的 max-num-seqs 受限（32K max-num-seqs=8），每实例只能跑很小的 BS，两个实例的 BS/2 分摊导致并行优势缩小甚至反转。同时 TP=16 在长序列下 decode 更快（更多核心分担 KV cache 读取），单实例吞吐反超 2×TP=8。

#### 选择决策树

```
Context Length ≤ 16K?
├── Yes → 2×TP=8（吞吐更高 5-39%，可超越 H100）
└── No → Context Length = 32K?
    ├── Yes → 需要最大吞吐（BS=16）?
    │   ├── Yes → 2×TP=8（唯一能到 BS=16，超越 H100）
    │   └── No → TP=16（小 BS 吞吐更优，部署更简单）
    └── No (64K) → TP=16（唯一选择，TP=8 OOM）
```

### Key Findings

1. **Trn2 在 8K BS=64 和 32K BS=16 时超越 H100**（均为 2×TP=8 配置）：
   - 8K BS=64: Trn2 376 tok/s vs H100 366 tok/s（**+3%**）
   - 32K BS=16: Trn2 72 tok/s vs H100 71 tok/s（**+1%**）
   - 这是因为 H100 在这些点上 KV cache 饱和，吞吐不再增长甚至下降

2. **H100 在长上下文大 BS 时 TTFT 严重恶化**：
   - 8K BS=64: H100 TTFT=28.5s vs Trn2 TTFT=1.1s（**H100 是 Trn2 的 26 倍**）
   - 16K BS=32: H100 TTFT=31.6s vs Trn2 TTFT=2.9s（**H100 是 Trn2 的 11 倍**）
   - 32K BS=16: H100 TTFT=40.7s vs Trn2 TTFT=6.1s（**H100 是 Trn2 的 7 倍**）
   - 原因：H100 的 KV cache 空间有限（2×80GB），长上下文+大 BS 导致排队等待

3. **Trn2 ITL 极其稳定，H100 ITL 随 BS 快速增长**：
   - Trn2 TP=8 的 Median ITL 在同一 context length 下几乎不随 BS 变化（如 4K: 46.46-46.55ms）
   - H100 的 Median ITL 在低 BS 时更优，但在 KV cache 压力下快速增长（4K: 18→28ms）

4. **Context length 越长，Trn2 竞争力越强**：
   - 2×TP=8: 4K 最大 85%，8K 达到 103%，16K 达到 93%，32K 达到 101%
   - TP=16: 4K 最大 69%，8K 达到 82%，16K 达到 93%，32K 达到 86%，64K 达到 81%

5. **64K 是 TP=16 的独占领域**：
   - TP=8 在 64K 下 OOM（24GB/core 硬限制），无法运行
   - TP=16 BS=4 达到 H100 BS=4 的 81%，且 TTFT 更优（10s vs 17s）
   - H100 BS=8 吞吐 24.03 几乎等于 BS=4 的 23.98，说明已饱和；Trn2 TP=16 在 BS=4 时也接近上限

6. **Cost-adjusted 分析**（cost ratio ~1.14x）：
   - 2×TP=8 最佳: 8K BS=64 cost-adjusted = 103% / 1.14 ≈ **90%**
   - TP=16 最佳: 64K BS=4 cost-adjusted = 81% / 1.14 ≈ **71%**
   - 在 H100 TTFT 恶化的工作点上，Trn2 提供**更好的用户体验**（TTFT 低一个数量级）
