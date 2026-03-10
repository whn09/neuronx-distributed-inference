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

python3 -m vllm.entrypoints.openai.api_server \
    --model "/opt/dlami/nvme/Seed-OSS-36B-Instruct" \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --max-num-seqs 16 \
    --no-enable-chunked-prefill \
    --no-enable-prefix-caching \
    --port 8000 \
    --additional-config '{"override_neuron_config": {"skip_warmup": true, "enable_bucketing": true, "context_encoding_buckets": [256, 512, 1024, 2048], "token_generation_buckets": [256, 512, 1024, 2048]}}'
```

关键参数说明：
- `--tensor-parallel-size 8`：使用 2 个 Trainium2 Chips（每个 Chip 在 LNC=2 下有 4 个逻辑核心）
- `--max-model-len 2048`：最大序列长度
- `--max-num-seqs 16`：最大并发序列数
- `enable_bucketing`：启用 bucket 优化编译效率

## Benchmark 方法

### 并发 1

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

vllm bench serve \
    --backend vllm \
    --model /opt/dlami/nvme/Seed-OSS-36B-Instruct \
    --endpoint /v1/completions \
    --dataset-name random \
    --num-prompts 16 \
    --random-input-len 900 \
    --random-output-len 90 \
    --random-range-ratio 0.03 \
    --max-concurrency 1
```

### 并发 16

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

vllm bench serve \
    --backend vllm \
    --model /opt/dlami/nvme/Seed-OSS-36B-Instruct \
    --endpoint /v1/completions \
    --dataset-name random \
    --num-prompts 256 \
    --random-input-len 900 \
    --random-output-len 90 \
    --random-range-ratio 0.03 \
    --max-concurrency 16
```

## Benchmark 结果

### 测试环境

| 项目 | 配置 |
|------|------|
| 实例类型 | trn2.48xlarge |
| Trainium2 Chips | 16（使用其中 2 个） |
| TP Degree | 8 |
| Neuron SDK | 2.23 |
| neuronx-distributed | 0.17 |
| neuronx-distributed-inference | 0.8 |
| vLLM | 0.13.0 |
| vllm-neuron | 0.4.1 |
| 输入长度 | 1024 tokens |
| 输出长度 | 256 tokens |

### 结果对比

测试参数：input=900, output=90, random-range-ratio=0.03

| 指标 | 并发 1 | 并发 16 |
|------|--------|---------|
| Successful requests | 16 | 256 |
| Failed requests | 0 | 0 |
| Benchmark duration (s) | 45.03 | 75.80 |
| Request throughput (req/s) | 0.36 | **3.38** |
| Output token throughput (tok/s) | 32.09 | **304.39** |
| Peak output token throughput (tok/s) | 34.00 | **544.00** |
| Total token throughput (tok/s) | 351.94 | **3347.94** |
| Mean TTFT (ms) | **134.60** | 323.77 |
| Median TTFT (ms) | **134.67** | 250.95 |
| P99 TTFT (ms) | **135.75** | 1703.99 |
| Mean TPOT (ms) | **30.00** | 49.19 |
| Median TPOT (ms) | **29.98** | 50.47 |
| P99 TPOT (ms) | **30.16** | 60.33 |
| Median ITL (ms) | 29.98 | 30.05 |
| P99 ITL (ms) | 30.69 | 408.24 |

### 结果分析

- **吞吐量线性扩展良好**：从并发 1 到并发 16，输出吞吐从 32 tok/s 提升至 304 tok/s，约 **9.5 倍**提升
- **单请求延迟优秀**：并发 1 时 TTFT 仅 135ms，TPOT 30ms
- **ITL 稳定**：并发 16 时 Median ITL（30.05ms）与并发 1（29.98ms）几乎一致
- **P99 ITL 存在毛刺**：并发 16 时 P99 ITL 408ms，原因是 Neuron 调度器将 Prefill 和 Decode 分开调度（见下文）
- **Peak 吞吐达到 544 tok/s**（输出）和 3348 tok/s（总），仅使用 2 个 Trainium2 Chips

### 调度机制说明

vllm-neuron 使用 ，其调度特点：

1. **Prefill 和 Decode 分开执行**：当有等待 prefill 的请求时，会暂停所有 decode，先完成 prefill
2. **Prefill 批大小为 1**：每次只 prefill 一个请求
3. 这意味着当多个请求同时到达时，需要逐个 prefill，后续请求的 TTFT 会增加
4. 建议 benchmark 时使用  参数，让输入输出长度有随机性，避免所有请求同时完成导致 TTFT 虚高
