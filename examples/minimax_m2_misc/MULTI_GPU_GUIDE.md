# MiniMax M2 多GPU测试指南

## 背景

MiniMax M2是一个大型MoE模型（~400GB BF16），单个GPU无法完全加载。本指南说明如何使用多GPU进行测试。

## 模型规格

- **参数量**: ~650B（62层，256个专家/层）
- **权重大小**: ~403GB（BF16）
- **内存需求**: 需要多个大容量GPU（如8x80GB A100或H100）

## 快速开始

### 自动多GPU分配（推荐）

最简单的方式，让transformers自动分配模型到所有可用GPU：

```bash
python test_layer_by_layer_gpu.py --test-type extract-all
```

这会：
- 自动检测所有可用GPU
- 智能分配模型层到不同GPU
- 显示每个GPU的内存使用情况

### 查看设备分配

运行时会显示详细信息：

```
检测到 8 个GPU
  GPU 0: NVIDIA A100-SXM4-80GB, 80.00 GB
  GPU 1: NVIDIA A100-SXM4-80GB, 80.00 GB
  ...

设备分配详情:
  cuda:0: 12 个模块
  cuda:1: 10 个模块
  cuda:2: 10 个模块
  ...

GPU内存使用:
  GPU 0: 已分配 58.32 GB, 已保留 60.00 GB
  GPU 1: 已分配 52.15 GB, 已保留 54.00 GB
  ...
```

## 高级用法

### 1. 限制每个GPU的内存使用

如果GPU被其他任务占用，可以限制使用量：

```bash
# 每个GPU最多使用40GB
python test_layer_by_layer_gpu.py --test-type extract-all --max-memory "40GB"

# 为不同GPU设置不同限制
python test_layer_by_layer_gpu.py --test-type extract-all --max-memory "0:60GB,1:60GB,2:40GB"
```

### 2. 使用不同的分配策略

```bash
# 平衡分配（尽量均衡各GPU负载）
python test_layer_by_layer_gpu.py --test-type extract-all --device-map balanced

# 平衡分配，少用GPU 0（为显示任务保留）
python test_layer_by_layer_gpu.py --test-type extract-all --device-map balanced_low_0

# 顺序分配（先填满GPU 0，再用GPU 1...）
python test_layer_by_layer_gpu.py --test-type extract-all --device-map sequential
```

### 3. 单GPU模式（如果你有足够大的GPU）

如果你有单个大容量GPU（如H200 141GB），可以使用单GPU模式：

```bash
python test_layer_by_layer_gpu.py --test-type extract-all --device-map cuda:0
```

### 4. 测试特定层

```bash
# 测试单层（layer 0）
python test_layer_by_layer_gpu.py --test-type single-layer --layer 0

# 测试多层（layers 0-5）
python test_layer_by_layer_gpu.py --test-type multi-layer --start-layer 0 --end-layer 5
```

## 常见问题

### Q1: OOM (Out of Memory) 错误

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate XXX GB
```

**解决方案**：

1. **检查GPU总内存**：
   ```bash
   nvidia-smi
   ```
   确保所有GPU的可用内存加起来 > 450GB（模型403GB + 推理开销）

2. **减少最大序列长度**：
   编辑脚本，减少`max_length`参数：
   ```python
   inputs, prompt_text = prepare_test_inputs(tokenizer, max_length=128)  # 从512减到128
   ```

3. **启用CPU offload**：
   如果GPU内存不够，可以将部分层放在CPU：
   ```python
   # 在load_model_gpu中修改
   model = AutoModelForCausalLM.from_pretrained(
       MODEL_PATH,
       device_map="auto",
       offload_folder="offload",  # 启用CPU offload
       offload_state_dict=True,   # offload未使用的权重
   )
   ```

### Q2: 模型加载很慢

**原因**：从磁盘加载400GB权重需要时间

**优化方案**：

1. **使用SSD**：确保模型存储在NVMe SSD上
2. **使用safetensors格式**：比PyTorch pickle快
3. **预热加载**：第一次加载后，权重可能在文件系统缓存中

### Q3: Hook在多GPU上不工作

**症状**：某些层的输出没有被记录

**原因**：Hooks需要正确处理跨设备数据传输

**解决方案**：已在代码中处理，hooks会自动将输出移到CPU：

```python
def create_hook(self, layer_name: str):
    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        # 自动移到CPU保存
        self.layer_outputs[layer_name] = hidden_states.detach().cpu()
    return hook
```

### Q4: 不同GPU之间的通信开销

**影响**：多GPU推理可能比单GPU慢（如果有足够大的单GPU）

**原因**：跨GPU的数据传输需要通过PCIe或NVLink

**优化建议**：
1. 使用NVLink连接的GPU（如A100/H100）
2. 使用`balanced`分配策略，减少跨GPU通信
3. 如果只是提取层输出（不需要多次推理），性能影响较小

## 内存估算

### 单次推理的内存需求

```
总内存 = 模型权重 + 激活值 + KV缓存

模型权重: 403GB (BF16)
激活值: ~batch_size × seq_len × hidden_size × num_layers × 2 bytes
        = 1 × 512 × 3072 × 62 × 2 / 1024^3
        ≈ 0.18 GB（可忽略）

KV缓存: ~2 × batch_size × num_layers × seq_len × num_kv_heads × head_dim × 2 bytes
        = 2 × 1 × 62 × 512 × 8 × 384 × 2 / 1024^3
        ≈ 0.23 GB（可忽略）

总计: ~403GB + 0.5GB ≈ 404GB
```

### 推荐GPU配置

| GPU配置 | 总内存 | 是否足够 | 备注 |
|---------|--------|----------|------|
| 8 × A100 (80GB) | 640GB | ✓ | 推荐，有余量 |
| 8 × A100 (40GB) | 320GB | ✗ | 不够 |
| 4 × H100 (80GB) | 320GB | ✗ | 不够 |
| 6 × H100 (80GB) | 480GB | ✓ | 可以，较紧 |
| 1 × H200 (141GB) | 141GB | ✗ | 不够（单GPU） |

## 性能优化

### 1. 使用Flash Attention（如果可用）

```python
# 在加载模型时添加
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",  # 需要flash-attn包
)
```

### 2. 使用torch.compile（PyTorch 2.0+）

```python
# 编译模型以加速推理
model = torch.compile(model, mode="reduce-overhead")
```

### 3. 批处理多个样本

如果需要测试多个输入，使用批处理：

```python
inputs = tokenizer([text1, text2, text3], padding=True, return_tensors="pt")
# batch_size = 3
```

## 监控工具

### 实时GPU监控

```bash
# 每秒更新一次
watch -n 1 nvidia-smi

# 更详细的信息
nvidia-smi dmon -s mu -c 100
```

### Python内监控

脚本已内置内存监控，会自动显示：
- 每个GPU的已分配内存
- 每个GPU的已保留内存
- 设备分配详情

## 故障排除

### 检查NCCL（多GPU通信库）

```bash
# 设置日志级别
export NCCL_DEBUG=INFO
python test_layer_by_layer_gpu.py --test-type extract-all
```

### 检查CUDA版本兼容性

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```

### 清理GPU缓存

如果遇到内存碎片问题：

```python
import torch
import gc

# 在Python中执行
torch.cuda.empty_cache()
gc.collect()
```

## 参考资料

- [Hugging Face 多GPU指南](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_many)
- [Accelerate库文档](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)
- [NVIDIA Multi-GPU最佳实践](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html)
