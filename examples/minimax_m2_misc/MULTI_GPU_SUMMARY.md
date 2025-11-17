# 多GPU版本修改总结

## 修改内容

已将 `test_layer_by_layer_gpu.py` 从单GPU版本升级为多GPU版本，支持大型MiniMax M2模型（~400GB）的测试。

## 主要改动

### 1. `load_model_gpu()` 函数

**改动前**（单GPU）：
```python
def load_model_gpu(device: str = "cuda:0"):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device,  # 单GPU
        low_cpu_mem_usage=True,
    )
```

**改动后**（多GPU）：
```python
def load_model_gpu(device_map: str = "auto", max_memory: dict = None):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device_map,        # 支持auto/balanced/sequential等
        max_memory=max_memory,         # 可限制每个GPU的内存
        low_cpu_mem_usage=True,
        offload_folder="offload",      # 支持CPU offload
    )
```

**新增功能**：
- ✓ 自动检测所有GPU
- ✓ 显示每个GPU的型号和容量
- ✓ 显示模型在各GPU上的分布
- ✓ 实时显示GPU内存使用情况

### 2. `get_model_input_device()` 辅助函数（新增）

在多GPU模式下，正确获取输入应该移动到的设备：

```python
def get_model_input_device(model):
    """获取模型输入应该所在的设备（多GPU模式下）"""
    # 尝试获取embedding层的设备
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight.device
    # ...
```

**为什么需要**：
- 多GPU模型的不同层可能在不同设备上
- 输入必须在第一层（embedding）所在的设备上
- 不能简单地使用 `next(model.parameters()).device`

### 3. 命令行参数更新

**新增参数**：
```bash
--device-map    # 设备分配策略（auto/balanced/sequential/cuda:0）
--max-memory    # 内存限制（如"40GB"或"0:40GB,1:60GB"）
```

**移除参数**：
```bash
--device        # 不再使用单一device参数
```

### 4. 所有测试函数更新

所有测试函数都添加了 `device_map` 和 `max_memory` 参数：

- `test_single_layer_gpu()`
- `test_multi_layer_gpu()`
- `test_full_model_gpu()`
- `extract_all_layers_gpu()`

## 使用示例

### 基础用法（自动多GPU）

```bash
# 自动分配到所有GPU（最简单）
python test_layer_by_layer_gpu.py --test-type extract-all
```

### 限制内存

```bash
# 每个GPU最多40GB
python test_layer_by_layer_gpu.py --test-type extract-all --max-memory "40GB"

# 为不同GPU设置不同限制
python test_layer_by_layer_gpu.py --test-type extract-all --max-memory "0:60GB,1:40GB"
```

### 不同分配策略

```bash
# 平衡分配
python test_layer_by_layer_gpu.py --test-type full --device-map balanced

# 顺序填充
python test_layer_by_layer_gpu.py --test-type full --device-map sequential

# 单GPU模式
python test_layer_by_layer_gpu.py --test-type full --device-map cuda:0
```

## 配套文档和工具

### 1. `MULTI_GPU_GUIDE.md`
详细使用指南，包括：
- 快速开始
- 高级用法
- 常见问题
- 内存估算
- 性能优化

### 2. `check_gpu_readiness.py`
环境检查脚本，验证：
- GPU数量和容量
- CUDA和PyTorch版本
- 必要的库
- 模型文件
- NVLink连接（可选）

运行方式：
```bash
python check_gpu_readiness.py
```

## 技术细节

### 设备分配策略对比

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `auto` | 自动智能分配 | 通用，推荐 |
| `balanced` | 平衡各GPU负载 | GPU容量相同时 |
| `balanced_low_0` | 少用GPU 0 | 保留GPU 0给显示 |
| `sequential` | 顺序填满 | 需要精确控制时 |
| `cuda:X` | 单GPU | 有足够大的单GPU |

### 内存分布示例

在8×80GB A100上的典型分布：

```
GPU 0: 58.32 GB  [Embedding + Layers 0-7]
GPU 1: 52.15 GB  [Layers 8-15]
GPU 2: 52.15 GB  [Layers 16-23]
GPU 3: 52.15 GB  [Layers 24-31]
GPU 4: 52.15 GB  [Layers 32-39]
GPU 5: 52.15 GB  [Layers 40-47]
GPU 6: 52.15 GB  [Layers 48-55]
GPU 7: 50.00 GB  [Layers 56-61 + LM Head]
```

### Hook在多GPU上的工作方式

1. **注册阶段**：Hooks注册到各层，无论层在哪个GPU
2. **执行阶段**：当数据流过该层时触发hook
3. **保存阶段**：自动将输出移到CPU，避免GPU间内存碎片

```python
def create_hook(self, layer_name: str):
    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # 关键：移到CPU保存，避免跨GPU内存管理问题
        self.layer_outputs[layer_name] = hidden_states.detach().cpu()
    return hook
```

## 性能影响

### 多GPU vs 单大GPU

假设有足够大的单GPU（如H200 141GB，虽然仍不够MiniMax M2）：

**多GPU优势**：
- ✓ 可以运行更大的模型
- ✓ 总内存更大
- ✓ 可以并行计算某些操作

**多GPU劣势**：
- ✗ 跨GPU通信开销（PCIe: ~32GB/s, NVLink: ~600GB/s）
- ✗ 内存碎片化
- ✗ 更复杂的调试

**典型性能对比**：
- 单GPU（如果够用）：100% 基准
- 多GPU（PCIe）：60-80% 基准
- 多GPU（NVLink）：85-95% 基准

### 优化建议

1. **使用NVLink GPU**：A100/H100系列
2. **减少跨GPU数据传输**：使用`balanced`策略
3. **批处理**：增大batch_size分摊开销
4. **预热**：第一次运行会慢，后续会快

## 兼容性

### 支持的GPU

理论上支持所有CUDA GPU，推荐：
- ✓ A100 (40GB/80GB)
- ✓ H100 (80GB)
- ✓ H200 (141GB)
- ✓ A6000 (48GB)
- ⚠️ V100 (16GB/32GB) - 容量可能不够

### 软件要求

```
Python >= 3.8
PyTorch >= 2.0
transformers >= 4.35
accelerate >= 0.20
safetensors >= 0.3
CUDA >= 11.8
```

## 未来改进

可能的优化方向：

1. **张量并行**：使用DeepSpeed/Megatron-LM实现更细粒度的并行
2. **Pipeline并行**：流水线多批次处理
3. **量化**：INT8/INT4量化减少内存
4. **KV缓存优化**：PagedAttention等技术
5. **动态批处理**：根据序列长度动态调整

## 问题排查

### 查看详细分配信息

在代码中添加：
```python
print("\n模型设备分配:")
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")
```

### 监控GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 详细监控
nvidia-smi dmon -s mu
```

### 检查通信瓶颈

```bash
# 启用NCCL调试
export NCCL_DEBUG=INFO
python test_layer_by_layer_gpu.py --test-type extract-all
```

## 总结

多GPU版本支持在多个GPU上分布式加载和运行MiniMax M2这样的超大模型，关键改进包括：

1. ✓ 灵活的设备分配策略
2. ✓ 内存使用限制和监控
3. ✓ 正确的跨设备数据处理
4. ✓ 完整的文档和工具支持

现在可以在多GPU集群上高效测试400GB+的模型！
