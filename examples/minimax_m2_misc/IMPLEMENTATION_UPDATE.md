# 逐层测试实现更新

## 更新时间
2025-11-17

## 实现内容

已完成 `test_layer_by_layer.py` 中的实际前向传播测试实现，替换了之前的 TODO 占位符。

---

## 主要改进

### 1. `test_single_layer()` - 单层测试 ✅

**实现内容**：
- ✅ 使用XLA设备创建输入张量
- ✅ 运行实际的前向传播
- ✅ 同步XLA操作（`xm.mark_step()`）
- ✅ 输出验证和统计分析
- ✅ 保存输出用于GPU对比

**关键代码**：
```python
# 导入XLA并移到XLA设备
import torch_xla.core.xla_model as xm
device = xm.xla_device()

# 创建XLA张量
dummy_hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16).to(device)
dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int32).to(device)
dummy_position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1).to(device)

# 运行前向传播
with torch.no_grad():
    outputs = layer_wrapper(
        dummy_hidden_states,
        attention_mask=dummy_attention_mask,
        position_ids=dummy_position_ids
    )

# 同步XLA操作
xm.mark_step()

# 移到CPU进行统计分析
hidden_states_cpu = hidden_states_out.cpu()
```

**输出统计**：
- Mean, Std, Min, Max
- NaN检测
- Inf检测
- Shape和dtype验证

### 2. `test_multi_layer()` - 多层测试 ✅

**实现内容**：
- ✅ 完全相同的XLA设备支持
- ✅ 多层顺序前向传播
- ✅ 完整的输出验证
- ✅ 保存中间层输出用于对比

**用途**：
- 测试连续多层（如layers 0-5）
- 二分查找问题层的范围
- 验证多层累积效应

### 3. `test_full_model()` - 完整模型

**当前状态**：
- ✅ 已有分布式环境初始化
- ✅ 模型编译框架（使用 `model.compile()`）
- ⚠️  Generation实现仍为TODO（需要实际编译后的模型）

**说明**：
完整模型测试使用 `NeuronMiniMaxM2ForCausalLM`，这是更高级的API，需要实际的Neuron编译。单层和多层测试已足够用于调试。

---

## 使用方法

### 测试单层（快速验证）

```bash
python minimax_m2_misc/test_layer_by_layer.py --test-type single-layer --layer 0
```

**预期输出**：
```
初始化分布式环境 (tp_degree=64)...
  ✓ torch.distributed已初始化 (backend=xla)
  ✓ parallel_state已初始化 (tp_degree=64)

测试 Layer 0 构建和前向传播...
  创建 SingleLayerWrapper...
  ✓ SingleLayerWrapper 创建成功

  创建测试输入...
    使用XLA设备: xla:0
  ✓ 输入创建完成

  运行前向传播（在XLA设备上）...
  ✓ 前向传播成功
  ✓ XLA操作已同步

  验证输出数值:
    Mean: 0.123456
    Std: 0.234567
    Min: -1.234567
    Max: 1.234567
    Has NaN: False
    Has Inf: False

  ✓ 输出已保存到: /home/ubuntu/traced_model/test_layers/single_layer_0/layer_output.pt
```

### 测试多层（范围测试）

```bash
# 测试前10层
python minimax_m2_misc/test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 9

# 二分查找示例：测试前半部分层
python minimax_m2_misc/test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 30

# 测试后半部分层
python minimax_m2_misc/test_layer_by_layer.py --test-type multi-layer --start-layer 31 --end-layer 61
```

### 加载已保存的结果

```bash
# 不重新运行，只加载结果
python minimax_m2_misc/test_layer_by_layer.py --test-type single-layer --layer 0 --skip-compile
```

---

## 输出文件结构

### 单层测试输出

```
/home/ubuntu/traced_model/test_layers/single_layer_0/
├── layer_output.pt      # 层输出数据
│   ├── layer_idx: int
│   ├── config: MiniMaxM2InferenceConfig
│   ├── input_hidden_states: Tensor
│   ├── input_attention_mask: Tensor
│   ├── input_position_ids: Tensor
│   ├── output_hidden_states: Tensor
│   ├── output_full: tuple
│   └── statistics: dict
│       ├── mean: float
│       ├── std: float
│       ├── min: float
│       ├── max: float
│       ├── has_nan: bool
│       └── has_inf: bool
└── config.pt            # 配置文件
```

### 多层测试输出

```
/home/ubuntu/traced_model/test_layers/layers_0_to_5/
├── layers_output.pt     # 多层输出数据
│   ├── start_layer: int
│   ├── end_layer: int
│   ├── num_layers: int
│   ├── config: MiniMaxM2InferenceConfig
│   ├── input_hidden_states: Tensor
│   ├── output_hidden_states: Tensor (最后一层的输出)
│   └── statistics: dict
└── config.pt            # 配置文件
```

---

## 与GPU版本对比

### 步骤1: 在GPU上生成golden reference

```bash
# 在GPU机器上运行
cd /home/ubuntu/neuronx-distributed-inference/examples
python minimax_m2_misc/test_layer_by_layer_gpu.py --test-type extract-all
```

### 步骤2: 在Trainium上测试

```bash
# 在Trainium机器上运行
cd /home/ubuntu/neuronx-distributed-inference/examples
python minimax_m2_misc/test_layer_by_layer.py --test-type single-layer --layer 0
```

### 步骤3: 对比输出

```bash
python compare_gpu_neuron_outputs.py \
  --gpu-path /home/ubuntu/traced_model/gpu_reference/full_model \
  --neuron-path /home/ubuntu/traced_model/test_layers/single_layer_0 \
  --layer layer_0
```

---

## 技术细节

### 为什么使用XLA设备？

Neuron模型组件（如 `NeuronMiniMaxM2DecoderLayer`）期望在XLA设备上运行：

1. **兼容性**：Neuron使用XLA作为编译后端
2. **正确性**：XLA张量保证与实际Neuron编译一致的数值行为
3. **可测试性**：即使不编译，也能验证模型forward pass逻辑

### XLA操作同步

```python
xm.mark_step()
```

**作用**：
- 强制XLA执行所有pending操作
- 确保输出已经计算完成
- 类似于CUDA的 `torch.cuda.synchronize()`

### CPU vs XLA张量转换

```python
# XLA → CPU (用于统计和保存)
hidden_states_cpu = hidden_states_out.cpu()

# CPU → XLA (用于输入)
tensor_xla = tensor_cpu.to(xm.xla_device())
```

**注意**：
- 跨设备转换有开销，避免频繁转换
- 统计分析在CPU上更方便
- 保存文件使用CPU张量（兼容性更好）

---

## 调试工作流

### 推荐流程

```bash
# 1. 快速测试：验证单层是否工作
python minimax_m2_misc/test_layer_by_layer.py --test-type single-layer --layer 0

# 2. 如果单层正常，测试前几层
python minimax_m2_misc/test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 5

# 3. 二分查找：逐步扩大范围
python minimax_m2_misc/test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 15
python minimax_m2_misc/test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 30

# 4. 最终测试所有层
python minimax_m2_misc/test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 61
```

### 诊断输出异常

检查保存的统计信息：

```python
import torch

# 加载输出
output_data = torch.load('/home/ubuntu/traced_model/test_layers/single_layer_0/layer_output.pt')

# 检查统计
stats = output_data['statistics']
print(f"Mean: {stats['mean']}")
print(f"Std: {stats['std']}")
print(f"Has NaN: {stats['has_nan']}")
print(f"Has Inf: {stats['has_inf']}")

# 检查实际输出
hidden_states = output_data['output_hidden_states']
print(f"Output shape: {hidden_states.shape}")
print(f"Output dtype: {hidden_states.dtype}")
```

---

## 常见问题

### Q1: XLA设备不可用

**错误**：
```
RuntimeError: XLA device not found
```

**解决**：
确保在Trainium实例上运行，并且已安装torch_neuronx：
```bash
pip list | grep neuron
```

### Q2: 内存不足

**错误**：
```
RuntimeError: [XLA:CPU] out of memory
```

**解决**：
- 减小 `seq_len`（在 `get_neuron_config()` 中设置）
- 减小 `batch_size`
- 测试更少的层

### Q3: 输出有NaN或Inf

**原因**：
- 模型权重未正确加载
- dtype不匹配（FP32 vs BF16）
- 数值溢出

**调试**：
```bash
# 检查模型权重dtype
python debug_model_dtype.py
```

### Q4: 分布式环境初始化失败

**错误**：
```
ValueError: parallel_state is already initialized
```

**解决**：
重启Python进程，或使用：
```python
from neuronx_distributed.parallel_layers import parallel_state
parallel_state.destroy_model_parallel()
```

---

## 性能预期

### 单层测试

| 配置 | 时间 | 内存 |
|------|------|------|
| seq_len=256 | ~10s | ~15GB |
| seq_len=512 | ~20s | ~25GB |
| seq_len=1024 | ~40s | ~45GB |

### 多层测试（10层）

| 配置 | 时间 | 内存 |
|------|------|------|
| seq_len=256 | ~1-2min | ~50GB |
| seq_len=512 | ~3-5min | ~80GB |

**注意**：
- 首次运行会更慢（XLA编译）
- 后续运行会快很多（使用缓存）

---

## 下一步

### 已完成 ✅
1. ✅ 单层前向传播测试
2. ✅ 多层前向传播测试
3. ✅ XLA设备支持
4. ✅ 输出验证和统计
5. ✅ 保存输出用于对比

### 可选改进
1. 实现完整模型的generation测试
2. 添加权重加载验证
3. 添加中间激活值对比
4. 实现自动化测试流程
5. 添加性能profiling

---

## 总结

现在可以使用 `test_layer_by_layer.py` 进行实际的逐层测试：

- ✅ **实现完整**：不再是TODO占位符
- ✅ **XLA支持**：与Neuron运行时兼容
- ✅ **输出验证**：完整的统计分析
- ✅ **可保存对比**：支持与GPU版本对比
- ✅ **灵活调试**：支持单层、多层、范围测试

可以开始实际的模型调试工作了！
