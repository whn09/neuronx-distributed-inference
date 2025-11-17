# 逐层输出记录工具 V2

基于 `generation_minimax_m2_demo_v2.py` 的简洁版本，用于记录完整模型每一层的输出。

## 快速开始

### 如果已有编译好的模型

```bash
cd /home/ubuntu/neuronx-distributed-inference/examples
python minimax_m2_misc/test_layer_by_layer_v2.py --skip-compile
```

### 如果需要编译模型

```bash
python minimax_m2_misc/test_layer_by_layer_v2.py
```

## 输出文件

所有输出保存在 `/home/ubuntu/traced_model/test_layers/full_model/`：

```
full_model/
├── embedding_output.pt      # Embedding层输出
├── layer_0_output.pt         # 第0层输出
├── layer_1_output.pt         # 第1层输出
├── ...
├── layer_61_output.pt        # 第61层输出
├── final_norm_output.pt      # Final Norm层输出
├── statistics.pt             # 统计信息
└── metadata.pt               # 元数据
```

## 分析输出

### 查看统计信息

```python
import torch

# 加载统计信息
stats = torch.load('/home/ubuntu/traced_model/test_layers/full_model/statistics.pt')

# 查看所有层
for layer_name in sorted(stats.keys()):
    if 'layer_' in layer_name:
        stat = stats[layer_name]
        print(f"{layer_name}: Mean={stat['mean']:.6f}, NaN={stat['has_nan']}")
```

### 查看特定层输出

```python
import torch

# 加载第30层输出
layer_30 = torch.load('/home/ubuntu/traced_model/test_layers/full_model/layer_30_output.pt')

print(f"Shape: {layer_30.shape}")
print(f"Mean: {layer_30.float().mean():.6f}")
print(f"Std: {layer_30.float().std():.6f}")
```

### 检查异常

```python
import torch

stats = torch.load('/home/ubuntu/traced_model/test_layers/full_model/statistics.pt')

# 找出有问题的层
for layer_name, stat in stats.items():
    if 'layer_' in layer_name:
        if stat.get('has_nan', False) or stat.get('has_inf', False):
            print(f"❌ {layer_name}: NaN={stat['has_nan']}, Inf={stat['has_inf']}")
```

## 与GPU版本对比

### 1. 在GPU上生成参考输出

```bash
# 在GPU机器上运行
python minimax_m2_misc/test_layer_by_layer_gpu.py --test-type extract-all
```

### 2. 对比输出

```python
import torch

# 加载Neuron和GPU的同一层输出
neuron = torch.load('/home/ubuntu/traced_model/test_layers/full_model/layer_30_output.pt')
gpu = torch.load('/home/ubuntu/traced_model/gpu_reference/full_model/layer_30_output.pt')

# 计算差异
diff = (neuron - gpu).abs()
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")

# 计算相似度
cosine_sim = torch.nn.functional.cosine_similarity(
    neuron.flatten(), gpu.flatten(), dim=0
)
print(f"Cosine similarity: {cosine_sim:.6f}")
```

## 配置说明

配置与 `generation_minimax_m2_demo_v2.py` 完全一致：

- **tp_degree**: 64 (必须是 num_key_value_heads=8 的倍数)
- **batch_size**: 1
- **max_context_length**: 1024
- **seq_len**: 1024
- **blockwise_matmul_config**: `{'use_torch_block_wise': True}`

## 代码结构

```python
test_layer_by_layer_v2.py
├── LayerOutputRecorder        # 记录层输出的类
│   ├── register_hooks()       # 注册hooks
│   ├── get_statistics()       # 获取统计信息
│   └── save_outputs()         # 保存输出
└── test_with_layer_recording  # 主函数
    ├── 步骤1: 编译模型（可选）
    ├── 步骤2: 加载模型
    ├── 步骤3: 注册hooks
    ├── 步骤4: 准备输入
    ├── 步骤5: 运行推理
    ├── 步骤6: 显示统计
    └── 步骤7: 保存输出
```

## 与原版本的区别

| 功能 | test_layer_by_layer.py (旧) | test_layer_by_layer_v2.py (新) |
|------|------------------------------|--------------------------------|
| 单层测试 | ✓ 支持 | ✗ 移除（无法编译） |
| 多层测试 | ✓ 支持 | ✗ 移除（无法编译） |
| 完整模型 | ✓ 支持 | ✓ 支持 |
| 代码行数 | ~680行 | ~280行 |
| 配置来源 | 自定义 | 与demo一致 |
| 依赖 | argparse复杂 | 简单 |

## 注意事项

1. **内存**: 记录所有62层输出需要几GB磁盘空间
2. **时间**: 首次运行需要编译，可能需要30-60分钟
3. **兼容性**: 输出格式与GPU版本兼容，可直接对比
4. **XLA张量**: 输出会自动尝试转换为CPU张量

## 故障排除

### 问题1: 编译失败

确保使用正确的模型路径：
```python
MODEL_PATH = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
TRACED_MODEL_PATH = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights-v2/"
```

### 问题2: 加载失败

检查是否已编译：
```bash
ls -lh /home/ubuntu/traced_model/MiniMax-M2-BF16-weights-v2/
```

### 问题3: Hooks不工作

检查模型结构：
```python
print(dir(model.model))
print(len(model.model.layers))
```

## 总结

这个V2版本：
- ✅ 基于官方demo，配置完全一致
- ✅ 代码简洁，易于理解和维护
- ✅ 移除了不可行的单层/多层测试
- ✅ 专注于完整模型的逐层输出记录
- ✅ 提供完整的统计和分析功能

现在可以直接运行测试了！
