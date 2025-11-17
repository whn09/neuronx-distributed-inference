# 完整模型逐层输出记录指南

## 概述

由于MoE模型的单层权重（4.5GB）超过了Neuron编译器的4GB限制，我们使用**完整模型**来记录每一层的输出。

完整模型可以正常编译和运行，因为：
- 自动分片到多个NeuronCores
- 每个Core只处理部分权重
- 绕过单张量4GB限制

---

## 使用方法

### 步骤1: 编译完整模型（如果还没编译）

```bash
cd /home/ubuntu/neuronx-distributed-inference/examples

# 编译模型（只需要做一次）
python minimax_m2_misc/test_layer_by_layer.py --test-type full
```

**预计时间**：第一次编译可能需要30-60分钟

### 步骤2: 运行推理并记录每层输出

如果已经有编译好的模型，使用 `--skip-compile`：

```bash
python minimax_m2_misc/test_layer_by_layer.py --test-type full --skip-compile
```

### 步骤3: 查看输出

输出会保存到：
```
/home/ubuntu/traced_model/test_layers/full_model/
├── layer_0_output.pt       # 第0层输出
├── layer_1_output.pt       # 第1层输出
├── ...
├── layer_61_output.pt      # 第61层输出
├── embedding_output.pt     # Embedding层输出
├── final_norm_output.pt    # Final Norm层输出
└── metadata.pt             # 元数据（包含统计信息）
```

---

## 输出内容

### 运行时显示

```
关键层统计信息:
  embedding:
    Shape: [1, 128, 3072]
    Mean: 0.123456, Std: 0.234567
    Range: [-1.234567, 1.234567]
    Has NaN: False, Has Inf: False

  layer_0:
    Shape: [1, 128, 3072]
    Mean: 0.123456, Std: 0.234567
    Range: [-1.234567, 1.234567]
    Has NaN: False, Has Inf: False

  layer_30:
    Shape: [1, 128, 3072]
    Mean: 0.123456, Std: 0.234567
    Range: [-1.234567, 1.234567]
    Has NaN: False, Has Inf: False

  ...
```

### 保存的文件

每个 `layer_X_output.pt` 文件包含该层的输出张量：

```python
import torch

# 加载某一层的输出
layer_0_output = torch.load('/home/ubuntu/traced_model/test_layers/full_model/layer_0_output.pt')

print(f"Shape: {layer_0_output.shape}")
print(f"Mean: {layer_0_output.float().mean()}")
print(f"Std: {layer_0_output.float().std()}")
```

`metadata.pt` 包含：
- 输入prompt
- 输入token IDs
- 所有层的统计信息
- 模型配置

---

## 分析每层输出

### 方法1: 查看统计信息

```python
import torch

# 加载元数据
metadata = torch.load('/home/ubuntu/traced_model/test_layers/full_model/metadata.pt')

# 查看所有层的统计
stats = metadata['statistics']

for layer_name, stat in sorted(stats.items()):
    if 'layer_' in layer_name:  # 只看decoder layers
        print(f"{layer_name}: Mean={stat['mean']:.6f}, Has NaN={stat['has_nan']}")
```

### 方法2: 检查特定层

```python
import torch

# 加载特定层的输出
layer_30 = torch.load('/home/ubuntu/traced_model/test_layers/full_model/layer_30_output.pt')

# 检查数值分布
print(f"Min: {layer_30.min()}")
print(f"Max: {layer_30.max()}")
print(f"Mean: {layer_30.float().mean()}")
print(f"Has NaN: {torch.isnan(layer_30).any()}")
print(f"Has Inf: {torch.isinf(layer_30).any()}")

# 检查特定位置的值
print(f"First token features (前5维): {layer_30[0, 0, :5]}")
```

### 方法3: 对比相邻层

```python
import torch

# 加载相邻两层
layer_30 = torch.load('/home/ubuntu/traced_model/test_layers/full_model/layer_30_output.pt')
layer_31 = torch.load('/home/ubuntu/traced_model/test_layers/full_model/layer_31_output.pt')

# 计算差异
diff = (layer_31 - layer_30).abs()
print(f"Max diff: {diff.max()}")
print(f"Mean diff: {diff.mean()}")

# 如果差异太大，可能有问题
if diff.max() > 10.0:
    print("⚠️  警告：相邻层差异过大！")
```

---

## 与GPU版本对比

### 步骤1: 在GPU上运行

```bash
# 在GPU机器上
python minimax_m2_misc/test_layer_by_layer_gpu.py --test-type extract-all
```

这会生成：
```
/home/ubuntu/traced_model/gpu_reference/full_model/
├── layer_0_output.pt
├── layer_1_output.pt
├── ...
```

### 步骤2: 对比输出

```python
import torch

# 加载Neuron和GPU的同一层输出
neuron_layer_30 = torch.load('/home/ubuntu/traced_model/test_layers/full_model/layer_30_output.pt')
gpu_layer_30 = torch.load('/home/ubuntu/traced_model/gpu_reference/full_model/layer_30_output.pt')

# 计算差异
diff = (neuron_layer_30 - gpu_layer_30).abs()
relative_diff = diff / (gpu_layer_30.abs() + 1e-8)

print(f"绝对差异:")
print(f"  Max: {diff.max():.6f}")
print(f"  Mean: {diff.mean():.6f}")
print(f"  Median: {diff.median():.6f}")

print(f"\n相对差异:")
print(f"  Max: {relative_diff.max():.6f}")
print(f"  Mean: {relative_diff.mean():.6f}")

# 计算相似度
cosine_sim = torch.nn.functional.cosine_similarity(
    neuron_layer_30.flatten(),
    gpu_layer_30.flatten(),
    dim=0
)
print(f"\nCosine Similarity: {cosine_sim:.6f}")

# 良好范围：
# - Max abs diff < 0.01
# - Mean abs diff < 0.001
# - Cosine similarity > 0.99
```

---

## 常见问题排查

### Q1: 如何找到有问题的层？

```python
import torch

metadata = torch.load('/home/ubuntu/traced_model/test_layers/full_model/metadata.pt')
stats = metadata['statistics']

# 检查每一层
problem_layers = []
for layer_name, stat in stats.items():
    if 'layer_' in layer_name:
        # 检查异常值
        if stat.get('has_nan', False) or stat.get('has_inf', False):
            problem_layers.append(layer_name)
            print(f"❌ {layer_name}: Has NaN={stat['has_nan']}, Has Inf={stat['has_inf']}")

        # 检查数值范围
        if abs(stat.get('mean', 0)) > 10.0:
            problem_layers.append(layer_name)
            print(f"⚠️  {layer_name}: Mean={stat['mean']:.6f} (异常大)")

if not problem_layers:
    print("✓ 所有层的输出看起来正常")
else:
    print(f"\n发现 {len(problem_layers)} 个有问题的层")
```

### Q2: 某一层的输出突然变化很大怎么办？

```python
import torch

# 加载前后几层
prev_layer = torch.load(f'/home/ubuntu/traced_model/test_layers/full_model/layer_{N-1}_output.pt')
current_layer = torch.load(f'/home/ubuntu/traced_model/test_layers/full_model/layer_{N}_output.pt')
next_layer = torch.load(f'/home/ubuntu/traced_model/test_layers/full_model/layer_{N+1}_output.pt')

# 计算变化幅度
change_in = (current_layer - prev_layer).abs().mean()
change_out = (next_layer - current_layer).abs().mean()

print(f"Layer {N-1} -> {N}: {change_in:.6f}")
print(f"Layer {N} -> {N+1}: {change_out:.6f}")

# 如果变化异常大，检查该层的具体输出
print(f"\nLayer {N} 统计:")
print(f"  Mean: {current_layer.float().mean():.6f}")
print(f"  Std: {current_layer.float().std():.6f}")
print(f"  Min: {current_layer.min():.6f}")
print(f"  Max: {current_layer.max():.6f}")
```

### Q3: 如何可视化某一层的输出分布？

```python
import torch
import matplotlib.pyplot as plt

layer_30 = torch.load('/home/ubuntu/traced_model/test_layers/full_model/layer_30_output.pt')

# 展平并画直方图
values = layer_30.flatten().cpu().numpy()

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(values, bins=100)
plt.title('Layer 30 Output Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.plot(values[:1000])
plt.title('First 1000 values')
plt.xlabel('Index')
plt.ylabel('Value')

plt.subplot(1, 3, 3)
plt.imshow(layer_30[0, :, :100].cpu().numpy(), aspect='auto', cmap='viridis')
plt.title('Layer 30 Output Heatmap (first 100 features)')
plt.xlabel('Feature dim')
plt.ylabel('Sequence position')
plt.colorbar()

plt.tight_layout()
plt.savefig('layer_30_analysis.png')
print("✓ 图表已保存到 layer_30_analysis.png")
```

---

## 完整工作流示例

```bash
# 1. 编译模型（第一次运行）
python minimax_m2_misc/test_layer_by_layer.py --test-type full

# 2. 运行推理并记录所有层输出
python minimax_m2_misc/test_layer_by_layer.py --test-type full --skip-compile

# 3. 分析输出（Python脚本）
python <<EOF
import torch

# 加载元数据
metadata = torch.load('/home/ubuntu/traced_model/test_layers/full_model/metadata.pt')
stats = metadata['statistics']

# 打印所有层的摘要
print("所有层统计摘要:\n")
for layer_name in sorted(stats.keys()):
    if 'layer_' in layer_name:
        stat = stats[layer_name]
        status = "✓" if not stat.get('has_nan') and not stat.get('has_inf') else "✗"
        print(f"{status} {layer_name:15s} Mean={stat['mean']:8.4f} Std={stat['std']:8.4f}")
EOF

# 4. 与GPU版本对比（如果有的话）
python compare_gpu_neuron_outputs.py \
  --gpu-path /home/ubuntu/traced_model/gpu_reference/full_model \
  --neuron-path /home/ubuntu/traced_model/test_layers/full_model \
  --layer-pattern "layer_*" \
  --report-path /home/ubuntu/comparison_reports/
```

---

## 注意事项

1. **内存需求**：记录所有62层的输出需要大量内存，确保有足够的磁盘空间（预计几GB）

2. **XLA张量**：输出会自动转换为CPU张量保存，但如果转换失败会保留XLA张量引用

3. **首次运行**：第一次运行推理时，XLA需要编译图，会比较慢（几分钟）

4. **Hook开销**：注册hooks会略微降低推理速度，但对于调试来说是值得的

5. **批量处理**：如果要测试多个不同的输入，可以修改prompt并多次运行

---

## 总结

使用完整模型的优势：
- ✅ 可以正常编译和运行（自动分片）
- ✅ 记录所有62层的输出
- ✅ 与GPU版本格式兼容，便于对比
- ✅ 包含详细的统计信息

使用此方法可以：
1. 定位哪一层出现了异常
2. 对比Neuron vs GPU的差异
3. 验证模型逻辑的正确性
4. 调试数值稳定性问题

现在就可以开始测试了！
