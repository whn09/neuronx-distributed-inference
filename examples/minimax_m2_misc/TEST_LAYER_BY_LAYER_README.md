# MiniMax M2 逐层测试指南

## 背景

当整体编译和测试模型时发现结果不对，但不知道问题出在哪里。这个工具可以帮助你逐层测试模型，精确定位问题所在。

## 测试策略

MiniMax M2 模型结构：
```
NeuronMiniMaxM2ForCausalLM
├── Embedding Layer
├── 62 x Decoder Layers
│   ├── Input LayerNorm (RMSNorm)
│   ├── Self-Attention
│   │   ├── Q/K/V Projection
│   │   ├── Rotary Embedding
│   │   └── Output Projection
│   ├── Post-Attention LayerNorm (RMSNorm)
│   └── MoE (256 experts)
│       ├── Gate
│       └── Expert FFN layers
├── Final LayerNorm
└── LM Head
```

### 逐层测试流程

1. **测试单个组件** - 最细粒度，测试Attention、MoE、RMSNorm等单个组件
2. **测试单层Decoder** - 测试单个完整的decoder layer（包含attention + MoE）
3. **测试多层范围** - 测试layers 0-5, 6-10等小范围
4. **测试完整模型** - 最后测试整个62层模型

## 使用方法

### 1. 测试单个Decoder Layer

```bash
# 编译并测试第0层
python test_layer_by_layer.py --test-type single-layer --layer 0

# 编译并测试第10层
python test_layer_by_layer.py --test-type single-layer --layer 10

# 跳过编译，只测试已编译的第0层
python test_layer_by_layer.py --test-type single-layer --layer 0 --skip-compile
```

**优点**: 可以快速定位到具体是哪一层有问题

### 2. 测试多层范围

```bash
# 编译并测试前5层（layers 0-4）
python test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 4

# 测试中间层（layers 20-25）
python test_layer_by_layer.py --test-type multi-layer --start-layer 20 --end-layer 25

# 测试最后几层（layers 58-61）
python test_layer_by_layer.py --test-type multi-layer --start-layer 58 --end-layer 61
```

**优点**: 可以二分查找问题层的范围

### 3. 测试完整模型

```bash
# 编译并测试完整模型（所有62层）
python test_layer_by_layer.py --test-type full

# 只测试已编译的完整模型
python test_layer_by_layer.py --test-type full --skip-compile
```

### 4. 测试单个组件（TODO）

```bash
# 测试Attention模块
python test_layer_by_layer.py --test-type component --component attention

# 测试MoE模块
python test_layer_by_layer.py --test-type component --component moe

# 测试RMSNorm模块
python test_layer_by_layer.py --test-type component --component rmsnorm
```

### 5. 对比层输出（TODO）

```bash
# 对比第0层、第30层和第61层的输出
python test_layer_by_layer.py --test-type compare --compare-layers 0,30,61
```

## 调试建议

### 推荐的测试顺序

1. **快速验证**: 先测试单层（layer 0）
   ```bash
   python test_layer_by_layer.py --test-type single-layer --layer 0
   ```

2. **范围定位**: 如果单层正常，测试小范围（5层）
   ```bash
   python test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 4
   ```

3. **二分查找**: 逐步扩大范围或使用二分法
   ```bash
   # 测试前半部分（0-30层）
   python test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 30

   # 测试后半部分（31-61层）
   python test_layer_by_layer.py --test-type multi-layer --start-layer 31 --end-layer 61
   ```

4. **完整测试**: 最后测试完整模型
   ```bash
   python test_layer_by_layer.py --test-type full
   ```

### 如何判断测试结果

检查输出中的这些信息：

1. **编译是否成功**
   - 是否有编译错误或警告
   - 编译时间是否合理

2. **输出数值是否正常**
   - 检查输出的均值、方差、最小值、最大值
   - 是否出现NaN或Inf
   - 数值范围是否合理（例如不应该全是0）

3. **输出shape是否正确**
   - 检查hidden_states的shape是否为 `(batch_size, seq_len, hidden_size)`
   - 检查past_key_value的shape

### 常见问题诊断

| 症状 | 可能原因 | 测试方法 |
|------|----------|----------|
| 输出全是0或很小的值 | 权重加载问题、量化问题 | 测试单层，检查权重是否正确加载 |
| 输出出现NaN/Inf | 数值溢出、除零错误 | 逐组件测试，定位到具体操作 |
| 某些层正常，某些层异常 | 特定层的权重或配置问题 | 对比正常层和异常层的差异 |
| 编译时间过长 | MoE专家数量太多 | 减少测试的层数或调整配置 |

## 输出文件结构

```
/home/ubuntu/traced_model/test_layers/
├── single_layer_0/          # 单层测试输出
│   ├── config.pt
│   └── [neuronx编译产物]
├── single_layer_10/
├── layers_0_to_4/           # 多层测试输出
│   ├── config.pt
│   └── [neuronx编译产物]
├── layers_20_to_25/
└── full_model/              # 完整模型输出
    ├── config.pt
    └── [neuronx编译产物]
```

## GPU版本测试（Golden Reference）

为了精确定位问题，需要先在GPU上运行模型并保存输出作为参考基准。

### 1. 在GPU上提取所有层输出

```bash
# 在有GPU的机器上运行
python test_layer_by_layer_gpu.py --test-type extract-all
```

这会保存：
- 所有62层的输出
- Embedding和Final Norm的输出
- 完整的generation结果
- 每层的统计信息

### 2. 测试GPU版本的单层/多层

```bash
# 测试单层
python test_layer_by_layer_gpu.py --test-type single-layer --layer 0

# 测试多层
python test_layer_by_layer_gpu.py --test-type multi-layer --start-layer 0 --end-layer 5

# 测试完整模型
python test_layer_by_layer_gpu.py --test-type full
```

### 3. GPU输出文件结构

```
/home/ubuntu/traced_model/gpu_reference/
├── single_layer_0/
│   ├── layer_0_output.pt       # 该层的输出tensor
│   └── metadata.pt              # 包含input_ids, config等元数据
├── layers_0_to_5/
│   ├── layer_0_output.pt
│   ├── layer_1_output.pt
│   ├── ...
│   └── metadata.pt
└── full_model/
    ├── embedding_output.pt
    ├── layer_0_output.pt
    ├── layer_1_output.pt
    ├── ...
    ├── layer_61_output.pt
    ├── final_norm_output.pt
    ├── generation_output.pt
    └── metadata.pt
```

## GPU vs Neuron 输出对比

生成GPU参考输出后，可以与Neuron版本对比：

### 1. 对比单层输出

```bash
python compare_gpu_neuron_outputs.py \
  --gpu-path /home/ubuntu/traced_model/gpu_reference/single_layer_0 \
  --neuron-path /home/ubuntu/traced_model/test_layers/single_layer_0 \
  --layer layer_0
```

### 2. 对比多层输出

```bash
python compare_gpu_neuron_outputs.py \
  --gpu-path /home/ubuntu/traced_model/gpu_reference/full_model \
  --neuron-path /home/ubuntu/traced_model/test_layers/full_model \
  --layer-pattern "layer_*"
```

### 3. 生成详细报告

```bash
python compare_gpu_neuron_outputs.py \
  --gpu-path /home/ubuntu/traced_model/gpu_reference/full_model \
  --neuron-path /home/ubuntu/traced_model/test_layers/full_model \
  --report-path /home/ubuntu/comparison_reports/
```

这会生成：
- `comparison_report.json` - 每一层的详细差异指标
- `comparison_summary.json` - 整体统计摘要

### 4. 对比指标说明

对比工具会计算以下指标：

| 指标 | 说明 | 良好范围 |
|------|------|----------|
| max_abs_diff | 最大绝对差异 | < 0.01 |
| mean_abs_diff | 平均绝对差异 | < 0.001 |
| similarity_percentage | 相似元素百分比 | > 95% |
| cosine_similarity | 余弦相似度 | > 0.99 |
| pearson_correlation | Pearson相关系数 | > 0.99 |

**异常情况警告**：
- NaN或Inf值
- 相似度 < 95%
- 最大绝对差异 > 1.0

## 完整调试工作流

### 推荐的完整流程

```bash
# Step 1: 在GPU上生成golden reference
python test_layer_by_layer_gpu.py --test-type extract-all

# Step 2: 在Neuron上测试单层（快速验证）
python test_layer_by_layer.py --test-type single-layer --layer 0

# Step 3: 对比GPU vs Neuron单层输出
python compare_gpu_neuron_outputs.py \
  --gpu-path /home/ubuntu/traced_model/gpu_reference/full_model \
  --neuron-path /home/ubuntu/traced_model/test_layers/single_layer_0 \
  --layer layer_0

# Step 4: 如果单层有问题，逐组件调试
# 如果单层正常，测试多层范围（二分查找）
python test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 30

# Step 5: 对比多层输出，定位问题层
python compare_gpu_neuron_outputs.py \
  --gpu-path /home/ubuntu/traced_model/gpu_reference/full_model \
  --neuron-path /home/ubuntu/traced_model/test_layers/layers_0_to_30 \
  --report-path /home/ubuntu/comparison_reports/layers_0_to_30

# Step 6: 精确定位到问题层后，深入分析该层的权重、配置等
```

## 下一步优化（TODO）

当前脚本提供了测试框架，以下功能需要进一步实现：

1. **实现torch_neuronx.trace()调用** - 实际的Neuron模型编译逻辑
2. **实现推理和输出验证** - 加载编译后的Neuron模型并进行推理
3. **实现单组件测试** - 独立测试Attention、MoE等组件
4. **添加权重对比** - 对比GPU和Neuron版本加载的权重是否一致
5. **添加可视化** - 生成差异热图，直观显示哪些层有问题

## 脚本文件说明

- `test_layer_by_layer.py` - Neuron版本逐层测试（需要AWS Inferentia/Trainium）
- `test_layer_by_layer_gpu.py` - GPU版本测试，生成golden reference
- `compare_gpu_neuron_outputs.py` - GPU vs Neuron输出对比工具
- `TEST_LAYER_BY_LAYER_README.md` - 本文档

## 参考

- 原始demo: `generation_minimax_m2_demo_v2.py`
- 模型实现: `src/neuronx_distributed_inference/models/minimax_m2/modeling_minimax_m2_v2.py`
- GPU模型实现: `src/neuronx_distributed_inference/models/minimax_m2/modeling_minimax_m2_gpu.py`
