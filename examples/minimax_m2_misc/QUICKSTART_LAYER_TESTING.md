# 快速开始：逐层测试MiniMax M2

## 目标

通过逐层测试定位模型推理问题，对比GPU和Neuron版本的输出差异。

## 前置条件

1. **GPU环境**（用于生成golden reference）
   - CUDA可用的GPU
   - 足够内存加载MiniMax M2模型（~100GB）
   - 模型路径：`/home/ubuntu/model_hf/MiniMax-M2-BF16/`

2. **Neuron环境**（用于实际测试）
   - AWS Inferentia或Trainium实例
   - neuronx-distributed-inference已安装

## 第一步：GPU上生成参考输出

在**有GPU的机器**上运行：

```bash
cd /home/ubuntu/neuronx-distributed-inference/examples

# 生成完整模型的所有层输出（作为golden reference）
python test_layer_by_layer_gpu.py --test-type extract-all
```

这个过程会：
- 加载GPU版本的MiniMax M2模型
- 使用相同的输入prompt运行模型
- 保存所有62层的输出到 `/home/ubuntu/traced_model/gpu_reference/full_model/`
- 每层保存为独立的`.pt`文件

**预期输出**：
```
加载GPU模型: /home/ubuntu/model_hf/MiniMax-M2-BF16/
  模型配置: 62 layers, 256 experts
  ✓ 模型加载完成

注册hooks...
  ✓ 注册hook: embedding
  ✓ 注册hook: layer_0
  ✓ 注册hook: layer_1
  ...
  ✓ 注册hook: layer_61
  ✓ 注册hook: final_norm

运行模型...
✓ 前向传播完成

层输出统计摘要:
  记录的层数: 64
  embedding: Mean=0.123456, Range=[-2.345, 3.456]
  layer_0: Mean=0.234567, Range=[-1.234, 2.345]
  ...

保存输出...
  ✓ 保存: /home/ubuntu/traced_model/gpu_reference/full_model/embedding_output.pt
  ✓ 保存: /home/ubuntu/traced_model/gpu_reference/full_model/layer_0_output.pt
  ...
```

## 第二步：Neuron上测试单层

在**Neuron实例**上，先测试单个层来快速验证：

```bash
cd /home/ubuntu/neuronx-distributed-inference/examples

# 测试第0层
python test_layer_by_layer.py --test-type single-layer --layer 0
```

**注意**：当前脚本是框架，需要补充实际的编译和推理逻辑。

## 第三步：对比GPU vs Neuron输出

假设你已经有GPU和Neuron的输出，运行对比：

```bash
# 对比单层输出
python compare_gpu_neuron_outputs.py \
  --gpu-path /home/ubuntu/traced_model/gpu_reference/full_model \
  --neuron-path /home/ubuntu/traced_model/test_layers/single_layer_0 \
  --layer layer_0
```

**预期输出**：
```
对比层: layer_0
  GPU路径: /home/ubuntu/traced_model/gpu_reference/full_model
  Neuron路径: /home/ubuntu/traced_model/test_layers/single_layer_0
  ✓ GPU输出 shape: torch.Size([1, 256, 5120]), dtype: torch.bfloat16
  ✓ Neuron输出 shape: torch.Size([1, 256, 5120]), dtype: torch.bfloat16

  关键指标:
    Shape匹配: True
    AllClose (rtol=0.001, atol=0.001): True
    相似度: 98.50%
    余弦相似度: 0.999234
    Pearson相关: 0.998765
    最大绝对差异: 5.234e-03
    平均绝对差异: 1.234e-04
    最大相对差异: 2.345e-02

  ✓ 输出匹配良好
```

## 第四步：根据结果诊断

### 场景A：单层测试通过

如果layer_0的对比结果良好（相似度>95%），继续测试更多层：

```bash
# 测试前5层
python test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 4

# 对比输出
python compare_gpu_neuron_outputs.py \
  --gpu-path /home/ubuntu/traced_model/gpu_reference/full_model \
  --neuron-path /home/ubuntu/traced_model/test_layers/layers_0_to_4 \
  --layer-pattern "layer_*" \
  --report-path /home/ubuntu/comparison_reports/
```

### 场景B：单层测试失败

如果layer_0就有问题（相似度<90%），需要更细粒度的调试：

1. **检查输入是否一致**
   ```python
   # 加载metadata检查
   import torch
   gpu_meta = torch.load('/home/ubuntu/traced_model/gpu_reference/full_model/metadata.pt')
   neuron_meta = torch.load('/home/ubuntu/traced_model/test_layers/single_layer_0/metadata.pt')

   print("GPU input_ids:", gpu_meta['input_ids'])
   print("Neuron input_ids:", neuron_meta['input_ids'])
   ```

2. **检查权重加载**
   - 验证Neuron版本是否正确加载了BF16权重
   - 检查是否有FP8量化问题

3. **逐组件测试**
   ```bash
   # 独立测试Attention模块
   python test_layer_by_layer.py --test-type component --component attention

   # 独立测试MoE模块
   python test_layer_by_layer.py --test-type component --component moe
   ```

### 场景C：前几层正常，后面层异常

使用二分查找定位问题层范围：

```bash
# 测试前半部分（0-30层）
python test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 30

# 如果前半部分正常，测试后半部分
python test_layer_by_layer.py --test-type multi-layer --start-layer 31 --end-layer 61

# 继续细分，例如测试15-30层
python test_layer_by_layer.py --test-type multi-layer --start-layer 15 --end-layer 30
```

## 第五步：生成完整对比报告

测试所有层后，生成详细报告：

```bash
python compare_gpu_neuron_outputs.py \
  --gpu-path /home/ubuntu/traced_model/gpu_reference/full_model \
  --neuron-path /home/ubuntu/traced_model/test_layers/full_model \
  --layer-pattern "layer_*" \
  --report-path /home/ubuntu/comparison_reports/full_model
```

查看报告：

```bash
# 查看摘要
cat /home/ubuntu/comparison_reports/full_model/comparison_summary.json

# 查看详细报告（可以用jq美化输出）
cat /home/ubuntu/comparison_reports/full_model/comparison_report.json | jq '.'
```

## 常见问题排查

### Q1: GPU脚本加载模型时OOM

**解决方案**：
```python
# 修改test_layer_by_layer_gpu.py中的加载方式
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,  # 使用bfloat16而不是float32
    device_map="auto",            # 自动分配到多个GPU
    low_cpu_mem_usage=True,       # 减少CPU内存使用
)
```

### Q2: 对比时发现shape不匹配

**可能原因**：
- 输入序列长度不一致
- Neuron版本使用了不同的padding

**解决方案**：
检查metadata.pt中的input_ids shape，确保使用相同的输入。

### Q3: 所有层的相似度都很低（<50%）

**可能原因**：
- 权重加载错误
- 使用了不同的模型checkpoint
- 数据类型转换问题

**解决方案**：
1. 确认GPU和Neuron使用相同的model_path
2. 检查Neuron版本的权重加载逻辑
3. 对比第一层的权重：
   ```python
   gpu_weight = gpu_model.model.layers[0].self_attn.q_proj.weight
   neuron_weight = neuron_model.model.layers[0].self_attn.q_proj.weight
   print("Weight差异:", (gpu_weight - neuron_weight).abs().max())
   ```

### Q4: NaN或Inf出现在某些层

**可能原因**：
- 数值溢出
- RMSNorm的eps太小
- 除零错误

**解决方案**：
1. 检查该层前一层的输出是否正常
2. 增大RMSNorm的eps值
3. 使用更高精度（float32）进行计算

## 下一步

1. **补充Neuron编译逻辑**：在`test_layer_by_layer.py`中实现实际的`torch_neuronx.trace()`调用
2. **添加权重对比**：独立对比GPU和Neuron加载的权重是否一致
3. **可视化差异**：生成热图显示哪些层、哪些位置差异最大
4. **自动化测试**：编写脚本自动运行完整的测试流程

## 获取帮助

如果遇到问题，请查看：
- 详细文档：`TEST_LAYER_BY_LAYER_README.md`
- 原始demo：`generation_minimax_m2_demo_v2.py`
- NeuronX分布式推理文档：https://awsdocs-neuron.readthedocs-hosted.com/
