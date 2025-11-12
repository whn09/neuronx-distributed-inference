# FP8 Quantization: weight_scale_inv vs scale

## 🎯 核心概念速查

```
weight_scale_inv (HuggingFace) = 1 / scale (Neuron)
```

---

## 📊 完整流程图

```
训练阶段 (Training)
─────────────────────────────────────────────────────────────
  原始权重 (bfloat16/float32)
  [6144, 3072] 矩阵
  值范围: 例如 [-15.2, +28.7]

         │ FP8 量化
         ↓

  1. 计算缩放因子
     abs_max = 28.7
     fp8_max = 448.0
     scale = abs_max / fp8_max = 0.0641

  2. 量化权重
     quantized_weight = original_weight / scale
     现在值范围: [-237.4, +448.0]  ✓ 适合FP8范围

  3. 转换为FP8格式
     weight_fp8 = quantized_weight.to(torch.float8_e4m3fn)

  4. 保存时计算倒数 (为了推理时快速!)
     weight_scale_inv = 1 / scale = 15.625


存储在Checkpoint
─────────────────────────────────────────────────────────────
  📁 model.safetensors

  layers.0.self_attn.q_proj.weight          ← FP8量化权重
    dtype: float8_e4m3fn
    shape: [6144, 3072]
    size: 6144 × 3072 × 1 byte = 18.4 MB

  layers.0.self_attn.q_proj.weight_scale_inv  ← Scale的倒数
    dtype: float32
    shape: [48, 24]  (block-wise: 每128×128一个scale)
    size: 48 × 24 × 4 bytes = 4.6 KB


推理阶段 (Inference)
─────────────────────────────────────────────────────────────

方法1: 使用HuggingFace命名 (weight_scale_inv)

  dequantized = weight_fp8 * weight_scale_inv
                ↑            ↑
              量化权重      1/scale (已预计算)

  = (original / scale) * (1 / scale)  ✗ 错误!

  实际上HF在推理时:
  dequantized = weight_fp8 * weight_scale_inv
              = (original / scale) * scale  ← weight_scale_inv在他们那边的语义不同!


方法2: 使用Neuron命名 (scale)

  Neuron框架首先转换:
    neuron_scale = 1.0 / hf_weight_scale_inv
                 = 1.0 / (1/scale)
                 = scale

  然后推理:
    dequantized = weight_fp8 * neuron_scale
                = (original / scale) * scale
                = original  ✓ 正确恢复!
```

---

## 🔢 数学关系

```python
# 训练时 (Quantization)
scale = abs_max(weight) / fp8_max              # scale用于量化
weight_fp8 = weight / scale                    # 量化
weight_scale_inv = 1 / scale                   # 保存倒数

# 推理时 (Dequantization)
# HF格式 → Neuron格式转换
neuron_scale = 1 / weight_scale_inv            # 转回scale
             = 1 / (1/scale)
             = scale

# 恢复原始权重
weight_original = weight_fp8 * neuron_scale
                = (weight/scale) * scale
                = weight
```

---

## 🧱 Block-wise量化示例

对于 `layers.0.self_attn.q_proj.weight`:

```
权重矩阵: [6144, 3072]
块大小: [128, 128]

分块方式:
┌─────┬─────┬─────┬───┬─────┐
│ B00 │ B01 │ B02 │...│ B023│  每个Bxy是128×128的块
├─────┼─────┼─────┼───┼─────┤
│ B10 │ B11 │ B12 │...│ B123│
├─────┼─────┼─────┼───┼─────┤
│ B20 │ B21 │ B22 │...│ B223│
│ ... │ ... │ ... │...│ ... │
├─────┼─────┼─────┼───┼─────┤
│ B470│ B471│ B472│...│B4723│
└─────┴─────┴─────┴───┴─────┘
   ↑                      ↑
  24列块                24列
  (3072/128)

48行块 (6144/128)

每个块有独立的scale:
scale_matrix shape: [48, 24]

scale[0,0] 用于 Block B00
scale[0,1] 用于 Block B01
scale[1,0] 用于 Block B10
...

量化:
  B00_fp8 = B00 / scale[0,0]
  B01_fp8 = B01 / scale[0,1]
  ...

反量化:
  B00_original = B00_fp8 * scale[0,0]
  B01_original = B01_fp8 * scale[0,1]
  ...
```

---

## 💾 存储效率对比

### 原始模型 (bfloat16)
```
layers.0.self_attn.q_proj.weight: [6144, 3072]
存储: 6144 × 3072 × 2 bytes = 36.9 MB
```

### FP8量化模型
```
layers.0.self_attn.q_proj.weight: [6144, 3072] (float8_e4m3fn)
存储: 6144 × 3072 × 1 byte = 18.4 MB

layers.0.self_attn.q_proj.weight_scale_inv: [48, 24] (float32)
存储: 48 × 24 × 4 bytes = 4.6 KB

总计: 18.4 MB + 4.6 KB ≈ 18.4 MB

节省: (36.9 - 18.4) / 36.9 = 50.1% ✓
```

整个模型:
- 原始: 48,239个bfloat16参数 ≈ 96 GB
- FP8量化: 47,864个FP8权重 + 47,864个scale ≈ 50 GB
- **节省约 46 GB 内存!**

---

## ⚡ 推理性能提升

1. **内存带宽**: FP8是bfloat16的一半大小 → 加载速度快2倍
2. **计算速度**: 硬件加速的FP8 GEMM (矩阵乘法)比bfloat16快
3. **KV Cache**: Attention的K/V缓存也使用FP8 → 可支持更长上下文

---

## 🔍 您遇到的错误分析

### ❌ 错误配置
```python
neuron_config = MoENeuronConfig(
    quantized_mlp_kernel_enabled=True,
    modules_to_not_convert=["lm_head", "self_attn"],  # ← 错误!
)
```

系统逻辑:
```
if "self_attn" in modules_to_not_convert:
    # 不期望attention层有scale参数
    expected_scale = None
else:
    # 期望有scale参数
    expected_scale = load_scale_from_checkpoint()

# 加载权重
weight = load_weight()
scale = load_scale() if needs_scale else None

# 验证维度
if scale is not None:
    assert scale.shape[axis] == weight.shape[axis]  # ← 在这里失败!
```

问题:
1. Checkpoint里**有** `q_proj.weight_scale_inv` (因为attention层确实被FP8量化了)
2. 系统加载并转换为 `scale`
3. 但因为 `"self_attn"` 在 `modules_to_not_convert` 中
4. 某些代码路径期望 `scale = None`
5. 维度验证时发现 `scale` 不是 `None` → AssertionError!

### ✅ 正确配置
```python
neuron_config = MoENeuronConfig(
    quantized_mlp_kernel_enabled=True,
    modules_to_not_convert=["lm_head"],  # 只排除真正未量化的层
)
```

这样系统知道:
- Attention层**有**FP8量化 → 期望并正确处理 scale 参数
- MoE Expert层**有**FP8量化 → 期望并正确处理 scale 参数
- lm_head **没有**FP8量化 → 不期望 scale 参数

---

## 📚 总结表

| 术语 | 数学含义 | 用途 | 位置 |
|------|----------|------|------|
| **scale** | `abs_max / fp8_max` | 量化除数 | 训练时计算 |
| **weight_scale_inv** | `1 / scale` | 反量化乘数 | HF checkpoint |
| **neuron_scale** | `1 / weight_scale_inv = scale` | 反量化乘数 | Neuron推理 |

关键转换:
```python
# HF checkpoint → Neuron framework
neuron_scale = 1.0 / hf_weight_scale_inv
```

推理公式:
```python
# 恢复原始权重
original_weight = quantized_weight_fp8 * neuron_scale
```

---

## 🎓 延伸阅读

FP8量化的优势:
1. 内存占用减半
2. 计算速度提升 (硬件加速)
3. 精度损失很小 (特别是block-wise量化)
4. 适合大模型推理

为什么不用INT8?
- FP8保留浮点格式，更容易处理大范围的值
- FP8在Transformer模型上的精度损失比INT8小
- 新一代AI加速器(如Neuron)对FP8有原生支持

MiniMax-M2的量化策略:
- Attention层: FP8 ✅
- MoE Expert层: FP8 ✅
- Router (gate): FP32 (需要高精度做expert选择)
- Embedding/LM Head: bfloat16 (输入输出保持高精度)
- LayerNorm: bfloat16 (归一化需要高精度)
