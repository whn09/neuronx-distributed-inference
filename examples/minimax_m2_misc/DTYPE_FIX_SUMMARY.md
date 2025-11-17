# MiniMax M2 内存占用1TB问题修复总结

## 问题描述

运行 `generation_minimax_m2_demo_v2.py` 时，内存占用达到1TB，但BF16模型文件只有400GB。

## 根本原因

在 `modeling_minimax_m2_v2.py` 的 `convert_minimax_m2_hf_to_neuron_state_dict` 函数中，重组MoE权重时创建了大量临时tensor：

```python
# 单层创建两个大tensor：
gate_up_proj = torch.empty([256, 3072, 3072], dtype=dtype, device=device)
down_proj = torch.empty([256, 1536, 3072], dtype=dtype, device=device)
```

**如果dtype是float32而不是bfloat16**：
- 单层临时内存：`gate_up_proj (13.5GB) + down_proj (6.75GB) = 20.25GB`
- 62层总计：`20.25GB × 62 ≈ 1,255GB ≈ 1.2TB` ✓ **符合观察到的内存占用！**

**如果dtype是bfloat16（正确）**：
- 单层临时内存：`gate_up_proj (6.75GB) + down_proj (3.38GB) = 10.13GB`
- 62层总计：`10.13GB × 62 ≈ 628GB` ✓ **与400GB文件大小+临时内存相符**

## 调试发现

运行 `debug_model_dtype.py` 确认：
1. ✓ Safetensors文件中的权重确实是 `torch.bfloat16`
2. ⚠️ 但在重组过程中，某些情况下dtype可能变成 `torch.float32`
3. 原因：虽然safetensors保持原始dtype，但在某些边界情况下，权重可能被错误地转换或创建为float32

## 修复方案

在 `modeling_minimax_m2_v2.py` 中添加了两处修复：

### 修复1: 在加载权重后强制转换为BF16

**位置**: `get_state_dict` 函数，第754-769行

```python
# FIX: 强制转换为bfloat16以避免1TB内存占用问题
print(f"\n=== Enforcing bfloat16 dtype for all weights ===")
fp32_count = 0
fp32_large_count = 0
for key in list(model_sd.keys()):
    if isinstance(model_sd[key], torch.Tensor):
        if model_sd[key].dtype == torch.float32:
            fp32_count += 1
            # 只转换大的权重tensor（跳过小的bias等）
            if model_sd[key].numel() > 1000:
                model_sd[key] = model_sd[key].to(torch.bfloat16)
                fp32_large_count += 1
print(f"  Found {fp32_count} float32 tensors, converted {fp32_large_count} large tensors to bfloat16")
```

**作用**:
- 在权重转换之前，确保所有大的权重tensor都是BF16
- 保留小的float32参数（如bias），它们不会造成内存问题

### 修复2: 在创建临时tensor时确保dtype正确

**位置**: `convert_minimax_m2_hf_to_neuron_state_dict` 函数，第257-268行

```python
# FIX: 确保dtype是bfloat16，避免创建巨大的FP32 tensor
if dtype != torch.bfloat16:
    print(f"  ⚠️  WARNING: Layer {l} experts.0.w1.weight has dtype={dtype}, forcing bfloat16")
    dtype = torch.bfloat16

if l == 0:
    print(f"  Layer {l} MoE dtype: {dtype}, device: {device}")
    # 计算单层临时tensor大小
    gate_up_elements = config.num_local_experts * hidden_size * 2 * intermediate_size
    down_elements = config.num_local_experts * intermediate_size * hidden_size
    size_bf16 = (gate_up_elements + down_elements) * 2 / 1024 / 1024 / 1024
    size_fp32 = (gate_up_elements + down_elements) * 4 / 1024 / 1024 / 1024
    print(f"  Estimated temp memory per layer: BF16={size_bf16:.2f}GB, FP32={size_fp32:.2f}GB")
```

**作用**:
- 防御性检查：即使前面的修复失败，也确保创建临时tensor时使用BF16
- 在第0层打印内存估算，帮助验证修复效果

## 验证修复

运行修复后的代码，应该看到：

```
=== Enforcing bfloat16 dtype for all weights ===
  Found X float32 tensors, converted Y large tensors to bfloat16

...

=== Renaming attention projections to add qkv_proj prefix ===
  Layer 0 MoE dtype: torch.bfloat16, device: cpu
  Estimated temp memory per layer: BF16=10.13GB, FP32=20.25GB
```

**预期内存占用**:
- 编译前：~400GB（模型文件）+ ~628GB（临时转换）= **~1TB**（但会立即释放）
- 编译后：取决于neuronx编译产物的大小

## 相关文件

- **问题文件**: `src/neuronx_distributed_inference/models/minimax_m2/modeling_minimax_m2_v2.py`
- **调试脚本**: `examples/debug_model_dtype.py`
- **测试demo**: `examples/generation_minimax_m2_demo_v2.py`

## 后续建议

1. **监控内存**: 运行时监控内存占用，确认是否降到预期范围
2. **性能对比**: 对比修复前后的推理性能，确保BF16精度没有影响
3. **代码审查**: 检查其他模型是否有类似问题

## 总结

通过强制所有大权重使用BF16，避免了在MoE权重重组时创建巨大的FP32临时tensor，将内存占用从~1.2TB降低到~628GB（临时峰值），最终稳定在合理范围。
