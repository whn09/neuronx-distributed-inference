"""
调试脚本：检查模型加载时的dtype
用于诊断内存占用1TB的问题
"""

import torch
from safetensors import safe_open
from pathlib import Path
import json

MODEL_PATH = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"

def check_safetensors_dtype():
    """检查safetensors文件中实际的dtype"""
    print("\n=== 检查safetensors文件的dtype ===")

    # 读取index
    index_path = Path(MODEL_PATH) / "model.safetensors.index.json"
    with open(index_path, 'r') as f:
        index = json.load(f)

    # 检查第一个shard文件
    shard_files = sorted(set(index['weight_map'].values()))
    first_shard = shard_files[0]
    print(f"\n检查文件: {first_shard}")

    shard_path = Path(MODEL_PATH) / first_shard
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"  包含 {len(keys)} 个参数")

        # 检查前几个权重的dtype
        print(f"\n  前10个参数的dtype:")
        for i, key in enumerate(keys[:10]):
            tensor = f.get_tensor(key)
            print(f"    {key}")
            print(f"      shape: {tensor.shape}, dtype: {tensor.dtype}, size: {tensor.numel() * tensor.element_size() / 1024 / 1024:.2f} MB")

        # 特别检查MoE expert的权重
        expert_keys = [k for k in keys if 'block_sparse_moe.experts' in k and 'w1.weight' in k]
        if expert_keys:
            expert_key = expert_keys[0]
            tensor = f.get_tensor(expert_key)
            print(f"\n  MoE Expert权重示例: {expert_key}")
            print(f"      shape: {tensor.shape}, dtype: {tensor.dtype}")
            print(f"      单个权重大小: {tensor.numel() * tensor.element_size() / 1024 / 1024:.2f} MB")

            # 计算如果是float32的大小
            size_fp32 = tensor.numel() * 4 / 1024 / 1024
            size_bf16 = tensor.numel() * 2 / 1024 / 1024
            print(f"      如果是FP32: {size_fp32:.2f} MB")
            print(f"      如果是BF16: {size_bf16:.2f} MB")


def estimate_memory_usage():
    """估算内存使用"""
    print("\n\n=== 估算内存使用 ===")

    # MiniMax M2 配置
    num_layers = 62
    num_experts = 256
    hidden_size = 5120
    intermediate_size = 1536

    # 计算单层MoE的大小
    # gate_up_proj: [num_experts, hidden_size, 2 * intermediate_size]
    gate_up_proj_elements = num_experts * hidden_size * 2 * intermediate_size

    # down_proj: [num_experts, intermediate_size, hidden_size]
    down_proj_elements = num_experts * intermediate_size * hidden_size

    total_elements_per_layer = gate_up_proj_elements + down_proj_elements

    print(f"\n配置:")
    print(f"  层数: {num_layers}")
    print(f"  专家数: {num_experts}")
    print(f"  隐藏维度: {hidden_size}")
    print(f"  中间维度: {intermediate_size}")

    print(f"\n单层MoE大小:")
    print(f"  gate_up_proj元素数: {gate_up_proj_elements:,}")
    print(f"  down_proj元素数: {down_proj_elements:,}")
    print(f"  总元素数: {total_elements_per_layer:,}")

    # FP32
    size_fp32_per_layer = total_elements_per_layer * 4 / 1024 / 1024 / 1024
    size_fp32_total = size_fp32_per_layer * num_layers
    print(f"\n如果使用FP32:")
    print(f"  单层: {size_fp32_per_layer:.2f} GB")
    print(f"  总共: {size_fp32_total:.2f} GB")

    # BF16
    size_bf16_per_layer = total_elements_per_layer * 2 / 1024 / 1024 / 1024
    size_bf16_total = size_bf16_per_layer * num_layers
    print(f"\n如果使用BF16:")
    print(f"  单层: {size_bf16_per_layer:.2f} GB")
    print(f"  总共: {size_bf16_total:.2f} GB")

    print(f"\n差异: {size_fp32_total - size_bf16_total:.2f} GB")

    print(f"\n结论:")
    if size_fp32_total > 900:
        print(f"  ⚠️  FP32会占用 {size_fp32_total:.0f} GB ≈ 1TB （符合观察到的内存占用！）")
    if size_bf16_total < 500:
        print(f"  ✓ BF16只需 {size_bf16_total:.0f} GB （符合模型文件大小）")


def check_conversion_code():
    """检查转换代码中可能导致dtype改变的地方"""
    print("\n\n=== 可能的问题点 ===")

    print("\n1. safetensors加载 (modeling_minimax_m2_v2.py:735)")
    print("   model_sd[key] = f.get_tensor(key)")
    print("   → safetensors应该保持原始dtype，这里应该没问题")

    print("\n2. 获取dtype (modeling_minimax_m2_v2.py:255)")
    print("   dtype = neuron_state_dict[f'layers.{l}.block_sparse_moe.experts.0.w1.weight'].dtype")
    print("   → 如果这里的dtype不是bf16，后续创建的tensor就会是错误的类型")

    print("\n3. 创建临时tensor (modeling_minimax_m2_v2.py:258-264)")
    print("   gate_up_proj = torch.empty(..., dtype=dtype, device=device)")
    print("   → 使用上一步获取的dtype，如果dtype错误，这里就会创建FP32的大tensor")

    print("\n建议修复:")
    print("  在get_state_dict函数开始处添加类型转换:")
    print("    # 确保所有权重都是BF16")
    print("    for key in model_sd.keys():")
    print("        if isinstance(model_sd[key], torch.Tensor) and model_sd[key].dtype == torch.float32:")
    print("            model_sd[key] = model_sd[key].to(torch.bfloat16)")


if __name__ == "__main__":
    print("="*60)
    print("MiniMax M2 内存占用调试")
    print("="*60)

    check_safetensors_dtype()
    estimate_memory_usage()
    check_conversion_code()

    print("\n" + "="*60)
    print("调试完成")
    print("="*60)
