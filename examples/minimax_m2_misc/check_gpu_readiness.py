"""
检查GPU环境是否准备好运行MiniMax M2模型

运行这个脚本来验证：
1. GPU数量和容量
2. CUDA和PyTorch版本
3. 必要的库
4. 内存是否足够
"""

import sys
from pathlib import Path

def check_cuda():
    """检查CUDA和PyTorch"""
    print("\n=== 检查CUDA和PyTorch ===")
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        print(f"✓ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA版本: {torch.version.cuda}")
            print(f"✓ cuDNN版本: {torch.backends.cudnn.version()}")
        return torch.cuda.is_available()
    except ImportError:
        print("✗ PyTorch未安装")
        return False


def check_gpus():
    """检查GPU配置"""
    print("\n=== 检查GPU配置 ===")
    try:
        import torch
        if not torch.cuda.is_available():
            print("✗ 未检测到CUDA GPU")
            return False, 0, 0

        num_gpus = torch.cuda.device_count()
        print(f"✓ 检测到 {num_gpus} 个GPU\n")

        total_memory = 0
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            gpu_memory = props.total_memory / (1024 ** 3)  # GB
            total_memory += gpu_memory
            print(f"  GPU {i}:")
            print(f"    名称: {props.name}")
            print(f"    显存: {gpu_memory:.2f} GB")
            print(f"    计算能力: {props.major}.{props.minor}")
            print(f"    多处理器数量: {props.multi_processor_count}")

        print(f"\n  总显存: {total_memory:.2f} GB")

        # 检查是否足够
        required_memory = 450  # GB（模型403GB + 开销）
        if total_memory >= required_memory:
            print(f"  ✓ 总显存足够（需要 ~{required_memory}GB）")
            return True, num_gpus, total_memory
        else:
            print(f"  ⚠️  总显存可能不够（需要 ~{required_memory}GB，当前 {total_memory:.2f}GB）")
            print(f"  建议：使用更多或更大的GPU，或启用CPU offload")
            return False, num_gpus, total_memory

    except Exception as e:
        print(f"✗ 检查GPU时出错: {e}")
        return False, 0, 0


def check_libraries():
    """检查必要的库"""
    print("\n=== 检查必要的库 ===")

    libraries = {
        'transformers': 'Hugging Face Transformers',
        'accelerate': 'Hugging Face Accelerate (多GPU支持)',
        'safetensors': 'Safetensors (快速加载)',
    }

    all_installed = True
    for lib, desc in libraries.items():
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {desc}: {version}")
        except ImportError:
            print(f"✗ {desc}: 未安装")
            print(f"  安装命令: pip install {lib}")
            all_installed = False

    return all_installed


def check_model_files():
    """检查模型文件"""
    print("\n=== 检查模型文件 ===")

    model_path = Path("/home/ubuntu/model_hf/MiniMax-M2-BF16/")

    if not model_path.exists():
        print(f"✗ 模型路径不存在: {model_path}")
        return False

    print(f"✓ 模型路径: {model_path}")

    # 检查关键文件
    required_files = [
        'config.json',
        'tokenizer_config.json',
        'model.safetensors.index.json',
    ]

    all_exist = True
    for filename in required_files:
        filepath = model_path / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 ** 2)  # MB
            print(f"  ✓ {filename} ({size:.2f} MB)")
        else:
            print(f"  ✗ {filename} 缺失")
            all_exist = False

    # 检查safetensors分片数量
    shard_files = list(model_path.glob("model-*.safetensors"))
    if shard_files:
        total_size = sum(f.stat().st_size for f in shard_files) / (1024 ** 3)  # GB
        print(f"  ✓ 找到 {len(shard_files)} 个safetensors分片")
        print(f"  ✓ 总大小: {total_size:.2f} GB")
    else:
        print(f"  ✗ 未找到safetensors分片文件")
        all_exist = False

    return all_exist


def check_nvlink():
    """检查NVLink连接（可选，影响多GPU性能）"""
    print("\n=== 检查NVLink（可选）===")

    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', 'nvlink', '--status'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            if 'Active' in result.stdout:
                print("✓ NVLink已启用（多GPU通信更快）")
                return True
            else:
                print("⚠️  NVLink未启用（将使用PCIe，速度较慢）")
                return False
        else:
            print("ℹ️  无法检查NVLink状态（可能不支持）")
            return None
    except Exception as e:
        print(f"ℹ️  跳过NVLink检查: {e}")
        return None


def estimate_performance():
    """估算性能"""
    print("\n=== 性能估算 ===")

    try:
        import torch
        if not torch.cuda.is_available():
            return

        num_gpus = torch.cuda.device_count()

        # 假设参数
        seq_len = 512
        batch_size = 1

        # 估算推理时间（非常粗略）
        # 基于经验：大型MoE模型在多GPU上约 100-500 tokens/sec
        estimated_tokens_per_sec = 100 * num_gpus * 0.5  # 保守估计
        time_per_token = 1.0 / estimated_tokens_per_sec

        print(f"  保守估算 (基于 {num_gpus} GPU):")
        print(f"    推理速度: ~{estimated_tokens_per_sec:.0f} tokens/sec")
        print(f"    每个token时间: ~{time_per_token*1000:.1f} ms")
        print(f"    生成100个tokens: ~{time_per_token*100:.1f} 秒")

        print(f"\n  注意: 实际性能取决于:")
        print(f"    - GPU类型（A100/H100等）")
        print(f"    - NVLink连接")
        print(f"    - 序列长度")
        print(f"    - MoE路由效率")

    except Exception as e:
        print(f"无法估算性能: {e}")


def main():
    print("="*60)
    print("MiniMax M2 多GPU环境检查")
    print("="*60)

    results = []

    # 1. 检查CUDA
    cuda_ok = check_cuda()
    results.append(("CUDA环境", cuda_ok))

    if not cuda_ok:
        print("\n❌ CUDA不可用，无法继续检查")
        print("\n请确保:")
        print("  1. 安装了NVIDIA驱动")
        print("  2. 安装了CUDA Toolkit")
        print("  3. 安装了GPU版本的PyTorch")
        return

    # 2. 检查GPU
    gpu_ok, num_gpus, total_memory = check_gpus()
    results.append(("GPU配置", gpu_ok))

    # 3. 检查库
    libs_ok = check_libraries()
    results.append(("必要的库", libs_ok))

    # 4. 检查模型文件
    model_ok = check_model_files()
    results.append(("模型文件", model_ok))

    # 5. 检查NVLink（可选）
    nvlink_ok = check_nvlink()
    if nvlink_ok is not None:
        results.append(("NVLink", nvlink_ok))

    # 6. 性能估算
    if gpu_ok:
        estimate_performance()

    # 总结
    print("\n" + "="*60)
    print("检查总结")
    print("="*60 + "\n")

    for check_name, status in results:
        icon = "✓" if status else "✗"
        print(f"  {icon} {check_name}")

    all_ok = all(status for _, status in results if _ != "NVLink")

    print("\n" + "="*60)
    if all_ok:
        print("✓ 环境准备就绪！")
        print("\n可以运行:")
        print("  python test_layer_by_layer_gpu.py --test-type extract-all")
    else:
        print("⚠️  环境未完全准备好")
        print("\n请修复上述问题后再运行")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
