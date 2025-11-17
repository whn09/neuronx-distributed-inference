"""
GPU版本逐层测试脚本 - MiniMax M2 Model
用于生成golden reference输出，与Neuron版本对比

测试策略：
1. 在GPU上运行模型（使用transformers库的原生实现）
2. 保存每一层的输出作为golden reference
3. 与Neuron版本的输出进行对比

使用方法：
python test_layer_by_layer_gpu.py --test-type [single-layer|multi-layer|full|extract-all]
"""

import argparse
import torch
import gc
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

# 配置路径
MODEL_PATH = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
OUTPUT_BASE = "/home/ubuntu/traced_model/gpu_reference/"


class LayerOutputRecorder:
    """记录每一层的输出"""

    def __init__(self):
        self.layer_outputs = {}
        self.hooks = []

    def create_hook(self, layer_name: str):
        """创建hook函数"""
        def hook(module, input, output):
            # 处理不同类型的输出
            if isinstance(output, tuple):
                # 对于DecoderLayer，输出是(hidden_states, present_key_value, ...)
                hidden_states = output[0]
            else:
                hidden_states = output

            # 保存到CPU以节省GPU内存
            if isinstance(hidden_states, torch.Tensor):
                self.layer_outputs[layer_name] = hidden_states.detach().cpu()
            else:
                self.layer_outputs[layer_name] = hidden_states

        return hook

    def register_hooks(self, model, layer_indices: List[int] = None):
        """为指定的层注册hooks"""
        if layer_indices is None:
            # 默认记录所有层
            layer_indices = list(range(len(model.model.layers)))

        for idx in layer_indices:
            if idx < len(model.model.layers):
                layer_name = f"layer_{idx}"
                hook = model.model.layers[idx].register_forward_hook(
                    self.create_hook(layer_name)
                )
                self.hooks.append(hook)
                print(f"  ✓ 注册hook: {layer_name}")

        # 也记录embedding和final norm的输出
        if hasattr(model.model, 'embed_tokens'):
            hook = model.model.embed_tokens.register_forward_hook(
                self.create_hook("embedding")
            )
            self.hooks.append(hook)
            print(f"  ✓ 注册hook: embedding")

        if hasattr(model.model, 'norm'):
            hook = model.model.norm.register_forward_hook(
                self.create_hook("final_norm")
            )
            self.hooks.append(hook)
            print(f"  ✓ 注册hook: final_norm")

    def remove_hooks(self):
        """移除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def save_outputs(self, output_path: Path, metadata: Dict = None):
        """保存输出到文件"""
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存每一层的输出
        for layer_name, output in self.layer_outputs.items():
            save_path = output_path / f"{layer_name}_output.pt"
            torch.save(output, save_path)
            print(f"  ✓ 保存: {save_path}")

        # 保存元数据
        if metadata:
            metadata_path = output_path / "metadata.pt"
            torch.save(metadata, metadata_path)
            print(f"  ✓ 保存元数据: {metadata_path}")

    def get_statistics(self) -> Dict:
        """获取输出的统计信息"""
        stats = {}
        for layer_name, output in self.layer_outputs.items():
            if isinstance(output, torch.Tensor):
                stats[layer_name] = {
                    'shape': list(output.shape),
                    'dtype': str(output.dtype),
                    'mean': float(output.float().mean()),
                    'std': float(output.float().std()),
                    'min': float(output.min()),
                    'max': float(output.max()),
                    'has_nan': bool(torch.isnan(output).any()),
                    'has_inf': bool(torch.isinf(output).any()),
                }
        return stats


def load_model_gpu(device_map: str = "auto", max_memory: dict = None):
    """
    加载GPU模型（支持多GPU）

    Args:
        device_map: 设备分配策略
            - "auto": 自动分配到所有可用GPU（推荐用于大模型）
            - "balanced": 平衡分配到所有GPU
            - "balanced_low_0": 平衡分配，尽量少占用GPU 0
            - "sequential": 按顺序填满GPU
            - "cuda:0": 单GPU模式
        max_memory: 每个设备的最大内存限制，例如 {0: "40GB", 1: "40GB"}
    """
    print(f"\n加载GPU模型: {MODEL_PATH}")
    print(f"设备分配策略: {device_map}")

    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU模式")
        device_map = "cpu"
    else:
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个GPU")

        # 打印每个GPU的信息
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1024**3  # 转换为GB
            print(f"  GPU {i}: {props.name}, {total_memory:.2f} GB")

    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"  模型配置: {config.num_hidden_layers} layers, {config.num_local_experts} experts")

    # 估算模型大小
    # hidden_size=3072, intermediate_size=1536, num_layers=62, num_experts=256
    # 每层约 6.5GB (BF16)，总共约 403GB
    estimated_size_gb = 403
    print(f"  估算模型大小: ~{estimated_size_gb} GB (BF16)")

    # 加载模型 - 使用bfloat16以节省内存，自动分配到多GPU
    print(f"  加载模型权重（多GPU模式）...")
    print(f"  这可能需要几分钟时间...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=device_map,  # 自动分配到多个GPU
        max_memory=max_memory,  # 可选：限制每个GPU的内存使用
        low_cpu_mem_usage=True,  # 降低CPU内存使用
        offload_folder="offload",  # CPU offload目录（如果需要）
        trust_remote_code=True
    )

    model.eval()
    print(f"  ✓ 模型加载完成")

    # 打印实际的设备分配情况
    if device_map == "auto" or device_map == "balanced":
        print(f"\n  设备分配详情:")
        if hasattr(model, 'hf_device_map'):
            device_distribution = {}
            for name, device in model.hf_device_map.items():
                device_distribution[device] = device_distribution.get(device, 0) + 1

            for device, count in sorted(device_distribution.items()):
                print(f"    {device}: {count} 个模块")

    # 打印GPU内存使用情况
    if torch.cuda.is_available():
        print(f"\n  GPU内存使用:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"    GPU {i}: 已分配 {allocated:.2f} GB, 已保留 {reserved:.2f} GB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, config


def get_model_input_device(model):
    """
    获取模型输入应该所在的设备（多GPU模式下）

    对于多GPU模型，输入应该移动到第一个层所在的设备
    """
    # 尝试获取embedding层的设备
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight.device

    # 如果有device_map，找到第一个设备
    if hasattr(model, 'hf_device_map'):
        # 找到embed_tokens的设备
        for name, device in model.hf_device_map.items():
            if 'embed_tokens' in name:
                return torch.device(device)

    # 回退到第一个参数的设备
    return next(model.parameters()).device


def prepare_test_inputs(tokenizer, prompt: str = None, max_length: int = 512):
    """准备测试输入"""
    if prompt is None:
        prompt = "Give me a short introduction to large language models."

    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    inputs = tokenizer([text], padding=True, return_tensors="pt")

    # 截断到指定长度
    if inputs.input_ids.shape[1] > max_length:
        inputs.input_ids = inputs.input_ids[:, :max_length]
        inputs.attention_mask = inputs.attention_mask[:, :max_length]

    return inputs, text


def test_single_layer_gpu(layer_idx: int, save_outputs: bool = True, device_map: str = "auto", max_memory: dict = None):
    """测试单个decoder layer"""
    print(f"\n{'='*60}")
    print(f"GPU测试 - 单层: Layer {layer_idx}")
    print(f"{'='*60}\n")

    model, tokenizer, config = load_model_gpu(device_map=device_map, max_memory=max_memory)

    # 准备输入
    inputs, prompt_text = prepare_test_inputs(tokenizer, max_length=256)
    print(f"\n输入prompt (前100字符): {prompt_text[:100]}...")
    print(f"输入shape: {inputs.input_ids.shape}")

    # 移动到正确的设备（多GPU模式下是embedding层所在设备）
    device = get_model_input_device(model)
    print(f"输入设备: {device}")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 记录指定层的输出
    recorder = LayerOutputRecorder()
    recorder.register_hooks(model, layer_indices=[layer_idx])

    # 运行模型
    print(f"\n运行模型...")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"✓ 前向传播完成")

    # 获取统计信息
    stats = recorder.get_statistics()
    print(f"\n层输出统计:")
    for layer_name, stat in stats.items():
        print(f"  {layer_name}:")
        print(f"    Shape: {stat['shape']}")
        print(f"    Mean: {stat['mean']:.6f}, Std: {stat['std']:.6f}")
        print(f"    Range: [{stat['min']:.6f}, {stat['max']:.6f}]")
        print(f"    Has NaN: {stat['has_nan']}, Has Inf: {stat['has_inf']}")

    # 保存输出
    if save_outputs:
        output_path = Path(OUTPUT_BASE) / f"single_layer_{layer_idx}"
        metadata = {
            'layer_idx': layer_idx,
            'prompt': prompt_text,
            'input_ids': inputs['input_ids'].cpu(),
            'attention_mask': inputs['attention_mask'].cpu(),
            'config': config,
            'statistics': stats,
        }
        recorder.save_outputs(output_path, metadata)

    recorder.remove_hooks()

    return recorder.layer_outputs, stats


def test_multi_layer_gpu(start_layer: int, end_layer: int, save_outputs: bool = True, device_map: str = "auto", max_memory: dict = None):
    """测试多层decoder layers"""
    print(f"\n{'='*60}")
    print(f"GPU测试 - 多层: Layers {start_layer}-{end_layer}")
    print(f"{'='*60}\n")

    model, tokenizer, config = load_model_gpu(device_map=device_map, max_memory=max_memory)

    # 准备输入
    inputs, prompt_text = prepare_test_inputs(tokenizer, max_length=512)
    print(f"\n输入prompt (前100字符): {prompt_text[:100]}...")
    print(f"输入shape: {inputs.input_ids.shape}")

    # 移动到正确的设备（多GPU模式下是embedding层所在设备）
    device = get_model_input_device(model)
    print(f"输入设备: {device}")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 记录指定范围的层
    layer_indices = list(range(start_layer, end_layer + 1))
    recorder = LayerOutputRecorder()
    recorder.register_hooks(model, layer_indices=layer_indices)

    # 运行模型
    print(f"\n运行模型...")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"✓ 前向传播完成")

    # 获取统计信息
    stats = recorder.get_statistics()
    print(f"\n层输出统计:")
    for layer_name in sorted(stats.keys()):
        stat = stats[layer_name]
        print(f"  {layer_name}:")
        print(f"    Shape: {stat['shape']}, Mean: {stat['mean']:.6f}, Range: [{stat['min']:.6f}, {stat['max']:.6f}]")

    # 保存输出
    if save_outputs:
        output_path = Path(OUTPUT_BASE) / f"layers_{start_layer}_to_{end_layer}"
        metadata = {
            'start_layer': start_layer,
            'end_layer': end_layer,
            'prompt': prompt_text,
            'input_ids': inputs['input_ids'].cpu(),
            'attention_mask': inputs['attention_mask'].cpu(),
            'config': config,
            'statistics': stats,
        }
        recorder.save_outputs(output_path, metadata)

    recorder.remove_hooks()

    return recorder.layer_outputs, stats


def test_full_model_gpu(save_outputs: bool = True, save_generation: bool = True, device_map: str = "auto", max_memory: dict = None):
    """测试完整模型"""
    print(f"\n{'='*60}")
    print(f"GPU测试 - 完整模型")
    print(f"{'='*60}\n")

    model, tokenizer, config = load_model_gpu(device_map=device_map, max_memory=max_memory)

    # 准备输入
    inputs, prompt_text = prepare_test_inputs(tokenizer, max_length=128)  # 用较短的输入以节省内存
    print(f"\n输入prompt (前100字符): {prompt_text[:100]}...")
    print(f"输入shape: {inputs.input_ids.shape}")

    # 移动到正确的设备（多GPU模式下是embedding层所在设备）
    device = get_model_input_device(model)
    print(f"输入设备: {device}")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 记录所有层
    recorder = LayerOutputRecorder()
    recorder.register_hooks(model)  # 记录所有层

    # 运行模型
    print(f"\n运行模型...")
    with torch.no_grad():
        outputs = model(**inputs)

    print(f"✓ 前向传播完成")

    # 获取统计信息
    stats = recorder.get_statistics()
    print(f"\n层输出统计摘要:")
    print(f"  记录的层数: {len(stats)}")

    # 显示几个关键层的统计
    key_layers = ['embedding', 'layer_0', 'layer_30', 'layer_61', 'final_norm']
    for layer_name in key_layers:
        if layer_name in stats:
            stat = stats[layer_name]
            print(f"  {layer_name}: Mean={stat['mean']:.6f}, Range=[{stat['min']:.6f}, {stat['max']:.6f}]")

    # 保存输出
    if save_outputs:
        output_path = Path(OUTPUT_BASE) / "full_model"
        metadata = {
            'prompt': prompt_text,
            'input_ids': inputs['input_ids'].cpu(),
            'attention_mask': inputs['attention_mask'].cpu(),
            'output_logits': outputs.logits.cpu() if hasattr(outputs, 'logits') else None,
            'config': config,
            'statistics': stats,
        }
        recorder.save_outputs(output_path, metadata)

    # 可选：测试generation
    if save_generation:
        print(f"\n测试generation...")
        with torch.no_grad():
            generated_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.6,
                top_k=20,
                top_p=0.95,
            )

        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"\n生成的文本 (前200字符):")
        print(f"{generated_text[0][:200]}...")

        # 保存生成结果
        output_path = Path(OUTPUT_BASE) / "full_model"
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save({
            'generated_ids': generated_ids.cpu(),
            'generated_text': generated_text,
        }, output_path / "generation_output.pt")

    recorder.remove_hooks()

    return recorder.layer_outputs, stats


def extract_all_layers_gpu(device_map: str = "auto", max_memory: dict = None):
    """提取所有层的输出，用于详细对比"""
    print(f"\n{'='*60}")
    print(f"GPU测试 - 提取所有层输出")
    print(f"{'='*60}\n")

    # 直接调用full_model测试
    return test_full_model_gpu(save_outputs=True, save_generation=True, device_map=device_map, max_memory=max_memory)


def main():
    parser = argparse.ArgumentParser(
        description="GPU版本逐层测试MiniMax M2模型（支持多GPU）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用所有GPU自动分配（推荐）
  python test_layer_by_layer_gpu.py --test-type extract-all

  # 使用特定GPU（单GPU模式）
  python test_layer_by_layer_gpu.py --test-type full --device-map cuda:0

  # 限制每个GPU的内存使用
  python test_layer_by_layer_gpu.py --test-type full --max-memory "40GB"

  # 平衡分配到所有GPU
  python test_layer_by_layer_gpu.py --test-type extract-all --device-map balanced
        """
    )
    parser.add_argument(
        '--test-type',
        type=str,
        choices=['single-layer', 'multi-layer', 'full', 'extract-all'],
        required=True,
        help='测试类型'
    )
    parser.add_argument('--layer', type=int, help='层索引（用于single-layer测试）')
    parser.add_argument('--start-layer', type=int, default=0, help='起始层（用于multi-layer测试）')
    parser.add_argument('--end-layer', type=int, default=5, help='结束层（用于multi-layer测试）')
    parser.add_argument('--no-save', action='store_true', help='不保存输出文件')
    parser.add_argument(
        '--device-map',
        type=str,
        default='auto',
        help='设备分配策略: auto(自动), balanced(平衡), sequential(顺序), cuda:0(单GPU)'
    )
    parser.add_argument(
        '--max-memory',
        type=str,
        help='每个GPU的最大内存限制，例如: "40GB" 或 "0:40GB,1:40GB"'
    )

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# MiniMax M2 GPU版本测试工具（多GPU支持）")
    print(f"{'#'*60}\n")

    # 解析max_memory参数
    max_memory = None
    if args.max_memory:
        if ',' in args.max_memory:
            # 格式: "0:40GB,1:40GB"
            max_memory = {}
            for item in args.max_memory.split(','):
                gpu_id, mem = item.split(':')
                max_memory[int(gpu_id)] = mem
        else:
            # 格式: "40GB" - 应用到所有GPU
            num_gpus = torch.cuda.device_count()
            max_memory = {i: args.max_memory for i in range(num_gpus)}
        print(f"内存限制: {max_memory}\n")

    try:
        if args.test_type == 'single-layer':
            if args.layer is None:
                print("错误: --layer 参数必须指定")
                return
            test_single_layer_gpu(
                args.layer,
                save_outputs=not args.no_save,
                device_map=args.device_map,
                max_memory=max_memory
            )

        elif args.test_type == 'multi-layer':
            test_multi_layer_gpu(
                args.start_layer,
                args.end_layer,
                save_outputs=not args.no_save,
                device_map=args.device_map,
                max_memory=max_memory
            )

        elif args.test_type == 'full':
            test_full_model_gpu(
                save_outputs=not args.no_save,
                device_map=args.device_map,
                max_memory=max_memory
            )

        elif args.test_type == 'extract-all':
            extract_all_layers_gpu(
                device_map=args.device_map,
                max_memory=max_memory
            )

        print(f"\n{'='*60}")
        print(f"GPU测试完成!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
