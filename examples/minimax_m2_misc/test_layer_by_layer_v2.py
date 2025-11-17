"""
基于 generation_minimax_m2_demo_v2.py 的逐层输出记录脚本

用途：运行完整模型并记录每一层的输出，用于调试和对比

使用方法：
    # 编译并测试
    python test_layer_by_layer_v2.py

    # 只测试（跳过编译）
    python test_layer_by_layer_v2.py --skip-compile
"""

import argparse
import torch
from pathlib import Path

from transformers import AutoTokenizer

from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_v2 import (
    MiniMaxM2InferenceConfig,
    NeuronMiniMaxM2ForCausalLM
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# 路径配置
MODEL_PATH = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
TRACED_MODEL_PATH = "/home/ubuntu/traced_model/MiniMax-M2-BF16-weights-v2/"
OUTPUT_PATH = "/home/ubuntu/traced_model/test_layers/full_model/"

torch.manual_seed(0)


def test_with_layer_recording(skip_compile=False):
    """运行完整模型并记录每层输出"""

    if not skip_compile:
        print("\n" + "="*60)
        print("步骤 1: 编译模型")
        print("="*60)

        # 配置（与 generation_minimax_m2_demo_v2.py 完全一致）
        neuron_config = MoENeuronConfig(
            tp_degree=64,
            batch_size=1,
            max_context_length=1024,
            seq_len=1024,
            on_device_sampling_config=OnDeviceSamplingConfig(
                do_sample=True, temperature=0.6, top_k=20, top_p=0.95
            ),
            enable_bucketing=False,
            flash_decoding_enabled=False,
            save_sharded_checkpoint=True,
            blockwise_matmul_config={'use_torch_block_wise': True},
            record_layer_outputs=True,  # ← 启用层输出记录！
        )

        config = MiniMaxM2InferenceConfig(
            neuron_config,
            load_config=load_pretrained_config(MODEL_PATH),
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
        tokenizer.pad_token = tokenizer.eos_token

        # 编译模型
        print("\n编译模型...")
        model = NeuronMiniMaxM2ForCausalLM(MODEL_PATH, config)
        model.compile(TRACED_MODEL_PATH)
        tokenizer.save_pretrained(TRACED_MODEL_PATH)
        print("✓ 模型编译完成")

    # 加载模型
    print("\n" + "="*60)
    print("步骤 2: 加载模型")
    print("="*60)

    model = NeuronMiniMaxM2ForCausalLM(TRACED_MODEL_PATH)
    model.load(TRACED_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TRACED_MODEL_PATH)
    print("✓ 模型加载完成")

    # 准备输入
    print("\n" + "="*60)
    print("步骤 3: 准备输入")
    print("="*60)

    prompt = "Give me a short introduction to large language models."
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    inputs = tokenizer([text], padding=True, return_tensors="pt")

    # 添加position_ids（模型需要）
    seq_len = inputs.input_ids.shape[1]
    inputs['position_ids'] = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(inputs.input_ids.shape[0], -1)

    print(f"  Prompt: {prompt}")
    print(f"  Input shape: {inputs.input_ids.shape}")
    print(f"  Position IDs shape: {inputs.position_ids.shape}")
    print(f"  Input tokens (前10个): {inputs.input_ids[0, :10].tolist()}")

    # 运行推理
    print("\n" + "="*60)
    print("步骤 4: 运行推理")
    print("="*60)

    try:
        with torch.no_grad():
            outputs = model(**inputs)

        print("✓ 推理成功")
        print(f"  输出类型: {type(outputs)}")

        # 从模型属性中读取layer_hidden_states
        # 尝试从不同的位置读取
        layer_hidden_states = None

        # 尝试1: 直接从model读取
        if hasattr(model, 'layer_hidden_states'):
            layer_hidden_states = model.layer_hidden_states
            print(f"  ✓ 从 model.layer_hidden_states 读取")
        # 尝试2: 从model.models[0]读取（编译后的结构）
        elif hasattr(model, 'models') and len(model.models) > 0:
            if hasattr(model.models[0], 'layer_hidden_states'):
                layer_hidden_states = model.models[0].layer_hidden_states
                print(f"  ✓ 从 model.models[0].layer_hidden_states 读取")
            # 尝试3: 从model.models[0].model读取
            elif hasattr(model.models[0], 'model') and hasattr(model.models[0].model, 'layer_hidden_states'):
                layer_hidden_states = model.models[0].model.layer_hidden_states
                print(f"  ✓ 从 model.models[0].model.layer_hidden_states 读取")

        print(f"  Layer hidden states 类型: {type(layer_hidden_states)}")

        if isinstance(layer_hidden_states, list) and len(layer_hidden_states) > 0:
            print(f"  ✓ 成功获取层输出！共 {len(layer_hidden_states)} 层")
            print(f"    第一层shape: {layer_hidden_states[0].shape if len(layer_hidden_states) > 0 else 'N/A'}")
        else:
            print(f"  ⚠️  layer_hidden_states 不是list或为空: {layer_hidden_states}")

    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 显示统计信息
    print("\n" + "="*60)
    print("步骤 5: 层输出统计")
    print("="*60)

    if not isinstance(layer_hidden_states, list):
        print("✗ 未能获取层输出，跳过统计")
        return

    # 计算每层的统计信息
    stats = {}
    for idx, hidden_states in enumerate(layer_hidden_states):
        layer_name = f"layer_{idx}" if idx > 0 else "embedding"
        if isinstance(hidden_states, torch.Tensor):
            try:
                stats[layer_name] = {
                    'shape': list(hidden_states.shape),
                    'dtype': str(hidden_states.dtype),
                    'mean': float(hidden_states.float().mean()),
                    'std': float(hidden_states.float().std()),
                    'min': float(hidden_states.min()),
                    'max': float(hidden_states.max()),
                    'has_nan': bool(torch.isnan(hidden_states).any()),
                    'has_inf': bool(torch.isinf(hidden_states).any()),
                }
            except Exception as e:
                stats[layer_name] = {'error': str(e)}

    print(f"\n记录的层数: {len(stats)}")

    # 显示关键层
    print("\n关键层统计:")
    key_layers = ['embedding', 'layer_0', 'layer_30', 'layer_61', 'final_norm']
    for layer_name in key_layers:
        if layer_name in stats:
            stat = stats[layer_name]
            if 'error' not in stat:
                print(f"\n  {layer_name}:")
                print(f"    Shape: {stat['shape']}")
                print(f"    dtype: {stat['dtype']}")
                print(f"    Mean: {stat['mean']:.6f}, Std: {stat['std']:.6f}")
                print(f"    Range: [{stat['min']:.6f}, {stat['max']:.6f}]")
                print(f"    Has NaN: {stat['has_nan']}, Has Inf: {stat['has_inf']}")

    # 检查异常层
    print("\n检查异常层:")
    problem_layers = []
    for layer_name, stat in stats.items():
        if 'layer_' in layer_name and 'error' not in stat:
            if stat.get('has_nan', False) or stat.get('has_inf', False):
                problem_layers.append(layer_name)
                print(f"  ✗ {layer_name}: Has NaN={stat['has_nan']}, Has Inf={stat['has_inf']}")

    if not problem_layers:
        print("  ✓ 所有层的输出正常（无NaN或Inf）")

    # 保存输出
    print("\n" + "="*60)
    print("步骤 6: 保存输出")
    print("="*60)

    # 保存层输出
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n保存层输出到 {output_path}...")

    # 保存每一层的输出
    for idx, hidden_states in enumerate(layer_hidden_states):
        layer_name = f"layer_{idx}" if idx > 0 else "embedding"
        save_path = output_path / f"{layer_name}_output.pt"
        try:
            # 尝试转换到CPU保存
            if isinstance(hidden_states, torch.Tensor):
                torch.save(hidden_states.cpu(), save_path)
        except:
            # 如果无法转换到CPU，直接保存
            torch.save(hidden_states, save_path)

    print(f"  ✓ 保存了 {len(layer_hidden_states)} 个层的输出")

    # 保存元数据
    metadata = {
        'prompt': prompt,
        'input_ids': inputs.input_ids.cpu() if torch.is_tensor(inputs.input_ids) else inputs.input_ids,
        'attention_mask': inputs.attention_mask.cpu() if torch.is_tensor(inputs.attention_mask) else inputs.attention_mask,
        'statistics': stats,
    }

    # 保存统计信息
    stats_path = output_path / "statistics.pt"
    torch.save(stats, stats_path)
    print(f"  ✓ 保存了统计信息到 {stats_path}")

    metadata_path = output_path / "metadata.pt"
    torch.save(metadata, metadata_path)
    print(f"  ✓ 保存了元数据到 {metadata_path}")

    # 总结
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    print(f"\n输出文件位置: {OUTPUT_PATH}")
    print(f"  - layer_X_output.pt: 各层输出")
    print(f"  - statistics.pt: 统计信息")
    print(f"  - metadata.pt: 元数据（输入、配置等）")
    print(f"\n记录的层数: {len(stats)}")

    if problem_layers:
        print(f"\n⚠️  发现 {len(problem_layers)} 个异常层:")
        for layer in problem_layers:
            print(f"  - {layer}")
    else:
        print("\n✓ 所有层输出正常")


def main():
    parser = argparse.ArgumentParser(
        description="运行完整模型并记录每层输出（基于 generation_minimax_m2_demo_v2.py）"
    )
    parser.add_argument(
        '--skip-compile',
        action='store_true',
        help='跳过编译，直接加载已编译的模型'
    )

    args = parser.parse_args()

    print("="*60)
    print("MiniMax M2 逐层输出记录")
    print("="*60)

    test_with_layer_recording(skip_compile=args.skip_compile)


if __name__ == "__main__":
    main()
