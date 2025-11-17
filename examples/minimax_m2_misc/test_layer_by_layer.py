"""
逐层测试脚本 - MiniMax M2 Model
用于定位模型编译和推理问题

测试策略：
1. 测试单个组件（Embedding, Attention, MoE, RMSNorm）
2. 测试单个decoder layer
3. 测试多层decoder layers（可配置范围）
4. 测试完整模型

使用方法：
python test_layer_by_layer.py --test-type [component|single-layer|multi-layer|full]
"""

import argparse
import torch
import gc
import os
from pathlib import Path

from transformers import AutoTokenizer, AutoConfig
from neuronx_distributed_inference.models.config import MoENeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_v2 import (
    MiniMaxM2InferenceConfig,
    NeuronMiniMaxM2ForCausalLM,
    NeuronMiniMaxM2DecoderLayer,
)
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

# 导入neuronx_distributed用于初始化分布式环境
from neuronx_distributed.parallel_layers import parallel_state


# 配置路径
MODEL_PATH = "/home/ubuntu/model_hf/MiniMax-M2-BF16/"
TRACED_MODEL_BASE = "/home/ubuntu/traced_model/test_layers/"

# 全局标志：是否已初始化分布式环境
_DISTRIBUTED_INITIALIZED = False


def initialize_distributed_env(tp_degree=64):
    """
    初始化neuronx_distributed分布式环境

    Neuron模型需要在分布式环境中初始化，即使是单进程也需要
    """
    global _DISTRIBUTED_INITIALIZED

    if _DISTRIBUTED_INITIALIZED:
        print("  ✓ 分布式环境已初始化，跳过")
        return

    print(f"\n初始化分布式环境 (tp_degree={tp_degree})...")

    # 设置环境变量（如果未设置）
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'

    # 初始化torch.distributed（即使是单进程）
    if not torch.distributed.is_initialized():
        try:
            torch.distributed.init_process_group(backend='xla')
            print("  ✓ torch.distributed已初始化 (backend=xla)")
        except Exception as e:
            print(f"  ⚠️  torch.distributed初始化失败: {e}")
            print("  尝试不使用分布式环境...")

    # 初始化neuronx_distributed的parallel_state
    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_degree,
        )
        print(f"  ✓ parallel_state已初始化 (tp_degree={tp_degree})")
        _DISTRIBUTED_INITIALIZED = True
    except Exception as e:
        print(f"  ✗ parallel_state初始化失败: {e}")
        raise


def get_neuron_config(batch_size=1, max_context_length=512, seq_len=512, tp_degree=1):
    """创建基础neuron配置

    Args:
        batch_size: batch size
        max_context_length: maximum context length
        seq_len: sequence length
        tp_degree: tensor parallel degree (use 1 for single-process testing, 64 for multi-process)
    """
    return MoENeuronConfig(
        tp_degree=tp_degree,
        batch_size=batch_size,
        max_context_length=max_context_length,
        seq_len=seq_len,
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=True, temperature=0.6, top_k=20, top_p=0.95
        ),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        save_sharded_checkpoint=True,
        blockwise_matmul_config={'use_torch_block_wise': True},
    )


class SingleLayerWrapper(torch.nn.Module):
    """包装单个decoder layer用于独立测试"""

    def __init__(self, config: MiniMaxM2InferenceConfig, layer_idx: int):
        super().__init__()
        self.layer = NeuronMiniMaxM2DecoderLayer(config, layer_idx)
        self.layer_idx = layer_idx

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        return self.layer(hidden_states, attention_mask, position_ids, past_key_value)


class MultiLayerWrapper(torch.nn.Module):
    """包装多个decoder layers用于范围测试"""

    def __init__(self, config: MiniMaxM2InferenceConfig, start_layer: int, end_layer: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            NeuronMiniMaxM2DecoderLayer(config, i)
            for i in range(start_layer, end_layer + 1)
        ])
        self.start_layer = start_layer
        self.end_layer = end_layer

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        for layer in self.layers:
            outputs = layer(hidden_states, attention_mask, position_ids, past_key_value)
            hidden_states = outputs[0]
            past_key_value = outputs[1] if len(outputs) > 1 else None
        return outputs


class LayerOutputRecorder:
    """记录每一层的输出（用于完整模型）"""

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

            # 保存输出（注意：在Neuron上可能是XLA张量）
            if isinstance(hidden_states, torch.Tensor):
                # 尝试移到CPU，如果是XLA张量会自动转换
                try:
                    self.layer_outputs[layer_name] = hidden_states.detach().cpu()
                except:
                    # 如果无法移到CPU，直接保存引用
                    self.layer_outputs[layer_name] = hidden_states.detach()
            else:
                self.layer_outputs[layer_name] = hidden_states

        return hook

    def register_hooks(self, model, layer_indices: list = None):
        """为指定的层注册hooks"""
        # 获取模型的layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            print("  ⚠️  无法找到模型的layers")
            return

        if layer_indices is None:
            # 默认记录所有层
            layer_indices = list(range(len(layers)))

        for idx in layer_indices:
            if idx < len(layers):
                layer_name = f"layer_{idx}"
                hook = layers[idx].register_forward_hook(
                    self.create_hook(layer_name)
                )
                self.hooks.append(hook)
                print(f"  ✓ 注册hook: {layer_name}")

        # 也记录embedding和final norm的输出
        if hasattr(model, 'model'):
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

    def save_outputs(self, output_path: Path, metadata: dict = None):
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

    def get_statistics(self) -> dict:
        """获取输出的统计信息"""
        stats = {}
        for layer_name, output in self.layer_outputs.items():
            if isinstance(output, torch.Tensor):
                try:
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
                except Exception as e:
                    stats[layer_name] = {'error': str(e)}
        return stats


def test_single_component(component_name: str):
    """测试单个组件（Attention, MoE, RMSNorm等）"""
    print(f"\n{'='*60}")
    print(f"测试组件: {component_name}")
    print(f"{'='*60}\n")

    # TODO: 实现单个组件的独立测试
    # 例如：只测试Attention模块的forward pass
    print(f"[TODO] 实现 {component_name} 组件测试")
    print("提示: 需要创建dummy输入数据并测试组件的forward pass")


def test_single_layer(layer_idx: int, compile_model: bool = True):
    """测试单个decoder layer"""
    print(f"\n{'='*60}")
    print(f"测试单层: Layer {layer_idx}")
    print(f"{'='*60}\n")

    # 创建配置
    neuron_config = get_neuron_config(max_context_length=256, seq_len=256)

    # !!! 重要：在创建模型组件之前初始化分布式环境
    initialize_distributed_env(tp_degree=neuron_config.tp_degree)

    config = MiniMaxM2InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )

    output_path = Path(TRACED_MODEL_BASE) / f"single_layer_{layer_idx}"
    output_path.mkdir(parents=True, exist_ok=True)

    if compile_model:
        print(f"\n测试 Layer {layer_idx} 构建和前向传播...")

        # 创建单层包装器
        print(f"\n  创建 SingleLayerWrapper...")
        layer_wrapper = SingleLayerWrapper(config, layer_idx)
        print(f"  ✓ SingleLayerWrapper 创建成功")

        # 注意：我们在CPU上测试，不使用XLA编译
        # 原因：MoE层的权重（4.5GB）超过了Neuron编译器的4GB单张量限制
        # 这个测试只验证模型逻辑的正确性，不进行实际的Neuron编译
        device = 'cpu'
        print(f"  使用设备: {device} (CPU测试，不进行Neuron编译)")

        # 创建示例输入
        batch_size = config.neuron_config.batch_size
        seq_len = config.neuron_config.seq_len
        hidden_size = config.hidden_size

        print(f"\n  创建测试输入...")
        print(f"    batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")

        dummy_hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
        dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int32)
        dummy_position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)

        print(f"  ✓ 输入创建完成:")
        print(f"    hidden_states: {dummy_hidden_states.shape}, dtype={dummy_hidden_states.dtype}, device={dummy_hidden_states.device}")
        print(f"    attention_mask: {dummy_attention_mask.shape}, dtype={dummy_attention_mask.dtype}")
        print(f"    position_ids: {dummy_position_ids.shape}, dtype={dummy_position_ids.dtype}")

        # 运行前向传播（在CPU上）
        print(f"\n  运行前向传播（在CPU上）...")
        try:
            with torch.no_grad():
                outputs = layer_wrapper(
                    dummy_hidden_states,
                    attention_mask=dummy_attention_mask,
                    position_ids=dummy_position_ids
                )

            print(f"  ✓ 前向传播成功")
            print(f"  输出类型: {type(outputs)}")

            # 解析输出
            if isinstance(outputs, tuple):
                hidden_states_out = outputs[0]
                print(f"  输出tuple长度: {len(outputs)}")
                print(f"  hidden_states输出: {hidden_states_out.shape}, dtype={hidden_states_out.dtype}")

                # 验证输出
                print(f"\n  验证输出数值:")
                print(f"    Mean: {hidden_states_out.float().mean():.6f}")
                print(f"    Std: {hidden_states_out.float().std():.6f}")
                print(f"    Min: {hidden_states_out.min():.6f}")
                print(f"    Max: {hidden_states_out.max():.6f}")
                print(f"    Has NaN: {torch.isnan(hidden_states_out).any().item()}")
                print(f"    Has Inf: {torch.isinf(hidden_states_out).any().item()}")

                # 保存输出用于对比
                output_data = {
                    'layer_idx': layer_idx,
                    'config': config,
                    'input_hidden_states': dummy_hidden_states,
                    'input_attention_mask': dummy_attention_mask,
                    'input_position_ids': dummy_position_ids,
                    'output_hidden_states': hidden_states_out,
                    'output_full': outputs,
                    'statistics': {
                        'mean': float(hidden_states_out.float().mean()),
                        'std': float(hidden_states_out.float().std()),
                        'min': float(hidden_states_out.min()),
                        'max': float(hidden_states_out.max()),
                        'has_nan': bool(torch.isnan(hidden_states_out).any()),
                        'has_inf': bool(torch.isinf(hidden_states_out).any()),
                    }
                }

                torch.save(output_data, output_path / 'layer_output.pt')
                print(f"\n  ✓ 输出已保存到: {output_path / 'layer_output.pt'}")

            else:
                print(f"  警告: 输出不是tuple类型")

        except Exception as e:
            print(f"\n  ✗ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 保存配置
        torch.save({
            'config': config,
            'layer_idx': layer_idx,
        }, output_path / 'config.pt')

        print(f"\n✓ Layer {layer_idx} 测试完成")

        # 注意事项
        print(f"\n注意:")
        print(f"  - 这是CPU上的测试，验证模型组件构建和前向传播")
        print(f"  - 由于MoE权重（4.5GB）超过Neuron编译器的4GB限制，无法进行XLA编译")
        print(f"  - 输出已保存，可用于与GPU版本对比")
        print(f"  - 此测试主要用于验证模型逻辑正确性，不测试Neuron编译")

    else:
        # 加载已保存的输出
        print(f"\n加载已保存的测试结果...")

        output_file = output_path / 'layer_output.pt'
        if output_file.exists():
            output_data = torch.load(output_file)
            print(f"  ✓ 加载成功: {output_file}")
            print(f"\n  统计信息:")
            stats = output_data['statistics']
            for key, value in stats.items():
                print(f"    {key}: {value}")
        else:
            print(f"  ✗ 未找到输出文件: {output_file}")
            print(f"  请先运行编译/测试步骤")


def test_multi_layer(start_layer: int, end_layer: int, compile_model: bool = True):
    """测试多层decoder layers"""
    print(f"\n{'='*60}")
    print(f"测试多层: Layers {start_layer}-{end_layer}")
    print(f"{'='*60}\n")

    # 创建配置
    neuron_config = get_neuron_config(max_context_length=512, seq_len=512)

    # !!! 重要：在创建模型组件之前初始化分布式环境
    initialize_distributed_env(tp_degree=neuron_config.tp_degree)

    config = MiniMaxM2InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )

    output_path = Path(TRACED_MODEL_BASE) / f"layers_{start_layer}_to_{end_layer}"
    output_path.mkdir(parents=True, exist_ok=True)

    if compile_model:
        print(f"\n测试 Layers {start_layer}-{end_layer} 构建和前向传播...")

        num_layers = end_layer - start_layer + 1
        print(f"  层数: {num_layers}")

        # 创建多层包装器
        print(f"\n  创建 MultiLayerWrapper...")
        multi_layer_wrapper = MultiLayerWrapper(config, start_layer, end_layer)
        print(f"  ✓ MultiLayerWrapper 创建成功")

        # 注意：我们在CPU上测试，不使用XLA编译
        device = 'cpu'
        print(f"  使用设备: {device} (CPU测试，不进行Neuron编译)")

        # 创建示例输入
        batch_size = config.neuron_config.batch_size
        seq_len = config.neuron_config.seq_len
        hidden_size = config.hidden_size

        print(f"\n  创建测试输入...")
        print(f"    batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")

        dummy_hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
        dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.int32)
        dummy_position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)

        print(f"  ✓ 输入创建完成:")
        print(f"    hidden_states: {dummy_hidden_states.shape}, dtype={dummy_hidden_states.dtype}, device={dummy_hidden_states.device}")
        print(f"    attention_mask: {dummy_attention_mask.shape}, dtype={dummy_attention_mask.dtype}")
        print(f"    position_ids: {dummy_position_ids.shape}, dtype={dummy_position_ids.dtype}")

        # 运行前向传播（在CPU上）
        print(f"\n  运行前向传播（{num_layers}层，在CPU上）...")
        try:
            with torch.no_grad():
                outputs = multi_layer_wrapper(
                    dummy_hidden_states,
                    attention_mask=dummy_attention_mask,
                    position_ids=dummy_position_ids
                )

            print(f"  ✓ 前向传播成功")
            print(f"  输出类型: {type(outputs)}")

            # 解析输出
            if isinstance(outputs, tuple):
                hidden_states_out = outputs[0]
                print(f"  输出tuple长度: {len(outputs)}")
                print(f"  hidden_states输出: {hidden_states_out.shape}, dtype={hidden_states_out.dtype}")

                # 验证输出
                print(f"\n  验证输出数值:")
                print(f"    Mean: {hidden_states_out.float().mean():.6f}")
                print(f"    Std: {hidden_states_out.float().std():.6f}")
                print(f"    Min: {hidden_states_out.min():.6f}")
                print(f"    Max: {hidden_states_out.max():.6f}")
                print(f"    Has NaN: {torch.isnan(hidden_states_out).any().item()}")
                print(f"    Has Inf: {torch.isinf(hidden_states_out).any().item()}")

                # 保存输出用于对比
                output_data = {
                    'start_layer': start_layer,
                    'end_layer': end_layer,
                    'num_layers': num_layers,
                    'config': config,
                    'input_hidden_states': dummy_hidden_states,
                    'input_attention_mask': dummy_attention_mask,
                    'input_position_ids': dummy_position_ids,
                    'output_hidden_states': hidden_states_out,
                    'output_full': outputs,
                    'statistics': {
                        'mean': float(hidden_states_out.float().mean()),
                        'std': float(hidden_states_out.float().std()),
                        'min': float(hidden_states_out.min()),
                        'max': float(hidden_states_out.max()),
                        'has_nan': bool(torch.isnan(hidden_states_out).any()),
                        'has_inf': bool(torch.isinf(hidden_states_out).any()),
                    }
                }

                torch.save(output_data, output_path / 'layers_output.pt')
                print(f"\n  ✓ 输出已保存到: {output_path / 'layers_output.pt'}")

            else:
                print(f"  警告: 输出不是tuple类型")

        except Exception as e:
            print(f"\n  ✗ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 保存配置
        torch.save({
            'config': config,
            'start_layer': start_layer,
            'end_layer': end_layer,
        }, output_path / 'config.pt')

        print(f"\n✓ Layers {start_layer}-{end_layer} 测试完成")
        print(f"\n注意:")
        print(f"  - 这是CPU上的测试，验证多层组件构建和前向传播")
        print(f"  - 可以用于二分查找问题层的范围")
        print(f"  - 输出已保存，可用于与GPU版本对比")
        print(f"  - 注意：由于MoE权重超过4GB，无法在Neuron上编译单层")

    else:
        # 加载已保存的输出
        print(f"\n加载已保存的测试结果...")

        output_file = output_path / 'layers_output.pt'
        if output_file.exists():
            output_data = torch.load(output_file)
            print(f"  ✓ 加载成功: {output_file}")
            print(f"\n  层范围: {output_data['start_layer']}-{output_data['end_layer']}")
            print(f"  统计信息:")
            stats = output_data['statistics']
            for key, value in stats.items():
                print(f"    {key}: {value}")
        else:
            print(f"  ✗ 未找到输出文件: {output_file}")
            print(f"  请先运行测试步骤")


def test_full_model(compile_model: bool = True):
    """测试完整模型（参考原始demo）"""
    print(f"\n{'='*60}")
    print(f"测试完整模型")
    print(f"{'='*60}\n")

    # 创建配置
    neuron_config = get_neuron_config(max_context_length=1024, seq_len=1024)

    # !!! 重要：在创建模型组件之前初始化分布式环境
    initialize_distributed_env(tp_degree=neuron_config.tp_degree)

    config = MiniMaxM2InferenceConfig(
        neuron_config,
        load_config=load_pretrained_config(MODEL_PATH),
    )

    output_path = Path(TRACED_MODEL_BASE) / "full_model"
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token

    if compile_model:
        print(f"\n编译完整模型...")
        model = NeuronMiniMaxM2ForCausalLM(MODEL_PATH, config)
        model.compile(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        print(f"✓ 完整模型编译完成")

    # 测试推理
    print(f"\n加载完整模型...")
    model = NeuronMiniMaxM2ForCausalLM(str(output_path))
    model.load(str(output_path))
    print(f"✓ 模型加载完成")

    # 注册hooks记录每一层的输出
    print(f"\n注册hooks记录每一层的输出...")
    recorder = LayerOutputRecorder()
    recorder.register_hooks(model)  # 记录所有层
    print(f"✓ Hooks注册完成")

    # 简单推理测试
    prompt = "Hello, how are you?"
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

    inputs = tokenizer([text], padding=True, return_tensors="pt")
    print(f"\n运行推理...")
    print(f"  输入shape: {inputs.input_ids.shape}")
    print(f"  输入tokens (前10个): {inputs.input_ids[0, :10].tolist()}")

    # 运行前向传播
    try:
        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✓ 推理成功")
        print(f"  输出shape: {outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}")

        # 获取并显示统计信息
        print(f"\n获取层输出统计...")
        stats = recorder.get_statistics()
        print(f"  记录的层数: {len(stats)}")

        # 显示几个关键层的统计
        key_layers = ['embedding', 'layer_0', 'layer_30', 'layer_61', 'final_norm']
        print(f"\n关键层统计信息:")
        for layer_name in key_layers:
            if layer_name in stats:
                stat = stats[layer_name]
                if 'error' not in stat:
                    print(f"  {layer_name}:")
                    print(f"    Shape: {stat['shape']}")
                    print(f"    Mean: {stat['mean']:.6f}, Std: {stat['std']:.6f}")
                    print(f"    Range: [{stat['min']:.6f}, {stat['max']:.6f}]")
                    print(f"    Has NaN: {stat['has_nan']}, Has Inf: {stat['has_inf']}")

        # 保存所有层的输出
        print(f"\n保存层输出...")
        metadata = {
            'prompt': prompt,
            'input_ids': inputs.input_ids.cpu() if torch.is_tensor(inputs.input_ids) else inputs.input_ids,
            'attention_mask': inputs.attention_mask.cpu() if torch.is_tensor(inputs.attention_mask) else inputs.attention_mask,
            'config': config,
            'statistics': stats,
        }

        # 如果有logits输出，也保存
        if hasattr(outputs, 'logits'):
            try:
                metadata['output_logits'] = outputs.logits.cpu()
            except:
                print(f"  ⚠️  无法保存logits（可能是XLA张量）")

        recorder.save_outputs(output_path, metadata)
        print(f"✓ 所有层输出已保存到: {output_path}")

    except Exception as e:
        print(f"✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 移除hooks
        recorder.remove_hooks()
        print(f"\n✓ Hooks已移除")


def compare_layer_outputs(layer_indices: list):
    """对比多个层的输出，帮助定位问题层"""
    print(f"\n{'='*60}")
    print(f"对比层输出")
    print(f"{'='*60}\n")

    print(f"对比的层: {layer_indices}")
    print(f"[TODO] 实现层输出对比逻辑")
    print("提示:")
    print("  1. 使用相同的输入数据")
    print("  2. 记录每层的输出统计信息（mean, std, min, max）")
    print("  3. 检测异常输出（NaN, Inf, 或异常大的值）")


def main():
    parser = argparse.ArgumentParser(description="逐层测试MiniMax M2模型")
    parser.add_argument(
        '--test-type',
        type=str,
        choices=['component', 'single-layer', 'multi-layer', 'full', 'compare'],
        required=True,
        help='测试类型'
    )
    parser.add_argument('--component', type=str, help='组件名称（用于component测试）')
    parser.add_argument('--layer', type=int, help='层索引（用于single-layer测试）')
    parser.add_argument('--start-layer', type=int, default=0, help='起始层（用于multi-layer测试）')
    parser.add_argument('--end-layer', type=int, default=5, help='结束层（用于multi-layer测试）')
    parser.add_argument('--skip-compile', action='store_true', help='跳过编译，直接加载已编译模型')
    parser.add_argument('--compare-layers', type=str, help='对比的层列表，逗号分隔（例如：0,10,20）')

    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"# MiniMax M2 逐层测试工具")
    print(f"{'#'*60}\n")

    try:
        if args.test_type == 'component':
            if not args.component:
                print("错误: --component 参数必须指定")
                return
            test_single_component(args.component)

        elif args.test_type == 'single-layer':
            if args.layer is None:
                print("错误: --layer 参数必须指定")
                return
            test_single_layer(args.layer, compile_model=not args.skip_compile)

        elif args.test_type == 'multi-layer':
            test_multi_layer(
                args.start_layer,
                args.end_layer,
                compile_model=not args.skip_compile
            )

        elif args.test_type == 'full':
            test_full_model(compile_model=not args.skip_compile)

        elif args.test_type == 'compare':
            if not args.compare_layers:
                print("错误: --compare-layers 参数必须指定")
                return
            layer_indices = [int(x.strip()) for x in args.compare_layers.split(',')]
            compare_layer_outputs(layer_indices)

        print(f"\n{'='*60}")
        print(f"测试完成!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
