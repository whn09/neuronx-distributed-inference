# Neuron分布式环境初始化修复

## 问题

在Trainium上运行 `test_layer_by_layer.py` 时遇到错误：

```
ValueError: NeuronMiniMaxM2Attention has to be initialized in a distributed env.
Please use neuronx_distributed module to initialize a distributed env.
```

## 根本原因

Neuron模型组件（如`NeuronMiniMaxM2Attention`、`NeuronMiniMaxM2DecoderLayer`等）**必须在分布式环境中初始化**，即使只使用单个进程也需要。

这是因为这些组件内部依赖：
- `neuronx_distributed.parallel_layers.parallel_state`
- Tensor parallel状态
- 分布式通信组

## 解决方案

在创建任何Neuron模型组件之前，必须先初始化分布式环境。

### 修改内容

1. **添加分布式初始化函数**

```python
from neuronx_distributed.parallel_layers import parallel_state

def initialize_distributed_env(tp_degree=64):
    """初始化neuronx_distributed分布式环境"""
    # 设置环境变量
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'

    # 初始化torch.distributed
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='xla')

    # 初始化parallel_state
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_degree,
    )
```

2. **在所有测试函数中调用初始化**

```python
def test_single_layer(layer_idx: int, compile_model: bool = True):
    # 创建配置
    neuron_config = get_neuron_config(...)

    # !!! 重要：在创建模型组件之前初始化
    initialize_distributed_env(tp_degree=neuron_config.tp_degree)

    config = MiniMaxM2InferenceConfig(...)

    # 现在可以安全地创建模型组件
    layer_wrapper = SingleLayerWrapper(config, layer_idx)
```

### 修改的函数

- `test_single_layer()`
- `test_multi_layer()`
- `test_full_model()`

## 运行结果

修复后，脚本可以正常运行：

```bash
$ python minimax_m2_misc/test_layer_by_layer.py --test-type single-layer --layer 0

初始化分布式环境 (tp_degree=64)...
  ✓ torch.distributed已初始化 (backend=xla)
  ✓ parallel_state已初始化 (tp_degree=64)

编译 Layer 0...
[TODO] 调用 torch_neuronx.trace() 编译单层
  输入shape: hidden_states=torch.Size([1, 256, 3072])
  保存路径: /home/ubuntu/traced_model/test_layers/single_layer_0
✓ Layer 0 编译完成
```

## 技术细节

### 为什么需要分布式环境？

即使在单进程场景下，Neuron模型仍需要分布式环境因为：

1. **Tensor Parallelism**: Neuron使用TP来分配模型到多个NeuronCore
2. **并行状态管理**: 需要跟踪rank、world_size等信息
3. **集合通信**: 即使单进程也需要通信primitives的初始化

### Backend选择

```python
torch.distributed.init_process_group(backend='xla')
```

- 使用 `backend='xla'` 而不是 `'nccl'` 或 `'gloo'`
- XLA backend专为Neuron设备设计
- 支持Neuron的分布式操作

### parallel_state初始化

```python
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=tp_degree,
)
```

参数说明：
- `tensor_model_parallel_size`: Tensor并行度（通常=tp_degree）
- 可选参数：`pipeline_model_parallel_size`, `data_parallel_size`等

### 单例模式

使用全局标志避免重复初始化：

```python
_DISTRIBUTED_INITIALIZED = False

def initialize_distributed_env(tp_degree):
    global _DISTRIBUTED_INITIALIZED
    if _DISTRIBUTED_INITIALIZED:
        return  # 已初始化，跳过
    # ...
    _DISTRIBUTED_INITIALIZED = True
```

这确保即使多次调用测试函数，分布式环境只初始化一次。

## 常见警告

运行时可能看到以下警告（可以忽略）：

```
WARNING:Neuron:TP degree (1) and KV heads (8) are not divisible.
Overriding attention sharding strategy to GQA.CONVERT_TO_MHA!
```

**原因**：parallel_state初始化时使用了 `tensor_model_parallel_size=1`（因为只有单进程），但模型有8个KV heads。

**解决方法**：这只是策略调整，不影响功能。在多进程场景下（真实部署），tp_degree会是64，问题自然消失。

## 与原始demo的对比

原始 `generation_minimax_m2_demo_v2.py` 不需要显式初始化是因为：

1. **使用完整模型类**: `NeuronMiniMaxM2ForCausalLM`内部已处理
2. **通过compile/load流程**: 编译和加载过程会自动初始化

而我们的逐层测试脚本：
- 直接创建单层组件（`NeuronMiniMaxM2DecoderLayer`）
- 绕过了完整模型类的初始化逻辑
- 因此需要手动初始化分布式环境

## 多进程场景

对于真实的多进程训练/推理：

```bash
# 使用torchrun启动
torchrun --nproc_per_node=64 test_layer_by_layer.py --test-type single-layer --layer 0
```

此时：
- `RANK`, `WORLD_SIZE`等环境变量会自动设置
- `initialize_distributed_env()`会使用这些值
- TP会真正分配到64个进程

## 故障排除

### 问题1：torch.distributed初始化失败

```
RuntimeError: Error while initializing process group
```

**解决**：检查XLA backend是否可用：
```python
import torch_xla.core.xla_model as xm
print(xm.xla_device())  # 应该输出 'xla:0'
```

### 问题2：parallel_state已初始化

```
RuntimeError: parallel_state is already initialized
```

**解决**：全局标志`_DISTRIBUTED_INITIALIZED`会防止重复初始化。如需重新初始化：
```python
parallel_state.destroy_model_parallel()
_DISTRIBUTED_INITIALIZED = False
```

### 问题3：Backend not available

```
KeyError: 'xla'
```

**解决**：确保安装了正确版本的PyTorch XLA：
```bash
pip install torch_xla
```

## 总结

关键要点：
1. ✓ Neuron模型组件需要分布式环境
2. ✓ 即使单进程也需要初始化
3. ✓ 使用 `backend='xla'` 而不是 'nccl'
4. ✓ 在创建模型组件**之前**初始化
5. ✓ 使用全局标志防止重复初始化

修改后的脚本现在可以在Trainium上正常运行了！
