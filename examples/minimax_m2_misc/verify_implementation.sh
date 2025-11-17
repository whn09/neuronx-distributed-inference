#!/bin/bash

# 验证逐层测试实现脚本
# 用于快速验证test_layer_by_layer.py的实现是否正常工作

set -e  # 遇到错误立即退出

echo "=========================================="
echo "验证逐层测试实现"
echo "=========================================="
echo ""

# 进入正确的目录
cd /home/ubuntu/neuronx-distributed-inference/examples

# 测试1: 单层测试（Layer 0）
echo "测试1: 单层测试 (Layer 0)"
echo "------------------------------------------"
python minimax_m2_misc/test_layer_by_layer.py --test-type single-layer --layer 0

if [ $? -eq 0 ]; then
    echo "✓ 单层测试成功"
else
    echo "✗ 单层测试失败"
    exit 1
fi

echo ""
echo ""

# 检查输出文件是否存在
OUTPUT_DIR="/home/ubuntu/traced_model/test_layers/single_layer_0"
if [ -f "$OUTPUT_DIR/layer_output.pt" ] && [ -f "$OUTPUT_DIR/config.pt" ]; then
    echo "✓ 输出文件已生成:"
    ls -lh "$OUTPUT_DIR/"
else
    echo "✗ 输出文件未生成"
    exit 1
fi

echo ""
echo ""

# 测试2: 加载已保存的输出
echo "测试2: 加载已保存的输出"
echo "------------------------------------------"
python minimax_m2_misc/test_layer_by_layer.py --test-type single-layer --layer 0 --skip-compile

if [ $? -eq 0 ]; then
    echo "✓ 加载输出成功"
else
    echo "✗ 加载输出失败"
    exit 1
fi

echo ""
echo ""

# 测试3: 验证输出内容
echo "测试3: 验证输出内容"
echo "------------------------------------------"
python -c "
import torch
import sys

output_path = '/home/ubuntu/traced_model/test_layers/single_layer_0/layer_output.pt'
try:
    data = torch.load(output_path)

    # 检查必需的字段
    required_fields = ['layer_idx', 'config', 'output_hidden_states', 'statistics']
    for field in required_fields:
        if field not in data:
            print(f'✗ 缺少字段: {field}')
            sys.exit(1)

    print('✓ 所有必需字段都存在')

    # 检查统计信息
    stats = data['statistics']
    print(f'  Layer {data[\"layer_idx\"]} 统计信息:')
    print(f'    Mean: {stats[\"mean\"]:.6f}')
    print(f'    Std: {stats[\"std\"]:.6f}')
    print(f'    Has NaN: {stats[\"has_nan\"]}')
    print(f'    Has Inf: {stats[\"has_inf\"]}')

    # 检查是否有异常值
    if stats['has_nan'] or stats['has_inf']:
        print('⚠️  警告: 输出包含NaN或Inf')
        sys.exit(1)

    print('✓ 输出数值正常')

except Exception as e:
    print(f'✗ 加载或验证失败: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✓ 输出验证成功"
else
    echo "✗ 输出验证失败"
    exit 1
fi

echo ""
echo ""

# 测试4: 多层测试（可选，比较慢）
echo "测试4: 多层测试 (Layers 0-1, 快速验证)"
echo "------------------------------------------"
python minimax_m2_misc/test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 1

if [ $? -eq 0 ]; then
    echo "✓ 多层测试成功"
else
    echo "✗ 多层测试失败"
    exit 1
fi

echo ""
echo ""

# 最终总结
echo "=========================================="
echo "所有验证测试通过! ✓"
echo "=========================================="
echo ""
echo "实现已就绪，可以开始实际调试工作："
echo ""
echo "  1. 单层测试:"
echo "     python minimax_m2_misc/test_layer_by_layer.py --test-type single-layer --layer <N>"
echo ""
echo "  2. 多层测试:"
echo "     python minimax_m2_misc/test_layer_by_layer.py --test-type multi-layer --start-layer 0 --end-layer 5"
echo ""
echo "  3. 与GPU输出对比:"
echo "     python compare_gpu_neuron_outputs.py --gpu-path <GPU_PATH> --neuron-path <NEURON_PATH> --layer layer_0"
echo ""
echo "详细文档请参考: minimax_m2_misc/IMPLEMENTATION_UPDATE.md"
echo ""
