"""
对比GPU和Neuron版本的输出差异

用于定位模型推理问题：
1. 加载GPU golden reference输出
2. 加载Neuron版本输出
3. 逐层对比差异
4. 生成详细的差异报告

使用方法：
python compare_gpu_neuron_outputs.py --gpu-path <gpu_output_dir> --neuron-path <neuron_output_dir>
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import json


def load_layer_output(output_path: Path, layer_name: str) -> torch.Tensor:
    """加载单层输出"""
    file_path = output_path / f"{layer_name}_output.pt"
    if not file_path.exists():
        raise FileNotFoundError(f"未找到输出文件: {file_path}")
    return torch.load(file_path)


def load_metadata(output_path: Path) -> Dict:
    """加载元数据"""
    metadata_path = output_path / "metadata.pt"
    if metadata_path.exists():
        return torch.load(metadata_path)
    return {}


def compute_difference_metrics(
    gpu_output: torch.Tensor,
    neuron_output: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> Dict:
    """计算两个输出之间的差异指标"""

    # 确保两个tensor在CPU上且为float32进行精确计算
    gpu_output = gpu_output.cpu().float()
    neuron_output = neuron_output.cpu().float()

    # 基本检查
    metrics = {
        'gpu_shape': list(gpu_output.shape),
        'neuron_shape': list(neuron_output.shape),
        'shape_match': gpu_output.shape == neuron_output.shape,
    }

    if not metrics['shape_match']:
        metrics['error'] = "Shape不匹配"
        return metrics

    # 计算差异
    diff = gpu_output - neuron_output
    abs_diff = torch.abs(diff)
    rel_diff = abs_diff / (torch.abs(gpu_output) + 1e-8)

    # 统计指标
    metrics.update({
        'max_abs_diff': float(abs_diff.max()),
        'mean_abs_diff': float(abs_diff.mean()),
        'median_abs_diff': float(abs_diff.median()),
        'std_abs_diff': float(abs_diff.std()),

        'max_rel_diff': float(rel_diff.max()),
        'mean_rel_diff': float(rel_diff.mean()),
        'median_rel_diff': float(rel_diff.median()),

        'gpu_mean': float(gpu_output.mean()),
        'gpu_std': float(gpu_output.std()),
        'gpu_min': float(gpu_output.min()),
        'gpu_max': float(gpu_output.max()),

        'neuron_mean': float(neuron_output.mean()),
        'neuron_std': float(neuron_output.std()),
        'neuron_min': float(neuron_output.min()),
        'neuron_max': float(neuron_output.max()),

        'gpu_has_nan': bool(torch.isnan(gpu_output).any()),
        'gpu_has_inf': bool(torch.isinf(gpu_output).any()),
        'neuron_has_nan': bool(torch.isnan(neuron_output).any()),
        'neuron_has_inf': bool(torch.isinf(neuron_output).any()),
    })

    # 使用torch.allclose检查
    metrics['allclose'] = bool(torch.allclose(
        gpu_output, neuron_output, rtol=rtol, atol=atol
    ))

    # 计算相似度百分比
    close_elements = torch.isclose(
        gpu_output, neuron_output, rtol=rtol, atol=atol
    )
    metrics['similarity_percentage'] = float(close_elements.float().mean() * 100)

    # 余弦相似度
    gpu_flat = gpu_output.flatten()
    neuron_flat = neuron_output.flatten()
    cosine_sim = torch.nn.functional.cosine_similarity(
        gpu_flat.unsqueeze(0),
        neuron_flat.unsqueeze(0)
    )
    metrics['cosine_similarity'] = float(cosine_sim)

    # Pearson相关系数
    gpu_centered = gpu_flat - gpu_flat.mean()
    neuron_centered = neuron_flat - neuron_flat.mean()
    pearson = (gpu_centered * neuron_centered).sum() / (
        gpu_centered.norm() * neuron_centered.norm() + 1e-8
    )
    metrics['pearson_correlation'] = float(pearson)

    return metrics


def compare_single_layer(
    gpu_path: Path,
    neuron_path: Path,
    layer_name: str,
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> Dict:
    """对比单层输出"""

    print(f"\n对比层: {layer_name}")
    print(f"  GPU路径: {gpu_path}")
    print(f"  Neuron路径: {neuron_path}")

    try:
        # 加载输出
        gpu_output = load_layer_output(gpu_path, layer_name)
        neuron_output = load_layer_output(neuron_path, layer_name)

        print(f"  ✓ GPU输出 shape: {gpu_output.shape}, dtype: {gpu_output.dtype}")
        print(f"  ✓ Neuron输出 shape: {neuron_output.shape}, dtype: {neuron_output.dtype}")

        # 计算差异
        metrics = compute_difference_metrics(gpu_output, neuron_output, rtol, atol)

        # 打印关键指标
        print(f"\n  关键指标:")
        print(f"    Shape匹配: {metrics['shape_match']}")
        if metrics['shape_match']:
            print(f"    AllClose (rtol={rtol}, atol={atol}): {metrics['allclose']}")
            print(f"    相似度: {metrics['similarity_percentage']:.2f}%")
            print(f"    余弦相似度: {metrics['cosine_similarity']:.6f}")
            print(f"    Pearson相关: {metrics['pearson_correlation']:.6f}")
            print(f"    最大绝对差异: {metrics['max_abs_diff']:.6e}")
            print(f"    平均绝对差异: {metrics['mean_abs_diff']:.6e}")
            print(f"    最大相对差异: {metrics['max_rel_diff']:.6e}")

            # 异常值检测
            issues = []
            if metrics['gpu_has_nan'] or metrics['neuron_has_nan']:
                issues.append("包含NaN")
            if metrics['gpu_has_inf'] or metrics['neuron_has_inf']:
                issues.append("包含Inf")
            if metrics['max_abs_diff'] > 1.0:
                issues.append(f"大差异 (max={metrics['max_abs_diff']:.2f})")
            if metrics['similarity_percentage'] < 95.0:
                issues.append(f"相似度低 ({metrics['similarity_percentage']:.1f}%)")

            if issues:
                print(f"\n  ⚠️  发现问题: {', '.join(issues)}")
            else:
                print(f"\n  ✓ 输出匹配良好")

        metrics['layer_name'] = layer_name
        return metrics

    except Exception as e:
        print(f"  ❌ 对比失败: {e}")
        return {
            'layer_name': layer_name,
            'error': str(e)
        }


def compare_multiple_layers(
    gpu_path: Path,
    neuron_path: Path,
    layer_pattern: str = "layer_*",
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> List[Dict]:
    """对比多个层的输出"""

    print(f"\n{'='*60}")
    print(f"对比多层输出")
    print(f"{'='*60}")

    # 查找所有GPU输出文件
    gpu_files = sorted(gpu_path.glob(f"{layer_pattern}_output.pt"))
    layer_names = [f.stem.replace("_output", "") for f in gpu_files]

    print(f"\n找到 {len(layer_names)} 个层输出")

    results = []
    for layer_name in layer_names:
        metrics = compare_single_layer(gpu_path, neuron_path, layer_name, rtol, atol)
        results.append(metrics)

    return results


def generate_comparison_report(
    results: List[Dict],
    output_path: Path = None
):
    """生成对比报告"""

    print(f"\n{'='*60}")
    print(f"对比报告摘要")
    print(f"{'='*60}\n")

    # 统计
    total_layers = len(results)
    successful_comparisons = [r for r in results if 'error' not in r]
    failed_comparisons = [r for r in results if 'error' in r]

    print(f"总层数: {total_layers}")
    print(f"成功对比: {len(successful_comparisons)}")
    print(f"失败对比: {len(failed_comparisons)}")

    if failed_comparisons:
        print(f"\n失败的层:")
        for r in failed_comparisons:
            print(f"  {r['layer_name']}: {r['error']}")

    if successful_comparisons:
        # 找出差异最大的层
        sorted_by_diff = sorted(
            successful_comparisons,
            key=lambda x: x.get('max_abs_diff', float('inf')),
            reverse=True
        )

        print(f"\n差异最大的前5层:")
        for i, r in enumerate(sorted_by_diff[:5], 1):
            print(f"  {i}. {r['layer_name']}")
            print(f"     最大绝对差异: {r['max_abs_diff']:.6e}")
            print(f"     相似度: {r['similarity_percentage']:.2f}%")
            print(f"     余弦相似度: {r['cosine_similarity']:.6f}")

        # 找出相似度最低的层
        sorted_by_sim = sorted(
            successful_comparisons,
            key=lambda x: x.get('similarity_percentage', 0)
        )

        print(f"\n相似度最低的前5层:")
        for i, r in enumerate(sorted_by_sim[:5], 1):
            print(f"  {i}. {r['layer_name']}")
            print(f"     相似度: {r['similarity_percentage']:.2f}%")
            print(f"     余弦相似度: {r['cosine_similarity']:.6f}")

        # 整体统计
        all_max_diffs = [r['max_abs_diff'] for r in successful_comparisons]
        all_similarities = [r['similarity_percentage'] for r in successful_comparisons]
        all_cosines = [r['cosine_similarity'] for r in successful_comparisons]

        print(f"\n整体统计:")
        print(f"  最大绝对差异:")
        print(f"    最小: {min(all_max_diffs):.6e}")
        print(f"    最大: {max(all_max_diffs):.6e}")
        print(f"    平均: {np.mean(all_max_diffs):.6e}")
        print(f"    中位数: {np.median(all_max_diffs):.6e}")

        print(f"  相似度 (%):")
        print(f"    最小: {min(all_similarities):.2f}")
        print(f"    最大: {max(all_similarities):.2f}")
        print(f"    平均: {np.mean(all_similarities):.2f}")

        print(f"  余弦相似度:")
        print(f"    最小: {min(all_cosines):.6f}")
        print(f"    最大: {max(all_cosines):.6f}")
        print(f"    平均: {np.mean(all_cosines):.6f}")

    # 保存详细报告
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存JSON报告
        report_file = output_path / "comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ 详细报告已保存到: {report_file}")

        # 保存摘要
        if successful_comparisons:
            summary = {
                'total_layers': total_layers,
                'successful': len(successful_comparisons),
                'failed': len(failed_comparisons),
                'max_abs_diff_stats': {
                    'min': float(min(all_max_diffs)),
                    'max': float(max(all_max_diffs)),
                    'mean': float(np.mean(all_max_diffs)),
                    'median': float(np.median(all_max_diffs)),
                },
                'similarity_stats': {
                    'min': float(min(all_similarities)),
                    'max': float(max(all_similarities)),
                    'mean': float(np.mean(all_similarities)),
                },
                'cosine_similarity_stats': {
                    'min': float(min(all_cosines)),
                    'max': float(max(all_cosines)),
                    'mean': float(np.mean(all_cosines)),
                }
            }

            summary_file = output_path / "comparison_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✓ 摘要已保存到: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="对比GPU和Neuron版本的输出")
    parser.add_argument('--gpu-path', type=str, required=True, help='GPU输出目录')
    parser.add_argument('--neuron-path', type=str, required=True, help='Neuron输出目录')
    parser.add_argument('--layer', type=str, help='指定对比的层名称（例如：layer_0）')
    parser.add_argument('--layer-pattern', type=str, default='layer_*', help='层名称pattern（用于对比多层）')
    parser.add_argument('--rtol', type=float, default=1e-3, help='相对容差')
    parser.add_argument('--atol', type=float, default=1e-3, help='绝对容差')
    parser.add_argument('--report-path', type=str, help='报告输出路径')

    args = parser.parse_args()

    gpu_path = Path(args.gpu_path)
    neuron_path = Path(args.neuron_path)

    if not gpu_path.exists():
        print(f"错误: GPU输出路径不存在: {gpu_path}")
        return

    if not neuron_path.exists():
        print(f"错误: Neuron输出路径不存在: {neuron_path}")
        return

    print(f"\n{'#'*60}")
    print(f"# GPU vs Neuron 输出对比工具")
    print(f"{'#'*60}")

    try:
        if args.layer:
            # 对比单层
            metrics = compare_single_layer(
                gpu_path, neuron_path, args.layer, args.rtol, args.atol
            )
            results = [metrics]
        else:
            # 对比多层
            results = compare_multiple_layers(
                gpu_path, neuron_path, args.layer_pattern, args.rtol, args.atol
            )

        # 生成报告
        generate_comparison_report(results, args.report_path)

        print(f"\n{'='*60}")
        print(f"对比完成!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ 对比失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
