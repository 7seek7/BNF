#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数范围缩小工具

根据优化结果自动缩小参数范围，用于分批优化策略。
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """加载JSON文件"""
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
    """保存JSON文件"""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def refine_parameter_bounds(original_bounds_file: Path,
                           best_params_file: Path,
                           output_file: Path,
                           shrink_factor: float = 0.3) -> Dict[str, Any]:
    """
    根据最佳参数缩小参数范围

    Args:
        original_bounds_file: 原始参数范围JSON文件
        best_params_file: 优化结果JSON文件（包含最佳参数）
        output_file: 输出的缩小后参数范围JSON文件
        shrink_factor: 缩小因子（0.3 = 缩小到±30%宽度）

    Returns:
        缩小后的参数范围字典
    """
    print("=" * 70)
    print("参数范围缩小工具")
    print("=" * 70)

    # 加载原始范围
    print(f"\n加载数据...")
    print(f"  原始范围文件: {original_bounds_file}")
    original_bounds = load_json_file(original_bounds_file)
    print(f"  原始范围参数数量: {len(original_bounds)}")

    # 加载优化结果
    print(f"  优化结果文件: {best_params_file}")
    best_result = load_json_file(best_params_file)
    best_params = best_result.get('params', {})

    if not best_params:
        raise ValueError("优化结果文件中没有找到 'params' 字段")

    print(f"  最佳参数数量: {len(best_params)}")
    if 'fitness' in best_result:
        print(f"  当前最佳 fitness: {best_result['fitness']:.6f}")

    # 缩小范围
    print(f"\n计算缩小范围（缩小因子: {shrink_factor*100:.0f}%）...")
    refined_bounds = {}
    shrink_count = 0
    no_shrink_count = 0

    for param, value in best_params.items():
        if param not in original_bounds:
            print(f"  ⚠️  参数 '{param}' 不在原始范围中，跳过")
            no_shrink_count += 1
            continue

        original_bound = original_bounds[param]
        full_width = original_bound["max"] - original_bound["min"]
        new_width = full_width * shrink_factor

        # 计算新的边界
        new_min = max(original_bound["min"], value - new_width / 2)
        new_max = min(original_bound["max"], value + new_width / 2)

        refined_bounds[param] = {
            "min": round(new_min, 4),
            "max": round(new_max, 4)
        }

        shrink_count += 1

    print(f"  ✅ 已缩小 {shrink_count} 个参数范围")
    print(f"  ⚠️  跳过 {no_shrink_count} 个参数")

    # 保存结果
    save_json_file(refined_bounds, output_file)
    print(f"\n✅ 缩小后的参数范围已保存到: {output_file}")

    # 显示前后对比
    print("\n" + "=" * 70)
    print("参数范围对比")
    print("=" * 70)

    print(f"\n{'参数名':<30} {'原始范围':<20} {'缩小范围':<20} {'宽度变化':<15}")
    print("-" * 100)

    for param in sorted(refined_bounds.keys()):
        orig = original_bounds[param]
        refined = refined_bounds[param]
        orig_width = orig['max'] - orig['min']
        new_width = refined['max'] - refined['min']
        width_change = f"{((new_width/orig_width - 1) * 100):+.0f}%"

        print(f"{param:<30} "
              f"{orig['min']:.2f} - {orig['max']:.2f}   "
              f"{refined['min']:.2f} - {refined['max']:.2f}   "
              f"{width_change:<15}")

    print("\n" + "=" * 70)

    # 统计信息
    print("\n统计信息:")
    print(f"  缩小的参数数量: {shrink_count}")
    print(f"  平均范围缩小: {shrink_factor*100:.0f}%")
    print(f"  保持不变的参数: {no_shrink_count}")

    return refined_bounds

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='根据优化结果自动缩小参数范围',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python refine_bounds.py ../data/parameter_bounds.json ../optimizer_results/best_config.json ../data/refined_bounds.json
  python refine_bounds.py ... --shrink 0.2  # 缩小到±20%
        """
    )

    parser.add_argument('original_bounds', type=Path,
                       help='原始参数范围JSON文件')
    parser.add_argument('best_params', type=Path,
                       help='优化结果JSON文件（包含最佳参数）')
    parser.add_argument('output', type=Path,
                       help='输出文件路径（缩小后的参数范围）')
    parser.add_argument('--shrink', type=float, default=0.3,
                       help='缩小因子（默认: 0.3，即缩小到±30%%宽度）')

    args = parser.parse_args()

    try:
        refined_bounds = refine_parameter_bounds(
            original_bounds_file=args.original_bounds,
            best_params_file=args.best_params,
            output_file=args.output,
            shrink_factor=args.shrink
        )
        print("\n✅ 完成！")
        return 0

    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("请检查文件路径是否正确")
        return 1
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        return 1
    except Exception as e:
        if "JSON" in str(type(e)):
            print(f"\n❌ JSON解析错误: {e}")
            print("请检查文件格式是否正确")
        else:
            print(f"\n❌ 未知错误: {e}")
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
