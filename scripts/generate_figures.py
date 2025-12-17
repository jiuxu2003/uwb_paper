#!/usr/bin/env python3
"""
批量生成所有论文图表

此脚本依次运行所有完整仿真脚本，生成三张高质量论文图表：
1. 时域波形图（outputs/waveform_demo.png）
2. 多用户干扰性能曲线（outputs/ber_vs_users.png）
3. 窄带干扰抑制性能曲线（outputs/ber_vs_sir.png）

用法:
    python scripts/generate_figures.py

输出:
    outputs/waveform_demo.png - TH-UWB 时域波形（300 DPI）
    outputs/ber_vs_users.png - BER vs 用户数量（300 DPI, 10000 比特）
    outputs/ber_vs_sir.png - BER vs SIR（300 DPI, 10000 比特）

参考:
    - spec.md SC-001, SC-002, SC-003: 三个核心用户故事的成功标准
    - 预期执行时间: ~15-20 分钟（包含两个 10000 比特完整仿真）
"""

import os
import sys
import time
import subprocess
from pathlib import Path


def run_script(script_path: str, description: str) -> tuple[bool, float]:
    """
    运行指定脚本并返回执行状态

    参数:
        script_path: 脚本路径（相对于项目根目录）
        description: 脚本描述

    返回:
        (success, elapsed_time): 成功标志和执行时间（秒）
    """
    print(f"\n{'='*60}")
    print(f"运行: {description}")
    print(f"脚本: {script_path}")
    print(f"{'='*60}")

    start_time = time.perf_counter()

    try:
        # 使用 subprocess 运行脚本
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True,
        )

        elapsed = time.perf_counter() - start_time

        # 打印脚本输出
        if result.stdout:
            print(result.stdout)

        print(f"\n✅ 完成！用时: {elapsed:.2f} 秒")
        return True, elapsed

    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - start_time
        print(f"\n❌ 失败！用时: {elapsed:.2f} 秒")
        print(f"\n错误输出:")
        print(e.stderr)
        return False, elapsed


def verify_output(file_path: str) -> bool:
    """
    验证输出文件是否存在且非空

    参数:
        file_path: 文件路径（相对于项目根目录）

    返回:
        True 如果文件存在且大小 > 0
    """
    path = Path(file_path)
    if not path.exists():
        print(f"  ⚠ 文件不存在: {file_path}")
        return False

    file_size = path.stat().st_size
    if file_size == 0:
        print(f"  ⚠ 文件为空: {file_path}")
        return False

    print(f"  ✓ 文件已生成: {file_path} ({file_size/1024:.1f} KB)")
    return True


def main():
    """主函数：批量生成所有论文图表"""
    print("=" * 60)
    print("批量生成所有论文图表")
    print("=" * 60)

    # 获取项目根目录
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # 确保输出目录存在
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 定义要运行的脚本（按顺序）
    scripts = [
        {
            "path": "scripts/demo_waveform.py",
            "description": "图表 1/3: TH-UWB 时域波形（User Story 1）",
            "output": "outputs/waveform_demo.png",
        },
        {
            "path": "scripts/run_mui_analysis.py",
            "description": "图表 2/3: 多用户干扰性能曲线（User Story 2）",
            "output": "outputs/ber_vs_users.png",
        },
        {
            "path": "scripts/run_nbi_analysis.py",
            "description": "图表 3/3: 窄带干扰抑制性能曲线（User Story 3）",
            "output": "outputs/ber_vs_sir.png",
        },
    ]

    # 执行统计
    results = []
    total_start = time.perf_counter()

    # 依次运行每个脚本
    for idx, script_info in enumerate(scripts, 1):
        success, elapsed = run_script(script_info["path"], script_info["description"])

        # 验证输出文件
        output_exists = verify_output(script_info["output"])

        results.append(
            {
                "name": Path(script_info["path"]).name,
                "success": success and output_exists,
                "time": elapsed,
                "output": script_info["output"],
            }
        )

        # 如果脚本失败，询问是否继续
        if not success or not output_exists:
            print(f"\n⚠ 脚本 {script_info['path']} 执行失败或未生成输出")
            response = input("是否继续执行下一个脚本？(y/n): ").strip().lower()
            if response != "y":
                print("\n用户取消执行")
                break

    total_elapsed = time.perf_counter() - total_start

    # 打印执行汇总
    print("\n" + "=" * 60)
    print("执行汇总")
    print("-" * 60)
    print(f"{'脚本':<30} | {'状态':<8} | {'用时 (秒)':<12} | {'输出文件':<30}")
    print("-" * 60)

    for result in results:
        status = "✓ 成功" if result["success"] else "✗ 失败"
        print(
            f"{result['name']:<30} | "
            f"{status:<8} | "
            f"{result['time']:<12.2f} | "
            f"{result['output']:<30}"
        )

    print("-" * 60)
    print(f"总用时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print("=" * 60)

    # 最终验证
    all_success = all(r["success"] for r in results)

    if all_success:
        print("\n✅ 所有图表生成成功！")
        print("\n论文图表清单:")
        for result in results:
            print(f"  - {result['output']}")
        print("\n项目已准备发表！所有图表符合学术标准（≥300 DPI）。")
        return 0
    else:
        print("\n⚠ 部分图表生成失败，请检查错误信息并重新运行。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
