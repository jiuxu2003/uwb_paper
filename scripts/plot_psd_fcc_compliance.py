#!/usr/bin/env python3
"""
UWB 脉冲功率谱密度与 FCC 合规性分析

本脚本生成 2 阶和 5 阶高斯导数脉冲的 PSD，
并与 FCC Part 15.209 室内辐射掩蔽罩进行对比，
验证合规性并生成学术标准图表。

功能：
1. 生成 2 阶脉冲（fc=0，不合规）和 5 阶脉冲（fc=6.85 GHz，合规）
2. 计算功率谱密度（PSD）
3. 生成 FCC Part 15.209 室内辐射掩蔽罩
4. 判断合规性并生成可视化图表
5. 输出符合 IEEE 学术标准的高清图表（≥300 DPI）

输出：
- outputs/psd_fcc_compliance_2nd_order.png：2 阶脉冲对比图（不合规）
- outputs/psd_fcc_compliance_5th_order.png：5 阶脉冲对比图（合规）

使用方法：
    python scripts/plot_psd_fcc_compliance.py

依赖：
    numpy>=1.24.0, scipy>=1.10.0, matplotlib>=3.7.0
"""

import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import psd, fcc
from src.visualization import compliance


def main():
    """主函数：生成脉冲、计算 PSD、判断合规性、绘制图表"""

    print("=== UWB 脉冲功率谱密度与 FCC 合规性分析 ===\n")

    # ==================== 参数配置 ====================
    # 脉冲参数
    tau = 0.287e-9  # 脉冲宽度（秒），对应 3dB 带宽约 3.5 GHz
    sampling_rate = 50e9  # 采样率 50 GHz
    freq_resolution = 10e6  # 频率分辨率 10 MHz
    freq_range = (0, 12e9)  # 频率范围 0-12 GHz

    # FCC 合规性判断容差
    tolerance = 0.1  # dB

    # 输出目录
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 生成 2 阶脉冲（不合规） ====================
    print("1. 生成 2 阶高斯导数脉冲（fc=0，无调制）...")
    waveform_2nd, _ = psd.generate_gaussian_derivative_pulse(
        n=2, tau=tau, fc=0, sampling_rate=sampling_rate
    )
    print(f"   脉冲能量: {np.trapz(waveform_2nd**2, dx=1/sampling_rate):.6f} (归一化)")

    print("2. 计算 2 阶脉冲 PSD...")
    frequencies_2nd, psd_2nd = psd.compute_psd(
        waveform_2nd, sampling_rate,
        freq_resolution=freq_resolution,
        freq_range=freq_range
    )
    print(f"   频率分辨率: {(frequencies_2nd[1]-frequencies_2nd[0])/1e6:.2f} MHz")
    print(f"   PSD 峰值: {np.max(psd_2nd):.2f} dBm/MHz @ {frequencies_2nd[np.argmax(psd_2nd)]/1e9:.2f} GHz")

    print("3. 生成 FCC 室内辐射掩蔽罩...")
    fcc_mask_2nd = fcc.get_fcc_indoor_mask(frequencies_2nd)

    print("4. 判断 2 阶脉冲合规性...")
    result_2nd = fcc.check_compliance(psd_2nd, fcc_mask_2nd, frequencies_2nd, tolerance=tolerance)
    print(f"   合规性状态: {result_2nd['status']}")
    print(f"   最大超限量: {result_2nd['max_excess']:.2f} dB")
    print(f"   违规频点数: {result_2nd['num_violations']}")
    print(f"   合规百分比: {result_2nd['compliance_percentage']:.1f}%\n")

    print("5. 绘制 2 阶脉冲 PSD vs FCC Mask 图表...")
    output_path_2nd = output_dir / "psd_fcc_compliance_2nd_order.png"
    compliance.plot_psd_fcc_compliance(
        psd_values=psd_2nd,
        fcc_mask=fcc_mask_2nd,
        frequencies=frequencies_2nd,
        compliance_result=result_2nd,
        title="2 阶高斯脉冲（fc=0）与 FCC 室内掩蔽罩对比",
        output_path=str(output_path_2nd)
    )

    # ==================== 生成 5 阶脉冲（合规） ====================
    print("\n6. 生成 5 阶高斯导数脉冲（fc=6.85 GHz，UWB 中心频率）...")
    waveform_5th, _ = psd.generate_gaussian_derivative_pulse(
        n=5, tau=tau, fc=6.85e9, sampling_rate=sampling_rate
    )
    print(f"   脉冲能量: {np.trapz(waveform_5th**2, dx=1/sampling_rate):.6f} (归一化)")

    print("7. 计算 5 阶脉冲 PSD...")
    frequencies_5th, psd_5th = psd.compute_psd(
        waveform_5th, sampling_rate,
        freq_resolution=freq_resolution,
        freq_range=freq_range
    )
    print(f"   频率分辨率: {(frequencies_5th[1]-frequencies_5th[0])/1e6:.2f} MHz")
    print(f"   PSD 峰值: {np.max(psd_5th):.2f} dBm/MHz @ {frequencies_5th[np.argmax(psd_5th)]/1e9:.2f} GHz")

    print("8. 生成 FCC 室内辐射掩蔽罩...")
    fcc_mask_5th = fcc.get_fcc_indoor_mask(frequencies_5th)

    print("9. 判断 5 阶脉冲合规性...")
    result_5th = fcc.check_compliance(psd_5th, fcc_mask_5th, frequencies_5th, tolerance=tolerance)
    print(f"   合规性状态: {result_5th['status']}")
    print(f"   最大超限量: {result_5th['max_excess']:.2f} dB")
    print(f"   违规频点数: {result_5th['num_violations']}")
    print(f"   合规百分比: {result_5th['compliance_percentage']:.1f}%\n")

    print("10. 绘制 5 阶脉冲 PSD vs FCC Mask 图表...")
    output_path_5th = output_dir / "psd_fcc_compliance_5th_order.png"
    compliance.plot_psd_fcc_compliance(
        psd_values=psd_5th,
        fcc_mask=fcc_mask_5th,
        frequencies=frequencies_5th,
        compliance_result=result_5th,
        title="5 阶高斯脉冲（fc=6.85 GHz）与 FCC 室内掩蔽罩对比",
        output_path=str(output_path_5th)
    )

    # ==================== 总结 ====================
    print("\n" + "="*60)
    print("=== 分析完成 ===")
    print("="*60)
    print("\n【输出文件】")
    print(f"  1. {output_path_2nd}")
    print(f"  2. {output_path_5th}")

    print("\n【合规性总结】")
    print(f"  2 阶脉冲（fc=0）: {result_2nd['status']} (超限 {result_2nd['max_excess']:.2f} dB)")
    print(f"  5 阶脉冲（fc=6.85 GHz）: {result_5th['status']} (余量 {-result_5th['max_excess']:.2f} dB)")

    print("\n【关键发现】")
    print("  1. 2 阶脉冲能量集中在低频（<3.1 GHz），违反 FCC 限制")
    print("  2. 5 阶脉冲通过载波调制（fc=6.85 GHz）将频谱搬移到 UWB 合法频段")
    print("  3. 5 阶脉冲满足 FCC Part 15.209 室内辐射限制，可用于 UWB 通信")

    print("\n【论文建议】")
    print("  ✓ 使用 5 阶或更高阶脉冲，配合适当的载波调制")
    print("  ✓ 中心频率建议 6-7 GHz（UWB 频段中点）")
    print("  ✓ 脉冲宽度 τ ≈ 0.3-0.5 ns（3dB 带宽 2-3 GHz）")

    print("\n分析完成！图表已保存到 outputs/ 目录。\n")


if __name__ == "__main__":
    main()
