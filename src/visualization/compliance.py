"""
FCC 合规性可视化模块

本模块提供 PSD 与 FCC 掩蔽罩对比可视化功能。

功能：
1. 绘制脉冲 PSD 与 FCC 掩蔽罩对比图
2. 标注合规性状态和关键指标
3. 输出符合 IEEE 学术标准的高清图表（≥300 DPI）
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

from src.visualization import config


def plot_psd_fcc_compliance(
    psd_values: np.ndarray,
    fcc_mask: np.ndarray,
    frequencies: np.ndarray,
    compliance_result: Dict,
    title: str = "UWB 脉冲 PSD 与 FCC 室内掩蔽罩对比",
    output_path: Optional[str] = None
) -> None:
    """
    绘制脉冲 PSD 与 FCC 掩蔽罩对比图

    参数:
        psd_values: 脉冲 PSD（dBm/MHz），1D 数组
        fcc_mask: FCC 掩蔽罩（dBm/MHz），1D 数组
        frequencies: 频率轴（Hz），1D 数组
        compliance_result: 合规性结果字典，包含：
            - status: "Compliant" | "Non-compliant" | "Marginal Compliant"
            - max_excess: 最大超限量（dB）
            - num_violations: 违规频点数量
            - compliance_percentage: 合规频点百分比
        title: 图表标题，默认 "UWB 脉冲 PSD 与 FCC 室内掩蔽罩对比"
        output_path: 输出路径（PNG格式），如果为 None 则仅显示不保存

    前置条件:
        - len(psd_values) == len(fcc_mask) == len(frequencies)
        - compliance_result 包含所有必需字段

    后置条件:
        - 生成符合 IEEE 学术标准的图表（≥300 DPI）
        - 如果 output_path 不为 None，保存图表到指定路径

    图表规范（参见 research.md）:
        - 字体：Times New Roman + Noto Serif CJK（中文）
        - DPI：≥300（学术出版要求）
        - 尺寸：7x5.25 英寸（IEEE 双栏图）
        - 线条：PSD（蓝色实线），FCC Mask（红色虚线）
        - 网格：显示主网格线（灰色虚线，透明度0.3）
        - 图例：右上角或最佳位置
    """
    # 参数验证
    assert len(psd_values) == len(fcc_mask) == len(frequencies), \
        "psd_values, fcc_mask, frequencies 长度必须一致"
    assert all(key in compliance_result for key in ["status", "max_excess", "num_violations", "compliance_percentage"]), \
        "compliance_result 缺少必需字段"

    # 应用学术标准配置
    config.setup_academic_style()

    # 创建图表
    fig, ax = plt.subplots(figsize=(7, 5.25))

    # 转换频率为 GHz（更易读）
    freq_ghz = frequencies / 1e9

    # 绘制 PSD 曲线（蓝色实线）
    ax.plot(freq_ghz, psd_values,
            color='#0173B2', linewidth=1.5, label='脉冲 PSD',
            zorder=3)

    # 绘制 FCC 掩蔽罩（红色虚线）
    ax.plot(freq_ghz, fcc_mask,
            color='#D55E00', linewidth=2.0, linestyle='--', label='FCC 室内掩蔽罩',
            zorder=2)

    # 设置坐标轴标签
    ax.set_xlabel('频率 (GHz)', fontsize=12, fontweight='normal')
    ax.set_ylabel('功率谱密度 (dBm/MHz)', fontsize=12, fontweight='normal')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # 设置坐标轴范围
    ax.set_xlim(0, 12)  # 0-12 GHz

    # Y轴范围：根据PSD和FCC mask的范围自动调整
    y_min = min(np.min(psd_values), np.min(fcc_mask)) - 10
    y_max = max(np.max(psd_values), np.max(fcc_mask)) + 10
    ax.set_ylim(y_min, y_max)

    # 显示网格
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.3, zorder=1)

    # 添加图例
    ax.legend(loc='best', fontsize=10, framealpha=0.9, edgecolor='gray')

    # 标注合规性状态（文本框）
    status_color = {
        "Compliant": "green",
        "Non-compliant": "red",
        "Marginal Compliant": "orange"
    }.get(compliance_result["status"], "black")

    # 状态名称中英文映射
    status_name_cn = {
        "Compliant": "合规",
        "Non-compliant": "不合规",
        "Marginal Compliant": "临界合规"
    }.get(compliance_result["status"], compliance_result["status"])

    status_text = (
        f"状态: {status_name_cn}\n"
        f"最大超限量: {compliance_result['max_excess']:.2f} dB\n"
        f"违规频点数: {compliance_result['num_violations']}\n"
        f"合规百分比: {compliance_result['compliance_percentage']:.1f}%"
    )

    # 文本框位置：右下角
    ax.text(0.98, 0.02, status_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor=status_color, linewidth=2),
            zorder=4)

    # 调整布局（避免标签被裁剪）
    plt.tight_layout()

    # 保存或显示图表
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {output_path}")
    else:
        plt.show()

    # 关闭图表（释放内存）
    plt.close(fig)
