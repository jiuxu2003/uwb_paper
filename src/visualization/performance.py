"""
性能曲线可视化模块

提供 BER vs 用户数量、BER vs 信干比等性能分析图表生成功能（300 DPI）。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Optional
from pathlib import Path

# 导入学术标准配置（与 compliance.py 保持一致）
from src.visualization import config


def plot_ber_vs_users(
    user_counts: np.ndarray,
    ber_values: np.ndarray,
    snr_db: float = 10.0,
    title: str = "多用户干扰性能分析",
    xlabel: str = "用户数量",
    ylabel: str = "误码率 (BER)",
    save_path: Optional[str] = None,
    show: bool = True,
    show_title: bool = False,
    figsize: tuple = (8, 6),
) -> None:
    """
    生成 BER vs 用户数量性能曲线

    展示多用户干扰（MUI）对系统误码率的影响。横轴为用户数量（线性），纵轴为
    误码率（对数坐标），用于验证 BER 随用户数量单调上升的趋势（SC-002）。

    参数:
        user_counts: 用户数量数组，shape (N,)，例如 [1, 2, 3, 5, 7, 10]
        ber_values: 对应的 BER 值数组，shape (N,)，范围 [0, 1]
        snr_db: 信噪比（dB），用于图例显示，默认 10.0
        title: 图表标题
        xlabel: 横轴标签
        ylabel: 纵轴标签
        save_path: 保存路径（例如 "outputs/ber_vs_users.png"），None 表示不保存
        show: 是否显示图表窗口，默认 True（无头环境应设为 False）
        figsize: 图表尺寸（英寸），默认 (8, 6)

    前置条件:
        - len(user_counts) == len(ber_values)
        - user_counts 为递增序列
        - ber_values 范围 [0, 1]

    后置条件:
        - 生成 ≥300 DPI 的 PNG 图表（如果 save_path 非空）
        - 图表包含网格、轴标签、图例、标题
        - 纵轴使用对数坐标

    异常:
        ValueError: 如果输入数组长度不匹配或数据范围非法

    示例:
        >>> user_counts = np.array([1, 2, 3, 5, 7, 10])
        >>> ber_values = np.array([1e-3, 3e-3, 8e-3, 2e-2, 5e-2, 1e-1])
        >>> plot_ber_vs_users(
        ...     user_counts, ber_values,
        ...     snr_db=10.0,
        ...     save_path="outputs/ber_vs_users_demo.png",
        ...     show=False
        ... )

    参考:
        - research.md Section 6.2: BER vs 用户数量图表规范
        - tasks.md T036: plot_ber_vs_users() 实现要求
        - spec.md SC-002: 多用户干扰性能分析目标
    """
    # 验证输入
    if len(user_counts) != len(ber_values):
        raise ValueError(
            f"user_counts 和 ber_values 长度必须一致，"
            f"len(user_counts)={len(user_counts)}, len(ber_values)={len(ber_values)}"
        )

    if np.any(ber_values < 0) or np.any(ber_values > 1):
        raise ValueError(f"ber_values 必须在 [0, 1] 范围内")

    if len(user_counts) == 0:
        raise ValueError("user_counts 和 ber_values 不能为空")

    # 应用学术标准配置（与 compliance.py 保持一致）
    config.setup_academic_style()

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制 BER 曲线（对数坐标）
    ax.semilogy(
        user_counts,
        ber_values,
        "o-",  # 圆圈标记 + 实线
        linewidth=2,
        markersize=8,
        label=f"SNR={snr_db:.0f}dB",
        color="#1f77b4",  # Matplotlib 默认蓝色（colorblind-friendly）
    )

    # 设置轴标签和标题
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if show_title:
        ax.set_title(title, fontsize=14)

    # 启用网格（对数坐标需要 both 参数）
    ax.grid(True, which="both", alpha=0.3, linestyle="--")

    # 添加图例
    ax.legend(loc="best", fontsize=11)

    # 调整布局（避免标签被裁剪）
    plt.tight_layout()

    # 保存图表
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 图表已保存到 {save_path}")

    # 显示图表
    if show:
        plt.show()
    else:
        plt.close()


def plot_ber_vs_sir(
    sir_db: np.ndarray,
    ber_values: np.ndarray,
    title: str = "窄带干扰抑制性能分析",
    xlabel: str = "信干比 SIR (dB)",
    ylabel: str = "误码率 (BER)",
    save_path: Optional[str] = None,
    show: bool = True,
    show_title: bool = False,
    figsize: tuple = (8, 6),
) -> None:
    """
    生成 BER vs 信干比（SIR）性能曲线

    展示窄带干扰（NBI）对系统误码率的影响。横轴为 SIR（dB），纵轴为误码率
    （对数坐标），用于验证 UWB 抗干扰能力（SC-003）。

    参数:
        sir_db: 信干比数组（dB），shape (N,)，例如 [30, 20, 10, 0, -10]
        ber_values: 对应的 BER 值数组，shape (N,)，范围 [0, 1]
        title: 图表标题
        xlabel: 横轴标签
        ylabel: 纵轴标签
        save_path: 保存路径（例如 "outputs/ber_vs_sir.png"），None 表示不保存
        show: 是否显示图表窗口，默认 True（无头环境应设为 False）
        figsize: 图表尺寸（英寸），默认 (8, 6)

    前置条件:
        - len(sir_db) == len(ber_values)
        - ber_values 范围 [0, 1]

    后置条件:
        - 生成 ≥300 DPI 的 PNG 图表（如果 save_path 非空）
        - 图表包含网格、轴标签、图例、标题
        - 纵轴使用对数坐标

    异常:
        ValueError: 如果输入数组长度不匹配或数据范围非法

    示例:
        >>> sir_db = np.array([30, 20, 10, 0, -10])
        >>> ber_values = np.array([1e-4, 3e-4, 1e-3, 5e-3, 2e-2])
        >>> plot_ber_vs_sir(
        ...     sir_db, ber_values,
        ...     save_path="outputs/ber_vs_sir_demo.png",
        ...     show=False
        ... )

    参考:
        - research.md Section 6.2: BER vs SIR 图表规范
        - tasks.md T045: plot_ber_vs_sir() 实现要求（Phase 5 - US3）
        - spec.md SC-003: 窄带干扰抑制性能分析目标
    """
    # 验证输入
    if len(sir_db) != len(ber_values):
        raise ValueError(
            f"sir_db 和 ber_values 长度必须一致，"
            f"len(sir_db)={len(sir_db)}, len(ber_values)={len(ber_values)}"
        )

    if np.any(ber_values < 0) or np.any(ber_values > 1):
        raise ValueError(f"ber_values 必须在 [0, 1] 范围内")

    if len(sir_db) == 0:
        raise ValueError("sir_db 和 ber_values 不能为空")

    # 应用学术标准配置（与 compliance.py 保持一致）
    config.setup_academic_style()

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制 BER 曲线（对数坐标，红色方块标记）
    ax.semilogy(
        sir_db,
        ber_values,
        "s-",  # 方块标记 + 实线
        linewidth=2,
        markersize=8,
        label="单用户",
        color="red",  # 任务规范要求使用红色
    )

    # 设置轴标签和标题
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if show_title:
        ax.set_title(title, fontsize=14)

    # 启用网格（对数坐标需要 both 参数）
    ax.grid(True, which="both", alpha=0.3, linestyle="--")

    # 添加图例
    ax.legend(loc="best", fontsize=11)

    # 调整布局（避免标签被裁剪）
    plt.tight_layout()

    # 保存图表
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 图表已保存到 {save_path}")

    # 显示图表
    if show:
        plt.show()
    else:
        plt.close()
